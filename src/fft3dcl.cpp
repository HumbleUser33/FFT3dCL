// FFT3DCL VapourSynth plugin (API 4) — port of FFT3dGPU v0.8.6 for Avisynth+.
// Namespace: fft3dcl. Function: fft3dcl.FFT3DCL(clip, ...23 params...).
//
// Phase 2: CPU reference backend (fp32). OpenCL backend arrives in phase 3.
// Intentional differences from the original are listed in docs/porting-map.md.

#include "VapourSynth4.h"
#include "VSHelper4.h"

#include "params.h"
#include "cpu/reference.h"
#include "cl/gpu.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

using namespace fft3dcl;

namespace {

// Shared GPU context (one per process), lazily initialized.
GpuContext g_gpu;
std::once_flag g_gpuOnce;
bool g_gpuOk = false;

void initGpuOnce() {
  std::string err;
  g_gpuOk = g_gpu.init(&err);
}

struct FilterInstance {
  ScriptParams sp;
  InternalParams ip; // representative (per-plane grids built in PlaneRef)
  VSVideoInfo vi;
  VSNode* clip = nullptr; // borrowed, owned by the graph

  bool useGpu = false;
  std::vector<int> processedPlanes;
  std::vector<PlaneGeom> geoms;      // per processed plane
  std::vector<InternalParams> ips;   // per processed plane
  std::vector<std::unique_ptr<PlaneRef>> refs;
  std::vector<std::unique_ptr<GpuRef>> gpus; // parallel to refs when useGpu
  std::vector<std::map<int, std::shared_ptr<PlaneRef::Spectra>>> cache;
  std::vector<bool> degridDone;
  std::mutex mtx; // guards cache, refs scratch, degrid init
};

const VSFrame* VS_CC fft3dclGetFrame(int n, int activationReason, void* instanceData,
                                      void** frameData, VSFrameContext* frameCtx, VSCore* core,
                                      const VSAPI* vsapi);
void VS_CC fft3dclFree(void* instanceData, VSCore* core, const VSAPI* vsapi);

// ---- create ---------------------------------------------------------------

void VS_CC fft3dclCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core,
                          const VSAPI* vsapi) {
  int err = 0;
  VSNode* clip = vsapi->mapGetNode(in, "clip", 0, &err);
  if (err || !clip) {
    vsapi->mapSetError(out, "FFT3DCL: a video clip must be specified as the 'clip' argument");
    return;
  }
  const VSVideoInfo* srcvi = vsapi->getVideoInfo(clip);
  if (!vsh::isConstantVideoFormat(srcvi) ||
      (srcvi->format.numPlanes != 1 && srcvi->format.numPlanes != 3)) {
    vsapi->mapSetError(out, "FFT3DCL: only constant-format gray, YUV and planar RGB clips are supported");
    return;
  }
  const VSVideoFormat& f = srcvi->format;
  if (f.sampleType == stFloat && f.bitsPerSample != 32) {
    vsapi->mapSetError(out, "FFT3DCL: only 32-bit float clips supported (16-bit float not ported yet)");
    return;
  }
  if (f.sampleType == stInteger && (f.bitsPerSample < 8 || f.bitsPerSample > 16)) {
    vsapi->mapSetError(out, "FFT3DCL: integer bit depths 8-16 only");
    return;
  }

  ScriptParams sp;
  auto getF = [&](const char* key, float def) {
    int e = 0;
    double v = vsapi->mapGetFloat(in, key, 0, &e);
    return e ? def : (float)v;
  };
  auto getI = [&](const char* key, int64_t def) {
    int e = 0;
    int64_t v = vsapi->mapGetInt(in, key, 0, &e);
    return e ? def : v;
  };

  sp.sigma = getF("sigma", 2.0f);
  sp.beta = getF("beta", 1.0f);
  sp.bw = (int)getI("bw", 32);
  sp.bh = (int)getI("bh", 32);
  sp.bt = (int)getI("bt", 3);
  sp.sharpen = getF("sharpen", 0.0f);
  sp.plane = (int)getI("plane", 0);
  sp.mode = (int)getI("mode", 1);
  sp.bordersize = (int)getI("bordersize", 1);
  sp.precision = (int)getI("precision", 0);
  // degrid default: mode==1 ? 1.0 : 0.0 (FFT3dGPU.cpp:1266-1267)
  sp.degrid = getF("degrid", sp.mode == 1 ? 1.0f : 0.0f);
  sp.scutoff = getF("scutoff", 0.3f);
  sp.svr = getF("svr", 1.0f);
  sp.smin = getF("smin", 4.0f);
  sp.smax = getF("smax", 20.0f);
  sp.kratio = getF("kratio", 2.0f);
  sp.ow = (int)getI("ow", sp.bw / 2);
  sp.oh = (int)getI("oh", sp.bh / 2);
  sp.wintype = (int)getI("wintype", 0);
  sp.interlaced = (int)getI("interlaced", 0);
  sp.sigma2 = getF("sigma2", sp.sigma);
  sp.sigma3 = getF("sigma3", sp.sigma);
  sp.sigma4 = getF("sigma4", sp.sigma);
  sp.oldfft = (int)getI("oldfft", -1);
  // Backend selection (port-only, does not affect denoising):
  //   0 = CPU reference, 1 = OpenCL GPU (default), 2 = GPU with CPU fallback.
  int device = (int)getI("device", 2);

  // Plane semantics (FFT3dGPU.cpp:1234-1256).
  int plane = sp.plane;
  if (f.colorFamily == cfRGB)
    plane = 4;
  if (f.colorFamily == cfGray) {
    if (plane == 1) {
      vsapi->mapSetError(out, "FFT3DCL: cannot set chroma processing for a greyscale clip");
      return;
    }
    if (plane == 4)
      plane = 0;
  }
  if (plane == 2 || plane == 3)
    plane = 1;
  if (plane != 0 && plane != 1 && plane != 4) {
    vsapi->mapSetError(out, "FFT3DCL: plane must be 0 (luma), 1 (chroma) or 4 (all)");
    return;
  }

  std::vector<int> planes;
  if (f.colorFamily == cfGray || plane == 0)
    planes = {0};
  else if (plane == 1)
    planes = {1, 2};
  else
    planes = {0, 1, 2};

  auto inst = std::make_unique<FilterInstance>();
  inst->sp = sp;
  inst->vi = *srcvi;
  inst->clip = clip;

  if (device != 0) {
    std::call_once(g_gpuOnce, initGpuOnce);
    if (!g_gpuOk && device == 1) {
      vsapi->mapSetError(out, "FFT3DCL: no OpenCL GPU device available (device=1)");
      return;
    }
    inst->useGpu = g_gpuOk;
  }

  for (int p : planes) {
    int pw = srcvi->width >> (p ? f.subSamplingW : 0);
    int ph = srcvi->height >> (p ? f.subSamplingH : 0);
    InternalParams ip;
    const char* e = nullptr;
    if (!validateParams(sp, pw, ph, f.bitsPerSample, ip, &e)) {
      vsapi->mapSetError(out, e);
      return;
    }
    inst->ip = ip;
    inst->ips.push_back(ip);
    PlaneGeom g{pw, ph, f.bitsPerSample, f.sampleType, f.colorFamily == cfYUV && p > 0};
    inst->geoms.push_back(g);
    inst->refs.push_back(std::make_unique<PlaneRef>(ip, g));
    inst->cache.emplace_back();
    inst->degridDone.push_back(false);
  }
  if (inst->useGpu) {
    for (size_t k = 0; k < inst->refs.size(); k++) {
      auto gpu = std::make_unique<GpuRef>(g_gpu, inst->ips[k], inst->geoms[k], *inst->refs[k]);
      if (!gpu->ok()) {
        inst->useGpu = false; // fall back to CPU for the whole instance
        break;
      }
      inst->gpus.push_back(std::move(gpu));
    }
    if (!inst->useGpu)
      inst->gpus.clear();
  }
  inst->processedPlanes = planes;

  VSFilterDependency deps[] = {{clip, rpGeneral}};
  // Kalman (bt=0) is stateful: fmFrameState delivers frames in order, one at a
  // time — same sequential semantics as the original MT_SERIALIZED filter.
  int fm = (sp.bt == 0) ? fmFrameState : fmParallelRequests;
  vsapi->createVideoFilter(out, "FFT3DCL", srcvi, fft3dclGetFrame, fft3dclFree, fm,
                           deps, 1, inst.release(), core);
}

// ---- getframe -------------------------------------------------------------

// VapourSynth differs on out-of-range frame requests: newer cores clamp them,
// older ones (R79, shipped by StaxRip) abort the whole process with
// "Negative frame request by: <filter>". Clamp on our side so behaviour is
// identical on every core version.
inline int clampFrame(int n, int numFrames) {
  if (n < 0)
    return 0;
  if (n > numFrames - 1)
    return numFrames - 1;
  return n;
}

const VSFrame* VS_CC fft3dclGetFrame(int n, int activationReason, void* instanceData,
                                      void** frameData, VSFrameContext* frameCtx, VSCore* core,
                                      const VSAPI* vsapi) {
  FilterInstance* inst = static_cast<FilterInstance*>(instanceData);
  const InternalParams& ip = inst->ip;
  const VSVideoInfo& vi = inst->vi;
  VSNode* clip = inst->clip;

  if (activationReason == arInitial) {
    if (ip.bm >= 2) {
      // Request the clamped window as one contiguous range: same set of frames
      // as clamping each index individually, but without duplicate requests.
      const int lo = clampFrame(n - ip.current, vi.numFrames);
      const int hi = clampFrame(n - ip.current + ip.bt - 1, vi.numFrames);
      for (int f = lo; f <= hi; f++)
        vsapi->requestFrameFilter(f, clip, frameCtx);
    } else {
      vsapi->requestFrameFilter(clampFrame(n, vi.numFrames), clip, frameCtx);
    }
    return nullptr;
  }
  if (activationReason != arAllFramesReady)
    return nullptr;

  const VSFrame* src = vsapi->getFrameFilter(n, clip, frameCtx);
  VSFrame* dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, src, core);

  // Untouched planes are copied straight from the source frame.
  for (int p = 0; p < vi.format.numPlanes; p++) {
    bool processed = false;
    for (int q : inst->processedPlanes)
      if (q == p) { processed = true; break; }
    if (processed)
      continue;
    const uint8_t* sp = vsapi->getReadPtr(src, p);
    uint8_t* dp = vsapi->getWritePtr(dst, p);
    ptrdiff_t ss = vsapi->getStride(src, p), ds = vsapi->getStride(dst, p);
    int h = vsapi->getFrameHeight(src, p);
    int row = vsapi->getFrameWidth(src, p) * vi.format.bytesPerSample;
    for (int y = 0; y < h; y++)
      memcpy(dp + y * ds, sp + y * ss, row);
  }

  std::lock_guard<std::mutex> lock(inst->mtx);

  for (size_t k = 0; k < inst->processedPlanes.size(); k++) {
    int p = inst->processedPlanes[k];
    PlaneRef& ref = *inst->refs[k];

    // Degrid: gridsample pattern built once (white-frame FFT, no source frame).
    if (ip.degridOn && !inst->degridDone[k]) {
      ref.buildDegridSample();
      if (inst->useGpu && !inst->gpus[k]->uploadDegridSample(ref))
        inst->useGpu = false; // fall back to CPU
      inst->degridDone[k] = true;
    }

    if (inst->useGpu) {
      GpuRef& gpu = *inst->gpus[k];
      auto ensure = [&](int fn) {
        fn = clampFrame(fn, vi.numFrames);
        if (gpu.has(fn))
          return fn;
        const VSFrame* fr = vsapi->getFrameFilter(fn, clip, frameCtx);
        std::vector<float> plane;
        ref.upload(vsapi->getReadPtr(fr, p), vsapi->getStride(fr, p), plane);
        vsapi->freeFrame(fr);
        gpu.forward(fn, plane);
        return fn;
      };
      std::vector<int> window;
      if (ip.bm >= 2) {
        for (int i = 0; i < ip.bt; i++)
          window.push_back(ensure(n - ip.current + i));
      } else {
        window.push_back(ensure(n));
      }
      std::vector<float> outPlane;
      if (!gpu.process(window, outPlane)) {
        // Driver/OOM failure: disable GPU and fall through to the CPU path.
        inst->useGpu = false;
      } else {
        ref.download(outPlane, vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p));

        if (ip.bm >= 2) {
          int lo = n - ip.current - ip.bt, hi = n - ip.current + 2 * ip.bt;
          gpu.pruneCache(lo, hi);
        } else if (ip.bm != 0) {
          gpu.pruneCache(-1, -2); // clear all
        }
        continue;
      }
    }

    auto& cache = inst->cache[k];
    auto getSpectra = [&](int fn) -> std::shared_ptr<PlaneRef::Spectra> {
      fn = clampFrame(fn, vi.numFrames);
      auto it = cache.find(fn);
      if (it != cache.end())
        return it->second;
      const VSFrame* fr = vsapi->getFrameFilter(fn, clip, frameCtx);
      std::vector<float> plane;
      ref.upload(vsapi->getReadPtr(fr, p), vsapi->getStride(fr, p), plane);
      vsapi->freeFrame(fr);
      auto s = ref.computeSpectra(plane);
      cache[fn] = s;
      return s;
    };

    std::vector<std::shared_ptr<PlaneRef::Spectra>> window;
    if (ip.bm >= 2) {
      for (int i = 0; i < ip.bt; i++)
        window.push_back(getSpectra(n - ip.current + i));
    } else {
      window.push_back(getSpectra(n));
    }

    std::vector<float> outPlane;
    ref.filterAndSynthesize(window, outPlane);
    ref.download(outPlane, vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p));

    // Prune the spectra cache around the current request.
    if (ip.bm >= 2) {
      int lo = n - ip.current - ip.bt, hi = n - ip.current + 2 * ip.bt;
      for (auto it = cache.begin(); it != cache.end();)
        it = (it->first < lo || it->first > hi) ? cache.erase(it) : std::next(it);
    } else if (ip.bm != 0 && !cache.empty()) {
      cache.clear();
    }
  }

  vsapi->freeFrame(src);
  return dst;
}

void VS_CC fft3dclFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
  delete static_cast<FilterInstance*>(instanceData);
}

} // namespace

// ---- plugin entry ----------------------------------------------------------

static const char* const fft3dclArgs =
    "clip:vnode;"
    "sigma:float:opt;beta:float:opt;bw:int:opt;bh:int:opt;bt:int:opt;sharpen:float:opt;"
    "plane:int:opt;mode:int:opt;bordersize:int:opt;precision:int:opt;degrid:float:opt;"
    "scutoff:float:opt;svr:float:opt;smin:float:opt;smax:float:opt;kratio:float:opt;"
    "ow:int:opt;oh:int:opt;wintype:int:opt;interlaced:int:opt;sigma2:float:opt;"
    "sigma3:float:opt;sigma4:float:opt;oldfft:int:opt;device:int:opt";

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
  vspapi->configPlugin("com.github.humbleuser33.fft3dcl", "fft3dcl", "FFT3DCL denoiser (VS port)",
                       VS_MAKE_VERSION(0, 8), VAPOURSYNTH_API_VERSION, 0, plugin);
  vspapi->registerFunction("FFT3DCL", fft3dclArgs, "clip:vnode;", fft3dclCreate, nullptr, plugin);
}
