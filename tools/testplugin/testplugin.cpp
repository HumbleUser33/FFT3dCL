// Minimal pass-through VapourSynth plugin, used only to investigate memory
// behaviour: namespace tst, function tst.Passthru(clip, extra=<variant>).
#include "VapourSynth4.h"
#include "VSHelper4.h"

#include <cstring>

namespace {

struct Inst {
  VSNode* clip;
  int variant;
};

const VSFrame* VS_CC getFrame(int n, int activationReason, void* instanceData, void** frameData,
                              VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
  Inst* inst = static_cast<Inst*>(instanceData);
  if (activationReason == arInitial) {
    vsapi->requestFrameFilter(n, inst->clip, frameCtx);
    return nullptr;
  }
  if (activationReason != arAllFramesReady)
    return nullptr;
  const VSFrame* src = vsapi->getFrameFilter(n, inst->clip, frameCtx);
  if (inst->variant == 5) // hand the input reference to the core, allocate nothing
    return src;
  VSFrame* dst;
  if (inst->variant == 6) { // fresh frame with no property prototype, then copy the planes
    const VSVideoInfo* vi = vsapi->getVideoInfo(inst->clip);
    dst = vsapi->newVideoFrame(&vi->format, vi->width, vi->height, nullptr, core);
    for (int p = 0; p < vi->format.numPlanes; p++) {
      const uint8_t* rp = vsapi->getReadPtr(src, p);
      uint8_t* wp = vsapi->getWritePtr(dst, p);
      ptrdiff_t rs = vsapi->getStride(src, p), ws = vsapi->getStride(dst, p);
      int h = vsapi->getFrameHeight(src, p);
      size_t bytes = (size_t)vsapi->getFrameWidth(src, p) * ((vi->format.bitsPerSample + 7) / 8);
      for (int y = 0; y < h; y++)
        memcpy(wp + y * ws, rp + y * rs, bytes);
    }
  } else {
    dst = vsapi->copyFrame(src, core);
  }
  vsapi->freeFrame(src);
  return dst;
}

void VS_CC freeInstance(void* instanceData, VSCore* core, const VSAPI* vsapi) {
  Inst* inst = static_cast<Inst*>(instanceData);
  vsapi->freeNode(inst->clip); // the filter owns its dependency reference
  delete inst;
}

void VS_CC create(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
  int err = 0;
  VSNode* clip = vsapi->mapGetNode(in, "clip", 0, &err);
  if (err) {
    vsapi->mapSetError(out, "Passthru: no clip");
    return;
  }
  int variant = (int)vsapi->mapGetInt(in, "extra", 0, &err);
  if (err)
    variant = 0;
  const VSVideoInfo* vi = vsapi->getVideoInfo(clip);
  static const int patterns[] = {rpGeneral, rpNoFrameReuse, rpStrictSpatial, rpGeneral, rpGeneral,
                                 rpGeneral, rpGeneral};
  static const int modes[] = {fmParallel, fmParallel, fmParallel, fmParallelRequests, fmUnordered,
                              fmParallel, fmParallel};
  Inst* inst = new Inst{clip, variant};
  VSFilterDependency deps[] = {{clip, patterns[variant % 7]}};
  vsapi->createVideoFilter(out, "Passthru", vi, getFrame, freeInstance, modes[variant % 7], deps, 1,
                           inst, core);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
#ifdef TST_MD
  vspapi->configPlugin("com.example.tstmd", "tstmd", "memory test passthrough (dynamic CRT)",
                       VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
#else
  vspapi->configPlugin("com.example.tst", "tst", "memory test passthrough", VS_MAKE_VERSION(1, 0),
                       VAPOURSYNTH_API_VERSION, 0, plugin);
#endif
  vspapi->registerFunction("Passthru", "clip:vnode;extra:int:opt;", "clip:vnode;", create, nullptr,
                           plugin);
}
