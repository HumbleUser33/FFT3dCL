#include "gpu.h"
#include "kernels.h"
#include "../timing.h"
#include <atomic>
#include <cstdio>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace fft3dcl {

namespace {
// The driver's blocking calls (clFinish, clEnqueueReadBuffer with CL_TRUE,
// clWaitForEvents) busy-wait on Windows: they keep a whole CPU core spinning
// while the GPU works. One frame of this filter is tens of milliseconds of GPU
// time, and the encoder needs those cycles, so the wait is done on a kernel32
// event that the OpenCL completion callback signals.
void CL_CALLBACK gpuDone(cl_event, cl_int, void* ud) { SetEvent(static_cast<HANDLE>(ud)); }

bool waitForEvent(const ClApi& api, cl_command_queue queue, cl_event ev) {
  bool ok = false;
  HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (!h)
    return api.finish(queue) == CL_SUCCESS;
  if (api.setEventCallback(ev, CL_COMPLETE, gpuDone, h) == CL_SUCCESS)
    ok = WaitForSingleObject(h, 120000) == WAIT_OBJECT_0; // only a dead driver waits longer
  CloseHandle(h);
  api.releaseEvent(ev);
  if (!ok)
    ok = api.finish(queue) == CL_SUCCESS; // driver never called back: fall back
  return ok;
}
} // namespace

static cl_mem mkBuf(GpuContext& c, cl_uint flags, size_t sz, const void* host) {
  cl_int e = 0;
  cl_mem m = c.api.createBuffer(c.context, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), sz,
                                (void*)host, &e);
  return e == CL_SUCCESS ? m : nullptr;
}

static std::atomic<int> g_live{0};
static std::atomic<size_t> g_bytes{0};

int GpuRef::liveInstances() { return g_live.load(); }
size_t GpuRef::liveBytes() { return g_bytes.load(); }

cl_mem GpuRef::mkTracked(cl_uint flags, size_t sz, const void* host) {
  cl_mem m = mkBuf(*ctx_, flags, sz, host);
  if (m) {
    bytes_ += sz;
    g_bytes += sz;
  }
  return m;
}

GpuRef::GpuRef(GpuContext& ctx, const InternalParams& p, const PlaneGeom& g, const PlaneRef& cpu)
  : ctx_(&ctx), p_(p), g_(g) {
  ++g_live;
  nb_ = cpu.nblocks();
  fbw_ = cpu.fbw();
  planeSize_ = fbw_ * p_.bh;

  // FFT work-group: one block per group. The batched FFT has bw*bh/2 butterflies
  // per stage, so a group bigger than max(bw,bh) keeps every work-item busy;
  // 4x measured about 8% faster at 4K with 32x32 blocks, and never leaves
  // work-items without work because the loops below all stride by the group size.
  size_t m = (size_t)(p_.bw > p_.bh ? p_.bw : p_.bh);
  if (m > ctx.maxWorkGroupSize) {
    err_ = "block size exceeds device max work-group size";
    return;
  }
  wgFft_ = 4 * m;
  if (wgFft_ > (size_t)p_.bw * p_.bh / 2)
    wgFft_ = (size_t)p_.bw * p_.bh / 2;
  if (wgFft_ < m)
    wgFft_ = m;
  if (wgFft_ > ctx.maxWorkGroupSize)
    wgFft_ = ctx.maxWorkGroupSize;
  if ((size_t)p_.bw * p_.bh * 8 > 48u * 1024) {
    // Conservative local-memory bound (all target GPUs provide >=48KB...
    // actually >=32KB; 64x64 blocks need 32KB).
    if ((size_t)p_.bw * p_.bh * 8 > 32u * 1024) {
      err_ = "block too large for device local memory (max 64x64)";
      return;
    }
  }

  cl_int e = 0;
  const char* ksrc = kKernelSource();
  prog_ = ctx.api.createProgramWithSource(ctx.context, 1, &ksrc, nullptr, &e);
  if (!prog_ || e != CL_SUCCESS) {
    err_ = "clCreateProgramWithSource failed";
    return;
  }
  char opts[256];
  snprintf(opts, sizeof(opts),
           "-DLSZ=%d -DLSW=%d -DLP=%d -DLSZP=%d -DBM=%d -DDG=%d -DPAT=%d -DSHP=%d -DRAW=%d "
           "-cl-fp32-correctly-rounded-divide-sqrt",
           p_.bw * p_.bh, p_.bw > p_.bh ? p_.bw : p_.bh, p_.bw + 1, (p_.bw + 1) * p_.bh, p_.bm,
           p_.degridOn ? 1 : 0, p_.usePattern ? 1 : 0, p_.sharpenOn ? 1 : 0,
           g_.sampleType == 1 ? 0 : (g_.bits <= 8 ? 8 : 16));
  e = ctx.api.buildProgram(prog_, 1, &ctx.device, opts, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    char log[4096] = {};
    ctx.api.getProgramBuildInfo(prog_, ctx.device, 0x1183, sizeof(log), log, nullptr);
    err_ = std::string("kernel build failed: ") + log;
    return;
  }

  kFwd_ = ctx.api.createKernel(prog_, "fft2d_fwd", &e);
  kFil_ = ctx.api.createKernel(prog_, "filter", &e);
  kInv_ = ctx.api.createKernel(prog_, "fft2d_inv", &e);
  kSyn_ = ctx.api.createKernel(prog_, "synthesize", &e);
  kKalInit_ = ctx.api.createKernel(prog_, "kalman_init", &e);
  if (!kFwd_ || !kFil_ || !kInv_ || !kSyn_ || !kKalInit_) {
    err_ = "clCreateKernel failed";
    return;
  }

  // Integer samples: the forward kernel reads the bytes VapourSynth gave us and
  // the synthesis writes them back, so no conversion kernel is needed.
  rawPath_ = g_.sampleType != 1;
  bps_ = g_.sampleType == 1 ? 4 : (g_.bits <= 8 ? 1 : 2);

  const int W = g_.width, H = g_.height, bw = p_.bw, bh = p_.bh;
  bufBx_ = mkTracked(CL_MEM_READ_ONLY, sizeof(int) * nb_, cpu.bx().data());
  bufBy_ = mkTracked(CL_MEM_READ_ONLY, sizeof(int) * nb_, cpu.by().data());
  bufWinAna_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winAna().data());
  bufWinA_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynA().data());
  bufWinB_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynB().data());
  std::vector<unsigned char> par(nb_);
  for (int i = 0; i < nb_; i++)
    par[i] = cpu.par()[i];
  bufPar_ = mkTracked(CL_MEM_READ_ONLY, nb_, par.data());
  bufGs_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * 2 * planeSize_, nullptr);
  std::vector<float> pat = cpu.pattern();
  if (pat.empty()) pat.assign(planeSize_, 0.0f);
  bufPat_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * pat.size(), pat.data());
  std::vector<float> shp = cpu.sharpenLut();
  if (shp.empty()) shp.assign(planeSize_, 0.0f);
  bufShp_ = mkTracked(CL_MEM_READ_ONLY, sizeof(float) * shp.size(), shp.data());
  bufFilt_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
  bufBlocks_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * bw * bh, nullptr);
  if (rawPath_) {
    // With integer samples the bytes go straight in and out of the two kernels
    // that already touch them, so the float plane and the float output buffer
    // are not allocated at all: 66 MB of device memory saved per instance at 4K.
    bufRaw_ = mkTracked(CL_MEM_READ_WRITE, (size_t)bps_ * W * H, nullptr);
  } else {
    bufOut_ = mkTracked(CL_MEM_WRITE_ONLY, sizeof(float) * W * H, nullptr);
    bufPlane_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * W * H, nullptr);
  }
  if (p_.bm == 0) {
    bufKLast_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
    bufKCov_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * planeSize_, nullptr);
    bufKPro_ = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * planeSize_, nullptr);
    if (!bufKLast_ || !bufKCov_ || !bufKPro_) {
      err_ = "GPU alloc failed (kalman state)";
      return;
    }
  }
  if (!bufBx_ || !bufBy_ || !bufWinAna_ || !bufWinA_ || !bufWinB_ || !bufPar_ || !bufGs_ ||
      !bufPat_ || !bufShp_ || !bufFilt_ || !bufBlocks_ ||
      (rawPath_ ? !bufRaw_ : (!bufOut_ || !bufPlane_))) {
    err_ = "GPU alloc failed";
    return;
  }

  // Static kernel args (per-frame args set in forward/process).
  size_t W_ = W, H_ = H;
  (void)W_; (void)H_;
  int iw = W, ih = H, ibw = bw, ibh = bh, ifbw = fbw_, inb = nb_;
  float cn = cpu.chromaNorm();
  ctx.api.setKernelArg(kFwd_, 1, sizeof(cl_mem), &bufBx_);
  ctx.api.setKernelArg(kFwd_, 2, sizeof(cl_mem), &bufBy_);
  ctx.api.setKernelArg(kFwd_, 4, sizeof(cl_mem), &bufWinAna_);
  ctx.api.setKernelArg(kFwd_, 5, sizeof(int), &iw);
  ctx.api.setKernelArg(kFwd_, 6, sizeof(int), &ih);
  ctx.api.setKernelArg(kFwd_, 7, sizeof(int), &ibw);
  ctx.api.setKernelArg(kFwd_, 8, sizeof(int), &ibh);
  ctx.api.setKernelArg(kFwd_, 9, sizeof(int), &ifbw);
  ctx.api.setKernelArg(kFwd_, 10, sizeof(int), &inb);
  ctx.api.setKernelArg(kFwd_, 11, sizeof(float), &cn);
  // norm/invNorm/maxPix only matter for the integer formats; the values are the
  // ones the old unpack_u16/pack_u16 kernels were given.
  float norm = (rawPath_ && g_.bits > 8) ? (float)(1u << (16 - g_.bits)) / 65535.0f : 1.0f;
  float invNorm = rawPath_ ? 1.0f / (float)(1u << (16 - g_.bits)) : 1.0f;
  float maxPix = rawPath_ ? (float)((1u << g_.bits) - 1) : 0.0f;
  ctx.api.setKernelArg(kFwd_, 12, sizeof(float), &norm);

  ctx.api.setKernelArg(kInv_, 1, sizeof(cl_mem), &bufBlocks_);
  ctx.api.setKernelArg(kInv_, 2, sizeof(cl_mem), &bufWinA_);
  ctx.api.setKernelArg(kInv_, 3, sizeof(cl_mem), &bufWinB_);
  ctx.api.setKernelArg(kInv_, 4, sizeof(cl_mem), &bufPar_);
  ctx.api.setKernelArg(kInv_, 5, sizeof(int), &ibw);
  ctx.api.setKernelArg(kInv_, 6, sizeof(int), &ibh);
  ctx.api.setKernelArg(kInv_, 7, sizeof(int), &ifbw);
  ctx.api.setKernelArg(kInv_, 8, sizeof(int), &inb);

  int xoff = p_.mode == 0 ? -bw / 2 : -p_.ow;
  int xstep = p_.mode == 0 ? bw : bw - p_.ow;
  int yoff = p_.mode == 0 ? -bh / 2 : -p_.oh;
  int ystep = p_.mode == 0 ? bh : bh - p_.oh;
  int ncols = cpu.ncols(), nrows = cpu.nrows();
  ctx.api.setKernelArg(kSyn_, 0, sizeof(cl_mem), &bufBlocks_);
  // arg 1 (the output) is set per call: the float buffer or the raw bytes.
  ctx.api.setKernelArg(kSyn_, 2, sizeof(int), &iw);
  ctx.api.setKernelArg(kSyn_, 3, sizeof(int), &ih);
  ctx.api.setKernelArg(kSyn_, 4, sizeof(int), &ibw);
  ctx.api.setKernelArg(kSyn_, 5, sizeof(int), &ibh);
  ctx.api.setKernelArg(kSyn_, 6, sizeof(int), &xoff);
  ctx.api.setKernelArg(kSyn_, 7, sizeof(int), &xstep);
  ctx.api.setKernelArg(kSyn_, 8, sizeof(int), &yoff);
  ctx.api.setKernelArg(kSyn_, 9, sizeof(int), &ystep);
  ctx.api.setKernelArg(kSyn_, 10, sizeof(int), &ncols);
  ctx.api.setKernelArg(kSyn_, 11, sizeof(int), &nrows);
  ctx.api.setKernelArg(kSyn_, 12, sizeof(float), &cn);
  ctx.api.setKernelArg(kSyn_, 13, sizeof(float), &invNorm);
  ctx.api.setKernelArg(kSyn_, 14, sizeof(float), &maxPix);

  ctx.api.setKernelArg(kFil_, 4, sizeof(cl_mem), &bufFilt_);
  ctx.api.setKernelArg(kFil_, 5, sizeof(cl_mem), &bufGs_);
  ctx.api.setKernelArg(kFil_, 6, sizeof(cl_mem), &bufPat_);
  ctx.api.setKernelArg(kFil_, 7, sizeof(cl_mem), &bufShp_);
  ctx.api.setKernelArg(kFil_, 8, sizeof(cl_mem), &bufKLast_);
  ctx.api.setKernelArg(kFil_, 9, sizeof(cl_mem), &bufKCov_);
  ctx.api.setKernelArg(kFil_, 10, sizeof(cl_mem), &bufKPro_);
  ctx.api.setKernelArg(kFil_, 11, sizeof(int), &inb);
  ctx.api.setKernelArg(kFil_, 12, sizeof(int), &ifbw);
  ctx.api.setKernelArg(kFil_, 13, sizeof(int), &ibh);
  float sigmaX = p_.beta * p_.wienerS, sigmaY = p_.wienerS;
  ctx.api.setKernelArg(kFil_, 14, sizeof(float), &sigmaX);
  ctx.api.setKernelArg(kFil_, 15, sizeof(float), &sigmaY);
  ctx.api.setKernelArg(kFil_, 16, sizeof(float), &p_.betaX);
  ctx.api.setKernelArg(kFil_, 17, sizeof(float), &p_.beta);
  ctx.api.setKernelArg(kFil_, 18, sizeof(float), &p_.ssmin);
  ctx.api.setKernelArg(kFil_, 19, sizeof(float), &p_.ssmax);
  ctx.api.setKernelArg(kFil_, 20, sizeof(float), &p_.kalmanCN);
  ctx.api.setKernelArg(kFil_, 21, sizeof(float), &p_.kratio);

  ok_ = true;
}

GpuRef::~GpuRef() {
  if (!ctx_)
    return;
  --g_live;
  g_bytes -= bytes_;
  for (auto& kv : specCache_)
    ctx_->api.releaseMemObject(kv.second);
  specCache_.clear();
  for (cl_mem m : specPool_)
    ctx_->api.releaseMemObject(m);
  specPool_.clear();
  cl_mem all[] = {bufBx_, bufBy_, bufWinAna_, bufWinA_, bufWinB_, bufPar_, bufGs_, bufPat_,
                  bufShp_, bufFilt_, bufBlocks_, bufOut_, bufPlane_, bufRaw_, bufKLast_, bufKCov_,
                  bufKPro_};
  for (cl_mem m : all)
    if (m)
      ctx_->api.releaseMemObject(m);
  cl_kernel ks[] = {kFwd_, kFil_, kInv_, kSyn_, kKalInit_};
  for (cl_kernel k : ks)
    if (k)
      ctx_->api.releaseKernel(k);
  if (prog_)
    ctx_->api.releaseProgram(prog_);
}

cl_mem GpuRef::spectra(int fn) {
  auto it = specCache_.find(fn);
  if (it != specCache_.end())
    return it->second;
  // Reuse a retired buffer when possible: creating and releasing one per frame
  // makes the driver hold on to the memory.
  cl_mem m = nullptr;
  if (!specPool_.empty()) {
    m = specPool_.back();
    specPool_.pop_back();
  } else {
    m = mkTracked(CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
  }
  if (m)
    specCache_[fn] = m;
  return m;
}

// A spectra buffer that left the cache: keep a few for reuse, release the rest.
// The budget, not the count, is what matters: at 4K one buffer is 142 MiB, so
// four of them would hold 566 MiB of device memory for nothing.
void GpuRef::retireSpectra(cl_mem m) {
  const size_t budget = 256u << 20; // 256 MiB of recycled buffers is enough
  const size_t one = sizeof(float) * 2 * (size_t)nb_ * planeSize_;
  if (specPool_.size() * one + one <= budget)
    specPool_.push_back(m);
  else {
    ctx_->api.releaseMemObject(m);
    // The running total has to follow the release, otherwise the "N MiB of
    // device buffers" line counts buffers the driver already gave back and
    // over-states the footprint (it did: 1694 MiB reported for 1113 MiB live).
    bytes_ -= one;
    g_bytes -= one;
  }
}

bool GpuRef::launchFwd(cl_mem spec) {
  ctx_->api.setKernelArg(kFwd_, 0, sizeof(cl_mem), rawPath_ ? &bufRaw_ : &bufPlane_);
  ctx_->api.setKernelArg(kFwd_, 3, sizeof(cl_mem), &spec);
  size_t goff = 0, gsize = (size_t)nb_ * wgFft_, lsize = wgFft_;
  long long t0 = timeEnabled() ? nowNs() : 0;
  cl_int e = ctx_->api.enqueueNDRange(ctx_->queue, kFwd_, 1, &goff, &gsize, &lsize, 0, nullptr,
                                      nullptr);
  if (t0) { // diagnostic mode: drain the queue to time the kernel itself
    ctx_->api.finish(ctx_->queue);
    addStage(kGpuFwd, nowNs() - t0);
  }
  if (e != CL_SUCCESS) {
    err_ = "fft2d_fwd launch failed";
    return false;
  }
  return true;
}

bool GpuRef::forward(int fn, const std::vector<float>& plane) {
  if (!ok_)
    return false;
  cl_mem spec = spectra(fn);
  if (!spec) {
    err_ = "GPU spectra alloc failed";
    return false;
  }
  cl_int e;
  {
    StageTime t(kUpload);
    e = ctx_->api.enqueueWriteBuffer(ctx_->queue, bufPlane_, CL_TRUE, 0,
                                     sizeof(float) * g_.width * g_.height, plane.data(), 0, nullptr,
                                     nullptr);
  }
  if (e != CL_SUCCESS) {
    err_ = "enqueueWriteBuffer failed";
    return false;
  }
  StageTime t(kFwd);
  return launchFwd(spec);
}

bool GpuRef::forwardRaw(int fn, const void* raw, size_t bytes) {
  if (!ok_ || !rawPath_)
    return false;
  cl_mem spec = spectra(fn);
  if (!spec) {
    err_ = "GPU spectra alloc failed";
    return false;
  }
  cl_int e;
  {
    StageTime t(kUpload);
    // Blocking on purpose: the driver copies the plane here, so the caller does
    // not have to keep the source frame alive, and a non-blocking queue only
    // saved 0.2 ms in measurement while holding four 4K frames in RAM.
    e = ctx_->api.enqueueWriteBuffer(ctx_->queue, bufRaw_, CL_TRUE, 0, bytes, (void*)raw, 0,
                                     nullptr, nullptr);
  }
  if (e != CL_SUCCESS) {
    err_ = "enqueueWriteBuffer failed";
    return false;
  }
  StageTime t(kFwd);
  return launchFwd(spec);
}

bool GpuRef::uploadDegridSample(const PlaneRef& cpu) {
  if (!ok_)
    return false;
  const std::vector<float>& r = cpu.gsRe();
  const std::vector<float>& im = cpu.gsIm();
  std::vector<float> gs(2 * planeSize_);
  for (int i = 0; i < planeSize_; i++) {
    gs[2 * i] = r[i];
    gs[2 * i + 1] = im[i];
  }
  cl_int e = ctx_->api.enqueueWriteBuffer(ctx_->queue, bufGs_, CL_TRUE, 0, sizeof(float) * gs.size(),
                                          gs.data(), 0, nullptr, nullptr);
  return e == CL_SUCCESS;
}

bool GpuRef::runKernels(const std::vector<int>& window) {
  cl_mem s0 = spectra(window[0]);
  cl_mem s1 = window.size() > 1 ? spectra(window[1]) : s0;
  cl_mem s2 = window.size() > 2 ? spectra(window[2]) : s0;
  cl_mem s3 = window.size() > 3 ? spectra(window[3]) : s0;
  if (!s0 || !s1 || !s2 || !s3) {
    err_ = "missing spectra in window";
    return false;
  }
  cl_int e;
  if (p_.bm == 0 && kalmanFirst_) {
    int total = nb_ * planeSize_;
    ctx_->api.setKernelArg(kKalInit_, 0, sizeof(cl_mem), &s0);
    ctx_->api.setKernelArg(kKalInit_, 1, sizeof(cl_mem), &bufKLast_);
    ctx_->api.setKernelArg(kKalInit_, 2, sizeof(cl_mem), &bufKCov_);
    ctx_->api.setKernelArg(kKalInit_, 3, sizeof(cl_mem), &bufKPro_);
    ctx_->api.setKernelArg(kKalInit_, 4, sizeof(cl_mem), &bufPat_);
    ctx_->api.setKernelArg(kKalInit_, 5, sizeof(int), &total);
    ctx_->api.setKernelArg(kKalInit_, 6, sizeof(int), &planeSize_);
    ctx_->api.setKernelArg(kKalInit_, 7, sizeof(float), &p_.kalmanCN);
    size_t gs1 = ((size_t)total + 63) / 64 * 64;
    e = ctx_->api.enqueueNDRange(ctx_->queue, kKalInit_, 1, nullptr, &gs1, nullptr, 0, nullptr,
                                 nullptr);
    if (e != CL_SUCCESS) {
      err_ = "kalman_init launch failed";
      return false;
    }
    kalmanFirst_ = false;
  }
  ctx_->api.setKernelArg(kFil_, 0, sizeof(cl_mem), &s0);
  ctx_->api.setKernelArg(kFil_, 1, sizeof(cl_mem), &s1);
  ctx_->api.setKernelArg(kFil_, 2, sizeof(cl_mem), &s2);
  ctx_->api.setKernelArg(kFil_, 3, sizeof(cl_mem), &s3);
  size_t gsFil = (size_t)nb_ * planeSize_;
  long long tFil = timeEnabled() ? nowNs() : 0;
  e = ctx_->api.enqueueNDRange(ctx_->queue, kFil_, 1, nullptr, &gsFil, nullptr, 0, nullptr, nullptr);
  if (tFil) {
    ctx_->api.finish(ctx_->queue);
    addStage(kGpuFil, nowNs() - tFil);
  }
  if (e != CL_SUCCESS) {
    err_ = "filter launch failed";
    return false;
  }
  size_t goff = 0, gsize = (size_t)nb_ * wgFft_, lsize = wgFft_;
  ctx_->api.setKernelArg(kInv_, 0, sizeof(cl_mem), &bufFilt_);
  long long tInv = timeEnabled() ? nowNs() : 0;
  e = ctx_->api.enqueueNDRange(ctx_->queue, kInv_, 1, &goff, &gsize, &lsize, 0, nullptr, nullptr);
  if (tInv) {
    ctx_->api.finish(ctx_->queue);
    addStage(kGpuInv, nowNs() - tInv);
  }
  if (e != CL_SUCCESS) {
    err_ = "fft2d_inv launch failed";
    return false;
  }
  size_t gs2[2] = {(size_t)g_.width, (size_t)g_.height};
  ctx_->api.setKernelArg(kSyn_, 1, sizeof(cl_mem), rawPath_ ? &bufRaw_ : &bufOut_);
  long long tSyn = timeEnabled() ? nowNs() : 0;
  e = ctx_->api.enqueueNDRange(ctx_->queue, kSyn_, 2, nullptr, gs2, nullptr, 0, nullptr, nullptr);
  if (tSyn) {
    ctx_->api.finish(ctx_->queue);
    addStage(kGpuSyn, nowNs() - tSyn);
  }
  if (e != CL_SUCCESS) {
    err_ = "synthesize launch failed";
    return false;
  }
  return true;
}

bool GpuRef::process(const std::vector<int>& window, std::vector<float>& out) {
  if (!ok_)
    return false;
  {
    StageTime t(kRun);
    if (!runKernels(window))
      return false;
  }
  out.assign(g_.width * g_.height, 0.0f);
  StageTime t(kReadback);
  cl_event ev = nullptr;
  cl_int e = ctx_->api.enqueueReadBuffer(ctx_->queue, bufOut_, CL_FALSE, 0, sizeof(float) * out.size(),
                                         out.data(), 0, nullptr, &ev);
  if (e != CL_SUCCESS || !ev) {
    err_ = "readback failed";
    return false;
  }
  if (!waitForEvent(ctx_->api, ctx_->queue, ev)) {
    err_ = "readback failed";
    return false;
  }
  return true;
}

bool GpuRef::processRaw(const std::vector<int>& window, void* dst, size_t bytes) {
  if (!ok_ || !rawPath_)
    return false;
  {
    StageTime t(kRun);
    if (!runKernels(window))
      return false;
  }
  StageTime t(kReadback);
  cl_event ev = nullptr;
  cl_int e =
      ctx_->api.enqueueReadBuffer(ctx_->queue, bufRaw_, CL_FALSE, 0, bytes, dst, 0, nullptr, &ev);
  if (e != CL_SUCCESS || !ev) {
    err_ = "readback failed";
    return false;
  }
  if (!waitForEvent(ctx_->api, ctx_->queue, ev)) {
    err_ = "readback failed";
    return false;
  }
  return true;
}

void GpuRef::pruneCache(int lo, int hi) {
  for (auto it = specCache_.begin(); it != specCache_.end();) {
    bool keep = (lo < 0 || it->first >= lo) && (hi < 0 || it->first <= hi);
    if (keep) {
      ++it;
    } else {
      retireSpectra(it->second);
      it = specCache_.erase(it);
    }
  }
}

} // namespace fft3dcl
