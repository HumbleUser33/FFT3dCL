#include "gpu.h"
#include "kernels.h"
#include <cstdio>

namespace fft3dcl {

static cl_mem mkBuf(GpuContext& c, cl_uint flags, size_t sz, const void* host) {
  cl_int e = 0;
  cl_mem m = c.api.createBuffer(c.context, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), sz,
                                (void*)host, &e);
  return e == CL_SUCCESS ? m : nullptr;
}

GpuRef::GpuRef(GpuContext& ctx, const InternalParams& p, const PlaneGeom& g, const PlaneRef& cpu)
  : ctx_(&ctx), p_(p), g_(g) {
  nb_ = cpu.nblocks();
  fbw_ = cpu.fbw();
  planeSize_ = fbw_ * p_.bh;

  // FFT work-group: one block per group, max(bw,bh) work-items.
  wgFft_ = (size_t)(p_.bw > p_.bh ? p_.bw : p_.bh);
  if (wgFft_ > ctx.maxWorkGroupSize) {
    err_ = "block size exceeds device max work-group size";
    return;
  }
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
  char opts[160];
  snprintf(opts, sizeof(opts),
           "-DLSZ=%d -DBM=%d -DDG=%d -DPAT=%d -DSHP=%d -cl-fp32-correctly-rounded-divide-sqrt",
           p_.bw * p_.bh, p_.bm, p_.degridOn ? 1 : 0, p_.usePattern ? 1 : 0,
           p_.sharpenOn ? 1 : 0);
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

  const int W = g_.width, H = g_.height, bw = p_.bw, bh = p_.bh;
  bufBx_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(int) * nb_, cpu.bx().data());
  bufBy_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(int) * nb_, cpu.by().data());
  bufWinAna_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winAna().data());
  bufWinA_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynA().data());
  bufWinB_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynB().data());
  std::vector<unsigned char> par(nb_);
  for (int i = 0; i < nb_; i++)
    par[i] = cpu.par()[i];
  bufPar_ = mkBuf(ctx, CL_MEM_READ_ONLY, nb_, par.data());
  bufGs_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * 2 * planeSize_, nullptr);
  std::vector<float> pat = cpu.pattern();
  if (pat.empty()) pat.assign(planeSize_, 0.0f);
  bufPat_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * pat.size(), pat.data());
  std::vector<float> shp = cpu.sharpenLut();
  if (shp.empty()) shp.assign(planeSize_, 0.0f);
  bufShp_ = mkBuf(ctx, CL_MEM_READ_ONLY, sizeof(float) * shp.size(), shp.data());
  bufFilt_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
  bufBlocks_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * bw * bh, nullptr);
  bufOut_ = mkBuf(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * W * H, nullptr);
  bufPlane_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * W * H, nullptr);
  if (p_.bm == 0) {
    bufKLast_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
    bufKCov_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * planeSize_, nullptr);
    bufKPro_ = mkBuf(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb_ * planeSize_, nullptr);
    if (!bufKLast_ || !bufKCov_ || !bufKPro_) {
      err_ = "GPU alloc failed (kalman state)";
      return;
    }
  }
  if (!bufBx_ || !bufBy_ || !bufWinAna_ || !bufWinA_ || !bufWinB_ || !bufPar_ || !bufGs_ ||
      !bufPat_ || !bufShp_ || !bufFilt_ || !bufBlocks_ || !bufOut_ || !bufPlane_) {
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
  ctx.api.setKernelArg(kSyn_, 1, sizeof(cl_mem), &bufOut_);
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
  for (auto& kv : specCache_)
    ctx_->api.releaseMemObject(kv.second);
  specCache_.clear();
  for (cl_mem m : specPool_)
    ctx_->api.releaseMemObject(m);
  specPool_.clear();
  cl_mem all[] = {bufBx_, bufBy_, bufWinAna_, bufWinA_, bufWinB_, bufPar_, bufGs_, bufPat_,
                  bufShp_, bufFilt_, bufBlocks_, bufOut_, bufPlane_, bufKLast_, bufKCov_,
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
    m = mkBuf(*ctx_, CL_MEM_READ_WRITE, sizeof(float) * 2 * (size_t)nb_ * planeSize_, nullptr);
  }
  if (m)
    specCache_[fn] = m;
  return m;
}

// A spectra buffer that left the cache: keep a few for reuse, release the rest.
void GpuRef::retireSpectra(cl_mem m) {
  if (specPool_.size() < 4)
    specPool_.push_back(m);
  else
    ctx_->api.releaseMemObject(m);
}

bool GpuRef::forward(int fn, const std::vector<float>& plane) {
  if (!ok_)
    return false;
  cl_mem spec = spectra(fn);
  if (!spec) {
    err_ = "GPU spectra alloc failed";
    return false;
  }
  cl_int e = ctx_->api.enqueueWriteBuffer(ctx_->queue, bufPlane_, CL_TRUE, 0,
                                          sizeof(float) * g_.width * g_.height, plane.data(), 0,
                                          nullptr, nullptr);
  if (e != CL_SUCCESS) {
    err_ = "enqueueWriteBuffer failed";
    return false;
  }
  ctx_->api.setKernelArg(kFwd_, 0, sizeof(cl_mem), &bufPlane_);
  ctx_->api.setKernelArg(kFwd_, 3, sizeof(cl_mem), &spec);
  size_t goff = 0, gsize = (size_t)nb_ * wgFft_, lsize = wgFft_;
  e = ctx_->api.enqueueNDRange(ctx_->queue, kFwd_, 1, &goff, &gsize, &lsize, 0, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    err_ = "fft2d_fwd launch failed";
    return false;
  }
  return true;
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

bool GpuRef::process(const std::vector<int>& window, std::vector<float>& out) {
  if (!ok_)
    return false;
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
  e = ctx_->api.enqueueNDRange(ctx_->queue, kFil_, 1, nullptr, &gsFil, nullptr, 0, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    err_ = "filter launch failed";
    return false;
  }
  size_t goff = 0, gsize = (size_t)nb_ * wgFft_, lsize = wgFft_;
  ctx_->api.setKernelArg(kInv_, 0, sizeof(cl_mem), &bufFilt_);
  e = ctx_->api.enqueueNDRange(ctx_->queue, kInv_, 1, &goff, &gsize, &lsize, 0, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    err_ = "fft2d_inv launch failed";
    return false;
  }
  size_t gs2[2] = {(size_t)g_.width, (size_t)g_.height};
  e = ctx_->api.enqueueNDRange(ctx_->queue, kSyn_, 2, nullptr, gs2, nullptr, 0, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    err_ = "synthesize launch failed";
    return false;
  }
  out.assign(g_.width * g_.height, 0.0f);
  e = ctx_->api.enqueueReadBuffer(ctx_->queue, bufOut_, CL_TRUE, 0, sizeof(float) * out.size(),
                                  out.data(), 0, nullptr, nullptr);
  if (e != CL_SUCCESS) {
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
