// Standalone GPU-vs-CPU pipeline test: analysis+FFT, temporal filter
// (Wiener/degrid/sharpen), inverse+synthesis. Compares the OpenCL pipeline
// against the CPU reference on random frames.
#include "cl.h"
#include "kernels.h"
#include "params.h"
#include "reference.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace fft3dcl;

#define CHECK(expr)                                                            \
  do {                                                                         \
    cl_int e_ = (expr);                                                        \
    if (e_ != CL_SUCCESS) {                                                    \
      printf("OpenCL error %d at %s:%d\n", e_, __FILE__, __LINE__);            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static cl_mem makeBuf(const GpuContext& gpu, cl_uint flags, size_t sz, const void* host) {
  cl_int e = 0;
  cl_mem m = gpu.api.createBuffer(gpu.context, flags | (host ? CL_MEM_COPY_HOST_PTR : 0), sz,
                                  (void*)host, &e);
  return e == CL_SUCCESS ? m : nullptr;
}

int main(int argc, char** argv) {
  const int W = 256, H = 192;
  ScriptParams sp; // defaults: mode 1, bw=bh=32, ow=oh=16, bt=3, degrid=1
  if (argc > 1) sp.bt = atoi(argv[1]);
  if (argc > 2) sp.sharpen = (float)atof(argv[2]);
  if (argc > 3) sp.degrid = (float)atof(argv[3]);
  if (argc > 4) sp.sigma2 = (float)atof(argv[4]); // enables pattern when != sigma
  InternalParams ip;
  const char* err = nullptr;
  if (!validateParams(sp, W, H, 8, ip, &err)) {
    printf("params: %s\n", err);
    return 1;
  }

  PlaneGeom g;
  g.width = W; g.height = H; g.bits = 8; g.sampleType = 0; g.chroma = false;
  PlaneRef cpu(ip, g);
  if (ip.degridOn)
    cpu.buildDegridSample();

  const int nframes = ip.bm >= 2 ? ip.bt : 1;
  std::vector<std::vector<float>> planes(nframes, std::vector<float>(W * H));
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& pl : planes)
    for (auto& v : pl)
      v = dist(rng);

  // CPU reference.
  std::vector<std::shared_ptr<PlaneRef::Spectra>> cspec(nframes);
  for (int f = 0; f < nframes; f++)
    cspec[f] = cpu.computeSpectra(planes[f]);
  std::vector<float> cpuOut(W * H, 0.0f);
  cpu.filterAndSynthesize(cspec, cpuOut);

  // GPU context + program.
  GpuContext gpu;
  std::string gerr;
  if (!gpu.init(&gerr)) {
    printf("GPU init failed: %s\n", gerr.c_str());
    return 1;
  }
  printf("device: %s (maxwg %zu)\n", gpu.deviceName.c_str(), gpu.maxWorkGroupSize);

  const int nb = cpu.nblocks(), fbw = cpu.fbw(), bw = ip.bw, bh = ip.bh;
  const int planeSize = fbw * bh;
  const int ncols = cpu.ncols(), nrows = cpu.nrows();

  size_t localNeed = (size_t)bw * bh * 8;
  if (localNeed > 48u * 1024) {
    printf("block too large for typical local memory (%zu bytes)\n", localNeed);
    return 1;
  }

  cl_int e = 0;
  std::vector<cl_mem> bufPlane(nframes), bufSpec(nframes);
  for (int f = 0; f < nframes; f++) {
    bufPlane[f] = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * W * H, planes[f].data());
    bufSpec[f] = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * 2 * nb * planeSize, nullptr);
    if (!bufPlane[f] || !bufSpec[f]) { printf("alloc failed\n"); return 1; }
  }
  cl_mem bufBx = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(int) * nb, cpu.bx().data());
  cl_mem bufBy = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(int) * nb, cpu.by().data());
  cl_mem bufWinAna = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winAna().data());
  cl_mem bufWinA = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynA().data());
  cl_mem bufWinB = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * bw * bh, cpu.winSynB().data());
  std::vector<unsigned char> par(nb);
  for (int i = 0; i < nb; i++)
    par[i] = cpu.par()[i];
  cl_mem bufPar = makeBuf(gpu, CL_MEM_READ_ONLY, nb, par.data());
  cl_mem bufFilt = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * 2 * nb * planeSize, nullptr);
  cl_mem bufBlocks = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * (size_t)nb * bw * bh, nullptr);
  cl_mem bufOut = makeBuf(gpu, CL_MEM_WRITE_ONLY, sizeof(float) * W * H, nullptr);
  // degrid sample: single planeSize copy (identical for all blocks).
  std::vector<float> gs(2 * planeSize, 0.0f);
  if (ip.degridOn)
    for (int i = 0; i < planeSize; i++) {
      gs[2 * i] = cpu.gsRe()[i];
      gs[2 * i + 1] = cpu.gsIm()[i];
    }
  cl_mem bufGs = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * gs.size(), gs.data());
  std::vector<float> pat = cpu.pattern();
  if (pat.empty()) pat.assign(planeSize, 0.0f);
  cl_mem bufPat = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * pat.size(), pat.data());
  std::vector<float> shp = cpu.sharpenLut();
  if (shp.empty()) shp.assign(planeSize, 0.0f);
  cl_mem bufShp = makeBuf(gpu, CL_MEM_READ_ONLY, sizeof(float) * shp.size(), shp.data());
  cl_mem bufKLast = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * 2 * nb * planeSize, nullptr);
  cl_mem bufKCov = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * nb * planeSize, nullptr);
  cl_mem bufKPro = makeBuf(gpu, CL_MEM_READ_WRITE, sizeof(float) * nb * planeSize, nullptr);

  const char* ksrc = kKernelSource();
  cl_program prog = gpu.api.createProgramWithSource(gpu.context, 1, &ksrc, nullptr, &e);
  CHECK(e);
  char opts[256];
  snprintf(opts, sizeof(opts),
           "-DLSZ=%d -DLSW=%d -DLP=%d -DLSZP=%d -DBM=%d -DDG=%d -DPAT=%d -DSHP=%d -DRAW=0 "
           "-cl-fp32-correctly-rounded-divide-sqrt",
           bw * bh, bw > bh ? bw : bh, bw + 1, (bw + 1) * bh, ip.bm, ip.degridOn ? 1 : 0,
           ip.usePattern ? 1 : 0, ip.sharpenOn ? 1 : 0);
  e = gpu.api.buildProgram(prog, 1, &gpu.device, opts, nullptr, nullptr);
  if (e != CL_SUCCESS) {
    char log[8192] = {};
    gpu.api.getProgramBuildInfo(prog, gpu.device, 0x1183, sizeof(log), log, nullptr);
    printf("build failed (%d):\n%s\n", e, log);
    return 1;
  }

  int wg = bw > bh ? bw : bh;
  if (wg > (int)gpu.maxWorkGroupSize) wg = (int)gpu.maxWorkGroupSize;
  size_t goff = 0, gsize, lsize = wg;

  // Forward FFT per frame.
  cl_kernel kFwd = gpu.api.createKernel(prog, "fft2d_fwd", &e);
  CHECK(e);
  CHECK(gpu.api.setKernelArg(kFwd, 0, sizeof(cl_mem), &bufPlane[0])); // replaced per frame
  CHECK(gpu.api.setKernelArg(kFwd, 1, sizeof(cl_mem), &bufBx));
  CHECK(gpu.api.setKernelArg(kFwd, 2, sizeof(cl_mem), &bufBy));
  CHECK(gpu.api.setKernelArg(kFwd, 4, sizeof(cl_mem), &bufWinAna));
  CHECK(gpu.api.setKernelArg(kFwd, 5, sizeof(int), &W));
  CHECK(gpu.api.setKernelArg(kFwd, 6, sizeof(int), &H));
  CHECK(gpu.api.setKernelArg(kFwd, 7, sizeof(int), &bw));
  CHECK(gpu.api.setKernelArg(kFwd, 8, sizeof(int), &bh));
  CHECK(gpu.api.setKernelArg(kFwd, 9, sizeof(int), &fbw));
  CHECK(gpu.api.setKernelArg(kFwd, 10, sizeof(int), &nb));
  float cn = cpu.chromaNorm();
  CHECK(gpu.api.setKernelArg(kFwd, 11, sizeof(float), &cn));
  float one = 1.0f, zero = 0.0f; // norm/invNorm/maxPix: unused with float samples
  CHECK(gpu.api.setKernelArg(kFwd, 12, sizeof(float), &one));
  gsize = (size_t)nb * wg;
  for (int f = 0; f < nframes; f++) {
    CHECK(gpu.api.setKernelArg(kFwd, 0, sizeof(cl_mem), &bufPlane[f]));
    CHECK(gpu.api.setKernelArg(kFwd, 3, sizeof(cl_mem), &bufSpec[f]));
    CHECK(gpu.api.enqueueNDRange(gpu.queue, kFwd, 1, &goff, &gsize, &lsize, 0, nullptr, nullptr));
  }

  // Kalman init on the first frame.
  if (ip.bm == 0) {
    cl_kernel ki = gpu.api.createKernel(prog, "kalman_init", &e);
    CHECK(e);
    CHECK(gpu.api.setKernelArg(ki, 0, sizeof(cl_mem), &bufSpec[0]));
    CHECK(gpu.api.setKernelArg(ki, 1, sizeof(cl_mem), &bufKLast));
    CHECK(gpu.api.setKernelArg(ki, 2, sizeof(cl_mem), &bufKCov));
    CHECK(gpu.api.setKernelArg(ki, 3, sizeof(cl_mem), &bufKPro));
    CHECK(gpu.api.setKernelArg(ki, 4, sizeof(cl_mem), &bufPat));
    int total = nb * planeSize;
    CHECK(gpu.api.setKernelArg(ki, 5, sizeof(int), &total));
    CHECK(gpu.api.setKernelArg(ki, 6, sizeof(int), &planeSize));
    CHECK(gpu.api.setKernelArg(ki, 7, sizeof(float), &ip.kalmanCN));
    size_t gs1 = ((size_t)total + 63) / 64 * 64;
    CHECK(gpu.api.enqueueNDRange(gpu.queue, ki, 1, nullptr, &gs1, nullptr, 0, nullptr, nullptr));
    gpu.api.releaseKernel(ki);
  }

  // Temporal filter + sharpen.
  cl_kernel kFil = gpu.api.createKernel(prog, "filter", &e);
  CHECK(e);
  cl_mem s0 = bufSpec[0], s1 = nframes > 1 ? bufSpec[1] : bufSpec[0];
  cl_mem s2 = nframes > 2 ? bufSpec[2] : bufSpec[0];
  cl_mem s3 = nframes > 3 ? bufSpec[3] : bufSpec[0];
  CHECK(gpu.api.setKernelArg(kFil, 0, sizeof(cl_mem), &s0));
  CHECK(gpu.api.setKernelArg(kFil, 1, sizeof(cl_mem), &s1));
  CHECK(gpu.api.setKernelArg(kFil, 2, sizeof(cl_mem), &s2));
  CHECK(gpu.api.setKernelArg(kFil, 3, sizeof(cl_mem), &s3));
  CHECK(gpu.api.setKernelArg(kFil, 4, sizeof(cl_mem), &bufFilt));
  CHECK(gpu.api.setKernelArg(kFil, 5, sizeof(cl_mem), &bufGs));
  CHECK(gpu.api.setKernelArg(kFil, 6, sizeof(cl_mem), &bufPat));
  CHECK(gpu.api.setKernelArg(kFil, 7, sizeof(cl_mem), &bufShp));
  CHECK(gpu.api.setKernelArg(kFil, 8, sizeof(cl_mem), &bufKLast));
  CHECK(gpu.api.setKernelArg(kFil, 9, sizeof(cl_mem), &bufKCov));
  CHECK(gpu.api.setKernelArg(kFil, 10, sizeof(cl_mem), &bufKPro));
  CHECK(gpu.api.setKernelArg(kFil, 11, sizeof(int), &nb));
  CHECK(gpu.api.setKernelArg(kFil, 12, sizeof(int), &fbw));
  CHECK(gpu.api.setKernelArg(kFil, 13, sizeof(int), &bh));
  CHECK(gpu.api.setKernelArg(kFil, 14, sizeof(float), &ip.wienerS));
  float sigmaX = ip.beta * ip.wienerS, sigmaY = ip.wienerS;
  // NOTE: kernel arg order is sigmaX, sigmaY, betaX, beta
  CHECK(gpu.api.setKernelArg(kFil, 14, sizeof(float), &sigmaX));
  CHECK(gpu.api.setKernelArg(kFil, 15, sizeof(float), &sigmaY));
  CHECK(gpu.api.setKernelArg(kFil, 16, sizeof(float), &ip.betaX));
  CHECK(gpu.api.setKernelArg(kFil, 17, sizeof(float), &ip.beta));
  CHECK(gpu.api.setKernelArg(kFil, 18, sizeof(float), &ip.ssmin));
  CHECK(gpu.api.setKernelArg(kFil, 19, sizeof(float), &ip.ssmax));
  CHECK(gpu.api.setKernelArg(kFil, 20, sizeof(float), &ip.kalmanCN));
  CHECK(gpu.api.setKernelArg(kFil, 21, sizeof(float), &ip.kratio));
  size_t gsFil = (size_t)nb * planeSize;
  CHECK(gpu.api.enqueueNDRange(gpu.queue, kFil, 1, nullptr, &gsFil, nullptr, 0, nullptr, nullptr));

  // Inverse FFT + synthesis window.
  cl_kernel kInv = gpu.api.createKernel(prog, "fft2d_inv", &e);
  CHECK(e);
  CHECK(gpu.api.setKernelArg(kInv, 0, sizeof(cl_mem), &bufFilt));
  CHECK(gpu.api.setKernelArg(kInv, 1, sizeof(cl_mem), &bufBlocks));
  CHECK(gpu.api.setKernelArg(kInv, 2, sizeof(cl_mem), &bufWinA));
  CHECK(gpu.api.setKernelArg(kInv, 3, sizeof(cl_mem), &bufWinB));
  CHECK(gpu.api.setKernelArg(kInv, 4, sizeof(cl_mem), &bufPar));
  CHECK(gpu.api.setKernelArg(kInv, 5, sizeof(int), &bw));
  CHECK(gpu.api.setKernelArg(kInv, 6, sizeof(int), &bh));
  CHECK(gpu.api.setKernelArg(kInv, 7, sizeof(int), &fbw));
  CHECK(gpu.api.setKernelArg(kInv, 8, sizeof(int), &nb));
  gsize = (size_t)nb * wg;
  CHECK(gpu.api.enqueueNDRange(gpu.queue, kInv, 1, &goff, &gsize, &lsize, 0, nullptr, nullptr));

  // Overlap-add synthesis.
  int xoff = -ip.ow, xstep = ip.bw - ip.ow, yoff = -ip.oh, ystep = ip.bh - ip.oh;
  if (ip.mode == 0) { xoff = -bw / 2; xstep = bw; yoff = -bh / 2; ystep = bh; }
  cl_kernel kSyn = gpu.api.createKernel(prog, "synthesize", &e);
  CHECK(e);
  CHECK(gpu.api.setKernelArg(kSyn, 0, sizeof(cl_mem), &bufBlocks));
  CHECK(gpu.api.setKernelArg(kSyn, 1, sizeof(cl_mem), &bufOut));
  CHECK(gpu.api.setKernelArg(kSyn, 2, sizeof(int), &W));
  CHECK(gpu.api.setKernelArg(kSyn, 3, sizeof(int), &H));
  CHECK(gpu.api.setKernelArg(kSyn, 4, sizeof(int), &bw));
  CHECK(gpu.api.setKernelArg(kSyn, 5, sizeof(int), &bh));
  CHECK(gpu.api.setKernelArg(kSyn, 6, sizeof(int), &xoff));
  CHECK(gpu.api.setKernelArg(kSyn, 7, sizeof(int), &xstep));
  CHECK(gpu.api.setKernelArg(kSyn, 8, sizeof(int), &yoff));
  CHECK(gpu.api.setKernelArg(kSyn, 9, sizeof(int), &ystep));
  CHECK(gpu.api.setKernelArg(kSyn, 10, sizeof(int), &ncols));
  CHECK(gpu.api.setKernelArg(kSyn, 11, sizeof(int), &nrows));
  CHECK(gpu.api.setKernelArg(kSyn, 12, sizeof(float), &cn));
  CHECK(gpu.api.setKernelArg(kSyn, 13, sizeof(float), &one));
  CHECK(gpu.api.setKernelArg(kSyn, 14, sizeof(float), &zero));
  size_t gs2[2] = {(size_t)W, (size_t)H};
  CHECK(gpu.api.enqueueNDRange(gpu.queue, kSyn, 2, nullptr, gs2, nullptr, 0, nullptr, nullptr));
  CHECK(gpu.api.finish(gpu.queue));

  std::vector<float> gpuOut(W * H);
  CHECK(gpu.api.enqueueReadBuffer(gpu.queue, bufOut, CL_TRUE, 0, sizeof(float) * W * H,
                                  gpuOut.data(), 0, nullptr, nullptr));

  double maxd = 0, sumd = 0;
  int worst = 0;
  for (int i = 0; i < W * H; i++) {
    double d = std::abs((double)gpuOut[i] - cpuOut[i]);
    sumd += d;
    if (d > maxd) { maxd = d; worst = i; }
  }
  printf("bt=%d sharpen=%g degrid=%g: GPU vs CPU maxdiff=%.6f (at %d,%d) mean=%.7f\n",
         ip.bm, sp.sharpen, sp.degrid, maxd, worst % W, worst / W, sumd / (W * H));
  printf("%s\n", maxd < 2e-4 ? "PASS" : "FAIL");

  gpu.api.releaseKernel(kFwd);
  gpu.api.releaseKernel(kFil);
  gpu.api.releaseKernel(kInv);
  gpu.api.releaseKernel(kSyn);
  gpu.api.releaseProgram(prog);
  for (int f = 0; f < nframes; f++) {
    gpu.api.releaseMemObject(bufPlane[f]);
    gpu.api.releaseMemObject(bufSpec[f]);
  }
  gpu.api.releaseMemObject(bufBx);
  gpu.api.releaseMemObject(bufBy);
  gpu.api.releaseMemObject(bufWinAna);
  gpu.api.releaseMemObject(bufWinA);
  gpu.api.releaseMemObject(bufWinB);
  gpu.api.releaseMemObject(bufPar);
  gpu.api.releaseMemObject(bufFilt);
  gpu.api.releaseMemObject(bufBlocks);
  gpu.api.releaseMemObject(bufOut);
  gpu.api.releaseMemObject(bufGs);
  gpu.api.releaseMemObject(bufPat);
  gpu.api.releaseMemObject(bufShp);
  gpu.api.releaseMemObject(bufKLast);
  gpu.api.releaseMemObject(bufKCov);
  gpu.api.releaseMemObject(bufKPro);
  gpu.shutdown();
  return maxd < 2e-4 ? 0 : 1;
}
