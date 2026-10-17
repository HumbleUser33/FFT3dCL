#pragma once
// GPU (OpenCL 1.2) backend mirroring PlaneRef's pipeline:
// forward spectra per frame (cached in device buffers), temporal filter +
// sharpen, inverse FFT + overlap-add synthesis. Geometry and tables are
// copied from the CPU PlaneRef so both backends agree exactly.

#include "cl.h"
#include "params.h"
#include "../cpu/reference.h"
#include <map>
#include <string>
#include <vector>

namespace fft3dgpu {

class GpuRef {
public:
  // Builds and compiles the program for this instance's parameter set.
  GpuRef(GpuContext& ctx, const InternalParams& p, const PlaneGeom& g, const PlaneRef& cpu);
  ~GpuRef();

  bool ok() const { return ok_; }
  const std::string& error() const { return err_; }

  // Forward FFT of an uploaded float plane; result cached under frame key fn.
  bool forward(int fn, const std::vector<float>& plane);
  bool has(int fn) const { return specCache_.count(fn) != 0; }

  // Degrid sample upload (from cpu.gsRe/gsIm, single planeSize copy).
  bool uploadDegridSample(const PlaneRef& cpu);

  // Full filter + inverse + synthesis for the given window frames (frame
  // numbers, in temporal order). out is W*H floats.
  bool process(const std::vector<int>& window, std::vector<float>& out);

  // Drop cached spectra outside [lo, hi]; -1 disables a bound.
  void pruneCache(int lo, int hi);

private:
  cl_mem spectra(int fn); // create-or-get device buffer for frame fn

  GpuContext* ctx_;
  InternalParams p_;
  PlaneGeom g_;
  bool ok_ = false;
  std::string err_;

  cl_program prog_ = nullptr;
  cl_kernel kFwd_ = nullptr, kFil_ = nullptr, kInv_ = nullptr, kSyn_ = nullptr,
            kKalInit_ = nullptr;

  cl_mem bufBx_ = nullptr, bufBy_ = nullptr, bufWinAna_ = nullptr, bufWinA_ = nullptr,
         bufWinB_ = nullptr, bufPar_ = nullptr, bufGs_ = nullptr, bufPat_ = nullptr,
         bufShp_ = nullptr, bufFilt_ = nullptr, bufBlocks_ = nullptr, bufOut_ = nullptr,
         bufPlane_ = nullptr, bufKLast_ = nullptr, bufKCov_ = nullptr, bufKPro_ = nullptr;

  std::map<int, cl_mem> specCache_;
  bool kalmanFirst_ = true;
  int nb_, fbw_, planeSize_;
  size_t wgFft_ = 0;
};

} // namespace fft3dgpu
