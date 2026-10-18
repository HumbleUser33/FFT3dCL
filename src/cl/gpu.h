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

namespace fft3dcl {

class GpuRef {
public:
  // Builds and compiles the program for this instance's parameter set.
  GpuRef(GpuContext& ctx, const InternalParams& p, const PlaneGeom& g, const PlaneRef& cpu);
  ~GpuRef();

  bool ok() const { return ok_; }
  const std::string& error() const { return err_; }

  // Process-wide totals, so a log can tell a real leak (instances still alive)
  // from the driver keeping freed memory (no instance alive, memory still used).
  static int liveInstances();
  static size_t liveBytes();

  // Forward FFT of an uploaded float plane; result cached under frame key fn.
  bool forward(int fn, const std::vector<float>& plane);
  // Same, but the plane is handed over as the raw 8/16-bit bytes VapourSynth
  // gives us and converted to float on the device. No CPU conversion at all.
  bool forwardRaw(int fn, const void* raw, size_t bytes);
  bool has(int fn) const { return specCache_.count(fn) != 0; }
  // True when this plane can use the raw path (integer samples).
  bool rawPath() const { return rawPath_; }
  int bytesPerSample() const { return bps_; }

  // Degrid sample upload (from cpu.gsRe/gsIm, single planeSize copy).
  bool uploadDegridSample(const PlaneRef& cpu);

  // Full filter + inverse + synthesis for the given window frames (frame
  // numbers, in temporal order). out is W*H floats.
  bool process(const std::vector<int>& window, std::vector<float>& out);
  // Same, but the result is converted back to raw 8/16-bit bytes on the device
  // and read straight into dst (which must hold W*H*bytesPerSample bytes).
  bool processRaw(const std::vector<int>& window, void* dst, size_t bytes);

  // Drop cached spectra outside [lo, hi]; -1 disables a bound.
  void pruneCache(int lo, int hi);

private:
  cl_mem spectra(int fn); // create-or-get device buffer for frame fn
  void retireSpectra(cl_mem m); // put a cache buffer back in the reuse pool
  cl_mem mkTracked(cl_uint flags, size_t sz, const void* host); // mkBuf + byte count
  bool launchFwd(cl_mem spec);  // fft2d_fwd: bufPlane_ -> spec
  bool unpack();                // raw bytes -> bufPlane_ as float
  bool pack();                  // bufOut_ float -> raw bytes
  bool runKernels(const std::vector<int>& window); // filter + inverse + synthesis

  GpuContext* ctx_;
  InternalParams p_;
  PlaneGeom g_;
  bool ok_ = false;
  std::string err_;

  cl_program prog_ = nullptr;
  cl_kernel kFwd_ = nullptr, kFil_ = nullptr, kInv_ = nullptr, kSyn_ = nullptr,
            kKalInit_ = nullptr, kUnpack_ = nullptr, kPack_ = nullptr;

  cl_mem bufBx_ = nullptr, bufBy_ = nullptr, bufWinAna_ = nullptr, bufWinA_ = nullptr,
         bufWinB_ = nullptr, bufPar_ = nullptr, bufGs_ = nullptr, bufPat_ = nullptr,
         bufShp_ = nullptr, bufFilt_ = nullptr, bufBlocks_ = nullptr, bufOut_ = nullptr,
         bufPlane_ = nullptr, bufRaw_ = nullptr, bufKLast_ = nullptr, bufKCov_ = nullptr,
         bufKPro_ = nullptr;

  bool rawPath_ = false; // integer samples: the raw byte path is available
  int bps_ = 4;          // bytes per sample in the raw buffers

  std::map<int, cl_mem> specCache_;
  size_t bytes_ = 0; // device bytes held by this instance
  // Retired spectra buffers. Reusing them avoids a create/release round-trip
  // per frame, which the driver answers by keeping the memory (visible as
  // slowly growing GPU memory across previews).
  std::vector<cl_mem> specPool_;
  bool kalmanFirst_ = true;
  int nb_, fbw_, planeSize_;
  size_t wgFft_ = 0;
};

} // namespace fft3dcl
