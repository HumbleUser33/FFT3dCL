#pragma once
// CPU reference implementation of the FFT3dGPU v0.8.6 pipeline.
// Faithful to the original's math (fp32 / precision=2 semantics):
// mode 0 and mode 1, bt -1..4, Wiener/Kalman/sharpen/degrid/wintype/pattern.
// This is the oracle for the later OpenCL port; speed is secondary.
//
// Not implemented yet (rejected at parameter validation): mode 2, interlaced.

#include "params.h"
#include "fft.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace fft3dcl {

struct PlaneGeom {
  int width = 0, height = 0;
  int bits = 8;
  int sampleType = 0; // 0 = integer, 1 = float
  bool chroma = false; // YUV chroma plane (mode 1 centering)
};

class PlaneRef {
public:
  PlaneRef(const InternalParams& p, const PlaneGeom& g);

  // Source plane (native format) -> internal float.
  void upload(const uint8_t* src, ptrdiff_t pitch, std::vector<float>& dst) const;
  // Internal float -> destination plane (native format).
  void download(const std::vector<float>& src, uint8_t* dst, ptrdiff_t pitch) const;

  // Per-frame block spectra: fbw columns x bh rows per block, re/im separate.
  struct Spectra {
    std::vector<float> re, im; // size nblocks * fbw * bh
  };
  std::shared_ptr<Spectra> computeSpectra(const std::vector<float>& plane);

  // Degrid: gridsample pattern from the synthetic white reference frame
  // (constructor equivalent): FFT(analysis window) / DC(window) * degrid.
  void buildDegridSample();

  // Filter + inverse FFT + overlap-add synthesis.
  // bm >= 2: window holds bt spectra, current frame at index p_.current.
  // bm <= 1: window holds one spectrum (the current frame).
  void filterAndSynthesize(const std::vector<std::shared_ptr<Spectra>>& window,
                           std::vector<float>& out);

  int nblocks() const { return static_cast<int>(bx_.size()); }

  // Geometry access for the GPU backend.
  const std::vector<int>& bx() const { return bx_; }
  const std::vector<int>& by() const { return by_; }
  const std::vector<uint8_t>& par() const { return par_; }
  const std::vector<float>& winAna() const { return winAna_; }
  const std::vector<float>& winSynA() const { return winSynA_; }
  const std::vector<float>& winSynB() const { return winSynB_; }
  int ncols() const { return ncols_; }
  int nrows() const { return ncols_ ? nblocks() / ncols_ : nblocks(); }
  int fbw() const { return fbw_; }
  float chromaNorm() const { return chromaNorm_; }
  // Filter tables (empty if the feature is off).
  const std::vector<float>& gsRe() const { return gsRe_; }
  const std::vector<float>& gsIm() const { return gsIm_; }
  const std::vector<float>& pattern() const { return pattern_; }
  const std::vector<float>& sharpenLut() const { return sharpen_; }

private:
  int specIdx(int b, int c, int r) const { return (b * fbw_ + c) * p_.bh + r; }

  InternalParams p_;
  PlaneGeom g_;
  int fbw_;

  // Block grid.
  std::vector<int> bx_, by_;
  std::vector<uint8_t> par_; // mode 1: 0 -> synthesis window A, 1 -> window B
  int ncols_ = 0;            // mode 1: blocks per row (0 for mode 0)

  // Tables.
  std::vector<float> winAna_;  // mode 0: w; mode 1: A (analysis)
  std::vector<float> winSynA_; // mode 1: synthesis for even-parity grid
  std::vector<float> winSynB_; // mode 1: synthesis for odd-parity grid
  std::vector<float> pattern_; // fbw x bh motif (empty = constant sigma)
  std::vector<float> sharpen_; // fbw x bh motif (empty = off)

  FftPlan rowPlan_, colPlan_;

  // Scratch (per instance, used inside filterAndSynthesize).
  std::vector<float> blockReal_;
  std::vector<std::complex<float>> fullSpec_;
  std::vector<float> filtRe_, filtIm_, corrRe_, corrIm_;

  // Degrid state.
  std::vector<float> gsRe_, gsIm_; // gridsample, nblocks*fbw*bh
  bool degridReady_ = false;

  // Kalman state (bm == 0).
  std::vector<float> kLastRe_, kLastIm_, kCovarRe_, kCovarIm_, kProcRe_, kProcIm_;
  bool kalmanFirst_ = true;

  float chromaNorm_ = 0.0f; // CHROMANORM (mode 1 chroma only)
};

} // namespace fft3dcl
