#pragma once
#include <complex>
#include <vector>

namespace fft3dgpu {

// 1D complex FFT, length = power of two. Stockham autosort radix-2.
// Twiddle factors are computed once in double and stored as float, matching
// the original's double CPU tables uploaded as float textures.
// forward(): unnormalized. inverse(): scaled by 1/n.
class FftPlan {
public:
  explicit FftPlan(int n);

  // out may alias in.
  void forward(const std::complex<float>* in, std::complex<float>* out) const;
  void inverse(const std::complex<float>* in, std::complex<float>* out) const;

  int size() const { return n_; }
  const std::vector<std::complex<float>>& twiddles() const { return tw_; }
  const std::vector<int>& reversal() const { return rev_; }

private:
  int n_;
  std::vector<std::complex<float>> tw_; // exp(-2*pi*i*k/n), k < n/2
  std::vector<int> rev_;                // bit-reversal permutation
};

// 2D real FFT of a bw x bh block (row-major real input).
// forward: full complex spectrum (unnormalized) in 'full' (bw*bh complex).
// inverse: rebuilds the Hermitian-symmetric full spectrum from the kept
// half-spectrum columns 0..fbw-1 (fbw = bw/2+1), then inverse-transforms
// with the 1/(bw*bh) normalization.
void fft2dRealForward(const float* block, int bw, int bh,
                      const FftPlan& rowPlan, const FftPlan& colPlan,
                      std::complex<float>* full);
void fft2dRealInverseHalf(const std::complex<float>* half, int bw, int bh, int fbw,
                          const FftPlan& rowPlan, const FftPlan& colPlan,
                          float* block);

} // namespace fft3dgpu
