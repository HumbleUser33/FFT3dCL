// Standalone round-trip test for the CPU FFT helpers.
#include "fft.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace fft3dgpu;

int main() {
  const int bw = 32, bh = 32, fbw = bw / 2 + 1;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  FftPlan rowPlan(bw), colPlan(bh);
  std::vector<float> block(bw * bh), back(bw * bh);
  std::vector<std::complex<float>> full(bw * bh), half(fbw * bh);

  for (auto& v : block)
    v = dist(rng);

  fft2dRealForward(block.data(), bw, bh, rowPlan, colPlan, full.data());

  // sanity: DC = sum
  float sum = 0;
  for (float v : block)
    sum += v;
  printf("DC re=%.4f im=%.4f (sum=%.4f)\n", full[0].real(), full[0].imag(), sum);

  for (int r = 0; r < bh; r++)
    for (int c = 0; c < fbw; c++)
      half[r * fbw + c] = full[r * bw + c];

  fft2dRealInverseHalf(half.data(), bw, bh, fbw, rowPlan, colPlan, back.data());

  float maxerr = 0;
  for (int i = 0; i < bw * bh; i++)
    maxerr = std::max(maxerr, std::fabs(block[i] - back[i]));
  printf("roundtrip maxerr = %.6f\n", maxerr);

  // 1D check
  std::vector<std::complex<float>> a(8), b(8), c(8);
  for (int i = 0; i < 8; i++)
    a[i] = std::complex<float>((float)i, 0);
  FftPlan p8(8);
  p8.forward(a.data(), b.data());
  printf("fwd[1] = %.4f %.4f (expect -4.0000 9.6569)\n", b[1].real(), b[1].imag());
  p8.inverse(b.data(), c.data());
  float e = 0;
  for (int i = 0; i < 8; i++)
    e = std::max(e, std::abs(a[i] - c[i]));
  printf("1d roundtrip maxerr = %.6f\n", e);
  return 0;
}
