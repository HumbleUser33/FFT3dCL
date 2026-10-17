#include "fft.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace fft3dgpu {

FftPlan::FftPlan(int n) : n_(n) {
  tw_.resize(n / 2);
  for (int k = 0; k < n / 2; k++) {
    double a = -2.0 * 3.14159265358979323846 * k / n;
    tw_[k] = std::complex<float>((float)std::cos(a), (float)std::sin(a));
  }
  rev_.resize(n);
  int bits = 0;
  while ((1 << bits) < n)
    bits++;
  for (int i = 0; i < n; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++)
      if (i & (1 << b))
        r |= 1 << (bits - 1 - b);
    rev_[i] = r;
  }
}

// Classic iterative Cooley-Tukey DIT: bit-reversal then butterflies.
static void transform(const FftPlan& plan, const std::complex<float>* in,
                      std::complex<float>* out, bool inverse) {
  const int n = plan.size();
  const std::vector<std::complex<float>>& tw = plan.twiddles();
  const std::vector<int>& rev = plan.reversal();
  std::vector<std::complex<float>> a(n);
  for (int i = 0; i < n; i++)
    a[i] = in[rev[i]];
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    const int step = n / len;
    for (int i = 0; i < n; i += len)
      for (int j = 0; j < half; j++) {
        std::complex<float> w = tw[j * step];
        if (inverse)
          w = std::conj(w);
        std::complex<float> t = w * a[i + j + half];
        std::complex<float> u = a[i + j];
        a[i + j] = u + t;
        a[i + j + half] = u - t;
      }
  }
  if (inverse) {
    const float invn = 1.0f / n;
    for (int i = 0; i < n; i++)
      out[i] = a[i] * invn;
  } else {
    for (int i = 0; i < n; i++)
      out[i] = a[i];
  }
}

void FftPlan::forward(const std::complex<float>* in, std::complex<float>* out) const {
  transform(*this, in, out, false);
}

void FftPlan::inverse(const std::complex<float>* in, std::complex<float>* out) const {
  transform(*this, in, out, true);
}

void fft2dRealForward(const float* block, int bw, int bh,
                      const FftPlan& rowPlan, const FftPlan& colPlan,
                      std::complex<float>* full) {
  // rows
  std::vector<std::complex<float>> row(bw), col(bh), colout(bh);
  for (int r = 0; r < bh; r++) {
    for (int c = 0; c < bw; c++)
      row[c] = std::complex<float>(block[r * bw + c], 0.0f);
    rowPlan.forward(row.data(), full + r * bw);
  }
  // columns
  for (int c = 0; c < bw; c++) {
    for (int r = 0; r < bh; r++)
      col[r] = full[r * bw + c];
    colPlan.forward(col.data(), colout.data());
    for (int r = 0; r < bh; r++)
      full[r * bw + c] = colout[r];
  }
}

void fft2dRealInverseHalf(const std::complex<float>* half, int bw, int bh, int fbw,
                          const FftPlan& rowPlan, const FftPlan& colPlan,
                          float* block) {
  // Rebuild the full Hermitian spectrum from columns 0..fbw-1.
  std::vector<std::complex<float>> full(bw * bh);
  for (int r = 0; r < bh; r++)
    for (int c = 0; c < fbw; c++)
      full[r * bw + c] = half[r * fbw + c];
  for (int r = 0; r < bh; r++) {
    int rm = (bh - r) % bh;
    for (int c = fbw; c < bw; c++) {
      int cm = bw - c; // in 1..bw/2-1
      full[r * bw + c] = std::conj(full[rm * bw + cm]);
    }
  }
  // columns inverse
  std::vector<std::complex<float>> col(bh), colout(bh);
  for (int c = 0; c < bw; c++) {
    for (int r = 0; r < bh; r++)
      col[r] = full[r * bw + c];
    colPlan.inverse(col.data(), colout.data());
    for (int r = 0; r < bh; r++)
      full[r * bw + c] = colout[r];
  }
  // rows inverse
  std::vector<std::complex<float>> rowout(bw);
  for (int r = 0; r < bh; r++) {
    rowPlan.inverse(full.data() + r * bw, rowout.data());
    for (int c = 0; c < bw; c++)
      block[r * bw + c] = rowout[c].real();
  }
}

} // namespace fft3dgpu
