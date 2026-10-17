#include "tables.h"
#include <cmath>

namespace fft3dgpu {

static const double pi = 3.14159265358979323846;

std::vector<float> buildWindowMode0(int bw, int bh) {
  std::vector<float> w(bw * bh);
  int off = 0;
  for (double n1 = -bh / 2.0 + 0.5; n1 < bh / 2.0; n1++) {
    double cosy = std::cos(n1 * pi / bh);
    for (double n2 = -bw / 2.0 + 0.5; n2 < bw / 2.0; n2++) {
      double cosx = std::cos(n2 * pi / bw);
      w[off++] = (float)std::sqrt(0.5 * (cosx * cosx + cosy * cosy));
    }
  }
  return w;
}

void buildWindowsMode1(int bw, int bh, int ow, int oh, int wintype,
                       std::vector<float>& A, std::vector<float>& SynA,
                       std::vector<float>& SynB) {
  std::vector<double> wanx(ow), wany(oh), wsynx(ow), wsyny(oh);
  switch (wintype) {
  case 1:
    for (int i = 0; i < ow; i++) {
      wanx[i] = std::sqrt(std::cos(pi * (i - ow + 0.5) / (ow * 2.0)));
      wsynx[i] = wanx[i] * wanx[i] * wanx[i];
    }
    for (int i = 0; i < oh; i++) {
      wany[i] = std::sqrt(std::cos(pi * (i - oh + 0.5) / (oh * 2.0)));
      wsyny[i] = wany[i] * wany[i] * wany[i];
    }
    break;
  case 2:
    for (int i = 0; i < ow; i++) {
      wanx[i] = 1.0;
      wsynx[i] = std::cos(pi * (i - ow + 0.5) / (ow * 2.0));
      wsynx[i] *= wsynx[i];
    }
    for (int i = 0; i < oh; i++) {
      wany[i] = 1.0;
      wsyny[i] = std::cos(pi * (i - oh + 0.5) / (oh * 2.0));
      wsyny[i] *= wsyny[i];
    }
    break;
  default: // wintype 0
    for (int i = 0; i < ow; i++) {
      wanx[i] = std::cos(pi * (i - ow + 0.5) / (ow * 2.0));
      wsynx[i] = wanx[i];
    }
    for (int i = 0; i < oh; i++) {
      wany[i] = std::cos(pi * (i - oh + 0.5) / (oh * 2.0));
      wsyny[i] = wany[i];
    }
    break;
  }

  // Horizontal ramps. Analysis map (MapAna) uses wanx/wany; synthesis map
  // (MapSyn) uses wsynx/wsyny with identical packing (filters.cpp:558-663):
  // xy components = rising ramp, zw components = reversed ramp, middle = 1.
  std::vector<double> wxA(bw), wxB(bw), wxSA(bw), wxSB(bw), wyA(bh), wySA(bh);
  for (int lx = 0; lx < ow; lx++) {
    wxA[lx] = wanx[lx];
    wxB[lx] = wanx[(ow - 1) - lx];
    wxSA[lx] = wsynx[lx];
    wxSB[lx] = wsynx[(ow - 1) - lx];
  }
  for (int lx = ow; lx < bw - ow; lx++) {
    wxA[lx] = 1.0;
    wxB[lx] = 1.0;
    wxSA[lx] = 1.0;
    wxSB[lx] = 1.0;
  }
  for (int lx = bw - ow; lx < bw; lx++) {
    int x = lx - (bw - ow);
    wxA[lx] = wanx[(ow - 1) - x]; // = wanx[bw-1-lx]
    wxB[lx] = wanx[x];
    wxSA[lx] = wsynx[(ow - 1) - x];
    wxSB[lx] = wsynx[x];
  }
  for (int ly = 0; ly < oh; ly++) {
    wyA[ly] = wany[ly];
    wySA[ly] = wsyny[ly];
  }
  for (int ly = oh; ly < bh - oh; ly++) {
    wyA[ly] = 1.0;
    wySA[ly] = 1.0;
  }
  for (int ly = bh - oh; ly < bh; ly++) {
    wyA[ly] = wany[bh - 1 - ly];
    wySA[ly] = wsyny[bh - 1 - ly];
  }

  A.resize(bw * bh);
  SynA.resize(bw * bh);
  SynB.resize(bw * bh);
  for (int ly = 0; ly < bh; ly++)
    for (int lx = 0; lx < bw; lx++) {
      A[ly * bw + lx] = (float)(wxA[lx] * wyA[ly]);
      SynA[ly * bw + lx] = (float)(wxSA[lx] * wySA[ly]);
      SynB[ly * bw + lx] = (float)(wxSB[lx] * wySA[ly]);
    }
}

std::vector<float> buildPattern(int bw, int bh, int bt, float sigma,
                                float sigma2, float sigma3, float sigma4,
                                bool usePattern) {
  if (!usePattern)
    return {};
  const int fbw = bw / 2 + 1;
  auto scale = [&](float s) {
    double v = s / 255.0;
    return (float)(v * v * bt * bw * bh);
  };
  float s = scale(sigma), s2 = scale(sigma2), s3 = scale(sigma3), s4 = scale(sigma4);
  float ft2 = (float)(std::sqrt(0.5) / 2.0);
  float ft3 = (float)(std::sqrt(0.5) / 4.0);
  std::vector<float> pat(fbw * bh);
  for (int h = 0; h < bh; h++) {
    float fy = (bh - 2.0f * std::fabs(h - bh / 2.0f)) / bh;
    for (int w = 0; w < fbw; w++) {
      float fx = w * 1.0f / fbw;
      float f = std::sqrt((fx * fx + fy * fy) * 0.5f);
      float p;
      if (f < ft3)
        p = s4 + (s3 - s4) * f / ft3;
      else if (f < ft2)
        p = s3 + (s2 - s3) * (f - ft3) / (ft2 - ft3);
      else
        p = s + (s2 - s) * (1.0f - f) / (1.0f - ft2);
      pat[h * fbw + w] = p;
    }
  }
  return pat;
}

std::vector<float> buildSharpenLUT(int bw, int bh, float strength, float svr,
                                   float scutoff) {
  if (strength == 0.0f)
    return {};
  const int fbw = bw / 2 + 1;
  std::vector<float> lut(fbw * bh);
  for (int j = 0; j < bh; j++) {
    int dj = j;
    if (j >= bh / 2)
      dj = bh - j;
    float d2v = (float)(dj * dj) * (svr * svr) / ((bh / 2.0f) * (bh / 2.0f));
    for (int i = 0; i < fbw; i++) {
      float d2 = d2v + (float)(i * i) / ((bw / 2.0f) * (bw / 2.0f));
      lut[j * fbw + i] = strength * (1.0f - std::exp(-d2 / (2.0f * scutoff * scutoff)));
    }
  }
  return lut;
}

} // namespace fft3dgpu
