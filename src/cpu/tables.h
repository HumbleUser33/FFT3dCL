#pragma once
#include <vector>

namespace fft3dgpu {

// All tables are built in double and stored as float, like the original's
// CPU tables uploaded to float textures.

// Mode 0 window, bw x bh, row-major:
//   w(lx,ly) = sqrt(0.5*(cosx^2 + cosy^2)),
//   cosx = cos((lx - bw/2 + 0.5)*pi/bw), cosy likewise (filters.cpp:157-185).
std::vector<float> buildWindowMode0(int bw, int bh);

// Mode 1 windows A, SynA, SynB (bw x bh each, row-major), from CreateFactorMap
// (filters.cpp:501-670). A = analysis for both grids (wanx/wany). SynA =
// synthesis for the even-parity grid (MapSyn xy components, wsynx/wsyny);
// SynB = synthesis for the odd-parity grid (MapSyn zw components). Separable:
// wx(lx)*wy(ly). wsynx differs from wanx for wintype 1 and 2.
void buildWindowsMode1(int bw, int bh, int ow, int oh, int wintype,
                       std::vector<float>& A, std::vector<float>& SynA,
                       std::vector<float>& SynB);

// Sigma pattern motif, fbw x bh (fbw = bw/2+1), FFT3dGPU.cpp:588-631.
// sigma2/3/4 are script units; squaring/scaling done here. Returns empty if
// usePattern is false.
std::vector<float> buildPattern(int bw, int bh, int bt, float sigma,
                                float sigma2, float sigma3, float sigma4,
                                bool usePattern);

// Sharpen factor LUT motif, fbw x bh, filters.cpp:935-976 (Fizick).
// strength = script sharpen. Returns empty if strength == 0.
std::vector<float> buildSharpenLUT(int bw, int bh, float strength, float svr,
                                   float scutoff);

} // namespace fft3dgpu
