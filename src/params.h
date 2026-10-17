#pragma once
// Parameter contract for FFT3DCL, names/defaults/clamping identical to
// FFT3dGPU v0.8.6 (FFT3dGPU.cpp:163-282, 1233-1303). See docs/params.md.

namespace fft3dcl {

// Script-level parameters exactly as the user passes them.
struct ScriptParams {
  float sigma = 2.0f;
  float beta = 1.0f;
  int bw = 32, bh = 32, bt = 3;
  float sharpen = 0.0f;
  int plane = 0;              // 0 luma, 1 chroma, 4 all (resolved per format)
  int mode = 1;               // default 1
  int bordersize = 1;
  int precision = 0;          // 0 FLOAT16, 1 FLOAT32_FFT, 2 FLOAT32_ALL
  float degrid = 1.0f;        // default: mode==1 ? 1.0 : 0.0
  float scutoff = 0.3f;
  float svr = 1.0f;
  float smin = 4.0f, smax = 20.0f;
  float kratio = 2.0f;
  int ow = 16, oh = 16;       // default: bw/2, bh/2 (of the *input* bw/bh)
  int wintype = 0;
  int interlaced = 0;
  float sigma2 = 2.0f, sigma3 = 2.0f, sigma4 = 2.0f; // default: = sigma
  int oldfft = -1;            // -1 = undefined (MEASURE), 0 = STOCKHAM, 1 = RADIX2LUT
};

// Internal state after constructor validation.
struct InternalParams {
  float sigma;                // sigma / 255
  float beta;
  int bw, bh;                 // rounded down to power of 2, clamped [4,512]
  int bm;                     // original bt (-1..4)
  int bt;                     // max(bm,1)
  float sharpen;
  bool sharpenOn;
  int mode;                   // 0..2 else 0
  int bordersize;
  int precision;              // final (forced 2 for >8 bit)
  float degrid;
  bool degridOn;
  float scutoff, svr, smin, smax; // smin/smax already /255
  float kratio;
  int ow, oh;                 // scaled, even, <= bw/2 / bh/2
  int wintype;                // 0..2 else 0
  int interlaced;
  float sigma2, sigma3, sigma4; // script units; pattern squaring done in tables
  bool usePattern;
  int current;                // index of frame n inside the temporal window
  int nx, ny, totw, toth;     // block grid for this plane size
  int oldfft;                 // -1 measure, 0 stockham, 1 radix2lut (phase 3)

  // Derived filter constants (FFT3dGPU.cpp:396-398, 413, 446-451).
  float wienerS;              // sigma^2 * bt * bw * bh
  float betaX;                // (beta-1)/beta
  float kalmanCN;             // sigma^2 * bw * bh  (no bt factor)
  float ssmin, ssmax;         // smin^2*bw*bh*bt, smax^2*bw*bh*bt
};

// FFT3dGPU.cpp:214-231: round down to power of two, then clamp [4,512].
inline int roundPow2Clamp(int v) {
  unsigned int logn = 0;
  for (unsigned int i = static_cast<unsigned int>(v - 1); i > 0; i /= 2, logn++)
    ;
  int r = 1;
  for (unsigned int i = 1; i <= logn; i++)
    r *= 2;
  if (r < 4) r = 4;
  if (r > 512) r = 512;
  return r;
}

// Validate and derive. width/height are the *plane* dimensions.
// Returns false and sets err on a hard error (bt out of range).
inline bool validateParams(const ScriptParams& sp, int width, int height,
                           int bitsPerSample, InternalParams& out,
                           const char** err) {
  out.sigma = sp.sigma / 255.0f;
  out.beta = sp.beta;
  out.bm = sp.bt;
  if (out.bm < -1 || out.bm > 4) {
    *err = "FFT3DCL: valid modes for bt are -1,0,1,2,3,4";
    return false;
  }
  out.bt = sp.bt > 0 ? sp.bt : 1;
  out.sharpen = sp.sharpen;
  out.sharpenOn = sp.sharpen != 0.0f;
  out.mode = (sp.mode < 0 || sp.mode > 2) ? 0 : sp.mode;
  out.bordersize = sp.bordersize;
  out.degrid = sp.degrid;
  out.degridOn = sp.degrid != 0.0f;
  out.scutoff = sp.scutoff;
  out.svr = sp.svr;
  out.smin = sp.smin / 255.0f;
  out.smax = sp.smax / 255.0f;
  out.kratio = sp.kratio;
  out.wintype = (sp.wintype < 0 || sp.wintype > 2) ? 0 : sp.wintype;
  out.interlaced = sp.interlaced;
  out.sigma2 = sp.sigma2;
  out.sigma3 = sp.sigma3;
  out.sigma4 = sp.sigma4;
  out.usePattern = (sp.sigma != sp.sigma2 || sp.sigma3 != sp.sigma4 || sp.sigma != sp.sigma3);
  out.oldfft = sp.oldfft;
  out.current = (out.bm == 1) ? 0 : (out.bm == 4) ? 2 : 1;

  // bw/bh power-of-two rounding, then ow/oh scaling (FFT3dGPU.cpp:213-242).
  out.bw = roundPow2Clamp(sp.bw);
  out.bh = roundPow2Clamp(sp.bh);
  out.ow = static_cast<int>(sp.ow * (float)out.bw / (float)sp.bw);
  out.ow &= 0xFFFFFFFE; // even
  out.oh = static_cast<int>(sp.oh * (float)out.bh / (float)sp.bh);
  if (out.oh * 2 > out.bh) out.oh = out.bh / 2;
  if (out.ow * 2 > out.bw) out.ow = out.bw / 2;

  // precision: >8 bit forces FLOAT32_ALL; out of range -> FLOAT32_FFT (l.244-249).
  int precision = sp.precision;
  if (bitsPerSample > 8) precision = 2;
  if (precision < 0 || precision > 2) precision = 1;
  out.precision = precision;

  // Block grid (FFT3dGPU.cpp:259-282). Mode 2 not ported yet.
  if (out.mode == 2) {
    *err = "FFT3DCL: mode 2 is not implemented yet in this port (use mode 0 or 1)";
    return false;
  }
  if (out.interlaced) {
    *err = "FFT3DCL: interlaced is not implemented yet in this port";
    return false;
  }
  if (out.mode == 0 || (out.bw == out.ow * 2 && out.bh == out.oh * 2)) {
    out.ny = static_cast<int>((height + out.bh * 1.5 - 1) / out.bh);
    out.nx = static_cast<int>((width + out.bw * 1.5 - 1) / out.bw);
  } else {
    out.nx = ((width + out.ow - 1) / (out.bw - out.ow) + 1 + 1) / 2;
    out.ny = ((height + out.oh - 1) / (out.bh - out.oh) + 1 + 1) / 2;
  }
  out.totw = out.nx * out.bw;
  out.toth = out.ny * out.bh;

  // Filter constants.
  out.wienerS = out.sigma * out.sigma * out.bt * out.bw * out.bh;
  out.betaX = (out.beta - 1.0f) / out.beta;
  out.kalmanCN = out.sigma * out.sigma * out.bw * out.bh;
  out.ssmin = out.smin * out.smin * out.bw * out.bh * out.bt;
  out.ssmax = out.smax * out.smax * out.bw * out.bh * out.bt;
  return true;
}

} // namespace fft3dcl
