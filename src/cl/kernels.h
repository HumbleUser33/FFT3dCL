#pragma once
// OpenCL C 1.2 kernel sources for the FFT3DCL port.
// Embedded as a string: no .cl file deployment needed.
// Portability rules: no fp64, no atomics, dynamic local memory,
// barriers only, work-group <= 1024.

namespace fft3dcl {

inline const char* kKernelSource() {
  return R"CL(
// ---- helpers ----------------------------------------------------------
#define PI 3.14159265358979f

// Pixel format of the plane the forward kernel reads and the synthesis writes,
// fixed at compile time: 0 = float samples, 8 = 8-bit bytes, 16 = 16-bit
// samples. Converting inside these two kernels removes the separate unpack and
// pack passes (two launches and two full passes over eight million pixels per
// 4K frame) and the 33 MB float plane they needed. The arithmetic is exactly
// the one of the PlaneRef::upload/download oracle: same fp32 multiply, same
// clamp order, floor() before the cast so the rounding does not depend on the
// device's float-to-integer conversion mode.
#if RAW == 8
  #define IN_T uchar
  #define OUT_T uchar
  #define TO_FLOAT(v) ((float)(v) * (1.0f / 255.0f))
#elif RAW == 16
  #define IN_T ushort
  #define OUT_T ushort
  #define TO_FLOAT(v) ((float)(v) * norm)
#else
  #define IN_T float
  #define OUT_T float
  #define TO_FLOAT(v) (v)
#endif

inline int mirror_i(int x, int w) {
  int p = x % (2 * w);
  if (p < 0) p += 2 * w;
  if (p >= w) p = 2 * w - 1 - p;
  return p;
}

inline float2 cmul(float2 a, float2 b) {
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Wiener helper (fft3dgpu.hlsl:643-664), per complex value.
inline void wienner(float* re, float* im, float sigmaX, float sigmaY, float betaX) {
  float psd = *re * *re + *im * *im + 0.0000000000000000000000000000000000001f;
  float psdInv = 1.0f / psd;
  float mulFac = betaX;
  if (1.0f / sigmaX > psdInv)
    mulFac = 1.0f - sigmaY * psdInv;
  *re *= mulFac;
  *im *= mulFac;
}

// Twiddle factors for one 1D FFT length, computed once per work-group instead
// of once per butterfly. The expression is character-identical to the one the
// butterflies used to evaluate, so every value is bit-for-bit the same; only
// the number of cos/sin calls changes (n-1 per group instead of n/2*log2(n)
// per row, i.e. about 80 times fewer for a 32x32 block).
// Stage len (2, 4, ... n) starts at offset len/2-1 and holds len/2 entries,
// so the table for length n needs n-1 entries.
__attribute__((always_inline))
void tw_fill(__local float2* wt, int n, int sign, int tid, int wg) {
  for (int len = 2; len <= n; len <<= 1) {
    int h = len >> 1;
    int step = n / len;
    for (int j = tid; j < h; j += wg) {
      float ang = (float)sign * 2.0f * PI * (float)(j * step) / (float)n;
      wt[(h - 1) + j] = (float2)(cos(ang), sin(ang));
    }
  }
}

// Radix-2 DIT FFT of length n (power of two), done for R transforms at once so
// every work-item has work in every stage and the group only needs one barrier
// per stage instead of one per transform and stage (12 barriers per 32x32 block
// instead of 384). The butterflies, their twiddle factors and the order of the
// arithmetic are exactly the ones of the previous per-transform version, so the
// result is bit-identical; only the mapping to work-items changed.
// Element k of transform r lives at L[r * rowPitch + k * stride].
__attribute__((always_inline))
void fft_batch(__local float2* restrict L, int R, int rowPitch, int stride, int n, int sign,
               int tid, int wg, __local const float2* restrict wt) {
  int bits = 0;
  while ((1 << bits) < n) bits++;
  // bit-reversal permutation: each disjoint pair (k, rev(k)), k < rev(k), of
  // every transform is swapped by exactly one work-item.
  for (int i = tid; i < R * n; i += wg) {
    int r = i / n, k = i - r * n;
    int rev = 0;
    for (int b = 0; b < bits; b++)
      if (k & (1 << b)) rev |= 1 << (bits - 1 - b);
    if (k < rev) {
      __local float2* pa = &L[r * rowPitch + k * stride];
      __local float2* pc = &L[r * rowPitch + rev * stride];
      float2 t = *pa;
      *pa = *pc;
      *pc = t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int len = 2; len <= n; len <<= 1) {
    int h = len >> 1;
    for (int bi = tid; bi < R * (n / 2); bi += wg) {
      int r = bi / (n / 2), q = bi - r * (n / 2);
      int blk = q / h, j = q - blk * h;
      float2 w = wt[(h - 1) + j];
      int p = blk * len + j;
      __local float2* pa = &L[r * rowPitch + p * stride];
      __local float2* pb = &L[r * rowPitch + (p + h) * stride];
      float2 a = *pa;
      float2 t = cmul(w, *pb);
      *pa = a + t;
      *pb = a - t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sign > 0) {
    float invn = 1.0f / (float)n;
    for (int i = tid; i < R * n; i += wg) {
      int r = i / n, k = i - r * n;
      L[r * rowPitch + k * stride] *= invn;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// ---- debug: single-row FFT -------------------------------------------
__kernel void fft1d_debug(__global const float* src, __global float2* dst, int n) {
  __local float2 L[LSZ];
  __local float2 WT[LSZ];
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  for (int i = tid; i < n; i += wg)
    L[i] = (float2)(src[i], 0.0f);
  tw_fill(WT, n, -1, tid, wg);
  barrier(CLK_LOCAL_MEM_FENCE);
  fft_batch(L, 1, 0, 1, n, -1, tid, wg, WT);
  for (int i = tid; i < n; i += wg)
    dst[i] = L[i];
}

// ---- analysis + forward FFT ------------------------------------------
// One work-group per block. Loads the windowed block (mirror addressing,
// chroma centering), does the 2D forward FFT, stores the half-spectrum.
__kernel void fft2d_fwd(__global const IN_T* restrict plane,
                        __global const int* restrict bx,
                        __global const int* restrict by,
                        __global float2* restrict spec,
                        __global const float* restrict win,
                        int W, int H, int bw, int bh, int fbw, int nb,
                        float chromaNorm, float norm) {
  __local float2 L[LSZP]; // LSZP = (bw+1)*bh: the spare column (LP = bw+1) shifts
                          // every row by one element, which breaks the 32-way
                          // bank conflict the strided column FFTs used to hit
  __local float2 WT[2 * LSW]; // twiddles: LSW for the rows (bw), LSW for the columns (bh)
  int b = get_group_id(0);
  if (b >= nb) return;
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  int x0 = bx[b], y0 = by[b];
  int n = bw * bh;
  for (int idx = tid; idx < n; idx += wg) {
    int ly = idx / bw, lx = idx - ly * bw;
    float v = TO_FLOAT(plane[mirror_i(y0 + ly, H) * W + mirror_i(x0 + lx, W)]);
    v -= chromaNorm;
    L[ly * LP + lx] = (float2)(v * win[idx], 0.0f);
  }
  tw_fill(WT, bw, -1, tid, wg);
  tw_fill(&WT[LSW], bh, -1, tid, wg);
  barrier(CLK_LOCAL_MEM_FENCE);
  fft_batch(L, bh, LP, 1, bw, -1, tid, wg, WT);        // every row in one pass
  // Only the first fbw columns are ever stored (the rest is the Hermitian
  // mirror), so the remaining bw-fbw column transforms are not computed.
  fft_batch(L, fbw, 1, LP, bh, -1, tid, wg, &WT[LSW]); // stored columns only
  __global float2* dst = &spec[(long)b * fbw * bh];
  for (int idx = tid; idx < fbw * bh; idx += wg) {
    int r = idx / fbw, c = idx - r * fbw;
    dst[idx] = L[r * LP + c];
  }
}
)CL"
// MSVC truncates a single string literal past 64 KiB, so the source is handed
// over as three adjacent literals that the compiler concatenates.
R"CL(
// ---- temporal filter + sharpen ----------------------------------------
// One work-item per (block, frequency bin). BM selects the temporal mode
// (compile-time -DBM=1|2|3|4|0|-1). DG/-DPAT/-DSHP enable degrid,
// sigma pattern and sharpen. Mirrors PlaneRef::filterAndSynthesize.
__kernel void filter(__global const float2* restrict s0,
                     __global const float2* restrict s1,
                     __global const float2* restrict s2,
                     __global const float2* restrict s3,
                     __global float2* restrict filt,
                     __global const float2* restrict gs,     // degrid sample (planeSize)
                     __global const float* restrict pattern, // planeSize (or dummy)
                     __global const float* restrict sharp,   // planeSize (or dummy)
                     __global float2* restrict kLast,
                     __global float* restrict kCovar,
                     __global float* restrict kProc,
                     int nb, int fbw, int bh,
                     float sigmaX, float sigmaY, float betaX, float beta,
                     float ssmin, float ssmax, float kalmanCN, float kratio) {
  int planeSize = fbw * bh;
  int gid = get_global_id(0);
  if (gid >= nb * planeSize) return;
  int b = gid / planeSize;
  int i = gid - b * planeSize;
  int r = i / fbw, c = i - r * fbw;

  // degrid correction for this bin: current-frame DC * gridsample
  float dRe = 0.0f, dIm = 0.0f;
#if DG
  // current-frame DC: window[current] -> s1 for bm 2/3, s2 for bm 4, s0 for bm<=1
  float2 dc =
#if BM == 4
      s2[b * planeSize];
#elif BM >= 2
      s1[b * planeSize];
#else
      s0[b * planeSize];
#endif
  float2 g = gs[i];
  dRe = dc.x * g.x;
  dIm = dc.x * g.y;
#endif

  float sX = sigmaX, sY = sigmaY;
#if PAT
  float pv = pattern[i];
  sX = pv * beta;
  sY = pv;
#endif

  float fr, fi; // filtered result
#if BM == 2
  float2 p = s0[b * planeSize + i], cu = s1[b * planeSize + i];
  float aRe = cu.x - p.x, aIm = cu.y - p.y;
  float bRe = cu.x + p.x, bIm = cu.y + p.y;
#if DG
  bRe -= 2.0f * dRe; bIm -= 2.0f * dIm;
#endif
  wienner(&aRe, &aIm, sX, sY, betaX);
  wienner(&bRe, &bIm, sX, sY, betaX);
  fr = (aRe + bRe) * 0.5f; fi = (aIm + bIm) * 0.5f;
#if DG
  fr += dRe; fi += dIm;
#endif
#elif BM == 3
  float2 p = s0[b * planeSize + i], cu = s1[b * planeSize + i], n = s2[b * planeSize + i];
  float t1Re = p.x + n.x, t1Im = p.y + n.y;
  float y1Re = cu.x - 0.5f * t1Re, y1Im = cu.y - 0.5f * t1Im;
  float dnRe = n.x - p.x, dnIm = n.y - p.y;
  float K = 0.8660254f;
  float y2Re = K * dnIm, y2Im = -K * dnRe;
  float d0Re = y1Re + y2Re, d0Im = y1Im + y2Im;
  float d1Re = cu.x + t1Re, d1Im = cu.y + t1Im;
  float d2Re = y1Re - y2Re, d2Im = y1Im - y2Im;
#if DG
  d1Re -= 3.0f * dRe; d1Im -= 3.0f * dIm;
#endif
  wienner(&d0Re, &d0Im, sX, sY, betaX);
  wienner(&d1Re, &d1Im, sX, sY, betaX);
  wienner(&d2Re, &d2Im, sX, sY, betaX);
  fr = (d0Re + d1Re + d2Re) * (1.0f / 3.0f);
  fi = (d0Im + d1Im + d2Im) * (1.0f / 3.0f);
#if DG
  fr += dRe; fi += dIm;
#endif
#elif BM == 4
  float2 p2 = s0[b * planeSize + i], p1 = s1[b * planeSize + i];
  float2 cu = s2[b * planeSize + i], n = s3[b * planeSize + i];
  float fp1r = -p2.x + p1.y + cu.x - n.y;
  float fp1i = -p2.y - p1.x + cu.y + n.x;
  float fcr = p2.x + p1.x + cu.x + n.x;
  float fci = p2.y + p1.y + cu.y + n.y;
  float fnr = -p2.x - p1.y + cu.x + n.y;
  float fni = -p2.y + p1.x + cu.y - n.x;
  float fp2r = p2.x - p1.x + cu.x - n.x;
  float fp2i = p2.y - p1.y + cu.y - n.y;
#if DG
  fp1r -= 4.0f * dRe; fp1i -= 4.0f * dIm;
#endif
  wienner(&fp2r, &fp2i, sX, sY, betaX);
  wienner(&fp1r, &fp1i, sX, sY, betaX);
  wienner(&fcr, &fci, sX, sY, betaX);
  wienner(&fnr, &fni, sX, sY, betaX);
  fr = (fp2r + fp1r + fcr + fnr) * 0.25f;
  fi = (fp2i + fp1i + fci + fni) * 0.25f;
#if DG
  fr += dRe; fi += dIm;
#endif
#elif BM == 1
  float2 cu = s0[b * planeSize + i];
  float re = cu.x, im = cu.y;
#if DG
  re -= dRe; im -= dIm;
#endif
  wienner(&re, &im, sX, sY, betaX);
#if DG
  re += dRe; im += dIm;
#endif
  fr = re; fi = im;
#elif BM == 0
  // Kalman: persistent state (kLast/kCovar/kProc), updated in place.
  float2 cu = s0[b * planeSize + i];
  float2 l = kLast[gid];
  float dRe2 = cu.x - l.x, dIm2 = cu.y - l.y;
  float mask = max(dRe2 * dRe2, dIm2 * dIm2);
  float cn = kalmanCN;
#if PAT
  cn = pattern[i];
#endif
  float s = kCovar[gid] + kProc[gid];
  float gain = s / (s + cn);
  float newProc = gain * gain * cn;
  float newCovar = (1.0f - gain) * s;
  float dstRe = gain * cu.x + (1.0f - gain) * l.x;
  float dstIm = gain * cu.y + (1.0f - gain) * l.y;
  if (mask > kratio * kratio * cn) {
    dstRe = cu.x; dstIm = cu.y;
    newCovar = cn; newProc = cn;
  }
  fr = dstRe; fi = dstIm;
  kCovar[gid] = newCovar; kProc[gid] = newProc;
  kLast[gid] = (float2)(dstRe, dstIm);
#else // BM == -1: no temporal filter
  float2 cu = s0[b * planeSize + i];
  fr = cu.x; fi = cu.y;
#endif

#if SHP
  {
    float re = fr, im = fi;
#if DG
    re -= dRe; im -= dIm;
#endif
    float psd = re * re + im * im + 0.0000000000000000000000000000000000001f;
    float factor = sharp[i];
    float g2 = 1.0f + factor * sqrt(psd * ssmax / ((psd + ssmin) * (psd + ssmax)));
    re *= g2; im *= g2;
#if DG
    re += dRe; im += dIm;
#endif
    fr = re; fi = im;
  }
#endif

  filt[gid] = (float2)(fr, fi);
}

// Kalman first-frame state init.
__kernel void kalman_init(__global const float2* restrict s0,
                          __global float2* restrict kLast,
                          __global float* restrict kCovar,
                          __global float* restrict kProc,
                          __global const float* restrict pattern,
                          int total, int planeSize, float kalmanCN) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  kLast[gid] = s0[gid];
  float cn = kalmanCN;
#if PAT
  cn = pattern[gid % planeSize];
#endif
  kCovar[gid] = cn;
  kProc[gid] = cn;
}

// ---- inverse FFT + synthesis window ----------------------------------
// One work-group per block. Expands the half-spectrum (Hermitian),
// inverse FFT, multiplies by the per-block synthesis window.
__kernel void fft2d_inv(__global const float2* restrict spec,
                        __global float* restrict blocks,
                        __global const float* restrict winA,
                        __global const float* restrict winB,
                        __global const uchar* restrict par,
                        int bw, int bh, int fbw, int nb) {
  __local float2 L[LSZP]; // padded layout, see fft2d_fwd
  __local float2 WT[2 * LSW]; // twiddles: LSW for the rows (bw), LSW for the columns (bh)
  int b = get_group_id(0);
  if (b >= nb) return;
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  __global const float* win = par[b] ? winB : winA;
  __global const float2* src = &spec[(long)b * fbw * bh];
  for (int idx = tid; idx < fbw * bh; idx += wg) {
    int r = idx / fbw, c = idx - r * fbw;
    L[r * LP + c] = src[idx];
  }
  tw_fill(WT, bw, +1, tid, wg);
  tw_fill(&WT[LSW], bh, +1, tid, wg);
  barrier(CLK_LOCAL_MEM_FENCE);
  // Hermitian completion: L[r][c] = conj(L[(bh-r)%bh][bw-c]) for c >= fbw
  for (int idx = tid; idx < (bw - fbw) * bh; idx += wg) {
    int r = idx / (bw - fbw), c = idx - r * (bw - fbw) + fbw;
    int rm = (bh - r) % bh, cm = bw - c;
    float2 v = L[rm * LP + cm];
    L[r * LP + c] = (float2)(v.x, -v.y);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  fft_batch(L, bh, LP, 1, bw, +1, tid, wg, WT);       // every row in one pass
  fft_batch(L, bw, 1, LP, bh, +1, tid, wg, &WT[LSW]); // every column in one pass
  __global float* dst = &blocks[(long)b * bw * bh];
  for (int idx = tid; idx < bw * bh; idx += wg) {
    int r = idx / bw, c = idx - r * bw;
    dst[idx] = L[r * LP + c].x * win[idx];
  }
}

// ---- overlap-add synthesis -------------------------------------------
// One work-item per output pixel; sums the covering blocks (grid is
// uniform: origin off + k*step, block size bw x bh).
__kernel void synthesize(__global const float* restrict blocks,
                         __global OUT_T* restrict out,
                         int W, int H, int bw, int bh,
                         int xoff, int xstep, int yoff, int ystep,
                         int ncols, int nrows, float chromaNorm,
                         float invNorm, float maxPix) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= W || y >= H) return;
  float sum = 0.0f;
  // candidate block indices: those with bx <= x < bx+bw
  int n0 = (int)floor((float)(x - xoff - bw + 1) / (float)xstep);
  int n1 = (int)floor((float)(x - xoff) / (float)xstep);
  int m0 = (int)floor((float)(y - yoff - bh + 1) / (float)ystep);
  int m1 = (int)floor((float)(y - yoff) / (float)ystep);
  if (n0 < 0) n0 = 0;
  if (m0 < 0) m0 = 0;
  if (n1 >= ncols) n1 = ncols - 1;
  if (m1 >= nrows) m1 = nrows - 1;
  for (int m = m0; m <= m1; m++) {
    int ly = y - (yoff + m * ystep);
    for (int n = n0; n <= n1; n++) {
      int lx = x - (xoff + n * xstep);
      if (lx < 0 || lx >= bw || ly < 0 || ly >= bh) continue;
      sum += blocks[((long)m * ncols + n) * bw * bh + ly * bw + lx];
    }
  }
  float v = sum + chromaNorm;
#if RAW == 8
  v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  out[y * W + x] = (uchar)floor(v * 255.0f + 0.5f);
#elif RAW == 16
  v = v * invNorm;
  v = v < maxPix ? v : maxPix;
  v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  out[y * W + x] = (ushort)floor(v * 65535.0f + 0.5f);
#else
  out[y * W + x] = v;
#endif
}
)CL";
}

} // namespace fft3dcl
