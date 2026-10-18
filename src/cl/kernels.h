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

// In-place radix-2 DIT FFT of length n (power of two) over strided
// local-memory elements, cooperatively by all work-items of the group.
// sign=-1 forward, sign=+1 inverse (with 1/n scaling).
__attribute__((always_inline))
void fft1d(__local float2* restrict base, int stride, int n, int sign, int tid, int wg) {
  int bits = 0;
  while ((1 << bits) < n) bits++;
  // bit-reversal permutation: each disjoint pair (i, rev(i)), i < rev(i),
  // is swapped by exactly one work-item.
  for (int i = tid; i < n; i += wg) {
    int r = 0;
    for (int b = 0; b < bits; b++)
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    if (i < r) {
      float2 t = base[i * stride];
      base[i * stride] = base[r * stride];
      base[r * stride] = t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int len = 2; len <= n; len <<= 1) {
    int h = len >> 1;
    int step = n / len;
    for (int bi = tid; bi < n / 2; bi += wg) {
      int blk = bi / h, j = bi - blk * h;
      float ang = (float)sign * 2.0f * PI * (float)(j * step) / (float)n;
      float2 w = (float2)(cos(ang), sin(ang));
      int p = blk * len + j;
      float2 a = base[p * stride];
      float2 t = cmul(w, base[(p + h) * stride]);
      base[p * stride] = a + t;
      base[(p + h) * stride] = a - t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (sign > 0) {
    float invn = 1.0f / (float)n;
    for (int i = tid; i < n; i += wg)
      base[i * stride] *= invn;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// ---- debug: single-row FFT -------------------------------------------
__kernel void fft1d_debug(__global const float* src, __global float2* dst, int n) {
  __local float2 L[LSZ];
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  for (int i = tid; i < n; i += wg)
    L[i] = (float2)(src[i], 0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  fft1d(L, 1, n, -1, tid, wg);
  for (int i = tid; i < n; i += wg)
    dst[i] = L[i];
}

// ---- analysis + forward FFT ------------------------------------------
// One work-group per block. Loads the windowed block (mirror addressing,
// chroma centering), does the 2D forward FFT, stores the half-spectrum.
__kernel void fft2d_fwd(__global const float* restrict plane,
                        __global const int* restrict bx,
                        __global const int* restrict by,
                        __global float2* restrict spec,
                        __global const float* restrict win,
                        int W, int H, int bw, int bh, int fbw, int nb,
                        float chromaNorm) {
  __local float2 L[LSZ]; // LSZ = bw*bh, set via -D at build time
  int b = get_group_id(0);
  if (b >= nb) return;
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  int x0 = bx[b], y0 = by[b];
  int n = bw * bh;
  for (int idx = tid; idx < n; idx += wg) {
    int ly = idx / bw, lx = idx - ly * bw;
    float v = plane[mirror_i(y0 + ly, H) * W + mirror_i(x0 + lx, W)];
    v -= chromaNorm;
    L[idx] = (float2)(v * win[idx], 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 0; r < bh; r++)
    fft1d(&L[r * bw], 1, bw, -1, tid, wg);
  for (int c = 0; c < bw; c++)
    fft1d(&L[c], bw, bh, -1, tid, wg);
  __global float2* dst = &spec[(long)b * fbw * bh];
  for (int idx = tid; idx < fbw * bh; idx += wg) {
    int r = idx / fbw, c = idx - r * fbw;
    dst[idx] = L[r * bw + c];
  }
}

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
  __local float2 L[LSZ]; // LSZ = bw*bh, set via -D at build time
  int b = get_group_id(0);
  if (b >= nb) return;
  int tid = get_local_id(0);
  int wg = get_local_size(0);
  __global const float* win = par[b] ? winB : winA;
  __global const float2* src = &spec[(long)b * fbw * bh];
  for (int idx = tid; idx < fbw * bh; idx += wg) {
    int r = idx / fbw, c = idx - r * fbw;
    L[r * bw + c] = src[idx];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Hermitian completion: L[r][c] = conj(L[(bh-r)%bh][bw-c]) for c >= fbw
  for (int idx = tid; idx < (bw - fbw) * bh; idx += wg) {
    int r = idx / (bw - fbw), c = idx - r * (bw - fbw) + fbw;
    int rm = (bh - r) % bh, cm = bw - c;
    float2 v = L[rm * bw + cm];
    L[r * bw + c] = (float2)(v.x, -v.y);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int r = 0; r < bh; r++)
    fft1d(&L[r * bw], 1, bw, +1, tid, wg);
  for (int c = 0; c < bw; c++)
    fft1d(&L[c], bw, bh, +1, tid, wg);
  __global float* dst = &blocks[(long)b * bw * bh];
  for (int idx = tid; idx < bw * bh; idx += wg)
    dst[idx] = L[idx].x * win[idx];
}

// ---- overlap-add synthesis -------------------------------------------
// One work-item per output pixel; sums the covering blocks (grid is
// uniform: origin off + k*step, block size bw x bh).
__kernel void synthesize(__global const float* restrict blocks,
                         __global float* restrict out,
                         int W, int H, int bw, int bh,
                         int xoff, int xstep, int yoff, int ystep,
                         int ncols, int nrows, float chromaNorm) {
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
  out[y * W + x] = sum + chromaNorm;
}

// Pixel conversion, done on the device so the host only has to copy bytes
// instead of converting eight million pixels twice per frame. The maths matches
// PlaneRef::upload/download exactly: same fp32 multiply, same clamp order, and
// floor() before the cast so the rounding does not depend on the device's
// float-to-integer conversion mode (a plain cast is round-to-nearest in OpenCL).
__kernel void unpack_u8(__global const uchar* restrict raw, __global float* restrict plane,
                        int n) {
  int i = get_global_id(0);
  if (i < n)
    plane[i] = (float)raw[i] * (1.0f / 255.0f);
}

__kernel void unpack_u16(__global const ushort* restrict raw, __global float* restrict plane,
                         int n, float norm) {
  int i = get_global_id(0);
  if (i < n)
    plane[i] = (float)raw[i] * norm;
}

__kernel void pack_u8(__global const float* restrict plane, __global uchar* restrict raw,
                      int n) {
  int i = get_global_id(0);
  if (i < n) {
    float v = plane[i];
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    raw[i] = (uchar)floor(v * 255.0f + 0.5f);
  }
}

__kernel void pack_u16(__global const float* restrict plane, __global ushort* restrict raw,
                       int n, float invNorm, float maxPix) {
  int i = get_global_id(0);
  if (i < n) {
    float v = plane[i] * invNorm;
    v = v < maxPix ? v : maxPix;
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    raw[i] = (ushort)floor(v * 65535.0f + 0.5f);
  }
}
)CL";
}

} // namespace fft3dcl
