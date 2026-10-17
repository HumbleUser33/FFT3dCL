#include "reference.h"
#include "tables.h"
#include <algorithm>
#include <cmath>
#include <complex>

namespace fft3dcl {

// DX9 MIRROR addressing for integer texel indices.
static inline int mirror_i(int x, int w) {
  int p = x % (2 * w);
  if (p < 0)
    p += 2 * w;
  if (p >= w)
    p = 2 * w - 1 - p;
  return p;
}

PlaneRef::PlaneRef(const InternalParams& p, const PlaneGeom& g)
    : p_(p), g_(g), fbw_(p.bw / 2 + 1), rowPlan_(p.bw), colPlan_(p.bh) {
  const int bw = p.bw, bh = p.bh;

  // Block list.
  if (p.mode == 0) {
    // Two grids: U at (i*bw, j*bh), S at (i*bw - bw/2, j*bh - bh/2).
    for (int grid = 0; grid < 2; grid++)
      for (int j = 0; j < p.ny; j++)
        for (int i = 0; i < p.nx; i++) {
          int off = grid ? -bw / 2 : 0, offy = grid ? -bh / 2 : 0;
          bx_.push_back(i * bw + off);
          by_.push_back(j * bh + offy);
          par_.push_back(0);
        }
    winAna_ = buildWindowMode0(bw, bh);
  } else {
    bool half = (bw == p.ow * 2 && bh == p.oh * 2);
    int ncols = 2 * p.nx;
    ncols_ = ncols;
    int nrows = 2 * p.ny;
    int ystep = bh - p.oh;
    for (int m = 0; m < nrows; m++)
      for (int n = 0; n < ncols; n++) {
        bx_.push_back(-p.ow + n * (bw - p.ow));
        by_.push_back(-p.oh + m * ystep);
        par_.push_back(static_cast<uint8_t>(n & 1));
      }
    std::vector<float> A, SynA, SynB;
    buildWindowsMode1(bw, bh, p.ow, p.oh, p.wintype, A, SynA, SynB);
    winAna_ = std::move(A);
    winSynA_ = std::move(SynA);
    // Synthesis window for the odd-parity (U) grid: the MapSyn zw components
    // sampled one (bw-ow) to the right (OQuad::CreateVBi factor offsets).
    // For the default ow=bw/2 this equals winSynA_ (COLA holds).
    winSynB_.resize(bw * bh);
    for (int ly = 0; ly < bh; ly++)
      for (int lx = 0; lx < bw; lx++)
        winSynB_[ly * bw + lx] = SynB[ly * bw + ((lx + bw - p.ow) % bw)];
  }

  pattern_ = buildPattern(bw, bh, p.bt, p.sigma * 255.0f, p.sigma2, p.sigma3, p.sigma4, p.usePattern);
  sharpen_ = buildSharpenLUT(bw, bh, p.sharpen, p.svr, p.scutoff);

  const int n = nblocks() * fbw_ * p_.bh;
  blockReal_.resize(bw * bh);
  fullSpec_.resize(bw * bh);
  filtRe_.resize(n);
  filtIm_.resize(n);
  corrRe_.resize(n);
  corrIm_.resize(n);

  // CHROMANORM: 128/255 at 8 bit, 32768/65535 at 10-16 bit, 0 for float
  // (fft3dgpu.hlsl:21-33). Mode 1 YUV chroma only (FFT3dGPU.cpp:462).
  if (p.mode == 1 && g.chroma) {
    if (g.sampleType == 1)
      chromaNorm_ = 0.0f;
    else if (g.bits <= 8)
      chromaNorm_ = 128.0f / 255.0f;
    else
      chromaNorm_ = 32768.0f / 65535.0f;
  }
}

void PlaneRef::upload(const uint8_t* src, ptrdiff_t pitch, std::vector<float>& dst) const {
  const int W = g_.width, H = g_.height;
  dst.resize(W * H);
  if (g_.sampleType == 1) { // float 32
    for (int y = 0; y < H; y++) {
      const float* row = reinterpret_cast<const float*>(src + y * pitch);
      for (int x = 0; x < W; x++)
        dst[y * W + x] = row[x];
    }
  } else if (g_.bits <= 8) { // FIXED2: v/255, no BITDEPTHNORM
    for (int y = 0; y < H; y++) {
      const uint8_t* row = src + y * pitch;
      for (int x = 0; x < W; x++)
        dst[y * W + x] = row[x] * (1.0f / 255.0f);
    }
  } else { // FIXED2_16: v/65535 * 2^(16-bits)
    const float norm = (float)(1u << (16 - g_.bits)) / 65535.0f;
    for (int y = 0; y < H; y++) {
      const uint16_t* row = reinterpret_cast<const uint16_t*>(src + y * pitch);
      for (int x = 0; x < W; x++)
        dst[y * W + x] = row[x] * norm;
    }
  }
}

void PlaneRef::download(const std::vector<float>& src, uint8_t* dst, ptrdiff_t pitch) const {
  const int W = g_.width, H = g_.height;
  if (g_.sampleType == 1) { // FLOAT4 RT: raw values
    for (int y = 0; y < H; y++) {
      float* row = reinterpret_cast<float*>(dst + y * pitch);
      for (int x = 0; x < W; x++)
        row[x] = src[y * W + x];
    }
  } else if (g_.bits <= 8) { // FIXED4 A8R8G8B8 UNORM
    for (int y = 0; y < H; y++) {
      uint8_t* row = dst + y * pitch;
      for (int x = 0; x < W; x++) {
        float v = src[y * W + x];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        row[x] = (uint8_t)std::lround(v * 255.0f);
      }
    }
  } else { // FIXED4_16: dst*INV_BITDEPTHNORM, min(dst,MAX), UNORM write
    const float invNorm = 1.0f / (float)(1u << (16 - g_.bits));
    const float maxPix = (float)((1u << g_.bits) - 1);
    for (int y = 0; y < H; y++) {
      uint16_t* row = reinterpret_cast<uint16_t*>(dst + y * pitch);
      for (int x = 0; x < W; x++) {
        float v = src[y * W + x] * invNorm;
        v = v < maxPix ? v : maxPix; // min(dst, MAX_PIXEL_VALUE)
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        row[x] = (uint16_t)std::lround(v * 65535.0f);
      }
    }
  }
}

std::shared_ptr<PlaneRef::Spectra> PlaneRef::computeSpectra(const std::vector<float>& plane) {
  auto s = std::make_shared<Spectra>();
  const int nb = nblocks(), fbw = fbw_, bw = p_.bw, bh = p_.bh, W = g_.width, H = g_.height;
  s->re.resize(nb * fbw * bh);
  s->im.resize(nb * fbw * bh);
  const bool center = chromaNorm_ != 0.0f;

  for (int b = 0; b < nb; b++) {
    const int x0 = bx_[b], y0 = by_[b];
    for (int ly = 0; ly < bh; ly++) {
      const float* prow = &plane[mirror_i(y0 + ly, H) * W];
      const float* wrow = &winAna_[ly * bw];
      float* brow = &blockReal_[ly * bw];
      for (int lx = 0; lx < bw; lx++) {
        float v = prow[mirror_i(x0 + lx, W)];
        if (center)
          v -= chromaNorm_;
        brow[lx] = v * wrow[lx];
      }
    }
    fft2dRealForward(blockReal_.data(), bw, bh, rowPlan_, colPlan_, fullSpec_.data());
    float* dRe = &s->re[b * fbw * bh];
    float* dIm = &s->im[b * fbw * bh];
    for (int r = 0; r < bh; r++)
      for (int c = 0; c < fbw; c++) {
        dRe[r * fbw + c] = fullSpec_[r * bw + c].real();
        dIm[r * fbw + c] = fullSpec_[r * bw + c].imag();
      }
  }
  return s;
}

void PlaneRef::buildDegridSample() {
  const int nb = nblocks(), fbw = fbw_, bh = p_.bh;
  const int planeSize = fbw * bh;
  // The original constructor clears a white (all 1.0) texture and FFTs it
  // (FFT3dGPU.cpp:529-564): gridsample = FFT(analysis window) / DC(window) *
  // degrid. A white frame makes every block identical, so the sample is the
  // same for all blocks. (Chroma centering cancels in the ratio.)
  std::vector<float> blk(winAna_);
  fft2dRealForward(blk.data(), p_.bw, bh, rowPlan_, colPlan_, fullSpec_.data());
  std::vector<float> gsR(planeSize), gsI(planeSize);
  const float dc = fullSpec_[0].real();
  for (int r = 0; r < bh; r++)
    for (int c = 0; c < fbw; c++) {
      const std::complex<float>& z = fullSpec_[r * p_.bw + c];
      gsR[r * fbw + c] = z.real() / dc * p_.degrid;
      gsI[r * fbw + c] = z.imag() / dc * p_.degrid;
    }
  gsRe_.assign(nb * planeSize, 0.0f);
  gsIm_.assign(nb * planeSize, 0.0f);
  for (int b = 0; b < nb; b++) {
    for (int i = 0; i < planeSize; i++) {
      gsRe_[b * planeSize + i] = gsR[i];
      gsIm_[b * planeSize + i] = gsI[i];
    }
  }
  degridReady_ = true;
}

// Wiener helper (fft3dgpu.hlsl:643-664), per complex value.
static inline void wienner(float& re, float& im, float sigmaX, float sigmaY, float betaX) {
  float psd = re * re + im * im + 0.0000000000000000000000000000000000001f;
  float psdInv = 1.0f / psd;
  float mulFac = betaX;
  if (1.0f / sigmaX > psdInv)
    mulFac = 1.0f - sigmaY * psdInv;
  re *= mulFac;
  im *= mulFac;
}

void PlaneRef::filterAndSynthesize(const std::vector<std::shared_ptr<Spectra>>& window,
                                   std::vector<float>& out) {
  const int nb = nblocks(), fbw = fbw_, bw = p_.bw, bh = p_.bh;
  const int W = g_.width, H = g_.height;
  const int planeSize = fbw * bh;
  out.assign(W * H, 0.0f);

  const Spectra& cur = *window[p_.bm >= 2 ? p_.current : 0];

  // Degrid runtime correction: FImgDegrid = current DC * gridsample.
  const bool dg = p_.degridOn && degridReady_;
  if (dg) {
    for (int b = 0; b < nb; b++) {
      const float dc = cur.re[b * planeSize];
      for (int i = 0; i < planeSize; i++) {
        corrRe_[b * planeSize + i] = dc * gsRe_[b * planeSize + i];
        corrIm_[b * planeSize + i] = dc * gsIm_[b * planeSize + i];
      }
    }
  }

  const float sigmaX = p_.beta * p_.wienerS;
  const float sigmaY = p_.wienerS;
  const bool pat = !pattern_.empty();

  // ---------- Phase 1: temporal filter -> filtRe_/filtIm_ (all blocks) ----------
  if (p_.bm >= 2) {
    const float K = 0.8660254f; // truncated literal from fft3dgpu.hlsl:723
    for (int r = 0; r < bh; r++)
      for (int c = 0; c < fbw; c++) {
        float sX = sigmaX, sY = sigmaY;
        if (pat) {
          float pv = pattern_[r * fbw + c];
          sX = pv * p_.beta;
          sY = pv;
        }
        for (int b = 0; b < nb; b++) {
          int idx = b * planeSize + r * fbw + c;
          float dRe = 0, dIm = 0;
          if (dg) { dRe = corrRe_[idx]; dIm = corrIm_[idx]; }

          if (p_.bm == 2) {
            // window[0]=prev, window[1]=current (current index 1)
            float pRe = window[0]->re[idx], pIm = window[0]->im[idx];
            float cRe = window[1]->re[idx], cIm = window[1]->im[idx];
            float aRe = cRe - pRe, aIm = cIm - pIm;
            float bRe = cRe + pRe, bIm = cIm + pIm;
            if (dg) { bRe -= 2.0f * dRe; bIm -= 2.0f * dIm; }
            wienner(aRe, aIm, sX, sY, p_.betaX);
            wienner(bRe, bIm, sX, sY, p_.betaX);
            // normalization 1/bt applies to the whole retval, factor included.
            filtRe_[idx] = (aRe + bRe) * 0.5f + (dg ? dRe : 0.0f);
            filtIm_[idx] = (aIm + bIm) * 0.5f + (dg ? dIm : 0.0f);
          } else if (p_.bm == 3) {
            // window[0]=prev1, window[1]=current, window[2]=next
            float pRe = window[0]->re[idx], pIm = window[0]->im[idx];
            float cRe = window[1]->re[idx], cIm = window[1]->im[idx];
            float nRe = window[2]->re[idx], nIm = window[2]->im[idx];
            float t1Re = pRe + nRe, t1Im = pIm + nIm;
            float y1Re = cRe - 0.5f * t1Re, y1Im = cIm - 0.5f * t1Im;
            float dnRe = nRe - pRe, dnIm = nIm - pIm;
            float y2Re = K * dnIm, y2Im = -K * dnRe; // -i*K*(n-p)
            float d0Re = y1Re + y2Re, d0Im = y1Im + y2Im;
            float d1Re = cRe + t1Re, d1Im = cIm + t1Im;
            float d2Re = y1Re - y2Re, d2Im = y1Im - y2Im;
            if (dg) { d1Re -= 3.0f * dRe; d1Im -= 3.0f * dIm; }
            wienner(d0Re, d0Im, sX, sY, p_.betaX);
            wienner(d1Re, d1Im, sX, sY, p_.betaX);
            wienner(d2Re, d2Im, sX, sY, p_.betaX);
            filtRe_[idx] = (d0Re + d1Re + d2Re) * (1.0f / 3.0f);
            filtIm_[idx] = (d0Im + d1Im + d2Im) * (1.0f / 3.0f);
            // 1/bt applies to the whole retval, degrid factor included -> add corr once.
            if (dg) { filtRe_[idx] += dRe; filtIm_[idx] += dIm; }
          } else { // bm == 4: window[0]=prev2, [1]=prev1, [2]=current, [3]=next
            float p2Re = window[0]->re[idx], p2Im = window[0]->im[idx];
            float p1Re = window[1]->re[idx], p1Im = window[1]->im[idx];
            float cRe = window[2]->re[idx], cIm = window[2]->im[idx];
            float nRe = window[3]->re[idx], nIm = window[3]->im[idx];
            // fp1 = -p2 - i*p1 + c + i*n ; fn = -p2 + i*p1 + c - i*n
            float fp1r = -p2Re + p1Im + cRe - nIm;
            float fp1i = -p2Im - p1Re + cIm + nRe;
            float fcr = p2Re + p1Re + cRe + nRe;
            float fci = p2Im + p1Im + cIm + nIm;
            float fnr = -p2Re - p1Im + cRe + nIm;
            float fni = -p2Im + p1Re + cIm - nRe;
            float fp2r = p2Re - p1Re + cRe - nRe;
            float fp2i = p2Im - p1Im + cIm - nIm;
            if (dg) { fp1r -= 4.0f * dRe; fp1i -= 4.0f * dIm; }
            wienner(fp2r, fp2i, sX, sY, p_.betaX);
            wienner(fp1r, fp1i, sX, sY, p_.betaX);
            wienner(fcr, fci, sX, sY, p_.betaX);
            wienner(fnr, fni, sX, sY, p_.betaX);
            filtRe_[idx] = (fp2r + fp1r + fcr + fnr) * 0.25f;
            filtIm_[idx] = (fp2i + fp1i + fci + fni) * 0.25f;
            if (dg) { filtRe_[idx] += dRe; filtIm_[idx] += dIm; } // 1/bt applies to factor too
          }
        }
      }
  } else if (p_.bm == 1) {
    for (int i = 0; i < nb * planeSize; i++) {
      float re = cur.re[i], im = cur.im[i];
      float dRe = 0, dIm = 0;
      if (dg) { dRe = corrRe_[i]; dIm = corrIm_[i]; re -= dRe; im -= dIm; }
      float sX = sigmaX, sY = sigmaY;
      if (pat) {
        float pv = pattern_[i % planeSize];
        sX = pv * p_.beta;
        sY = pv;
      }
      wienner(re, im, sX, sY, p_.betaX);
      if (dg) { re += dRe; im += dIm; }
      filtRe_[i] = re;
      filtIm_[i] = im;
    }
  } else if (p_.bm == 0) {
    // Kalman (fft3dgpu.hlsl:780-833), persistent state across frames.
    if (kalmanFirst_) {
      kLastRe_ = cur.re;
      kLastIm_ = cur.im;
      float cn0 = p_.kalmanCN;
      kCovarRe_.assign(nb * planeSize, cn0);
      kCovarIm_.assign(nb * planeSize, cn0);
      kProcRe_.assign(nb * planeSize, cn0);
      kProcIm_.assign(nb * planeSize, cn0);
      if (pat) {
        for (int b = 0; b < nb; b++)
          for (int i = 0; i < planeSize; i++) {
            float pv = pattern_[i];
            kCovarRe_[b * planeSize + i] = pv;
            kCovarIm_[b * planeSize + i] = pv;
            kProcRe_[b * planeSize + i] = pv;
            kProcIm_[b * planeSize + i] = pv;
          }
      }
      kalmanFirst_ = false;
    }
    const float kr2 = p_.kratio * p_.kratio;
    for (int i = 0; i < nb * planeSize; i++) {
      float re = cur.re[i], im = cur.im[i];
      float lRe = kLastRe_[i], lIm = kLastIm_[i];
      float dRe = re - lRe, dIm = im - lIm;
      float mask = std::max(dRe * dRe, dIm * dIm);
      float cn = pat ? pattern_[i % planeSize] : p_.kalmanCN;
      float s = kCovarRe_[i] + kProcRe_[i];
      float gain = s / (s + cn);
      float newProc = gain * gain * cn;
      float newCovar = (1.0f - gain) * s;
      float dstRe = gain * re + (1.0f - gain) * lRe;
      float dstIm = gain * im + (1.0f - gain) * lIm;
      if (mask > kr2 * cn) {
        dstRe = re;
        dstIm = im;
        newCovar = cn;
        newProc = cn;
      }
      filtRe_[i] = dstRe;
      filtIm_[i] = dstIm;
      kCovarRe_[i] = newCovar;
      kCovarIm_[i] = newCovar;
      kProcRe_[i] = newProc;
      kProcIm_[i] = newProc;
      kLastRe_[i] = dstRe;
      kLastIm_[i] = dstIm;
    }
  } else { // bm == -1: sharpen only
    for (int i = 0; i < nb * planeSize; i++) {
      filtRe_[i] = cur.re[i];
      filtIm_[i] = cur.im[i];
    }
  }

  // ---------- Phase 2: sharpen, inverse FFT, synthesis overlap-add ----------
  for (int b = 0; b < nb; b++) {
    if (p_.sharpenOn) {
      for (int r = 0; r < bh; r++)
        for (int c = 0; c < fbw; c++) {
          int idx = b * planeSize + r * fbw + c;
          float re = filtRe_[idx], im = filtIm_[idx];
          float dRe = 0, dIm = 0;
          if (dg) { dRe = corrRe_[idx]; dIm = corrIm_[idx]; re -= dRe; im -= dIm; }
          float psd = re * re + im * im + 0.0000000000000000000000000000000000001f;
          float factor = sharpen_[r * fbw + c];
          float g = 1.0f + factor * std::sqrt(psd * p_.ssmax / ((psd + p_.ssmin) * (psd + p_.ssmax)));
          re *= g;
          im *= g;
          if (dg) { re += dRe; im += dIm; }
          filtRe_[idx] = re;
          filtIm_[idx] = im;
        }
    }

    for (int i = 0; i < planeSize; i++)
      fullSpec_[i] = std::complex<float>(filtRe_[b * planeSize + i], filtIm_[b * planeSize + i]);
    fft2dRealInverseHalf(fullSpec_.data(), bw, bh, fbw, rowPlan_, colPlan_, blockReal_.data());

    const int x0 = bx_[b], y0 = by_[b];
    const float* win = (p_.mode == 1) ? (par_[b] ? winSynB_.data() : winSynA_.data()) : winAna_.data();
    for (int ly = 0; ly < bh; ly++) {
      int dy = y0 + ly;
      if (dy < 0 || dy >= H)
        continue;
      float* orow = &out[dy * W];
      const float* wrow = &win[ly * bw];
      const float* brow = &blockReal_[ly * bw];
      for (int lx = 0; lx < bw; lx++) {
        int dx = x0 + lx;
        if (dx < 0 || dx >= W)
          continue;
        orow[dx] += brow[lx] * wrow[lx];
      }
    }
  }

  // Chroma centering added back once per output pixel (mode 1).
  if (chromaNorm_ != 0.0f)
    for (int i = 0; i < W * H; i++)
      out[i] += chromaNorm_;
}

} // namespace fft3dcl
