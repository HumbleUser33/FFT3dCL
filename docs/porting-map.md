# FFT3DGPU → OpenCL porting map

Inventory of the original DX9 pipeline (v0.8.6), one section per stage. Formulas are NOT duplicated
here — they live in the three deep-dive docs; this file fixes the ORDER, the CONDITIONS and the
porting decisions.

- `docs/FFT3dGPU_cpu_shaders.md` — every filter class, shader entry, constants, window formulas
- `docs/FFT3dGPU_fft.md` — both GPU FFT algorithms, twiddle tables, normalization placement
- `docs/FFT3dGPU_textures.md` — texture formats, upload/download conversions, coordinate traps, memory

## 0. Global structure

One filter instance processes ONE plane group: luma only, or U+V (chroma), or G then B+R for RGB.
`plane=4` chains two instances sharing one destination frame (FFT3dGPUallPlane, FFT3dGPU.cpp:1188-1229).
Everything below runs per plane group, serialized.

Per output frame n (GetFrame, FFT3dGPU.cpp:759-1186):

```
for each plane in group:
  [A] block extraction + windowing        (ImgStream mode 0/2 | ImgStream2 mode 1)
  [B] 2D FFT per block                    (FFT2dRR = RADIX2LUT | FFT2dRR2 = STOCKHAM)
  [C] temporal stage (bm dependent)       (cached FFTs + fused Wiener | Kalman | none)
  [D] degrid correction                   (Minimize + GridCorrection, if degrid != 0)
  [E] Wiener / Kalman filtering           (Wienner2d/3d2/3d3/3d4 | KalmanMP)
  [F] sharpen (frequency domain)          (Sharpen, if sharpen != 0)
  [G] inverse 2D FFT                      (same FFT class, forward=false)
  [H] un-window + overlap-add + pack      (Float4toImg2* variants)
  [I] download + round/clamp to output    (DownloadFromTexture)
```

Constructor-time one-offs: degrid `gridsample` computation (l.515-570), sigma `Pattern` texture
(l.588-631), FFT algorithm choice when `oldfft` undefined (l.371-386), twiddle/factor LUT uploads.

## 1. Frame upload (CPU → GPU)

- 8 bit: 2 pixels per texel, UNORM (A8L8 / A8R8G8B8 family). 10-16 bit: 2 px/texel 16-bit UNORM.
  32f: 2 px/texel float2. Chroma planes are their own textures (never packed across planes).
- YUY2 path (packed) exists in the original — **not ported** (no packed formats in VapourSynth).
- Sampler state at every pass: ADDRESSU/V = MIRROR, point filtering (FFT3dGPU.cpp:816-820);
  Factor sampler temporarily WRAP during mode-1 passes. Border color 0 at device start only.
- **OpenCL decision**: no images. Plain `float`/`half`-storage buffers, explicit integer addressing,
  `mirror_i()` helper reproducing D3D mirror (no edge-texel duplication) exactly.

## 2. [A] Block extraction + analysis window

| mode | class | shaders | notes |
|---|---|---|---|
| 0, 2 | ImgStream | `Img2toFloat4`, `Img2toImg4` (mode 2 border) | 2 px/texel packing; offset (0.25·bw, 0.5·bh) packed texels = half block; window `sqrt(0.5(cosx²+cosy²))`; mode 2 adds border blocks (`sign·cos²`) |
| 1 | ImgStream2 | 8 variants `psImg2ToFloat4*` (SP/MP × half-overlap/other × interlaced) | two output textures (pair) with half-block horizontal offset; wintype 0/1/2 LUTs (FactorLUTana/syn); interlaced: OFFSET=bh, StretchRect band swaps |

Output: `img` (mode 0/2) or `imgp` pair, size `totw/2 × toth`, HALF4 or FLOAT4 per precision.
Details: cpu_shaders.md §6-7; window formulas textures.md §2.4-2.5.

## 3. [B]/[G] 2D FFT (forward and inverse)

- Two algorithms, selected by `oldfft` or measured at construction:
  - RADIX2LUT (`FFT2dRR`): radix-2 butterflies with LUT textures, DrawEven/DrawOdd quads.
  - STOCKHAM (`FFT2dRR2`): Stockham autosort, twiddles in vertex coords (HSP/VSP/HMP/VMP), C2R/R2C + CollectH/CollectV.
- Packing: one RGBA texel = two complex numbers `(x,z)` and `(y,w)`; two real images share one
  complex FFT (hence FFT textures are `totw/2` wide and spectra keep half columns, fbw=bw/2+1).
- Twiddles: computed on CPU in **double**, uploaded as textures (fft.md) — keep this in the port
  (host-side double tables → float, kills cross-vendor sin/cos differences).
- Normalization 1/N: RADIX2LUT folds it into the first inverse butterfly shader; STOCKHAM into the
  final `Combine`. Keep the same multiplication points.
- **OpenCL decision**: one work-group per block row/column Stockham in local memory; blocks ≤64 use
  local memory, larger fall back to global two-pass. Same stage order and same table values.

## 4. [C] Temporal handling (bm = bt)

| bm | path | shaders |
|---|---|---|
| −1 | no temporal, no denoise: FFT → [D] → [F] sharpen → inverse | — |
| 0 | Kalman, stateful across frames (`KalmanFilter`, ping/pong per plane) | `Kalman` (MRT) or `KalmanMP` (PASS1_K/PASS2_K multi-pass) |
| 1 | single frame Wiener | `Wienner2d` |
| 2..4 | FFT of each of the bt frames cached in LRU `GPUCache` (size bt); fused temporal+Wiener on (prev,current,next) | `Wienner3d2` (Hadamard), `Wienner3d3` (DFT3, const −0.8660254), `Wienner3d4` (Fizick variant) |

- Temporal window positions: frames `n − current + i`, `i=0..bt−1`, `current=(bm==1?0:bm==4?2:1)`.
- bt=2/3/4 also have non-fused `FFT2p/FFT3p/iFFT2p/iFFT3p` shaders (multi-pass path for 1-RT cards).
- **OpenCL decision**: fused shader becomes one kernel reading 3 buffers; GPUCache becomes a small
  LRU of device buffers; Kalman state = persistent buffers per plane.

## 5. [D] Degrid (degrid != 0)

- Constructor: FFT of first frame → `Minimize` (DC per block → `mintex` nx×ny) → `GridCorrection`
  → download → `gridsample` texture (FLOAT4). One-time.
- Per frame: `selectDC->Apply(current FFT)` → `mintex`; `calcgridcorrection->Apply(mintex, out, gridsample)`.
- Wiener shaders subtract `factor` (×3 in 3d3) before and add back after filtering (DEGRID define).
- Details: cpu_shaders.md §1-2.

## 6. [E] Wiener filtering

- Constants: `SIGMA=(beta·s, s)`, `BETA=((beta−1)/beta, beta)`, `s=(sigma/255)²·bt·bw·bh`.
- Pattern texture when sigma2/3/4 differ (formulas params.md + textures.md §7.1).
- Helper `Wienner()` (hlsl:645) — verified, see params.md. Epsilon `1e-30` literal — reproduce.
- Details: cpu_shaders.md §3.

## 7. [F] Sharpen

- Frequency-domain sharpen (Fizick fft3dfilter algorithm): LUT `strength·(1−exp(−d²/2·scutoff²))`,
  thresholds `SSMIN/SSMAX = (smin|smax/255)²·bw·bh·bt`, `svr` volume; runs on FFT textures, needs a
  re-FFT of its output when applied after Wiener (see GetFrame order l.860-865).
- Details: cpu_shaders.md §5.

## 8. [H] Synthesis: un-window + overlap-add + pack

- mode 0/2: `Float4toImg2` (+ `Img2toImg4` border for mode 2).
- mode 1: 8 variants `psFloat4ToImg2*` (half-overlap / arbitrary overlap via OQuad vertex buffers /
  interlaced via Z-buffer line mask D24X8 + `CopyTexToZBuffer`).
- Synthesis window LUT (FactorLUTsyn) applied here; normalization divide folded in (check per variant).
- **Known original bug**: MODE2 synthesis window overwritten (hlsl:155-156) — reproduce or fix
  consciously, flag in CHANGES-vs-original.md.
- Details: cpu_shaders.md §6-7, textures.md §6.

## 9. [I] Download + output conversion

- Download RT is FIXED4 family (4 samples/texel); UNORM write rounds to nearest and clamps [0,1] —
  reproduce with `rint` + clamp after scaling by (2^bits−1).
- 10-14 bit: double quantization quirk (textures.md §2.4) — reproduce, document.
- Sample order quirk at 16 bit (`/*0123 becomes 2103*/`, hlsl:984) is a memory-layout artifact to
  allow memcpy — irrelevant once OpenCL writes planar buffers directly.

## 10. Coordinate system traps (must match exactly)

1. Half-texel: quad texcoords `etaW = 0.5/W + left/W` (vertexbuffer.cpp:87-99) → integer center
   sampling in kernels; verify with impulse test.
2. Negative source RECTs = block offsets; MIRROR addressing does the edge padding. Port: explicit
   `mirror_i` on integer coords, same offsets (0.25·bw packed = bw/2 px, 0.5·bh).
3. `OFFSET` constants per shader (textures.md §8.4).
4. Interlaced: bh offset, band swaps, Z-buffer field mask — port as explicit field index math.

## 11. Precision / storage mapping

| precision | original storage | OpenCL port |
|---|---|---|
| 0 (8-bit default) | HALF4 (s10m5) RTs | fp32 buffers default; `-DSTORAGE_HALF` option using `vstore_half` for strict comparison |
| 1 | FLOAT4 FFT, HALF filter | same option split |
| 2 | FLOAT4 everywhere | fp32 (default for 10+ bits, forced l.244-246) |

Arithmetic is fp32 in all modes on modern GPUs; keep `FP_CONTRACT OFF` in validation builds.

## 12. Parameters → passes matrix

| param | affects |
|---|---|
| sigma, beta, sigma2/3/4 | [E] (and Pattern texture build) |
| bw, bh | [A][B][C][D][E][F][G][H] (block grid, tables, constants s, SSMIN/MAX, covarNoise) |
| bt | [C][E] selection, GPUCache size, s, sharpen constants |
| sharpen, svr, scutoff, smin, smax | [F] |
| plane | instance topology (luma/chroma), chroma resolution |
| mode, bordersize, ow, oh, wintype | [A][H], nx/ny, LUTs |
| precision | storage of [A]-[H] buffers |
| degrid | [D] + DEGRID define in [E] |
| kratio | [E] Kalman reset |
| interlaced | [A][H] variants, ny |
| oldfft | [B][G] algorithm |
| NVPerf | (dropped) |

## 13. Intentional differences vs original (to record in CHANGES-vs-original.md)

1. Prefetch frame index clamped to `[0, numFrames−1]` (fixes the documented negative-frame bug).
2. NVPerf parameter removed (user decision); `oldfft` kept.
3. YUY2 packed path not ported (VapourSynth has no packed YUV; planar equivalents are lossless).
4. Default precision fp32 for all bit depths; fp16 storage available as build/runtime option.
5. Device-lost recovery machinery not needed in OpenCL (no D3D device); errors surface as VS errors.
6. New optional `device` parameter (0=CPU, 1=GPU required, 2=GPU with CPU fallback, default 2).
   Backend selection only — it never changes the denoising math. The original had no equivalent
   (it was D3D9-only).
7. `bt=-1` produces the correct FFT round-trip instead of the original's black-luma bug.
8. Mode 0 is deterministic in the port; the original's mode 0 intermittently outputs black frames
   (texture-pool state bug), so mode 0 could not be validated against it.
9. General-o mode 1 (`ow*2 != bw` or `oh*2 != bh`): both backends use a uniform block grid that
   approximates the original's per-cell-parity synthesis offsets. Half-overlap (the default) is
   exact. See docs/validation-phase2.md.
