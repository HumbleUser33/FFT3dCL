# FFT3DGPU v0.8.6 — parameter specification (extracted from source)

Source of truth: `upstream/FFT3dGPU/FFT3dGPU/FFT3dGPU.cpp` (Create_fft3dGPU line 1233, constructor line 163),
`documentation/fft3dgpu.htm`. These names, defaults and semantics are the contract for the port.

AviSynth registration (FFT3dGPU.cpp:1359):

```
fft3dGPU(c[sigma]f[beta]f[bw]i[bh]i[bt]i[sharpen]f[plane]i[mode]i[bordersize]i[precision]i
         [NVPerf]b[degrid]f[scutoff]f[svr]f[smin]f[smax]f[kratio]f[ow]i[oh]i[wintype]i
         [interlaced]b[sigma2]f[sigma3]f[sigma4]f[oldfft]b)
```

## Parameter table

| # | Name | Type | Default | Accepted / clamped | Semantics (from source) |
|---|------|------|---------|--------------------|--------------------------|
| 0 | clip | clip | — | Y8, YV12, YV16, YV411, YV24, YUY2 (8 bit), planar RGB (8 bit), 10-16 bit planar, 32f planar (0.8.6) | Input. Packed RGB rejected (l.1243-1245). |
| 1 | sigma | float | 2.0 | any; internal `sigma/255` (l.172) | Wiener noise level, same meaning as fft3dfilter. Same units at every bit depth (l.392-394). |
| 2 | beta | float | 1.0 | any | Wiener over-subtraction factor. `BETA=((beta-1)/beta, beta)` (l.398). beta=1 → plain Wiener. |
| 3 | bw | int | 32 | power of 2; input rounded DOWN to power of 2, clamped [4,512] (l.214-222) | Block width (pixels). |
| 4 | bh | int | 32 | same as bw (l.223-231) | Block height. |
| 5 | bt | int | 3 | -1..4 else error "Valid modes for bt are -1,0,1,2,3,4" (l.232-233) | -1 sharpen only; 0 Kalman; 1 2D; 2 cur+prev; 3 prev+cur+next; 4 2 prev+cur+next. Internal `bm=_bt`, `bt=max(_bt,1)` (l.173). |
| 6 | sharpen | float | 0.0 | any; `sharpen != 0` enables (l.173) | + sharpens, − blurs. Applied in frequency domain (fft3dfilter algorithm). |
| 7 | plane | int | 0 | 0 luma; 1,2,3 chroma (2,3→1, l.1238-1239); 4 all. RGB: forced 4. Grey: 4→0, 1→error (l.1251-1256) | Which planes filtered. plane=4 builds two chained instances (luma + chroma) sharing one dst (FFT3dGPUallPlane). |
| 8 | mode | int | 1 | 0..2 else →0 (l.234-235) | 0: 1:1 blocks; 1: 2:1 overlap (uses ow/oh); 2: 1:1 + border (uses bordersize). |
| 9 | bordersize | int | 1 | used only mode 2 (nx/ny l.280-281) | Border size around blocks in mode 2. |
| 10 | precision | int | 0 (FLOAT16) | 0..2 else →1; **bits>8 forces 2** (l.244-249) | 0: half (s10m5) storage for FFT+filter; 1: float FFT, half filter; 2: float everywhere. Storage only — arithmetic is fp32 on GPU. |
| 11 | NVPerf | bool | false | — | NVPerfHUD debug hook (DirectInput + visible window). **Dropped in port** (user decision). |
| 12 | degrid | float | mode==1 ? 1.0 : 0.0 (l.1266-1267) | 0 disables | Block-grid removal strength. Only meaningful mode 1. Applied via gridsample (constructor, l.515-570) + per-frame Minimize/GridCorrection. |
| 13 | scutoff | float | 0.3 | >0 | Sharpen cutoff (Gaussian in frequency, Fizick code): LUT `strength*(1-exp(-d²/2·scutoff²))`. |
| 14 | svr | float | 1.0 | any | Sharpen volume. |
| 15 | smin | float | 4.0 | internal `/255` (l.187) | Sharpen min strength threshold (same units all depths). |
| 16 | smax | float | 20.0 | internal `/255` (l.188) | Sharpen max strength threshold. |
| 17 | kratio | float | 2.0 | >0 | Kalman reset threshold: reset where `mask > kratio²·covarNoise`. |
| 18 | ow | int | bw/2 | scaled `ow*=bw/_bw`, forced even (`&=0xFFFFFFFE`), clamped `ow*2<=bw` (l.236-242) | Horizontal block overlap, mode 1 only. Must be even. |
| 19 | oh | int | bh/2 | scaled, clamped `oh*2<=bh` (l.238-240) | Vertical overlap, mode 1 only. |
| 20 | wintype | int | 0 | 0..2 else →0 (l.254-255) | Window function (mode 1): 0 half-cosine both; 1 sqrt-cos analysis / cos^1.5 synthesis; 2 none analysis / raised-cosine synthesis. Formulas in FFT3dGPU_textures.md §2.5. |
| 21 | interlaced | bool | false | mode 1 only | Per-field block layout (ny doubled, OFFSET=bh, StretchRect + Z-buffer tricks, textures doc §6). |
| 22 | sigma2 | float | =sigma | — | Sigma at highest frequencies. Pattern used iff any of sigma2/3/4 differs (l.175). |
| 23 | sigma3 | float | =sigma | — | Sigma mid-high frequencies. |
| 24 | sigma4 | float | =sigma | — | Sigma at lowest frequencies. Pattern texture: SigmaToPatternTexture (l.588-631, formulas in textures doc §7.1). |
| 25 | oldfft | bool | undefined → MEASURE | true→RADIX2LUT, false→STOCKHAM (l.1235) | FFT algorithm choice. undefined → benchmark both at construction, pick faster (l.371-386). Kept in port (user decision). |

## Derived internal quantities (constructor)

- `sigma_i = sigma / 255`; `smin_i = smin/255`; `smax_i = smax/255` — everything normalized 0..1.
- `s = sigma_i² · bt · bw · bh`; Wiener constant `SIGMA = (beta·s, s)`; `BETA = ((beta−1)/beta, beta)` (l.396-398).
- Kalman `covarNoise = sigma_i² · bw · bh` — **no bt factor** (l.446-451).
- Sharpen constants: `SSMIN = smin_i²·bw·bh·bt`, `SSMAX = smax_i²·bw·bh·bt` (l.413).
- `current = (bm==1) ? 0 : (bm==4) ? 2 : 1` — index of frame n inside the temporal window (l.176).
- Block grid: mode≠2: `nx=(width+bw*1.5−1)/bw` (half overlap) or `nx=((width+ow−1)/(bw−ow)+2)/2`; mode 2: `nx=(width+(bw−4·border)*1.5−1)/(bw−4·border)`; same for y; interlaced doubles y (l.259-282). `totw=nx·bw`, `toth=ny·bh`.
- Chroma instances use `width/xRatioUV × height/yRatioUV` (l.1269-1270, 172).

## Wiener helper (fft3dgpu.hlsl:645-665, verified)

Per texel two complex values `(x,z)` and `(y,w)`:

```
PSDInv = 1 / (x²+z²  ,  y²+w²  + 1e-30)
MulFac = BETA.x (= (beta-1)/beta)
if (1/SIGMA.x > PSDInv) MulFac = 1 − SIGMA.y · PSDInv     // SIGMA.x=beta·s, SIGMA.y=s
return MulFac · src
```

With pattern (sigma2/3/4): `Sigma = (pattern·beta, pattern)` sampled from the Pattern texture.

## Temporal transforms (fft3dgpu.hlsl, verified)

- bt=2 (FFT2p/iFFT2p): Hadamard `X0=a+b`, `X1=a−b`; inverse `a−b` (with 1/2 folded in normalization).
- bt=3 (FFT3p/iFFT3p, also fused in Wienner3d3): DFT-3 with constant `−0.8660254` (= −√3/2, truncated literal — reproduce exactly).
- bt=4 (FFT3p-style variant / Wienner3d4): Fizick variant, see FFT3dGPU_cpu_shaders.md.

## Behavioural notes for the port

- Filter is MT_SERIALIZED (FFT3dGPU.h:45-46,74-77): fully serialized today.
- Prefetch bug (Ultim, README): after `GetFrame(n)` the filter uploads frame `nuploaded = lastn + (lastn − last)` (l.1093-1104) which can be negative or past the end after seeks. Port: clamp to `[0, numFrames−1]`, document as the one intentional behaviour fix.
- Kalman state persists across frames; on device lost it is `Restore()`d and GetFrame re-runs (l.1048-1064).
- Frame properties copied from frame n (v0.8.5+, l.1077-1079).
- Untouched planes BitBlt'd from source (l.1081-1091).

## Detailed references

- GPU pass classes, shaders, constants: `docs/FFT3dGPU_cpu_shaders.md`
- FFT algorithms, twiddle tables, normalization: `docs/FFT3dGPU_fft.md`
- Texture formats, upload/download conversions, coordinates, memory: `docs/FFT3dGPU_textures.md`
