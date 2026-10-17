# Phase 3 validation — OpenCL GPU backend

Date: 2026-10-17. The GPU backend (OpenCL C 1.2, runtime-loaded `OpenCL.dll`)
is validated two ways:

1. **GPU vs CPU oracle** (`tools/gpu_test.exe`, random planes, float domain):
   the full pipeline (analysis FFT, temporal filter, degrid, sharpen, inverse
   FFT, overlap-add) matches the CPU reference to **maxdiff ≤ 1e-6** for
   bt=1/2/3/4, Kalman (bt=0), sharpen on/off, degrid on/off, sigma pattern.
   This is ~4000× below one 8-bit LSB.

2. **GPU vs original FFT3dGPU 0.8.6** (`tools/run_test.ps1 -PortParams
   "device=1"`, FFV1 renders, sample-by-sample):

| Parameters | Identical | Max diff |
|---|---|---|
| default (mode 1, bt=3, degrid=1) | 97.1% | 1 |
| bt=1 | 97.1% | 1 |
| bt=2 | 96.0% | 1 |
| bt=4 | 84.7% | 2 |
| bt=0 (Kalman) | 94.0% | 1 |
| sharpen=1 | 90.8% | 1 |
| wintype=1 | 97.0% | 1 |
| wintype=2 | 96.8% | 1 |
| sigma2/3/4 pattern | 97.6% | 1 |
| beta=2 | 97.4% | 1 |
| plane=4 (luma+chroma) | 96.3% | 1 |
| bw=64,bh=64 | 93.5% | 1 |
| 10-bit default (yuv420p10le) | 94.8% | 4 |

These match the CPU reference's numbers (docs/validation-phase2.md) within a
couple of pixels, confirming the GPU path is numerically equivalent to the
validated oracle.

## Determinism / threading

- Kalman (bt=0) on GPU, `-t 1` vs `-t 8`: 100% bit-identical.
- The command queue is shared process-wide; each filter instance owns its
  kernels and buffers, and the per-instance mutex serializes submissions.

## Performance

1280x720, 32 frames, default params (includes decode + vspipe overhead):

- GPU (RTX 3050): 1.02 s  (~31 fps)
- CPU reference:   5.49 s  (~5.8 fps)

≈ **5.4× faster** end-to-end; the filter-only speedup is larger.

## Portability notes (AMD/Intel untested on this PC)

- Kernels are strict OpenCL C 1.2: no fp64, no atomics, no `extern __local`
  (static local arrays sized by `-DLSZ`), barriers only, work-groups ≤ 1024.
- `fft1d` uses float `cos/sin` twiddles (no double) — portable; the ~1e-7
  twiddle error is far below the fp32 noise floor already accepted.
- Local memory: 64x64 blocks need 32 KB (fits NVIDIA 48 KB, AMD 64 KB, Intel
  iGPU 32 KB borderline). Blocks > 64x64 are rejected with a clear error.
- Verified only on NVIDIA RTX 3050 (OpenCL 3.0 CUDA, runs the 1.2 source).
  AMD/Intel correctness rests on the portability rules above; cross-vendor
  bit-exactness is *not* guaranteed (different transcendentals), but the
  deviation is expected to stay within the ±1-2 LSB noise floor.

## Known limitation

General-o mode 1 (`ow*2 != bw` or `oh*2 != bh`) uses the same uniform block
grid as the CPU reference and is approximate (see validation-phase2.md). The
half-overlap default is exact.
