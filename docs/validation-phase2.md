# Phase 2 validation — CPU reference vs original FFT3dGPU 0.8.6

Date: 2026-10-17. Method: same lossless FFV1 clip rendered through
Avisynth+ with the original `FFT3dGPU.dll` (precision=2) and through
VapourSynth with the port; decoded to raw and compared sample-by-sample
(`tools/run_test.ps1`, `tools/compare.py`).

Clips: `tools/clip8.mkv` (256x192, 8 frames, YUV420P8, crop of Sample.mkv)
and `tools/clip10.mkv` (same, YUV420P10).

## Results (8-bit, precision=2 both sides)

| Parameters | Identical | Max diff |
|---|---|---|
| default (mode 1, bt=3, degrid=1) | 97.1% | 1 |
| bt=1, degrid=0 | 79.3% | 2 |
| bt=2 | 96.0% | 2 |
| bt=4 | 73.7% | 2 |
| bt=0 (Kalman) | 94.0% | 2 |
| sharpen=1 | 90.8% | 1 |
| wintype=1 | 97.0% | 1 |
| wintype=2 | 96.8% | 2 |
| sigma2/3/4 pattern | 97.6% | 1 |
| beta=2 | 97.4% | 1 |
| plane=1 (U) | 99.2% | 2 |
| plane=3 (V) | 99.2% | 2 |
| plane=4 (both chroma) | 96.3% | 2 |
| bw=64,bh=64 | 93.5% | 2 |
| bw=16,bh=16 | 97.4% | 1 |
| 10-bit default (yuv420p10le) | 94.8% | 4 |

Remaining differences are ±1-2 LSB (8-bit) / ±4 (10-bit): independent fp32
FFT implementations (original Stockham on GPU vs port Cooley-Tukey on CPU)
round differently. This is the expected noise floor.

## Determinism

- Port rendered twice: 100% bit-identical.
- Kalman (bt=0) with `-t 1` vs `-t 8`: 100% bit-identical, and matches the
  original at max 2. Frame-state delivery order is stable.

## Bugs found in the original during validation

1. **mode 0 is non-deterministic**: with precision 0 or 2 (and sometimes 1),
   the original outputs black luma randomly (texture-pool state bug). Only
   precision=1 reliably works. The port's mode 0 is deterministic; it cannot
   be validated against the original beyond the runs where the original
   happens to produce valid output.
2. **bt=-1 (no filter) outputs black luma** in the original (mode 0 and 1).
   The port implements the mathematically correct round-trip instead.
3. **gridsample is built from a white frame**, not from clip content
   (FFT3dGPU.cpp:529-531 clears the download texture to 0xFFFFFFFF). The
   port replicates this exactly.
4. **Degrid normalization**: the degrid factor is included in the 1/bt
   normalization of the fused temporal shaders (Wienner3d2/3/4), so the
   restored correction is `corr`, not `bt*corr`.

## Known limitation of the CPU reference

**General-o mode 1** (`ow*2 != bw` or `oh*2 != bh`): the port's uniform
block model reproduces the analysis geometry exactly but only approximates
the synthesis (78% identical, ow=8). The original's `OQuad::CreateVBi`
applies per-cell-parity factor offsets (x&1, y&1) that the uniform model
does not capture. This is a rare option (default is half-overlap). The
OpenCL port will replicate the original RECT/vertex-buffer structure
directly, so general-o will be exact there.

## Conclusion

The CPU reference is a faithful oracle for the production paths:
mode 1 half-overlap, all bt values, Wiener/Kalman/sharpen/degrid/pattern/
wintype, luma and chroma, 8 and 10 bit. Phase 3 (OpenCL) will be validated
against this reference.
