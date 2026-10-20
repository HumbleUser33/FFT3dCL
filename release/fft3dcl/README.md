# FFT3dCL — VapourSynth port (OpenCL GPU)

Modern port of the **FFT3dGPU v0.8.6** GPU denoiser (Avisynth+) to a **native
VapourSynth (API 4) plugin**, running on **NVIDIA, AMD and Intel** graphics
cards via OpenCL 1.2. The denoising method and all original parameters are
preserved exactly.

Namespace: `fft3dcl` — function: `core.fft3dcl.FFT3DCL(clip, ...)`.

---

## 1. Installation

Copy `fft3dcl.dll` into the VapourSynth plugins folder. The DLL is
**self-contained** (static CRT, no Visual C++ DLL required; OpenCL is loaded at
run time from the driver).

### With StaxRip
StaxRip ships its own VapourSynth. Two options:

- **Plugins folder** (recommended): drop `fft3dcl.dll` into
  `<StaxRip folder>\Apps\Plugins\VapourSynth\` (or your VapourSynth `plugins`
  folder). It will be loaded automatically.
- **Explicit load**: add at the top of the VapourSynth script (`*.vpy`):
  ```python
  core.std.LoadPlugin(r"C:\path\to\fft3dcl.dll")
  ```

### Quick check
```
vspipe --info your_script.vpy
```
must list the `FFT3DCL` filter without errors.

---

## 2. Usage

Minimal example (luma denoising, default settings identical to the original):

```python
import vapoursynth as vs
core = vs.core
clip = core.std.AcquireSource(...)          # your source
clip = core.fft3dcl.FFT3DCL(clip)
clip.set_output()
```

Example with typical parameters:

```python
clip = core.fft3dcl.FFT3DCL(
    clip,
    sigma=2.0,      # denoising strength
    bt=3,           # temporal window (-1..4; 0 = Kalman)
    beta=1.0,
    plane=0,        # 0 = luma, 1 = chroma, 4 = all
    mode=1,         # 0 or 1 (block overlap)
    bw=32, bh=32,   # block size (power of two)
    degrid=1.0,     # grid correction (mode 1)
    sharpen=0.0,
)
```

All original parameters are accepted: `sigma, beta, bw, bh, bt, sharpen, plane,
mode, bordersize, precision, degrid, scutoff, svr, smin, smax, kratio, ow, oh,
wintype, interlaced, sigma2, sigma3, sigma4, oldfft`.
(`NVPerf` was removed.)

### Port-specific parameter: `device`
Engine selection — **with no effect whatsoever on the denoising**:
- `device=2` (default): OpenCL GPU, with automatic fallback to the CPU if the
  GPU fails — the reason is written to the log, the fallback is never silent;
- `device=1`: GPU required — if the GPU cannot run the filter, the script stops
  with the reason instead of quietly becoming twenty times slower;
- `device=0`: CPU reference.

---

## 3. Notes

- **Formats**: 8 to 16-bit integer and 32-bit float, greyscale / planar YUV.
  10-bit HDR10 works (the original path forces `precision=2`).
- **Precision**: this port computes in fp32. Compared with the original, the
  difference is ±1 to 2 LSB (8-bit) / ±4 (10-bit) — rounding noise between
  independent FFT implementations. See `docs/validation-phase2.md` and
  `docs/validation-phase3.md`.
- **Determinism**: identical output on every run and for any thread count.
- **Speed**: about 5× faster on GPU than on CPU (measured on RTX 3050). On real
  4K HDR10 material: 14.3 frames per second on an RTX 3050, and the graphics card
  is busy for 25 ms of the 70 ms each frame takes.
- **AMD/Intel**: the kernels follow the OpenCL 1.2 portability rules (no fp64,
  no atomics, barriers only). Numerically validated on an NVIDIA RTX 3050;
  AMD/Intel compatibility rests on those rules.
- **Known limitation**: non-standard overlap (`ow`/`oh` ≠ half the block) is
  approximate; the default setting (half-block overlap) is exact.
- **Block size**: `bw`/`bh` of 64 need 34.3 KiB of on-chip memory per block,
  which every discrete card provides but some integrated Intel GPUs (32 KiB
  limit) do not. If the card refuses, the filter says so and uses the CPU
  (`device=2`) or stops (`device=1`). 32 and 64 run at the same speed and use the
  same graphics memory, so 32 is the safer choice.
- **Memory**: the filter keeps the frequency data of the frames in its temporal
  window, one buffer per frame, per plane, per filter. On 4K material that is
  about 960 MiB for one luma filter (`bt=3, bw=32`), 290 MiB per chroma plane, so
  `plane=4` costs about 1.7 GB and two chained filters about 2.7 GB. All of it is
  released when the clip is destroyed. If you suspect memory is not coming back,
  set the environment variable `FFT3DCL_LOG` to a file path: the plugin then
  writes one line per instance created and destroyed, with the number of live
  instances and the device memory they hold. If that counter returns to 0 while
  GPU memory stays high, the memory is held by the driver or by the host
  application, not by this filter.

## 4. Intentional differences vs the original
1. `bt=-1` performs the real FFT round-trip (the original output black).
2. Mode 0 is deterministic (the original sometimes output black).
3. A `device` parameter was added (engine selection only).
4. `NVPerf` removed; `oldfft` kept.

---

Coded by Qwen 3.8 Flash Next IQ3_S, tested by a human
