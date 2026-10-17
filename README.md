# FFT3dOpenCL

Modern port of **FFT3dGPU v0.8.6** (an Avisynth+-only, Direct3D 9 filter) to a
**native VapourSynth (API 4) plugin**, with an **OpenCL 1.2** backend that runs
on **NVIDIA, AMD and Intel** graphics cards.

The denoising method and **all 23 original parameters are preserved exactly**.
Only `NVPerf` was dropped (it was Direct3D 9 specific).

```python
clip = core.fft3dgpu.FFT3DGPU(clip, sigma=2.0, bt=3)
```

---

## Project status

| Item | Status |
|---|---|
| CPU reference backend | ✅ complete and validated |
| OpenCL GPU backend | ✅ complete and validated |
| VapourSynth API 4 plugin | ✅ |
| Match against original 0.8.6 | ✅ ±1–2 LSB (8-bit), ±4 (10-bit) |
| Determinism (threads, repeated runs) | ✅ 100 % identical |
| Real 4K HDR10 renders | ✅ |
| StaxRip integration (VapourSynth R79) | ✅ tested — see [`docs/validation-staxrip.md`](docs/validation-staxrip.md) |
| AMD / Intel | ⚠️ not tested (portability guaranteed by strict OpenCL 1.2 rules) |

The differences from the original come from two independent FFT implementations
(rounding noise), not from a change of algorithm. Full details in
[`docs/validation-phase2.md`](docs/validation-phase2.md),
[`docs/validation-phase3.md`](docs/validation-phase3.md),
[`docs/validation-phase5.md`](docs/validation-phase5.md) and
[`docs/validation-staxrip.md`](docs/validation-staxrip.md).

---

## Installation

The DLL is **self-contained**: the Visual C++ runtime is linked statically and
`OpenCL.dll` is loaded at run time by the graphics driver. Single dependency:
`KERNEL32.dll`.

Copy `release/fft3dgpu/fft3dgpu.dll` into the VapourSynth `plugins` folder, or
load it explicitly:

```python
core.std.LoadPlugin(r"C:\path\to\fft3dgpu.dll")
```

With **StaxRip**: drop the DLL into StaxRip's VapourSynth plugins folder
(`Apps\Plugins\VapourSynth\`), then call the filter from the script.
See [`release/fft3dgpu/README.md`](release/fft3dgpu/README.md) and
[`release/fft3dgpu/sample.vpy`](release/fft3dgpu/sample.vpy).

---

## Parameters

All original parameters are accepted:

`sigma`, `beta`, `bw`, `bh`, `bt`, `sharpen`, `plane`, `mode`, `bordersize`,
`precision`, `degrid`, `scutoff`, `svr`, `smin`, `smax`, `kratio`, `ow`, `oh`,
`wintype`, `interlaced`, `sigma2`, `sigma3`, `sigma4`, `oldfft`.

One parameter specific to this port, **with no effect on the denoising**:

| `device` | Behaviour |
|---|---|
| `2` (default) | OpenCL GPU, automatic fallback to the CPU if the GPU fails |
| `1` | GPU required (error if unavailable) |
| `0` | CPU reference |

Accepted formats: 8 to 16-bit integer and 32-bit float, greyscale or planar YUV
(10-bit HDR10 works).

---

## Performance

Measured on an RTX 3050: the GPU path is about **5× faster** than the CPU path
(1280×720 float: 1.02 s on GPU vs 5.49 s on CPU). Full-frame 4K fits in 8 GB of
video memory.

---

## Building

Windows, Visual Studio 2022 (MSVC), CMake. The VapourSynth headers are provided
in `third_party/`.

```
cmake -S . -B build
cmake --build build --config Release
```

The resulting DLL is `build/Release/fft3dgpu.dll`.

---

## Intentional differences from the original

Full list in [`docs/porting-map.md`](docs/porting-map.md). The main ones:

1. `bt=-1` performs the real FFT round-trip (the original output black luma).
2. Mode 0 is deterministic (the original sometimes output black frames).
3. A `device` parameter was added (engine selection only).
4. `NVPerf` removed; `oldfft` kept but ignored.
5. Non-standard overlap (`ow*2 != bw` or `oh*2 != bh`) is approximate:
   half-block overlap, which is the default, is exact.

---

## Licence

GNU GPL v2, like the original project
[pinterf/FFT3dGPU](https://github.com/pinterf/FFT3dGPU), from which this port is
derived.
