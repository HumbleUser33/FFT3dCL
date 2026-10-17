# FFT3dCL

Modern port of **FFT3dGPU v0.8.6** (an Avisynth+-only, Direct3D 9 filter) to a
**native VapourSynth (API 4) plugin**, with an **OpenCL 1.2** backend that runs
on **NVIDIA, AMD and Intel** graphics cards.

The denoising method and **all 23 original parameters are preserved exactly**.
Only `NVPerf` was dropped (it was Direct3D 9 specific).

```python
clip = core.fft3dcl.FFT3DCL(clip, sigma=2.0, bt=3)
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

Copy `release/fft3dcl/fft3dcl.dll` into the VapourSynth `plugins` folder, or
load it explicitly:

```python
core.std.LoadPlugin(r"C:\path\to\fft3dcl.dll")
```

With **StaxRip**: drop the DLL into StaxRip's VapourSynth plugins folder
(`Apps\Plugins\VapourSynth\`), then call the filter from the script.
See [`release/fft3dcl/README.md`](release/fft3dcl/README.md) and
[`release/fft3dcl/sample.vpy`](release/fft3dcl/sample.vpy).

---

## Parameters

All original parameters are accepted:

`sigma`, `beta`, `bw`, `bh`, `bt`, `sharpen`, `plane`, `mode`, `bordersize`,
`precision`, `degrid`, `scutoff`, `svr`, `smin`, `smax`, `kratio`, `ow`, `oh`,
`wintype`, `interlaced`, `sigma2`, `sigma3`, `sigma4`, `oldfft`.

Every option, with its default and what it does:

```python
clip = core.fft3dcl.FFT3DCL(
    clip,
    sigma=2.0,      # denoising strength (Wiener noise level, same units at any bit depth)
    beta=1.0,       # over-subtraction factor; 1.0 = plain Wiener
    bt=3,           # temporal window: -1 sharpen only, 0 Kalman, 1 2D,
                    # 2 current+previous, 3 prev+current+next, 4 two prev+current+next
    plane=0,        # 0 = luma, 1 = chroma, 4 = all
    mode=1,         # 0 = 1:1 blocks, 1 = half-block overlap (uses ow/oh), 2 = not ported
    bw=32, bh=32,   # block size, power of two, 4..512 (other values round down)
    ow=16, oh=16,   # block overlap, mode 1 only, must be even, at most bw/2 and bh/2
    degrid=1.0,     # block-grid removal, mode 1 only (default 1.0 in mode 1, else 0)
    sharpen=0.0,    # positive sharpens, negative blurs, applied in the frequency domain
    scutoff=0.3,    # sharpen cutoff (Gaussian in the frequency domain)
    svr=1.0,        # sharpen volume
    smin=4.0,       # sharpen minimum strength
    smax=20.0,      # sharpen maximum strength
    kratio=2.0,     # Kalman reset threshold (used with bt=0)
    bordersize=1,     # border around blocks, mode 2 only (mode 2 is not ported)
    precision=0,      # storage: 0 half, 1 mixed, 2 float; forced to 2 above 8-bit
    wintype=0,        # window function in mode 1: 0 half-cosine, 1 sqrt-cos, 2 raised-cosine
    sigma2=2.0,       # sigma at the highest frequencies; setting any of sigma2/3/4
    sigma3=2.0,       # to a value different from sigma enables the noise-pattern mode
    sigma4=2.0,       # sigma at the lowest frequencies
    oldfft=0,         # FFT implementation choice (accepted, no effect on the result)
)
```

Two parameters behave slightly differently here: `interlaced` is accepted but
rejected with an error (not ported yet), and `precision` only selects the
storage format — the arithmetic is 32-bit float everywhere.

One parameter specific to this port, **with no effect on the denoising**:

| `device` | Behaviour |
|---|---|
| `2` (default) | OpenCL GPU, automatic fallback to the CPU if the GPU fails |
| `1` | GPU required (error if unavailable) |
| `0` | CPU reference |

Accepted formats: 8 to 16-bit integer and 32-bit float, greyscale or planar YUV
(10-bit HDR10 works).

The full contract, with the exact clamping rules of the original, is in
[`docs/params.md`](docs/params.md).

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

The resulting DLL is `build/Release/fft3dcl.dll`.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | the plugin: VapourSynth entry point, CPU reference, OpenCL backend |
| `docs/` | parameter contract, porting map, validation reports, and the cartography notes of the original plugin |
| `tools/` | test harness (reference render vs port render, sample-by-sample diff) |
| `release/fft3dcl/` | ready-to-install DLL, install guide and sample script |
| `upstream/` | the original [pinterf/FFT3dGPU](https://github.com/pinterf/FFT3dGPU) v0.8.6 sources (Avisynth + Direct3D 9), unmodified |
| `archive/fr/` | French originals of the translated documents |
| `third_party/` | VapourSynth API headers used for the build |

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

## License

GNU GPL v2, like the original project
[pinterf/FFT3dGPU](https://github.com/pinterf/FFT3dGPU), from which this port is
derived.

---

Coded by Qwen 3.8 Flash Next IQ3_S, tested by a human
