# Changelog

## v1.1.1 — 2026-10-20

One setting, `bw=64`, was quietly running on the processor instead of the
graphics card, about twenty times slower. It now runs on the card. The pictures
are exactly the same as before.

### Fixed

- **`bw=64` really uses the graphics card now.** A leftover test program made the
  graphics code fail to build at that block size, and the filter then fell back
  to the processor without saying so.
- **The memory log counted too much.** Buffers that had already been given back
  were not subtracted, so the log reported 1694 MiB where 1113 MiB were in use.

### Changed

- **A graphics problem is reported instead of hidden.** With `device=1` the
  script stops and names the reason; with `device=2` a warning names the reason
  before the filter continues on the processor.
- **The README gives the real numbers**: how much graphics memory each filter
  holds on 4K material, and which block sizes a card can accept (64x64 needs
  34.3 KiB of on-chip memory, which some integrated Intel GPUs do not have).

### Speed

10 frames of a real 4K HDR10 clip (3840x2160, 10 bit), two filters in a row,
pass 1 with `plane=4, bw=64`, pass 2 with `plane=0, bw=32`, same machine:

| | v1.1.0 | v1.1.1 |
|---|---|---|
| time for the 10 frames | 58.1 s | 2.9 s |
| speed | 0.17 frames per second | 3.5 frames per second |
| filters running on the graphics card | 0 of 4 | 4 of 4 |

A repeat run measured 2.4 s. Decoding the same clip without any filter takes
2.4 s, so the denoising now costs about what decoding costs.

### Memory

Graphics memory held by the filter on 4K material, measured per filter:

| Filter | Graphics memory |
|---|---|
| one luma filter, `bt=3, bw=32` | 964 MiB |
| one luma filter, `bt=4, bw=64` | 1113 MiB |
| one chroma plane (U or V) | 291 MiB |
| one filter with `plane=4` (luma + U + V) | 1695 MiB |
| two filters in a row (`plane=4`, then luma) | 2659 MiB |

`bw=64` costs about the same as `bw=32`, in memory as in speed: with half-block
overlap, twice as many blocks means half as much data per block. Everything is
given back when the filter is destroyed.

### The pictures did not change

- Settings that already worked: 74 649 600 samples of 74 649 600 identical,
  maximum difference 0.
- `bw=64` (processor before, graphics card now): 100.00 % identical, maximum
  difference 1.
- Against the original FFT3dGPU 0.8.6: 97.14 % at the default settings, 94.83 %
  at 10 bit, 93.54 % at `bw=64` — the same numbers as before. Details in
  [`docs/validation-staxrip.md`](docs/validation-staxrip.md).

## v1.1.0 — 2026-10-18

Same pictures and the same parameters, but about twice as fast and noticeably
lighter on the processor and on graphics memory.

### Changed

- **The denoiser is about twice as fast.** The heavy math was reorganised so the
  graphics card does fewer, larger steps instead of many small ones. On a 4K HDR
  clip this one: 7.3 frames per second, now 14.3.
- **The filter no longer occupies a processor core while it waits.** It used to
  keep one core turning while the graphics card worked; it now waits quietly, so
  roughly 18 ms of processor time per 4K frame goes back to the encode.
- **Less graphics memory per filter.** The extra float copies of each plane are
  no longer needed: 281 MiB per filter instance on 4K material instead of 345,
  and the peak on a 4 GB card drops from 3.2 GB to 2.0 GB.

### Added

- **Optional timing log.** Setting the environment variable `FFT3DCL_TIME` to a
  file path writes how long each step of a frame takes, on the processor and on
  the graphics card. This is how the numbers above were measured.

### Verified

- **The pictures did not change at all.** The v1.0.0 build and this one were
  compared sample by sample on 8, 10 and 16 bit material: no sample differs.

### Performance report

24 frames of a real 4K HDR10 clip (3840x2160, 10 bit) with
`sigma=3, bt=4, bw=32, bh=32, ow=16, oh=16, sharpen=0.4`, on an NVIDIA RTX 3050
4 GB, three runs of each build.

| Measure | v1.0.0 | v1.1.0 |
|---|---|---|
| time for the 24 frames | 3.28 s | 1.68 s |
| speed | 7.3 fps | 14.3 fps |
| processor time used by the whole run | 9.85 s | 8.51 s |
| processor time the filter itself adds | about 2.4 s (99 ms per frame) | about 1.0 s (43 ms per frame) |
| graphics memory held by one filter | 345 MiB | 281 MiB |
| graphics memory peak, whole PC | 3220 MiB | 2043 MiB |
| program memory peak | 3425 MiB | 2955 MiB |

Decoding the same clip without the filter takes 7.48 s of processor time, which
is what the "the filter itself adds" line is measured against.

Where the time inside one frame goes now, and where it went before:

| Step, per frame | v1.0.0 | v1.1.0 |
|---|---|---|
| forward FFT (including the pixel conversion) | 43.6 ms | 8.7 ms |
| temporal filter | 5.4 ms | 5.2 ms |
| inverse FFT | 42.1 ms | 9.4 ms |
| putting the blocks back together | 1.7 ms | 1.6 ms |
| sending the frame to the graphics card | 2.9 ms | 2.7 ms |
| reading the result back | 3.5 ms | 3.0 ms |
| copying the planes the filter does not touch | 1.5 ms | 1.7 ms |
| **graphics card work per frame** | **92.8 ms** | **24.9 ms** |

Each change on its own, always compared with the build just before it, in the
same session. The machine itself moves by a few percent between runs.

| Change | wall time | speed | pictures |
|---|---|---|---|
| rotation factors computed once per group | 3.60 → 2.51 s | 6.7 → 9.6 fps | identical |
| one padded column, removes a memory bank conflict | 3.32 → 2.14 s | 7.2 → 11.2 fps | identical |
| all rows, then all columns, in a single pass | 3.37 → 1.84 s | 7.1 → 13.0 fps | identical |
| skip the mirror columns, larger work groups | 3.46 → 1.66 s | 6.9 → 14.6 fps | identical |
| pixel conversion inside the kernels | 1.80 → 1.71 s | 13.3 → 14.1 fps | identical, −64 MiB |
| waiting quietly instead of spinning | unchanged | 13.6 → 14.0 fps | identical, CPU 8.80 → 8.37 s |

The last two lines are not about speed: their gain is memory (64 MiB less per
filter, which matters on a 4 GB card) and processor time (18 ms per frame back
for the encode that follows).

Nothing moved by a single unit. `tools/ab_dll.ps1` renders the same clip with
both builds and compares every sample: 100 % identical, at 8, 10 and 16 bit, at
4K and at 1080p. Against the original FFT3dGPU 0.8.6 the fidelity is unchanged
too: 97.14 % of samples equal with the default settings, 96.30 % with `plane=4`,
94.83 % at 10 bit, 100 % with `bt=-1`.

## v1.0.0 — 2026-10-18

First release of FFT3dCL, a modern port of the FFT3dGPU 0.8.6 denoiser to a
native VapourSynth plugin. It runs on NVIDIA, AMD and Intel graphics cards
through OpenCL 1.2, keeps the original filter's parameters and denoising method,
and installs as a single DLL in StaxRip.

### Fixed

- **Memory was kept after closing a preview.** The filter now gives back the
  clip it borrows when it is destroyed, so opening and closing the preview no
  longer leaves hundreds of megabytes behind.

### Changed

- **Less memory while the filter runs.** Buffers that the GPU path never uses
  are no longer allocated, and the frame cache keeps only the frames it can
  actually reuse.
- **Pixel conversion moved to the GPU.** The filter sends plain 16-bit pixels to
  the graphics card instead of doing that arithmetic on the processor, which
  leaves more processor time for the encode that follows.
- **Less processor work per frame.** The final rounding of each pixel is now one
  vectorized calculation instead of a slow library call for every pixel.

### Added

- **Optional memory log.** Setting the environment variable `FFT3DCL_LOG` to a
  file path writes one line each time a filter instance is created or destroyed,
  with the graphics memory it holds.
- **Test scripts for StaxRip.** The `tools/` folder now contains the scripts
  that open and close the StaxRip preview and measure memory around it.
