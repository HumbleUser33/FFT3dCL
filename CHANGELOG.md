# Changelog

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
