# Changelog

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
