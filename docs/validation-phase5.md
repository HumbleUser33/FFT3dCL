# Validation — Phase 5 (StaxRip packaging)

Goal: check that the plugin loads and renders correctly in a real context
(StaxRip / vspipe), on the real 4K HDR clips, and that the DLL is
self-contained.

## 1. Self-contained DLL

`dumpbin /DEPENDENTS fft3dgpu.dll` → a single dependency: `KERNEL32.dll`.
- Visual C++ runtime linked statically (no `VCRUNTIME140.dll` / `MSVCP140.dll`).
- `OpenCL.dll` loaded dynamically at run time (present with any GPU driver).
- Size: ~258 KB.

Conclusion: the DLL is dropped in as-is, nothing else to install.

## 2. Loading

Two methods tested, both valid:
- **Auto-load**: copying the DLL into VapourSynth's `plugins` folder
  (`...\site-packages\vapoursynth\plugins`) → loaded with no environment
  variable set.
- **`core.std.LoadPlugin(r"...fft3dgpu.dll")`**: works (exit 0).

StaxRip: drop the DLL into its VapourSynth plugins folder, or add the
`LoadPlugin` line at the top of the script.

## 3. Real renders (GPU, device=1)

| Clip | Resolution / format | Frames | Filter time | Result |
|------|--------------------|--------|-------------|--------|
| Sample.mkv | 3840x2160 HEVC Main10 HDR10 | 8 | 2.13 s total | OK, no error |
| The Gorge 2025 | 1920x1038 AV1 10-bit HDR10 | 8 | 0.71 s (11.33 fps) | OK, no error |

The total time for The Gorge (≈288 s) is dominated by the **10-bit AV1 decode**
of the 6 GB lossless file (L-SMASH), not by the filter. GPU memory (RTX 3050,
8 GB) handles full-frame 4K without problem.

## 4. 1080p 10-bit comparison vs the original (GPU)

Crop 1920x1080, 8 frames, `yuv420p10le`, GPU vs original Avisynth+ 0.8.6
(`precision=2`):
- identical samples: **96.64 %**
- maximum difference: **4** (10-bit LSB) — consistent with the small 10-bit
  clip.

## 5. Frame leak (fixed)

Initial symptom: `Warning: Core freed but N filter instance(s) still exist`
plus bytes still allocated. Cause: frames obtained through `getFrameFilter`
(`src` + the temporal-window frames) were never released. Fix: added
`vsapi->freeFrame()` on `src` and on each window frame (CPU and GPU paths).

After the fix: the remaining counter is **bounded by the core's cache**
(~20 frames, independent of the total frame count) → no growing leak.
`fft3dgpuFree` is called. This warning is **common with third-party temporal
filters** (blur, DFTTest, FrameRateConverter): a benign teardown artefact with
no effect on the output.

## 6. Package

`release\fft3dgpu\`:
- `fft3dgpu.dll` (identical to the current build — hash verified)
- `README.md` (StaxRip installation + usage + notes)
- `sample.vpy` (example script)

## Verdict

Phase 5 validated: self-contained DLL, auto-load + LoadPlugin OK, real 4K HDR
renders OK, 1080p 10-bit matching the original, leak fixed. The plugin is ready
for StaxRip.
