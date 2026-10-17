# Validation — StaxRip integration (crash fix)

## Report

Installing `fft3dgpu.dll` as a StaxRip plugin and starting the preview on
`Sample.mkv` closed StaxRip immediately, with no error message. Reproduced on
two machines (RTX 3050 and RTX 4080), so it was not GPU-related.

## Root cause

StaxRip ships **VapourSynth R79 (API R4.2)**; the plugin had been developed and
tested against **R81**.

The filter requests a temporal window of frames around the frame being
produced. With the default `bt=3` it asks for frames `n-1, n, n+1`, so for the
first frame of a clip it requested **frame -1**. Near the end of a clip it also
requested frames past the last one.

VapourSynth differs on this:

| Core | Behaviour on an out-of-range request |
|---|---|
| R81 (used for all earlier testing) | silently clamps the index |
| R79 (shipped by StaxRip) | **fatal error**, process aborts |

Reproduced directly with StaxRip's own `vspipe`:

```
Fatal: Negative frame request by: FFT3DGPU
exit status 0xC0000409
```

The abort is a fast-fail, which is why StaxRip vanished without showing a
message.

## Fix

`src/fft3dgpu.cpp`: added `clampFrame()`, and the frame requests in the
`arInitial` branch of `getFrame` now clamp to `[0, numFrames-1]` before calling
`requestFrameFilter`. The window is requested as one contiguous clamped range,
which is the same set of frames as clamping each index individually but without
duplicate requests. The existing fetch-side clamps (`getSpectra`, GPU `ensure`)
now use the same helper.

The plugin no longer relies on the core being lenient, so it behaves the same on
every VapourSynth version.

## Verification

Using StaxRip's own VapourSynth (Core R79, API R4.2), 3840x2160 YUV420P10:

| Parameters | GPU (`device=1`) | CPU (`device=0`) |
|---|---|---|
| `bt=3` (default) | exit 0 | — |
| `bt=4` | exit 0 | exit 0 |
| `bt=2` | exit 0 | — |
| `bt=1` | exit 0 | — |
| `bt=0` (Kalman) | exit 0 | — |
| `bt=-1` | exit 0 | — |

Edge cases, both R79 and R81, all exit 0: clips of 1, 2 and 3 frames with
`bt=3` and `bt=4`.

Real source, StaxRip's VapourSynth + L-SMASH (`core.lsmas.LWLibavSource`) on
`Sample.mkv` (3840x2160 HEVC Main10): 6 frames in 1.36 s, exit 0.

No regression against the original 0.8.6 (GPU path, unchanged from the phase 2/3
figures):

| Parameters | Identical samples |
|---|---|
| default | 97.14 % |
| `bt=4` | 73.68 % |
| `bt=-1` | 100.00 % |
| `bt=0` | 94.01 % |
| `bt=2` | 96.03 % |
| `sharpen=0.5` | 94.38 % |
| `wintype=1` | 97.00 % |
| `plane=1` | 99.17 % |
| `sigma=4` | 97.55 % |
| 10-bit | 94.83 % |

## StaxRip installation notes

- **Install the plugin only once.** Two copies (for example in
  `Apps\Plugins\VS\...` and `Settings\Plugins\VapourSynth\`) make VapourSynth
  fail with `Plugin ... already loaded (com.github.pinterf.fft3dgpu)`.
- StaxRip loads its plugins with explicit `core.std.LoadPlugin(...)` lines, so
  the plugin namespace is `fft3dgpu` and the call is
  `core.fft3dgpu.FFT3DGPU(clip, ...)`.
- The plugin registers itself as API R4.3 and loads correctly on an R4.2 core;
  it only calls functions that exist since API 4.0.
