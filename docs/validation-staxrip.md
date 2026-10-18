# Validation — StaxRip integration (crash fix)

## Report

Installing `fft3dcl.dll` as a StaxRip plugin and starting the preview on
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

`src/fft3dcl.cpp`: added `clampFrame()`, and the frame requests in the
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
  the plugin namespace is `fft3dcl` and the call is
  `core.fft3dcl.FFT3DCL(clip, ...)`.
- The plugin registers itself as API R4.3 and loads correctly on an R4.2 core;
  it only calls functions that exist since API 4.0.

## Memory retention across previews (fixed)

### Report

Opening and closing a StaxRip preview repeatedly made the memory of the
`vspipe` process grow without ever coming back: about **830 MB per cycle** on
3840x2160 YUV420P10, while GPU memory stayed small. The same growth appeared
with BestSource (VapourSynth R81) and with L-SMASH (R79), in both the GPU and
the CPU path.

### How it was isolated

1. The filter's own buffers are released correctly: instrumenting the instance
   showed 410 MB of CPU scratch alive at teardown and 0 KB right after the
   instance was deleted, yet the process kept the 830 MB.
2. The growth did not depend on the number of frames, on `bt`, on `degrid`, on
   `sharpen`, on the frame mode, on the request pattern, or on the CRT linkage.
3. A 40-line pass-through plugin written from the official VapourSynth skeleton
   reproduced the same 830 MB per cycle; a well-behaved third-party temporal
   filter (`focus2.TemporalSoften2`) in the same process did not.
4. Diffing the two: the skeleton releases the input node in its free callback
   (`vsapi->freeNode(d->node)`), our filter did not.

### Root cause

In VapourSynth API 4 the filter **owns the reference to its dependency node**
and must return it in the free callback. We stored the node pointer but never
called `freeNode`, so the source node was never destroyed: its decoder and its
decoded-frame cache stayed in memory for the life of the process. Every new
script (every preview) added another one. This is also what produced the
`Core freed but N filter instance(s) still exist` message at shutdown.

### Fix

- `fft3dclFree` now calls `vsapi->freeNode(inst->clip)` before deleting the
  instance.
- `fft3dclCreate` wraps the reference in a small guard so that every error path
  (bad format, bad plane, invalid parameters) releases it too; the guard is
  disarmed once `createVideoFilter` has taken ownership.

### Verification

Private committed memory retained per create/render/destroy cycle, 4K clip:

| Environment | Before | After |
|---|---|---|
| VapourSynth R81 + BestSource, `device=2` | +833, +832 MiB | +0.6, +0.6 MiB |
| VapourSynth R81 + BestSource, `device=0` | +834, +832 MiB | -0.1, +0.6 MiB |
| StaxRip R79 + L-SMASH, `device=0` | +804, +801 MiB | +2.0, +2.4 MiB |

Pixels are unchanged: the previous build and the fixed one render
**100 % identical** output (max diff 0) for the default parameters, `bt=4
sharpen=0.4`, `bt=0`, `plane=4` and `mode=0`. The StaxRip R79 matrix (`bt=-1`,
`bt=0`, `bt=4` at the start, middle and end of the clip, a single frame,
`device=1`, `device=0`, `plane=4`) still exits 0, and the shutdown warning is
gone.

## Per-instance footprint (reduced)

### Report

With the leak gone, one 4K instance still touched a lot of memory while it was
alive, and GPU memory crept slowly from one preview to the next: about 1 GB of
device memory and 1.8 GB of working set on 3840x2160.

### Changes

1. **CPU scratch is allocated on demand.** `filtRe_`, `filtIm_`, `corrRe_` and
   `corrIm_` belong to the CPU path (`filterAndSynthesize`) but were sized in the
   constructor, so the OpenCL path paid 4 x 71 MiB at 4K for nothing. They are
   now sized on first use.
2. **Spectra buffers are recycled.** `pruneCache` used to call
   `clReleaseMemObject` on a spectra buffer and `spectra` a fresh
   `clCreateBuffer` for the next frame; the driver answers that pattern by
   keeping the freed memory. Retired buffers now go to a small reuse pool (256
   MiB worth) that `spectra` draws from, and the pool is released with the
   instance.
3. **The spectra cache window is `bt+2` frames instead of `3*bt+1`.** With
   `bt=3` that is 5 cached frames instead of 10, i.e. 710 MiB instead of 1420 MiB
   of device memory at 4K. The old window only helped requests far outside the
   current temporal window; sequential and prefetched access hit the new one
   just as well.

### Verification

4K clip, `bt=3`, four create/render/destroy cycles (VapourSynth R81 +
BestSource, `device=2`):

| Measure | Before | After |
|---|---|---|
| peak working set while rendering | 2767-2812 MiB | 1865 MiB |
| peak private committed while rendering | 3510-3619 MiB | 3001-3100 MiB |
| working set retained per cycle | +0 MiB | +0 MiB |
| private committed retained per cycle | +0.5 MiB | +0 to +14 MiB (noise) |
| peak private committed, `device=0` | ~3000 MiB | 2441 MiB |

Speed is unchanged: 16 frames at 4K render in 5.50 s before and 5.49 s after
with one request at a time, 5.36 s and 5.44 s with four parallel requests.

Pixels are unchanged: the build before these changes and the build after render
**100 % identical** output (max diff 0) for the default parameters, `bt=4`,
`bt=1`, `device=0`, 8-bit and 10-bit clips, and 1920x1080 10-bit with `bt=4`.
The StaxRip R79 matrix still exits 0 on all nine cases.

### Remaining footprint (not a leak)

A single 4K instance still peaks near 2.1 GB of private memory: the OpenCL
buffers, the host staging buffers and VapourSynth's frame cache. That memory is
released properly; shrinking it further means moving the pixel conversion to
the GPU (see the CPU audit below).

## Driving the real StaxRip preview (test method)

The measurements above use `vspipe`. To reproduce what the user sees, the real
StaxRip window is driven by UI Automation:

```powershell
# open the preview (the Preview menu item is invoked directly, no click needed)
& uia_do.ps1 -ProcessId <pid> -Name '^Preview$' -Action invoke
# close it again
SendMessageW(previewWindowHandle, WM_CLOSE, 0, 0)
```

Notes for whoever repeats this:

- UI Automation only sees windows on the **current virtual desktop**. If StaxRip
  is on another desktop, `FindAll` returns nothing and the tree looks empty.
- StaxRip hosts VapourSynth **in its own process** (no `vspipe` child), so the
  filter's memory shows up in the StaxRip process.
- One preview creates and destroys the filter **three times** (source script,
  view script, thumbnails), and the new instance is created *before* the old one
  is destroyed.

## GPU memory still climbs in StaxRip (not our buffers)

Same project, three preview open/close cycles, sampled once per second
(4K HDR10, `sigma=3, bt=4, bw=32, bh=32, ow=16, oh=16, sharpen=0.4`):

| Cycle | GPU memory peak while open | GPU memory after close |
|---|---|---|
| 1 | 3373 MiB | 2303 MiB |
| 2 | 3613 MiB | 2509 MiB |
| 3 | 3818 MiB | 2715 MiB |

About 200 MiB of GPU memory and 500 MiB of private memory stay behind per cycle,
which matches what is seen in Task Manager. The filter now logs its own state,
and that log says the memory is not ours:

```
08:00:41 FFT3DCL: created   - 1 GPU instance(s) alive, 329 MiB of device buffers
08:00:41 FFT3DCL: released  - 0 GPU instance(s) alive, 0 MiB of device buffers
```

Every instance is destroyed and every `cl_mem` is released before the preview
window closes. The same script, the same clip and the same parameters run in a
plain `python.exe` (StaxRip's own VapourSynth, ffms2, no cache clearing) keep
GPU memory flat: 3212, 3213, 3214 MiB after each cycle. So the growth comes from
the driver holding on to memory it was asked to free (WDDM defers it) and from
StaxRip's own preview cache, not from the plugin.

A process-wide buffer pool was tried to stop the driver round-trips. It made
things worse (peak 5.8 GB) because StaxRip creates the new graph before
destroying the old one, so the pool is always empty when it would be useful, and
it keeps 470 MiB alive in the meantime. It was reverted; the per-instance pool
stays.

## CPU used by the filter

24 frames of the real 4K HDR10 clip through StaxRip's VapourSynth, R79:

| Graph | Wall | CPU time | Cores busy | Speed |
|---|---|---|---|---|
| source only | 1.20 s | 7.86 s | 6.5 | 20.0 fps |
| source + FFT3DCL (GPU) | 4.18 s | 11.45 s | 2.7 | 5.7 fps |
| source + FFT3DCL (CPU path) | 52.30 s | 58.94 s | 1.1 | 0.5 fps |

The GPU path adds about 3.6 s of CPU for 24 frames, roughly 150 ms per frame,
and it also stops the decoder from using all cores (6.5 down to 2.7) because one
mutex covers the upload, the kernels and the download.

Two fixes landed:

1. `download()` used `std::lround` per pixel, a library call that cannot be
   vectorized. Replacing it with `(uint16_t)(v * 65535.0f + 0.5f)` on a value
   already clamped to `[0,1]` gives the same numbers and lets MSVC use SSE.
2. The 33 MiB staging vectors were allocated and zero-filled for every plane of
   every frame; they are now reused.

That took 24 frames from 4.51 s to 4.18 s wall and 11.89 s to 11.45 s of CPU.
The next step is the same conversion moved to the GPU: upload the plane as raw
16-bit and let a kernel expand it to float, and write the result back as 16-bit
on the device, which would remove most of the 150 ms per frame and let the
upload and download happen outside the mutex.
