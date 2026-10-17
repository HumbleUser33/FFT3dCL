# -*- coding: utf-8 -*-
# Example of FFT3DCL usage in a VapourSynth script (StaxRip or vspipe).
#
# If the DLL is not in VapourSynth's plugins folder, uncomment the LoadPlugin
# line below and adjust the path.

import vapoursynth as vs

core = vs.core

# core.std.LoadPlugin(r"C:\path\to\fft3dcl.dll")

# Source (BestSource example; in StaxRip the source is already defined).
clip = core.bs.VideoSource(source=r"C:\path\to\Sample.mkv")

# Optionally convert to 16-bit / an internal format if needed (not required:
# the filter accepts 8-16 bit integer and 32-bit float directly).

# FFT3DCL denoising (settings identical to the original 0.8.6).
clip = core.fft3dcl.FFT3DCL(
    clip,
    sigma=2.0,        # denoising strength
    bt=3,             # temporal window (-1..4, 0 = Kalman)
    beta=1.0,
    plane=0,          # 0 = luma only
    mode=1,           # block overlap
    bw=32, bh=32,     # block size (power of two)
    degrid=1.0,       # grid correction
    device=2,         # 2 = GPU with CPU fallback (default), 1 = GPU, 0 = CPU
)

clip.set_output()
