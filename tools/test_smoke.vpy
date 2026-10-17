import vapoursynth as vs
core = vs.core

# Tiny synthetic clip, YUV420P8, 5 frames
clip = core.std.BlankClip(width=64, height=48, length=5, format=vs.YUV420P8, color=[128, 128, 128])
clip = core.fft3dgpu.FFT3DGPU(clip)
clip.set_output()
