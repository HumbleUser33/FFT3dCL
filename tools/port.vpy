import vapoursynth as vs
core = vs.core

clip = core.bs.VideoSource(source=r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\tools\clip8.mkv")
clip = core.fft3dgpu.FFT3DGPU(clip, precision=2)
clip.set_output()
