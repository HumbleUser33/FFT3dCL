# Generates a 10-bit test clip: 256x192 crop, 8 frames, YUV420P10 (FFV1).
import vapoursynth as vs
core = vs.core

clip = core.bs.VideoSource(source=r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\Sample.mkv")
clip = core.std.Crop(clip, left=1600, top=800, right=1984, bottom=1168)
clip = core.std.Trim(clip, 0, 7)
clip = core.resize.Point(clip, format=vs.YUV420P10)
clip.set_output()
