LoadPlugin("C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\plugins\LSMASHSource.dll")
LoadPlugin("C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\plugins\FFT3dGPU.dll")
clip = LWLibavVideoSource(source="C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\tools\clip8.mkv")
clip = FFT3dGPU(clip, precision=2)
Return clip
