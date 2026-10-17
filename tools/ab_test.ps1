# A/B check: old fft3dgpu.dll vs new fft3dcl.dll must be bit-identical.
param(
  [string]$Params = "",
  [int]$W = 256, [int]$H = 192, [int]$N = 8,
  [string]$Clip = "tools\clip8.mkv",
  [string]$Fmt = "yuv420p"
)
$root = "C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS"
$ff = "$root\env\avs\ffmpeg.exe"
$vspipe = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\vspipe.exe"
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$clipFull = if ([System.IO.Path]::IsPathRooted($Clip)) { $Clip } else { "$root\$Clip" }
$env:VAPOURSYNTH_EXTRA_PLUGIN_PATH = "$root\build\Release"

foreach ($call in @("core.fft3dgpu.FFT3DGPU", "core.fft3dcl.FFT3DCL")) {
  $vpy = @"
import vapoursynth as vs
core = vs.core
clip = core.bs.VideoSource(source=r"$clipFull")
clip = $call(clip, precision=2$(if ($Params) { ", $Params" }))
clip.set_output()
"@
  $tag = if ($call -match "fft3dcl") { "new" } else { "old" }
  Set-Content -Path "tools\_ab_$tag.vpy" -Value $vpy -Encoding ASCII
  cmd /c "`"$vspipe`" -c y4m tools\_ab_$tag.vpy - 2>tools\_ab_${tag}_err.txt | `"$ff`" -y -v error -f yuv4mpegpipe -i pipe:0 -c:v ffv1 -an tools\_ab_$tag.mkv"
  if ($LASTEXITCODE -ne 0) { "RENDER FAILED ($tag)"; Get-Content tools\_ab_${tag}_err.txt | Select-Object -First 6; exit 1 }
}
"Params: '$Params'  $W x $H x $N  $Fmt"
& $py tools\compare.py tools\_ab_old.mkv tools\_ab_new.mkv $W $H $N $Fmt | Select-String -Pattern 'identical samples|max diff'
