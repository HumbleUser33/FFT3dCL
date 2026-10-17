# Renders ref (original FFT3dGPU) and port (VS plugin) with the same params, then compares.
# Usage: powershell -File tools\run_test.ps1 -Params "bt=-1, degrid=0" [-W 256 -H 192 -N 8]
param(
  [string]$Params = "",
  [string]$PortParams = "",
  [int]$W = 256, [int]$H = 192, [int]$N = 8,
  [string]$Clip = "tools\clip8.mkv",
  [string]$Fmt = "yuv420p"
)
$root = "C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS"
$ff = "$root\env\avs\ffmpeg.exe"
$vspipe = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\vspipe.exe"
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$clipFull = if ([System.IO.Path]::IsPathRooted($Clip)) { $Clip } else { "$root\$Clip" }

$avs = @"
LoadPlugin("$root\env\avs\plugins\LSMASHSource.dll")
LoadPlugin("$root\env\avs\plugins\FFT3dGPU.dll")
clip = LWLibavVideoSource(source="$clipFull")
clip = FFT3dGPU(clip, precision=2$(if ($Params) { ", $Params" }))
Return clip
"@
Set-Content -Path tools\_ref_tmp.avs -Value $avs -Encoding ASCII

$vpy = @"
import vapoursynth as vs
core = vs.core
clip = core.bs.VideoSource(source=r"$clipFull")
clip = core.fft3dgpu.FFT3DGPU(clip, precision=2$(if ($Params) { ", $Params" })$(if ($PortParams) { ", $PortParams" }))
clip.set_output()
"@
Set-Content -Path tools\_port_tmp.vpy -Value $vpy -Encoding ASCII

& $ff -y -v error -i tools\_ref_tmp.avs -c:v ffv1 -an tools\_ref_tmp.mkv 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { "REF render FAILED"; exit 1 }
$env:VAPOURSYNTH_EXTRA_PLUGIN_PATH = "$root\build\Release"
cmd /c "`"$vspipe`" -c y4m tools\_port_tmp.vpy - 2>tools\_port_err.txt | `"$ff`" -y -v error -f yuv4mpegpipe -i pipe:0 -c:v ffv1 -an tools\_port_tmp.mkv"
if ($LASTEXITCODE -ne 0) { "PORT render FAILED"; Get-Content tools\_port_err.txt | Select-Object -First 10; exit 1 }

& $py tools\compare.py tools\_ref_tmp.mkv tools\_port_tmp.mkv $W $H $N $Fmt
