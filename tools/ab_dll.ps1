# A/B pixel comparison between two fft3dcl.dll builds.
# Each build is loaded by explicit path in its own process, the same graph is
# rendered to FFV1 lossless, then every sample is compared. Use it to prove a
# performance change did not move a single pixel.
param(
  [string]$Old = "release\fft3dcl\fft3dcl.dll",
  [string]$New = "build\Release\fft3dcl.dll",
  [string]$Clip = "D:\Test Monkeys\Sample.mkv",
  [string]$Index = "D:\Test Monkeys\Sample.mkv_temp\temp.ffindex",
  [string]$Params = "sigma=3, bt=4, bw=32, bh=32, ow=16, oh=16, sharpen=0.4",
  [int]$Start = 0,
  [int]$End = 11,
  [int]$W = 3840,
  [int]$H = 2160,
  [string]$Fmt = "yuv420p10le"
)
$root = "C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS"
$vspipe = "D:\StaxRip-v2.52.6-x64\Apps\FrameServer\VapourSynth\Lib\site-packages\vapoursynth\vspipe.exe"
$ffms2 = "D:\StaxRip-v2.52.6-x64\Apps\Plugins\Dual\ffms2\ffms2.dll"
$ff = "$root\env\avs\ffmpeg.exe"
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$tmp = Join-Path $env:TEMP "fft3dcl_ab"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$n = $End - $Start + 1
if (-not [System.IO.Path]::IsPathRooted($Clip)) { $Clip = Join-Path $root $Clip }
if ($Index -and -not [System.IO.Path]::IsPathRooted($Index)) { $Index = Join-Path $root $Index }

foreach ($tag in @("old", "new")) {
  $dll = if ($tag -eq "old") { $Old } else { $New }
  if (-not [System.IO.Path]::IsPathRooted($dll)) { $dll = Join-Path $root $dll }
  $vpy = @"
import vapoursynth as vs
core = vs.core
core.std.LoadPlugin(r"$dll", altsearchpath=True)
core.std.LoadPlugin(r"$ffms2", altsearchpath=True)
clip = core.ffms2.Source(r"$Clip"$(if ($Index) { ", cachefile=r`"$Index`"" }))
clip = core.fft3dcl.FFT3DCL(clip, $Params)
clip.set_output()
"@
  $f = Join-Path $tmp "graph_$tag.vpy"
  Set-Content -Path $f -Value $vpy -Encoding ASCII
  $y4m = Join-Path $tmp "out_$tag.y4m"
  $err = Join-Path $tmp "err_$tag.txt"
  & $vspipe -c y4m -s $Start -e $End $f $y4m 2>$err
  if ($LASTEXITCODE -ne 0) {
    "RENDER FAILED ($tag)"; Get-Content $err | Select-Object -First 6; exit 1
  }
  $mkv = Join-Path $tmp "out_$tag.mkv"
  & $ff -y -v error -f yuv4mpegpipe -i $y4m -c:v ffv1 -an $mkv
  if ($LASTEXITCODE -ne 0) { "FFV1 FAILED ($tag)"; exit 1 }
  Remove-Item $y4m -Force
  "{0,-4} {1,-16} {2,8:N1} MB  {3}" -f $tag, (Split-Path $dll -Leaf), ((Get-Item $mkv).Length / 1MB), $dll
}
& $py "$root\tools\compare.py" (Join-Path $tmp "out_old.mkv") (Join-Path $tmp "out_new.mkv") `
    $W $H $n $Fmt | Select-String -Pattern 'identical samples|max diff|differing'
