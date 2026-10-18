# Sample the memory of the StaxRip preview chain once per second, plus the total
# GPU memory (the number Task Manager shows). Run it in the background while
# stax_preview_cycles.ps1 drives the preview.
param([int]$Seconds = 120, [string]$Out = "")
$ErrorActionPreference = "SilentlyContinue"
if (-not $Out) { $Out = Join-Path $env:TEMP "mem_sample.csv" }
"t;clock;phase;pid;name;ws_mib;priv_mib;cpu_s" | Set-Content -Path $Out -Encoding UTF8

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$lastGpu = [datetime]::MinValue
$gpu = 0
while ($stopwatch.Elapsed.TotalSeconds -lt $Seconds) {
  $phase = if (Test-Path "$env:TEMP\stax_phase.txt") { Get-Content "$env:TEMP\stax_phase.txt" -First 1 } else { "?" }
  if ((Get-Date) -gt $lastGpu) {
    $lastGpu = (Get-Date).AddSeconds(2)
    $g = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($g) { $gpu = [double]$g }
  }
  foreach ($p in Get-Process | Where-Object { $_.Name -match 'StaxRip|vspipe|mpv|python|ffplay' }) {
    try {
      $p.Refresh()
      "{0:N1};{1};{2};{3};{4};{5:N0};{6:N0};{7:N1}" -f `
        $stopwatch.Elapsed.TotalSeconds, (Get-Date -Format HH:mm:ss), $phase, $p.Id, $p.Name,
        ($p.WorkingSet64 / 1MB), ($p.PrivateMemorySize64 / 1MB), $p.TotalProcessorTime.TotalSeconds |
        Add-Content -Path $Out -Encoding UTF8
      "{0:N1};{1};{2};-1;VRAM_TOTAL;;{3:N0};;" -f `
        $stopwatch.Elapsed.TotalSeconds, (Get-Date -Format HH:mm:ss), $phase, $gpu |
        Add-Content -Path $Out -Encoding UTF8
    } catch {}
  }
  Start-Sleep -m 1000
}
"samples written to $Out"
