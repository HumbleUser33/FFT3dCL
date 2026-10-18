# Summarize the log written by stax_mem_sampler.ps1: peak working set, private
# memory and GPU memory per phase.
param([string]$Csv = "$env:TEMP\mem_sample.csv")
function Num($s) { $t = ($s -replace '[^\d\-]', ''); if ($t) { [double]$t } else { 0 } }
$rows = Import-Csv $Csv -Delimiter ';'
"lines: $($rows.Count)"
""
"--- per phase and process: max working set / private (MiB) ---"
$rows | Where-Object { $_.name -ne 'VRAM_TOTAL' } |
  Group-Object phase | Sort-Object Name | ForEach-Object {
    $ph = $_.Name
    $byProc = $_.Group | Group-Object name
    $line = ($byProc | ForEach-Object {
      $ws = ($_.Group | ForEach-Object { Num $_.ws_mib } | Measure-Object -Maximum).Maximum
      $pr = ($_.Group | ForEach-Object { Num $_.priv_mib } | Measure-Object -Maximum).Maximum
      "{0} WS={1:N0} Priv={2:N0}" -f $_.Name, $ws, $pr
    }) -join " | "
    $v = ($rows | Where-Object { $_.name -eq 'VRAM_TOTAL' -and $_.phase -eq $ph } |
          ForEach-Object { Num $_.priv_mib } | Measure-Object -Maximum).Maximum
    "{0,-18} {1}  VRAM={2:N0}" -f $ph, $line, $v
  }
""
"--- total GPU memory (MiB) per phase: first / last / max ---"
$rows | Where-Object { $_.name -eq 'VRAM_TOTAL' } | Group-Object phase | Sort-Object Name | ForEach-Object {
  $vals = $_.Group | ForEach-Object { Num $_.priv_mib }
  "{0,-18} start={1,6:N0}  end={2,6:N0}  max={3,6:N0}" -f $_.Name, $vals[0], $vals[-1], ($vals | Measure-Object -Maximum).Maximum
}
