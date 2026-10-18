# Open and close the StaxRip preview several times, marking each phase in a
# file that stax_mem_sampler.ps1 reads. Requires the preview to be reachable
# through the "Preview" menu item (StaxRip 2.52 opens it directly).
param(
  [Parameter(Mandatory)][int]$StaxPid,
  [int]$Cycles = 3,
  [int]$PlaySec = 25,
  [int]$CloseSec = 15
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
if (-not ("PostMsg" -as [type])) {
  Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PostMsg {
  [DllImport("user32.dll", CharSet = CharSet.Unicode)]
  public static extern IntPtr SendMessageW(IntPtr h, uint msg, IntPtr w, IntPtr l);
  public const uint WM_CLOSE = 0x0010;
}
'@
}
$AE = [System.Windows.Automation.AutomationElement]

function Set-Phase([string]$name) { Set-Content -Path "$env:TEMP\stax_phase.txt" -Value $name -Encoding ASCII }

function Find-PreviewWindow([int]$procId) {
  $cond = New-Object System.Windows.Automation.PropertyCondition(
    $AE::ProcessIdProperty, $procId)
  foreach ($w in $AE::RootElement.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)) {
    if ($w.Current.Name -like 'Preview*StaxRip*') { return $w }
  }
  return $null
}

function Open-Preview([int]$procId) {
  $p = Get-Process -Id $procId
  $win = $AE::FromHandle([IntPtr]$p.MainWindowHandle)
  $walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
  $stack = New-Object System.Collections.Stack
  $stack.Push($win)
  while ($stack.Count -gt 0) {
    $e = $stack.Pop()
    if ($e.Current.ControlType.Equals([System.Windows.Automation.ControlType]::MenuItem) -and
        $e.Current.Name -eq 'Preview') {
      $pat = $null
      if ($e.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pat)) {
        $pat.Invoke(); return $true
      }
    }
    $c = $walker.GetFirstChild($e)
    while ($c) { $stack.Push($c); $c = $walker.GetNextSibling($c) }
  }
  return $false
}

for ($i = 1; $i -le $Cycles; $i++) {
  Set-Phase "cycle${i}_open"
  if (-not (Open-Preview $StaxPid)) { "cycle $i : cannot open the preview"; exit 1 }
  Start-Sleep -s $PlaySec

  Set-Phase "cycle${i}_close"
  $pw = Find-PreviewWindow $StaxPid
  if ($pw) {
    [void][PostMsg]::SendMessageW([IntPtr]$pw.Current.NativeWindowHandle, [PostMsg]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
  } else { "cycle $i : preview window not found" }
  Start-Sleep -s $CloseSec
  Set-Phase "cycle${i}_idle"
}
Set-Phase "done"
"cycles finished"
