# Find an element in a StaxRip window by name and invoke or expand it.
# UI Automation only sees windows on the current virtual desktop: if StaxRip
# lives on another desktop the tree comes back empty.
param(
  [Parameter(Mandatory)][int]$ProcessId,
  [Parameter(Mandatory)][string]$Name,
  [ValidateSet("list", "invoke", "expand")]
  [string]$Action = "list",
  [int]$MaxDepth = 8
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$AE = [System.Windows.Automation.AutomationElement]

$p = Get-Process -Id $ProcessId
$win = $AE::FromHandle([IntPtr]$p.MainWindowHandle)
if (-not $win) { "no UI element for that window"; exit 1 }

$walker = [System.Windows.Automation.TreeWalker]::RawViewWalker
$found = @()
function Walk($el, $depth) {
  if ($depth -gt $script:MaxDepth) { return }
  $c = $walker.GetFirstChild($el)
  while ($c) {
    if ($c.Current.Name -match $script:Name) { $script:found += $c }
    Walk $c ($depth + 1)
    $c = $walker.GetNextSibling($c)
  }
}
Walk $win 1

if ($found.Count -eq 0) { "nothing matches '$Name'"; exit 1 }
foreach ($e in $found) {
  "  {0} '{1}'" -f $e.Current.ControlType.ProgrammaticName.Replace("ControlType.", ""), $e.Current.Name
}

if ($Action -eq "list") { exit 0 }
$target = $found[0]
if ($Action -eq "expand") {
  $pat = $null
  if ($target.TryGetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern, [ref]$pat)) {
    $pat.Expand(); "expanded: '$($target.Current.Name)'"
  } else { "no ExpandCollapse pattern" }
}
if ($Action -eq "invoke") {
  $pat = $null
  if ($target.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pat)) {
    $pat.Invoke(); "invoked: '$($target.Current.Name)'"
  } else { "no Invoke pattern" }
}
