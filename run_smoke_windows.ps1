param(
    [switch]$Gui,
    [int]$Steps = 120
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\setup_windows.ps1 first."
}

$script = Join-Path $Root "subsystems\testsimulation.py"

if ($Gui) {
    & $Python $script --steps $Steps
} else {
    & $Python $script --direct --steps $Steps --no-camera-window
}
