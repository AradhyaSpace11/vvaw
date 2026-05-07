$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\setup_windows.ps1 first."
}

& $Python (Join-Path $Root "check_gpu.py")
