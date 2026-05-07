param(
    [switch]$Cuda,
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$LocalPython = Join-Path $Root ".python312\python.exe"
$Cache = Join-Path $Root ".cache"
$PipCache = Join-Path $Cache "pip"
$TempDir = Join-Path $Cache "temp"

New-Item -ItemType Directory -Force -Path $PipCache, $TempDir | Out-Null
$env:PIP_CACHE_DIR = $PipCache
$env:TEMP = $TempDir
$env:TMP = $TempDir

function Get-BasePython {
    if (Test-Path $LocalPython) {
        return $LocalPython
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found. Install Python 3.10+ for Windows, then rerun this script."
}

if ($RecreateVenv -and (Test-Path $Venv)) {
    $resolvedRoot = (Resolve-Path $Root).Path
    $resolvedVenv = (Resolve-Path $Venv).Path
    if (-not $resolvedVenv.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove venv outside project root: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path $VenvPython)) {
    $basePython = Get-BasePython
    Write-Host "Creating .venv with $basePython"
    & $basePython -m venv $Venv
}

Write-Host "Installing requirements..."
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

if ($Cuda) {
    Write-Host "Installing CUDA-enabled PyTorch wheels..."
    & $VenvPython -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
}

Write-Host ""
Write-Host "VVAW Windows environment is ready."
Write-Host "Smoke test: .\run_smoke_windows.ps1"
Write-Host "v2 sim:     .\run_v2_windows.ps1 -Demo 5"
Write-Host "GPU check:  .\check_gpu_windows.ps1"
