param(
    [int]$Demo = 5,
    [double]$PhaseSpeed = 0.75,
    [int]$PhysicsSteps = 10,
    [switch]$Headless,
    [int]$MaxSteps = 0,
    [switch]$RequireCuda,
    [int]$GpuSentinelMb = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\setup_windows.ps1 first."
}

$script = Join-Path $Root "v2\infer.py"
$argsList = @(
    $script,
    "--demo", $Demo,
    "--phase-speed", $PhaseSpeed,
    "--physics-steps", $PhysicsSteps
)

if ($Headless) {
    $argsList += @("--direct", "--no-camera-window")
}

if ($MaxSteps -gt 0) {
    $argsList += @("--max-steps", $MaxSteps)
}

if ($RequireCuda) {
    $argsList += "--require-cuda"
}

if ($GpuSentinelMb -gt 0) {
    $argsList += @("--gpu-sentinel-mb", $GpuSentinelMb)
}

& $Python @argsList
