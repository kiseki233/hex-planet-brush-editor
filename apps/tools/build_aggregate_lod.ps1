param(
    [int]$Frequency = 1004,
    [Parameter(Mandatory=$true)][string]$MapName,
    [double]$Yaw = -0.35,
    [double]$Pitch = 0.25,
    [double]$Zoom = 1.0,
    [double]$TargetPixels = 96.0,
    [switch]$All
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptRoot "build_aggregate_lod.py"
$Arguments = @(
    $PythonScript,
    "--frequency", $Frequency,
    "--map-name", $MapName,
    "--yaw", $Yaw,
    "--pitch", $Pitch,
    "--zoom", $Zoom,
    "--target-pixels", $TargetPixels
)
if ($All) { $Arguments += "--all" }

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Python 3 was not found." }
& $Python.Source @Arguments
exit $LASTEXITCODE
