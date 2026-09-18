param(
    [int]$Frequency = 1004,
    [int]$TileSide = 16,
    [string]$MapName = "",
    [switch]$CreateMap,
    [switch]$VerifyMap,
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptRoot "build_production_plan.py"
$Arguments = @($PythonScript, "--frequency", $Frequency, "--tile-side", $TileSide)
if ($MapName) { $Arguments += @("--map-name", $MapName) }
if ($CreateMap) { $Arguments += "--create-map" }
if ($VerifyMap) { $Arguments += "--verify-map" }
if ($ForceRebuild) { $Arguments += "--force-rebuild" }

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Python 3 was not found." }
& $Python.Source @Arguments
exit $LASTEXITCODE
