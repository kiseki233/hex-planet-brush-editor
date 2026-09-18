param(
    [switch]$SkipWglProbe
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptRoot "run_windows_diagnostic.py"
$Arguments = @($PythonScript)
if ($SkipWglProbe) { $Arguments += "--skip-wgl-probe" }

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $Python) { throw "Python 3 was not found." }
& $Python.Source @Arguments
exit $LASTEXITCODE
