Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$appsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $appsRoot 'tools\run_acceptance.py'
$candidates = @('pwsh.exe', 'python.exe', 'py.exe')
$python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($null -eq $python) {
    $python = Get-Command py.exe -ErrorAction SilentlyContinue
}
if ($null -eq $python) {
    throw 'Python 3 was not found.'
}
if ($python.Name -eq 'py.exe') {
    & $python.Source -3 $script @args
} else {
    & $python.Source $script @args
}
exit $LASTEXITCODE
