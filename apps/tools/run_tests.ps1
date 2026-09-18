Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$appsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $appsRoot
try {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py.exe -3 -m unittest discover -s tests -v
    } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        & python.exe -m unittest discover -s tests -v
    } else {
        throw 'Python 3 was not found.'
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
