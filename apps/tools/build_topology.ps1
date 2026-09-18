Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$appsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $appsRoot
try {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        & py.exe -3 tools/build_topology.py @args
    } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        & python.exe tools/build_topology.py @args
    } else {
        throw 'Python 3 was not found.'
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
