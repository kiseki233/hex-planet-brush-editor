Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$appsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$scriptPath = Join-Path $appsRoot 'tools\build_chunk_map.py'

$candidates = @('py.exe', 'python.exe', 'python3.exe')
$python = $null
$pythonArgs = @()
foreach ($candidate in $candidates) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        continue
    }
    $python = $command.Source
    if ($candidate -eq 'py.exe') {
        $pythonArgs = @('-3')
    }
    break
}
if ($null -eq $python) {
    throw 'Python 3 was not found.'
}

& $python @pythonArgs $scriptPath @args
exit $LASTEXITCODE
