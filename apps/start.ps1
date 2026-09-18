Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$appsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $appsRoot
$artRoot = Join-Path $projectRoot 'art'
$brushRoot = Join-Path $artRoot 'brushes'
$mapRoot = Join-Path $artRoot 'maps'
$mainScript = Join-Path $appsRoot 'app\main.py'

New-Item -ItemType Directory -Force -Path $brushRoot | Out-Null
New-Item -ItemType Directory -Force -Path $mapRoot | Out-Null

$candidates = @(
    (Join-Path $appsRoot 'runtime\python\python.exe'),
    $env:KISEKI_PYTHON,
    'py.exe',
    'python.exe',
    'python3.exe'
)

$python = $null
$pythonArgs = @()
foreach ($candidate in $candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        continue
    }
    if ($candidate -eq 'py.exe') {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $python = $command.Source
            $pythonArgs = @('-3')
            break
        }
        continue
    }

    if ([System.IO.Path]::IsPathRooted($candidate)) {
        if (Test-Path -LiteralPath $candidate) {
            $python = $candidate
            break
        }
    } else {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $python = $command.Source
            break
        }
    }
}

if ($null -eq $python) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        '未检测到 Python 3。当前源码版不会自动安装环境。请安装 Python 3.10 或更高版本，并确保包含 tkinter。',
        '无法启动',
        'OK',
        'Error'
    ) | Out-Null
    exit 1
}

& $python @pythonArgs -c "import sys, tkinter; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        'Python 环境不符合要求。需要 Python 3.10 或更高版本，并且必须包含 tkinter。',
        '无法启动',
        'OK',
        'Error'
    ) | Out-Null
    exit 1
}

& $python @pythonArgs $mainScript
exit $LASTEXITCODE
