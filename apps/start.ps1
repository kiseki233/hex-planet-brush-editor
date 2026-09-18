Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$appsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $appsRoot
$artRoot = Join-Path $projectRoot 'art'
$brushRoot = Join-Path $artRoot 'brushes'
$mapRoot = Join-Path $artRoot 'maps'
$mainScript = Join-Path $appsRoot 'app\main.py'

# Launcher messages in Chinese, English and Japanese. The language matching the
# current Windows UI culture is shown first, the other two follow, so the dialog
# is readable no matter which locale the machine runs.
$messages = @{
    title = @{
        zh = '无法启动'
        en = 'Cannot start'
        ja = '起動できません'
    }
    pythonMissing = @{
        zh = '未检测到 Python 3。当前源码版不会自动安装环境。请安装 Python 3.10 或更高版本，并确保包含 tkinter。'
        en = 'Python 3 was not found. This source release does not install a runtime for you. Please install Python 3.10 or newer, including tkinter.'
        ja = 'Python 3 が見つかりません。このソース版は実行環境を自動インストールしません。tkinter を含む Python 3.10 以降をインストールしてください。'
    }
    pythonUnsuitable = @{
        zh = 'Python 环境不符合要求。需要 Python 3.10 或更高版本，并且必须包含 tkinter。'
        en = 'The Python environment does not meet the requirements. Python 3.10 or newer with tkinter is required.'
        ja = 'Python 環境が要件を満たしていません。tkinter を含む Python 3.10 以降が必要です。'
    }
}

function Get-PreferredLanguageOrder {
    $tag = ''
    try {
        $tag = [System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName
    } catch {
        $tag = ''
    }
    switch ($tag) {
        'zh' { return @('zh', 'en', 'ja') }
        'ja' { return @('ja', 'en', 'zh') }
        default { return @('en', 'zh', 'ja') }
    }
}

function Format-Message {
    param([hashtable]$Entry)
    $order = Get-PreferredLanguageOrder
    return (($order | ForEach-Object { $Entry[$_] }) -join "`r`n`r`n")
}

function Show-StartupError {
    param([hashtable]$Entry)
    $text = Format-Message -Entry $Entry
    $title = Format-Message -Entry $messages.title
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show($text, $title, 'OK', 'Error') | Out-Null
    } catch {
        # No WPF available (Server Core, constrained runspace): fall back to the console.
        Write-Host $text
    }
}

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
    Show-StartupError -Entry $messages.pythonMissing
    exit 1
}

& $python @pythonArgs -c "import sys, tkinter; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Show-StartupError -Entry $messages.pythonUnsuitable
    exit 1
}

& $python @pythonArgs $mainScript
exit $LASTEXITCODE
