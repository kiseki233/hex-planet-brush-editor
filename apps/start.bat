@echo off
setlocal
set "APPS_ROOT=%~dp0"
where pwsh.exe >nul 2>nul
if errorlevel 1 goto fallback
pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%APPS_ROOT%start.ps1"
exit /b %errorlevel%

:fallback
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%APPS_ROOT%start.ps1"
exit /b %errorlevel%
