@echo off
setlocal
if /I "%~1"=="--drive" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Drive
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
)
exit /b %errorlevel%
