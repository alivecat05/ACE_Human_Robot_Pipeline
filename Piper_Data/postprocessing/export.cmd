@echo off
setlocal
set "PYTHONUTF8=1"
set "PIPER_CONDA_ROOT=%~dp0..\.conda"
if exist "%~dp0..\.setup\conda-root.txt" set /p PIPER_CONDA_ROOT=<"%~dp0..\.setup\conda-root.txt"
set "PIPER_ENV=%PIPER_CONDA_ROOT%\envs\piper-data"
if not exist "%PIPER_ENV%\python.exe" (
  echo Run setup.cmd in the Piper_Data root folder first, or use Python from your manually activated Conda environment.
  exit /b 1
)
set "PATH=%PIPER_ENV%;%PIPER_ENV%\Library\bin;%PIPER_ENV%\Scripts;%PATH%"
"%PIPER_ENV%\python.exe" -m piper_lerobot.export %*
set "piper_exit=%errorlevel%"
exit /b %piper_exit%
