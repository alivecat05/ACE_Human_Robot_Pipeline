@echo off
setlocal
set "PYTHONUTF8=1"
set "PIPER_CONDA_ROOT=%~dp0..\.conda"
if exist "%~dp0..\.setup\conda-root.txt" set /p PIPER_CONDA_ROOT=<"%~dp0..\.setup\conda-root.txt"
set "PIPER_ENV=%PIPER_CONDA_ROOT%\envs\piper-data"
if not exist "%PIPER_ENV%\python.exe" (
  echo Run setup.cmd in the Piper_Data root folder first, or use Python from your manually activated Conda environment.
  set "piper_exit=1"
  goto failed
)
set "PATH=%PIPER_ENV%;%PIPER_ENV%\Library\bin;%PIPER_ENV%\Scripts;%PATH%"
"%PIPER_ENV%\python.exe" -m piper_capture.app %*
set "piper_exit=%errorlevel%"
if not "%piper_exit%"=="0" goto failed
exit /b %piper_exit%

:failed
echo.
echo PiperCapture failed to start or exited with an error. See the message above.
echo To install hardware SDKs, run setup.cmd --drive in the project root.
echo To try the interface without hardware, run recording\run.cmd --mock.
if not defined PIPER_NO_PAUSE pause
exit /b %piper_exit%
