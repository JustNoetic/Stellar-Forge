@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0..\Stellar Forge\venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0..\Stellar Forge\venv\Scripts\python.exe"
) else (
    where python.exe >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=python.exe"
    )
)

if "%PY_EXE%"=="" (
    echo [ERROR] Python 3.10+ not found in PATH or local venv.
    pause
    exit /b 1
)

"%PY_EXE%" scripts\package_release.py %*
set "PKG_RESULT=%errorlevel%"

if "%~1"=="" pause
exit /b %PKG_RESULT%
