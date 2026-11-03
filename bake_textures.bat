@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE="

rem 1. Check local repository venv
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0..\Stellar Forge\venv\Scripts\python.exe" (
    rem 2. Check sibling Stellar Forge venv (contains cupy-cuda12x)
    set "PY_EXE=%~dp0..\Stellar Forge\venv\Scripts\python.exe"
) else (
    rem 3. Fall back to system python
    where python.exe >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=python.exe"
    )
)

if "%PY_EXE%"=="" (
    echo [ERROR] No Python environment found.
    echo Please install Python 3.10+ with Pillow and NumPy, or create a venv.
    pause
    exit /b 1
)

rem Run texture baker with auto-selected CUDA support
"%PY_EXE%" scripts\bake_surface_maps.py %*
set "BAKE_RESULT=%errorlevel%"

if "%BAKE_RESULT%"=="0" (
    if exist "textures\Solar System\Earth\Earth_clouds.jpg" (
        if not exist "textures\Solar System\Earth\clouds\coverage.r8" (
            echo.
            echo Baking Earth cloud coverage mip pyramid...
            "%PY_EXE%" scripts\bake_cloud_maps.py "textures\Solar System\Earth\Earth_clouds.jpg" "textures\Solar System\Earth\clouds"
        )
    )
)

if "%~1"=="" pause
exit /b %BAKE_RESULT%
