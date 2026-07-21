@echo off
setlocal enabledelayedexpansion

:: Navigate to the directory of the batch script
cd /d "%~dp0"

echo ===================================================
echo   Stellar-Forge Environment Setup ^& Launcher
echo ===================================================

:: Verify if python inside virtual environment works
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [Stellar-Forge] Virtual environment is broken. Recreating...
        rmdir /s /q venv
        python -m venv venv
    )
) else (
    echo [Stellar-Forge] Virtual environment not found. Creating...
    if exist venv rmdir /s /q venv
    python -m venv venv
)

if !errorlevel! neq 0 (
    echo [Stellar-Forge] Failed to create virtual environment. Ensure Python is installed and in your PATH.
    pause
    exit /b 1
)

:: Activate the virtual environment
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [Stellar-Forge] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Check and install dependencies
echo [Stellar-Forge] Checking dependencies...
python -c "import sys, subprocess; pkgs={'numpy':'numpy','scipy':'scipy','moderngl':'moderngl','glfw':'glfw','pyrr':'pyrr','imgui':'imgui','numba':'numba','spiceypy':'spiceypy','requests':'requests','OpenGL':'PyOpenGL','PIL':'Pillow'}; missing=[p for m,p in pkgs.items() if subprocess.run([sys.executable,'-c',f'import {m}'], capture_output=True).returncode]; sys.exit(len(missing))" >nul 2>&1
if !errorlevel! neq 0 (
    echo [Stellar-Forge] Missing dependencies detected. Installing...
    python -c "import sys, subprocess; pkgs={'numpy':'numpy','scipy':'scipy','moderngl':'moderngl','glfw':'glfw','pyrr':'pyrr','imgui':'imgui','numba':'numba','spiceypy':'spiceypy','requests':'requests','OpenGL':'PyOpenGL','PIL':'Pillow'}; missing=[p for m,p in pkgs.items() if subprocess.run([sys.executable,'-c',f'import {m}'], capture_output=True).returncode]; subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)"
    if !errorlevel! neq 0 (
        echo [Stellar-Forge] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [Stellar-Forge] All dependencies are satisfied.
)

:: Run the simulation
set PYTHONPATH=%~dp0engine;%PYTHONPATH%
echo [Stellar-Forge] Starting Stellar-Forge simulation...
python engine\main.py
if !errorlevel! neq 0 (
    echo [Stellar-Forge] Simulation exited with error code !errorlevel!.
)

pause
