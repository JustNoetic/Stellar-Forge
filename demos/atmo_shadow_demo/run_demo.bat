@echo off
cd /d "%~dp0\..\.."
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe demos\atmo_shadow_demo\main.py
) else (
    python demos\atmo_shadow_demo\main.py
)
pause
