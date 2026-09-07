@echo off
setlocal enabledelayedexpansion

:: Navigate to the directory of the batch script (project root)
cd /d "%~dp0"

echo ===================================================
echo   Stellar-Forge Build Script
echo ===================================================
echo.

:: ── 1. Activate virtual environment ─────────────────────────────────────────
if not exist venv\Scripts\activate.bat (
    echo [Build] ERROR: venv not found. Run run.bat first to create it.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [Build] ERROR: Failed to activate venv.
    pause
    exit /b 1
)

:: ── 2. Install / upgrade PyInstaller ────────────────────────────────────────
echo [Build] Installing/upgrading PyInstaller...
pip install --upgrade pyinstaller --quiet
if !errorlevel! neq 0 (
    echo [Build] ERROR: Failed to install PyInstaller.
    pause
    exit /b 1
)

:: ── 3. Clean previous build artefacts ───────────────────────────────────────
echo [Build] Cleaning previous build...
if exist build         rmdir /s /q build
if exist dist\Stellar-Forge rmdir /s /q dist\Stellar-Forge

:: ── 4. Run PyInstaller ──────────────────────────────────────────────────────
echo [Build] Running PyInstaller (this may take a few minutes)...
pyinstaller stellar_forge.spec --noconfirm
if !errorlevel! neq 0 (
    echo.
    echo [Build] ERROR: PyInstaller failed. Check the output above for details.
    pause
    exit /b 1
)

:: ── 5. Copy external folders next to the .exe ───────────────────────────────
echo.
echo [Build] Copying external folders to dist\Stellar-Forge\...

:: textures/ — planet images, ring textures, bloom kernels (user-swappable)
echo [Build]   textures\
xcopy /E /I /Y /Q "textures" "dist\Stellar-Forge\textures"

:: data/ — system JSON, settings, ephemeris config
::   • data\kernels\ is massive (~4 GB); include it so the build is complete,
::     but users can remove kernels they don't need to save space.
echo [Build]   data\
xcopy /E /I /Y /Q "data" "dist\Stellar-Forge\data"

:: exports/ — screenshot output directory (create empty if it doesn't exist)
echo [Build]   exports\
if not exist "dist\Stellar-Forge\exports" mkdir "dist\Stellar-Forge\exports"
if exist exports (
    xcopy /E /I /Y /Q "exports" "dist\Stellar-Forge\exports"
)

:: imgui.ini — saved panel layout (optional; copy if present)
if exist imgui.ini (
    echo [Build]   imgui.ini
    copy /Y imgui.ini "dist\Stellar-Forge\imgui.ini" >nul
)

:: ── 6. Done ─────────────────────────────────────────────────────────────────
echo.
echo ===================================================
echo   Build complete!
echo   Output: dist\Stellar-Forge\Stellar-Forge.exe
echo ===================================================
echo.
echo To distribute, zip the entire dist\Stellar-Forge\ folder.
echo Users only need a GPU with OpenGL 3.3+ support.
echo.
pause
