@echo off
setlocal enabledelayedexpansion

:: Navigate to the directory of the batch script (project root)
cd /d "%~dp0"

:: ── Release version — update this before each release ──────────────────────
set VERSION=1.0.0
set ZIP_NAME=Stellar-Forge-v%VERSION%-Windows.zip

echo ===================================================
echo   Stellar-Forge Build Script  (v%VERSION%)
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
if exist build                   rmdir /s /q build
if exist dist\Stellar-Forge      rmdir /s /q dist\Stellar-Forge
if exist dist\Stellar-Forge-release rmdir /s /q dist\Stellar-Forge-release

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

:: data/ — system JSON, settings, ephemeris config + kernels (all of it; user
::          can delete kernels they don't need to save space)
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

:: ── 6. Stage release folder (copy without the heavy SPICE kernels) ──────────
echo.
echo [Build] Staging release zip (excluding SPICE kernels ~4 GB)...
set STAGE_DIR=dist\Stellar-Forge-release

:: Copy exe and _internal packages
xcopy /E /I /Y /Q "dist\Stellar-Forge\_internal" "%STAGE_DIR%\_internal\"
copy  /Y          "dist\Stellar-Forge\Stellar-Forge.exe" "%STAGE_DIR%\Stellar-Forge.exe" >nul

:: Copy user-editable external folders
xcopy /E /I /Y /Q "dist\Stellar-Forge\textures" "%STAGE_DIR%\textures\"
xcopy /E /I /Y /Q "dist\Stellar-Forge\exports"  "%STAGE_DIR%\exports\"

:: Copy data\ but skip kernels\ (robocopy /XD = eXclude Directory)
robocopy "dist\Stellar-Forge\data" "%STAGE_DIR%\data" /E /XD "kernels" /NJH /NJS /NFL /NDL >nul

if exist "dist\Stellar-Forge\imgui.ini" (
    copy /Y "dist\Stellar-Forge\imgui.ini" "%STAGE_DIR%\imgui.ini" >nul
)

:: ── 7. Zip the staged folder ─────────────────────────────────────────────────
echo [Build] Creating %ZIP_NAME%...
if exist "%ZIP_NAME%" del "%ZIP_NAME%"

powershell -NoProfile -Command ^
  "Compress-Archive -Path 'dist\Stellar-Forge-release\*' -DestinationPath '%ZIP_NAME%' -CompressionLevel Optimal"

if !errorlevel! neq 0 (
    echo [Build] WARNING: Zip creation failed. Staged folder is at %STAGE_DIR%\.
) else (
    echo [Build] Zip ready: %ZIP_NAME%
    rmdir /s /q "%STAGE_DIR%"
)

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo ===================================================
echo   Build complete!
echo   Full build : dist\Stellar-Forge\Stellar-Forge.exe
echo   Release zip: %ZIP_NAME%  (no SPICE kernels)
echo ===================================================
echo.
echo To publish: upload %ZIP_NAME% to a GitHub Release.
echo Users need a GPU with OpenGL 3.3+ — no Python required.
echo.
pause
