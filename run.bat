@echo off
setlocal

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --parallel
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

if exist "build\bin\StellarForge.exe" (
    .\build\bin\StellarForge.exe %*
) else (
    .\build\StellarForge.exe %*
)
