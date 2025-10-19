@echo off
REM Clean Build Script for KLT GPU Project (Windows)
REM This script removes all compiled object files and executables

echo 🧹 CLEANING KLT BUILD FILES
echo ============================

REM Change to klt directory if it exists
if exist "klt" (
    cd klt
    echo 📁 Changed to klt directory
)

REM Remove object files from build directory
if exist "build" (
    echo 🗑️  Removing object files from build/...
    del /q build\*.o 2>nul
    del /q build\*.a 2>nul
    echo ✅ Cleaned build/ directory
) else (
    echo ⚠️  build/ directory not found
)

REM Remove executables
echo 🗑️  Removing executables...
del /q example3.exe 2>nul
del /q example3_gpu.exe 2>nul
del /q example3_gpu_real.exe 2>nul
del /q convolve_cuda.exe 2>nul
del /q test_interpolation.exe 2>nul
del /q test_cuda.exe 2>nul

REM Remove any other common build artifacts
del /q *.o 2>nul
del /q *.a 2>nul
del /q *.so 2>nul
del /q *.dylib 2>nul

echo ✅ Cleaned all object files and executables

REM Show what's left
echo.
echo 📁 Remaining files:
dir /b *.c *.cu *.h *.pgm *.ppm *.txt *.ft 2>nul | findstr /v "File Not Found"

echo.
echo 🎯 Ready for clean compilation!
echo Run your compilation commands now.

pause
