#!/bin/bash

# Clean Build Script for KLT GPU Project
# This script removes all compiled object files and executables

echo "🧹 CLEANING KLT BUILD FILES"
echo "============================"

# Change to klt directory if it exists
if [ -d "klt" ]; then
    cd klt
    echo "📁 Changed to klt directory"
fi

# Remove object files from build directory
if [ -d "build" ]; then
    echo "🗑️  Removing object files from build/..."
    rm -f build/*.o
    rm -f build/*.a
    echo "✅ Cleaned build/ directory"
else
    echo "⚠️  build/ directory not found"
fi

# Remove executables
echo "🗑️  Removing executables..."
rm -f example3
rm -f example3_gpu
rm -f example3_gpu_real
rm -f convolve_cuda
rm -f test_interpolation
rm -f test_cuda

# Remove any other common build artifacts
rm -f *.o
rm -f *.a
rm -f *.so
rm -f *.dylib

echo "✅ Cleaned all object files and executables"

# Show what's left
echo ""
echo "📁 Remaining files:"
ls -la | grep -E "\.(c|cu|h|pgm|ppm|txt|ft)$" | head -10

echo ""
echo "🎯 Ready for clean compilation!"
echo "Run your compilation commands now."
