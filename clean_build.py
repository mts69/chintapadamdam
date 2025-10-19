#!/usr/bin/env python3
"""
Clean Build Script for KLT GPU Project
This script removes all compiled object files and executables
"""

import os
import glob
import shutil

def clean_build():
    print("🧹 CLEANING KLT BUILD FILES")
    print("=" * 40)
    
    # Change to klt directory if it exists
    if os.path.exists('klt'):
        os.chdir('klt')
        print("📁 Changed to klt directory")
    
    # Remove object files from build directory
    if os.path.exists('build'):
        print("🗑️  Removing object files from build/...")
        build_files = glob.glob('build/*.o') + glob.glob('build/*.a')
        for file in build_files:
            try:
                os.remove(file)
                print(f"  ✅ Removed {file}")
            except OSError as e:
                print(f"  ⚠️  Could not remove {file}: {e}")
        print("✅ Cleaned build/ directory")
    else:
        print("⚠️  build/ directory not found")
    
    # Remove executables
    print("🗑️  Removing executables...")
    executables = [
        'example3',
        'example3_gpu', 
        'example3_gpu_real',
        'convolve_cuda',
        'test_interpolation',
        'test_cuda'
    ]
    
    for exe in executables:
        if os.path.exists(exe):
            try:
                os.remove(exe)
                print(f"  ✅ Removed {exe}")
            except OSError as e:
                print(f"  ⚠️  Could not remove {exe}: {e}")
    
    # Remove any other common build artifacts
    print("🗑️  Removing other build artifacts...")
    artifacts = glob.glob('*.o') + glob.glob('*.a') + glob.glob('*.so') + glob.glob('*.dylib')
    for artifact in artifacts:
        try:
            os.remove(artifact)
            print(f"  ✅ Removed {artifact}")
        except OSError as e:
            print(f"  ⚠️  Could not remove {artifact}: {e}")
    
    print("✅ Cleaned all object files and executables")
    
    # Show what's left
    print("\n📁 Remaining source files:")
    source_files = glob.glob('*.c') + glob.glob('*.cu') + glob.glob('*.h')
    for file in sorted(source_files):
        print(f"  📄 {file}")
    
    print("\n🎯 Ready for clean compilation!")
    print("Run your compilation commands now.")

if __name__ == "__main__":
    clean_build()
