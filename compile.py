# 🚀 COMPILE KLT WITH GPU ACCELERATION (Convolution + Interpolation)
import os
import subprocess
import shutil

!git pull origin main

print("🚀 COMPILING KLT WITH FULL GPU ACCELERATION")
print("=" * 60)

# Start from /content/klt/
os.chdir('/content/klt')
print(f"📁 Working in: {os.getcwd()}")

# Check what files actually exist
print("\n🔍 Checking for source files...")
if os.path.exists('src'):
    src_files = os.listdir('src')
    print(f"📁 Files in src/: {src_files}")

    # Check for specific files we need
    required_files = ['convolve_gpu_functions.cu', 'interpolate_cuda.cu', 'example3_gpu_real.c']
    for file in required_files:
        if file in src_files:
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
else:
    print("❌ src/ directory not found")
    print("📁 Current directory contents:")
    for item in os.listdir('.'):
        print(f"  📄 {item}")

# Step 1: Compile GPU functions
print("\n1. Compiling GPU functions...")

# Compile convolution GPU functions
print("1a. Compiling convolution GPU functions...")
if not os.path.exists('src/convolve_gpu_functions.cu'):
    print("❌ src/convolve_gpu_functions.cu not found!")
    print("📁 Available files in src/:")
    if os.path.exists('src'):
        for file in os.listdir('src'):
            print(f"  📄 {file}")
    exit(1)

gpu_conv_cmd = [
    'nvcc',
    '-O3',
    '-std=c++11',
    '-arch=sm_75',  # Tesla T4
    '-I./include',
    '-c',
    'src/convolve_gpu_functions.cu',
    '-o', 'build/convolve_gpu_functions.o'
]

try:
    result = subprocess.run(gpu_conv_cmd, capture_output=True, text=True, check=True)
    print("✅ Convolution GPU functions compiled successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Convolution GPU compilation failed: {e}")
    print(f"Error: {e.stderr}")
    exit(1)

# Compile interpolation GPU functions
print("1b. Compiling interpolation GPU functions...")
if not os.path.exists('src/interpolate_cuda.cu'):
    print("❌ src/interpolate_cuda.cu not found!")
    print("📁 Available files in src/:")
    if os.path.exists('src'):
        for file in os.listdir('src'):
            print(f"  📄 {file}")
    exit(1)

gpu_interp_cmd = [
    'nvcc',
    '-O3',
    '-std=c++11',
    '-arch=sm_75',  # Tesla T4
    '-I./include',
    '-c',
    'src/interpolate_cuda.cu',
    '-o', 'build/interpolate_cuda.o'
]

try:
    result = subprocess.run(gpu_interp_cmd, capture_output=True, text=True, check=True)
    print("✅ Interpolation GPU functions compiled successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Interpolation GPU compilation failed: {e}")
    print(f"Error: {e.stderr}")
    exit(1)

# Step 2: Compile KLT library
print("\n2. Compiling KLT library...")
klt_sources = [
    'src/klt.c',
    'src/convolve.c',
    'src/error.c',
    'src/pnmio.c',
    'src/pyramid.c',
    'src/selectGoodFeatures.c',
    'src/storeFeatures.c',
    'src/trackFeatures.c',
    'src/klt_util.c',
    'src/writeFeatures.c'
]

object_files = []
for src in klt_sources:
    obj_file = src.replace('src/', 'build/').replace('.c', '.o')
    object_files.append(obj_file)

    compile_cmd = [
        'gcc',
        '-c',
        '-O3',
        '-DNDEBUG',
        '-I./include',
        '-o', obj_file,
        src
    ]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Compiled {src}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile {src}: {e}")
        print(f"Error: {e.stderr}")
        exit(1)

# Step 3: Create static library
print("\n3. Creating static library...")
ar_cmd = ['ar', 'rcs', 'build/libklt.a'] + object_files
try:
    result = subprocess.run(ar_cmd, capture_output=True, text=True, check=True)
    print("✅ Static library created!")
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to create library: {e}")
    exit(1)

# Step 4: Compile example3_gpu_real with GPU functions
print("\n4. Compiling example3_gpu_real with GPU support...")

# Check if main program exists
if not os.path.exists('src/example3_gpu_real.c'):
    print("❌ src/example3_gpu_real.c not found!")
    print("📁 Available files in src/:")
    if os.path.exists('src'):
        for file in os.listdir('src'):
            print(f"  📄 {file}")
    exit(1)

# Use nvcc for final linking (handles CUDA runtime automatically)
example3_cmd = [
    'nvcc',
    '-O3',
    '-std=c++11',
    '-arch=sm_75',
    '-I./include',
    '-o', 'example3_gpu_real',
    'src/example3_gpu_real.c',
    'build/convolve_gpu_functions.o',
    'build/interpolate_cuda.o',
    '-L./build',
    '-lklt',
    '-lm'
]

try:
    result = subprocess.run(example3_cmd, capture_output=True, text=True, check=True)
    print("✅ example3_gpu_real compiled successfully with GPU acceleration!")
    print("🚀 Both convolution and interpolation GPU functions linked!")
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to compile example3_gpu_real: {e}")
    print(f"Error: {e.stderr}")
    exit(1)

print("\n🎉 COMPILATION COMPLETE!")
print("✅ All components compiled successfully")
print("✅ Ready to run KLT with full GPU acceleration!")
print("🚀 GPU functions available: convolution + interpolation")


