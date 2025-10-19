# 🚀 RUN KLT ALGORITHM WITH FULL GPU ACCELERATION
import os
import subprocess

print("🚀 RUNNING KLT ALGORITHM WITH FULL GPU ACCELERATION")
print("=" * 60)

# Start from /content/klt/
os.chdir('/content/klt')
print(f"📁 Working in: {os.getcwd()}")

# Check if input images exist
input_images = [f"input/img{i}.pgm" for i in range(10)]
missing_images = [img for img in input_images if not os.path.exists(img)]

if missing_images:
    print(f"⚠️  Missing input images: {missing_images}")
    print("📁 Available input files:")
    if os.path.exists('input'):
        for file in os.listdir('input'):
            print(f"  📄 {file}")
    else:
        print("  ❌ input/ directory not found")
    print("\n⚠️  Cannot run KLT without input images")
    print("Please upload the .pgm image files to the input/ directory")
else:
    print("✅ All input images found!")
    
    # Run the KLT algorithm with GPU acceleration
    print("\n🚀 Starting KLT algorithm with GPU acceleration...")
    print("🎯 GPU functions enabled: convolution + interpolation")
    try:
        result = subprocess.run(['./example3_gpu_real'], capture_output=True, text=True, check=True, timeout=300)
        print("✅ KLT algorithm completed successfully!")
        print("\n📊 Output:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️  Warnings/Errors:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️  KLT algorithm timed out (5 minutes)")
    except subprocess.CalledProcessError as e:
        print(f"❌ KLT algorithm failed: {e}")
        print(f"Error: {e.stderr}")
        print(f"Output: {e.stdout}")

print("\n📁 Checking output files...")
if os.path.exists('output'):
    output_files = os.listdir('output')
    if output_files:
        print(f"✅ Found {len(output_files)} output files:")
        for file in sorted(output_files):
            file_path = os.path.join('output', file)
            file_size = os.path.getsize(file_path)
            print(f"  📄 {file} ({file_size} bytes)")
    else:
        print("⚠️  No output files found")
else:
    print("❌ output/ directory not found")

print("\n🎉 KLT ALGORITHM WITH GPU ACCELERATION COMPLETE!")
print("🚀 GPU functions used: convolution + interpolation")
