# 🚀 KLT CUDA Implementation for University GPU

This is a GPU-accelerated implementation of the KLT (Kanade-Lucas-Tomasi) algorithm with CUDA convolution kernels.

## 🎯 Features

- ✅ **Horizontal CUDA convolution kernel**
- ✅ **Vertical CUDA convolution kernel**  
- ✅ **Separable convolution pipeline**
- ✅ **Complete KLT algorithm integration**
- ✅ **Performance benchmarking**
- ✅ **Memory-safe implementation**

## 🔧 Quick Setup

### 1. Auto-Configure for Your GPU
```bash
chmod +x setup_gpu.sh
./setup_gpu.sh
```

### 2. Build Everything
```bash
make all
```

### 3. Test CUDA Convolution
```bash
make test-cuda
```

### 4. Test Complete KLT Algorithm
```bash
make test-klt
```

### 5. Run Performance Benchmark
```bash
make benchmark
```

## 📁 Project Structure

```
klt/
├── src/                    # Source files
│   ├── convolve_cuda.cu    # CUDA convolution kernels
│   ├── convolve.c          # CPU convolution
│   ├── klt.c              # Main KLT algorithm
│   ├── example3.c         # KLT example program
│   └── ...                # Other KLT source files
├── include/               # Header files
├── input/                 # Input images (PGM format)
├── output/                # Output files
├── build/                 # Build directory
├── Makefile.gpu          # Full Makefile with all options
├── setup_gpu.sh          # Auto-configuration script
└── README_GPU.md         # This file
```

## 🎮 Available Commands

### Build Commands
```bash
make all          # Build everything
make convolve_cuda # Build CUDA program only
make example3     # Build KLT example program
make clean        # Clean build files
```

### Test Commands
```bash
make test-cuda    # Test CUDA convolution kernels
make test-klt     # Test complete KLT algorithm
make benchmark    # Run performance comparison
```

### Utility Commands
```bash
make help         # Show all available commands
make gpu-info     # Show GPU information
make config       # Show current configuration
```

## 🔧 Manual GPU Configuration

If auto-detection doesn't work, edit the Makefile and set the correct CUDA architecture:

```makefile
# Common GPU architectures:
CUDA_ARCH = -arch=sm_50   # Maxwell (GTX 900 series)
CUDA_ARCH = -arch=sm_60   # Pascal (GTX 1000 series)
CUDA_ARCH = -arch=sm_70   # Volta (V100)
CUDA_ARCH = -arch=sm_75   # Turing (Tesla T4, RTX 2000)
CUDA_ARCH = -arch=sm_80   # Ampere (A100, RTX 3000)
CUDA_ARCH = -arch=sm_86   # Ampere (RTX 3000)
CUDA_ARCH = -arch=sm_89   # Ada Lovelace (RTX 4000)
```

## 📊 Expected Output

### CUDA Convolution Test
```
KLT CUDA Image Processing
==========================
Using CUDA device: [Your GPU Name]
Compute capability: X.X
Total global memory: XX.XX GB

Testing separable convolution (horizontal -> vertical)...
Step 1: Horizontal convolution...
Horizontal convolution: X.XXX ms
Step 2: Vertical convolution (using horizontal result)...
Vertical convolution: X.XXX ms
Complete separable convolution: X.XXX ms
✅ Both convolutions completed successfully!
```

### Complete KLT Algorithm
```
(KLT) Selecting the 150 best features from a 320 by 240 image...
    150 features found.
(KLT) Writing 150 features to PPM file: 'output/feat0.ppm'
(KLT) Tracking 150 features in a 320 by 240 image...
    140 features successfully tracked.
...
```

## 🚀 Performance Benefits

- **GPU Acceleration**: 10-100x speedup over CPU
- **Separable Convolution**: Memory-efficient processing
- **Parallel Processing**: Thousands of threads working simultaneously
- **Optimized Memory Access**: Coalesced memory patterns

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Check if CUDA is installed
   nvcc --version
   nvidia-smi
   ```

2. **Wrong GPU architecture**
   ```bash
   # Check your GPU compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   ```

3. **Permission denied**
   ```bash
   # Make scripts executable
   chmod +x setup_gpu.sh
   ```

4. **Missing dependencies**
   ```bash
   # Install required packages (Ubuntu/Debian)
   sudo apt-get install build-essential
   ```

## 📚 Technical Details

### CUDA Kernels Implemented
- `convolveImageHorizKernel`: Horizontal convolution
- `convolveImageVertKernel`: Vertical convolution
- `convolveSeparateCUDA`: Complete separable convolution

### Memory Management
- Proper CUDA memory allocation/deallocation
- Error checking with `CUDA_CHECK` macro
- Bounds checking for image dimensions

### Performance Optimization
- Coalesced memory access patterns
- Optimal thread block sizes (16x16)
- Efficient kernel launch configuration

## 🎓 University Usage

This implementation is designed for:
- **Computer Vision courses**
- **GPU programming research**
- **Performance optimization studies**
- **KLT algorithm understanding**

## 📄 License

This project is for educational purposes. Please cite the original KLT algorithm if used in research.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your GPU architecture
3. Ensure CUDA toolkit is properly installed
4. Check the build output for specific errors

---

**Happy GPU Computing! 🚀**
