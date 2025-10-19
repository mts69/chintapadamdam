######################################################################
# Choose your favorite C compiler
CC = gcc
NVCC = nvcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2) -I../include

######################################################################
# CUDA settings - adjust for your university's GPU
# Common GPU architectures:
# - sm_50: Maxwell (GTX 900 series)
# - sm_60: Pascal (GTX 1000 series) 
# - sm_70: Volta (V100)
# - sm_75: Turing (Tesla T4, RTX 2000 series)
# - sm_80: Ampere (RTX 3000 series, A100)
# - sm_86: Ampere (RTX 3000 series)
# - sm_89: Ada Lovelace (RTX 4000 series)

# Auto-detect GPU architecture or set manually
# Uncomment the line that matches your university's GPU:
# CUDA_ARCH = -arch=sm_50   # For Maxwell GPUs
# CUDA_ARCH = -arch=sm_60   # For Pascal GPUs  
# CUDA_ARCH = -arch=sm_70   # For Volta GPUs
# CUDA_ARCH = -arch=sm_75   # For Turing GPUs (Tesla T4, RTX 2000)
# CUDA_ARCH = -arch=sm_80   # For Ampere GPUs (A100, RTX 3000)
CUDA_ARCH = -arch=sm_86   # For Ampere GPUs (RTX 3000)
# CUDA_ARCH = -arch=sm_89   # For Ada Lovelace GPUs (RTX 4000)

CUDAFLAGS = -O3 -std=c++11 $(CUDA_ARCH) -Xcompiler -fPIC -I../include


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = ../src/example1.c ../src/example2.c ../src/example3.c ../src/example4.c ../src/example5.c
ARCH = ../src/convolve.c ../src/error.c ../src/pnmio.c ../src/pyramid.c ../src/selectGoodFeatures.c \
       ../src/storeFeatures.c ../src/trackFeatures.c ../src/klt.c ../src/klt_util.c ../src/writeFeatures.c
CUDA_SOURCES = ../src/convolve_cuda.cu ../src/interpolate_cuda.cu
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

all:  lib $(EXAMPLES:.c=) convolve_cuda test_interpolation

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ ../src/$@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ ../src/$@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ ../src/$@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ ../src/$@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ ../src/$@.c -L. -lklt $(LIB) -lm

# CUDA convolution program
convolve_cuda: $(CUDA_SOURCES)
	$(NVCC) $(CUDAFLAGS) -o $@ $< -lm

# CUDA interpolation test program
test_interpolation: ../src/test_interpolation.cu ../src/interpolate_cuda.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -lm

# Test CUDA convolution
test-cuda: convolve_cuda
	@echo "Testing CUDA convolution..."
	@echo "=========================="
	./convolve_cuda

# Test CUDA interpolation
test-interpolation: test_interpolation
	@echo "Testing CUDA interpolation..."
	@echo "============================="
	./test_interpolation

# Test complete KLT algorithm
test-klt: example3
	@echo "Testing complete KLT algorithm..."
	@echo "================================"
	./example3

# Run performance benchmark
benchmark: convolve_cuda example3
	@echo "Running performance benchmark..."
	@echo "=============================="
	@echo "CUDA Convolution Test:"
	@time ./convolve_cuda
	@echo ""
	@echo "Complete KLT Algorithm:"
	@time ./example3

# Show GPU information
gpu-info:
	@echo "GPU Information:"
	@echo "================"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits || echo "nvidia-smi not available"

# Show help
help:
	@echo "KLT CUDA Makefile Help"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  lib          - Build KLT library"
	@echo "  example1-5   - Build individual example programs"
	@echo "  convolve_cuda - Build CUDA convolution program"
	@echo "  test_interpolation - Build CUDA interpolation test"
	@echo "  test-cuda    - Test CUDA convolution"
	@echo "  test-interpolation - Test CUDA interpolation"
	@echo "  test-klt     - Test complete KLT algorithm"
	@echo "  benchmark   - Run performance comparison"
	@echo "  gpu-info     - Show GPU information"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help"
	@echo ""
	@echo "GPU Architecture Settings:"
	@echo "  Edit CUDA_ARCH in Makefile to match your GPU:"
	@echo "  - sm_50: Maxwell (GTX 900 series)"
	@echo "  - sm_60: Pascal (GTX 1000 series)"
	@echo "  - sm_70: Volta (V100)"
	@echo "  - sm_75: Turing (Tesla T4, RTX 2000)"
	@echo "  - sm_80: Ampere (A100, RTX 3000)"
	@echo "  - sm_86: Ampere (RTX 3000)"
	@echo "  - sm_89: Ada Lovelace (RTX 4000)"

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) convolve_cuda test_interpolation *.tar *.tar.gz libklt.a \
	      ../output/feat*.ppm ../output/features.ft ../output/features.txt



