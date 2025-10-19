#!/bin/bash

# KLT CUDA Setup Script for University GPU Environment
# This script helps configure the Makefile for your specific GPU

echo "🚀 KLT CUDA Setup for University GPU"
echo "====================================="
echo ""

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Detecting GPU information..."
    echo "=============================="
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits
    echo ""
    
    # Extract compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
    echo "Detected compute capability: $COMPUTE_CAP"
    
    # Determine CUDA architecture
    case $COMPUTE_CAP in
        "5.0"|"5.1"|"5.2")
            CUDA_ARCH="sm_50"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Maxwell)"
            ;;
        "6.0"|"6.1")
            CUDA_ARCH="sm_60"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Pascal)"
            ;;
        "7.0")
            CUDA_ARCH="sm_70"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Volta)"
            ;;
        "7.5")
            CUDA_ARCH="sm_75"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Turing)"
            ;;
        "8.0")
            CUDA_ARCH="sm_80"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Ampere)"
            ;;
        "8.6")
            CUDA_ARCH="sm_86"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Ampere)"
            ;;
        "8.9")
            CUDA_ARCH="sm_89"
            echo "→ Using CUDA architecture: $CUDA_ARCH (Ada Lovelace)"
            ;;
        *)
            echo "⚠️  Unknown compute capability: $COMPUTE_CAP"
            echo "→ Using default: sm_80 (Ampere)"
            CUDA_ARCH="sm_80"
            ;;
    esac
else
    echo "⚠️  nvidia-smi not found. Using default CUDA architecture."
    CUDA_ARCH="sm_80"
fi

echo ""
echo "🔧 Configuring Makefile..."
echo "=========================="

# Create a configured Makefile
cat > Makefile << EOF
# Makefile for KLT CUDA Implementation
# Auto-configured for University GPU Environment

# Compiler settings
CC = gcc
NVCC = nvcc
AR = ar

# CUDA architecture (auto-detected)
CUDA_ARCH = -arch=$CUDA_ARCH

# Compiler flags
CFLAGS = -O3 -Wall -DNDEBUG
CUDAFLAGS = -O3 -std=c++11 \$(CUDA_ARCH) -Xcompiler -fPIC
INCLUDES = -I./include
LIBRARIES = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include
INPUT_DIR = input
OUTPUT_DIR = output

# Source files
CPU_SOURCES = \$(SRC_DIR)/convolve.c \\
              \$(SRC_DIR)/error.c \\
              \$(SRC_DIR)/pnmio.c \\
              \$(SRC_DIR)/pyramid.c \\
              \$(SRC_DIR)/selectGoodFeatures.c \\
              \$(SRC_DIR)/storeFeatures.c \\
              \$(SRC_DIR)/trackFeatures.c \\
              \$(SRC_DIR)/klt.c \\
              \$(SRC_DIR)/klt_util.c \\
              \$(SRC_DIR)/writeFeatures.c

CUDA_SOURCES = \$(SRC_DIR)/convolve_cuda.cu

# Object files
CPU_OBJECTS = \$(CPU_SOURCES:\$(SRC_DIR)/%.c=\$(BUILD_DIR)/%.o)

# Targets
LIBRARY = \$(BUILD_DIR)/libklt.a
CUDA_PROGRAM = convolve_cuda
EXAMPLES_BIN = example1 example2 example3 example4 example5

# Default target
all: setup \$(LIBRARY) \$(CUDA_PROGRAM) \$(EXAMPLES_BIN)

# Create necessary directories
setup:
	@echo "Setting up build environment..."
	@mkdir -p \$(BUILD_DIR)
	@mkdir -p \$(OUTPUT_DIR)
	@echo "✅ Directories created"

# Compile CPU source files
\$(BUILD_DIR)/%.o: \$(SRC_DIR)/%.c
	@echo "Compiling \$<..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -c \$< -o \$@

# Create static library
\$(LIBRARY): \$(CPU_OBJECTS)
	@echo "Creating static library..."
	\$(AR) rcs \$@ \$^
	@echo "✅ Library created: \$@"

# Compile CUDA convolution program
\$(CUDA_PROGRAM): \$(SRC_DIR)/convolve_cuda.cu
	@echo "Compiling CUDA convolution program..."
	\$(NVCC) \$(CUDAFLAGS) \$(INCLUDES) -o \$@ \$< \$(LIBRARIES)
	@echo "✅ CUDA program created: \$@"

# Compile example programs
example1: \$(LIBRARY)
	@echo "Compiling example1..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -o \$@ \$(SRC_DIR)/\$@.c -L\$(BUILD_DIR) -lklt \$(LIBRARIES)

example2: \$(LIBRARY)
	@echo "Compiling example2..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -o \$@ \$(SRC_DIR)/\$@.c -L\$(BUILD_DIR) -lklt \$(LIBRARIES)

example3: \$(LIBRARY)
	@echo "Compiling example3..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -o \$@ \$(SRC_DIR)/\$@.c -L\$(BUILD_DIR) -lklt \$(LIBRARIES)

example4: \$(LIBRARY)
	@echo "Compiling example4..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -o \$@ \$(SRC_DIR)/\$@.c -L\$(BUILD_DIR) -lklt \$(LIBRARIES)

example5: \$(LIBRARY)
	@echo "Compiling example5..."
	\$(CC) \$(CFLAGS) \$(INCLUDES) -o \$@ \$(SRC_DIR)/\$@.c -L\$(BUILD_DIR) -lklt \$(LIBRARIES)

# Test CUDA program
test-cuda: \$(CUDA_PROGRAM)
	@echo "Testing CUDA convolution..."
	@echo "=========================="
	./\$(CUDA_PROGRAM)

# Test complete KLT algorithm
test-klt: example3
	@echo "Testing complete KLT algorithm..."
	@echo "================================"
	./example3

# Run performance comparison
benchmark: \$(CUDA_PROGRAM) example3
	@echo "Running performance benchmark..."
	@echo "=============================="
	@echo "CUDA Convolution Test:"
	@time ./\$(CUDA_PROGRAM)
	@echo ""
	@echo "Complete KLT Algorithm:"
	@time ./example3

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf \$(BUILD_DIR)
	rm -f \$(CUDA_PROGRAM) \$(EXAMPLES_BIN)
	rm -f \$(OUTPUT_DIR)/*.ppm \$(OUTPUT_DIR)/*.txt \$(OUTPUT_DIR)/*.ft
	@echo "✅ Cleaned"

# Show help
help:
	@echo "KLT CUDA Makefile Help"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  \$(CUDA_PROGRAM) - Build CUDA convolution program"
	@echo "  example1-5   - Build individual example programs"
	@echo "  test-cuda    - Test CUDA convolution"
	@echo "  test-klt     - Test complete KLT algorithm"
	@echo "  benchmark   - Run performance comparison"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help"

.PHONY: all setup test-cuda test-klt benchmark clean help
EOF

echo "✅ Makefile configured for $CUDA_ARCH"
echo ""
echo "🎯 Ready to build! Run:"
echo "  make all        # Build everything"
echo "  make test-cuda  # Test CUDA convolution"
echo "  make test-klt   # Test complete KLT algorithm"
echo "  make benchmark  # Run performance comparison"
echo ""
echo "📚 For more options, run: make help"
