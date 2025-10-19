# KLT GPU-Accelerated Makefile
# Replicates functionality of compile.py and run.py

# Compilers
CC = gcc
NVCC = nvcc

# Git authentication variables (set these as environment variables)
# export GIT_USERNAME=your_username
# export GIT_PASSWORD=your_personal_access_token
# export GIT_REPO=username/repository_name

# Compiler flags
CFLAGS = -O3 -DNDEBUG -I./include
CUDAFLAGS = -O3 -std=c++11 -arch=sm_75 -I./include

# Directories
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include
DATA_DIR = input
OUTPUT_DIR = output

# Source files
KLT_SOURCES = $(SRC_DIR)/klt.c \
              $(SRC_DIR)/convolve.c \
              $(SRC_DIR)/error.c \
              $(SRC_DIR)/pnmio.c \
              $(SRC_DIR)/pyramid.c \
              $(SRC_DIR)/selectGoodFeatures.c \
              $(SRC_DIR)/storeFeatures.c \
              $(SRC_DIR)/trackFeatures.c \
              $(SRC_DIR)/klt_util.c \
              $(SRC_DIR)/writeFeatures.c

CUDA_SOURCES = $(SRC_DIR)/convolve_gpu_functions.cu \
               $(SRC_DIR)/interpolate_cuda.cu

# Object files
KLT_OBJECTS = $(KLT_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
CUDA_OBJECTS = $(BUILD_DIR)/convolve_gpu_functions.o \
               $(BUILD_DIR)/interpolate_cuda.o

# Targets
LIBRARY = $(BUILD_DIR)/libklt.a
EXECUTABLE = example3_gpu_real

# Default target
all: gpu

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build GPU-accelerated version (replicates compile.py)
gpu: $(BUILD_DIR) $(LIBRARY) $(CUDA_OBJECTS) $(EXECUTABLE)
	@echo "🎉 GPU COMPILATION COMPLETE!"
	@echo "✅ All components compiled successfully"
	@echo "✅ Ready to run KLT with full GPU acceleration!"
	@echo "🚀 GPU functions available: convolution + interpolation"

# Compile KLT library sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) -c $(CFLAGS) -o $@ $<

# Compile CUDA sources
$(BUILD_DIR)/convolve_gpu_functions.o: $(SRC_DIR)/convolve_gpu_functions.cu
	@echo "Compiling convolution GPU functions..."
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

$(BUILD_DIR)/interpolate_cuda.o: $(SRC_DIR)/interpolate_cuda.cu
	@echo "Compiling interpolation GPU functions..."
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

# Create static library
$(LIBRARY): $(KLT_OBJECTS)
	@echo "Creating static library..."
	ar rcs $@ $^

# Link final executable
$(EXECUTABLE): $(SRC_DIR)/example3_gpu_real.c $(LIBRARY) $(CUDA_OBJECTS)
	@echo "Compiling example3_gpu_real with GPU support..."
	$(NVCC) $(CUDAFLAGS) -o $@ $< $(CUDA_OBJECTS) -L$(BUILD_DIR) -lklt -lm

# Run the GPU example (replicates run.py)
run: $(EXECUTABLE)
	@echo "🚀 RUNNING KLT ALGORITHM WITH FULL GPU ACCELERATION"
	@echo "===================================================="
	@echo "📁 Working in: $(PWD)"
	@echo "📁 Data directory: $(DATA_DIR)"
	@echo ""
	@echo "🔍 Checking for input images..."
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "❌ $(DATA_DIR)/ directory not found"; \
		echo "⚠️  Cannot run KLT without input images"; \
		echo "Please add .pgm image files to the $(DATA_DIR)/ directory"; \
		exit 1; \
	fi
	@echo "🔢 Calculating number of frames..."
	@nFrames=$$(ls $(DATA_DIR)/img*.pgm 2>/dev/null | wc -l); \
	if [ $$nFrames -eq 0 ]; then \
		echo "❌ No img*.pgm files found in $(DATA_DIR)/"; \
		echo "📁 Available files in $(DATA_DIR)/:"; \
		ls -la $(DATA_DIR)/ 2>/dev/null || echo "  ❌ Directory is empty"; \
		echo ""; \
		echo "⚠️  Cannot run KLT without input images"; \
		echo "Please add img*.pgm files to the $(DATA_DIR)/ directory"; \
		exit 1; \
	else \
		echo "✅ Found $$nFrames input images in $(DATA_DIR)/"; \
		echo "📊 Frame count: $$nFrames"; \
		echo ""; \
		echo "🚀 Starting KLT algorithm with GPU acceleration..."; \
		echo "🎯 GPU functions enabled: convolution + interpolation"; \
		echo "📁 Using data directory: $(DATA_DIR)"; \
		echo "🔢 Number of frames: $$nFrames"; \
		echo ""; \
		DATA_DIR=$(DATA_DIR) nFrames=$$nFrames timeout 300 ./$(EXECUTABLE) || echo "⚠️  KLT algorithm timed out (5 minutes)"; \
	fi
	@echo ""
	@echo "📁 Checking output files..."
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		output_count=$$(ls $(OUTPUT_DIR)/ 2>/dev/null | wc -l); \
		if [ $$output_count -gt 0 ]; then \
			echo "✅ Found $$output_count output files:"; \
			ls -la $(OUTPUT_DIR)/ | grep -v "^total" | while read line; do \
				echo "  📄 $$line"; \
			done; \
		else \
			echo "⚠️  No output files found"; \
		fi; \
	else \
		echo "❌ output/ directory not found"; \
	fi
	@echo ""
	@echo "🎉 KLT ALGORITHM WITH GPU ACCELERATION COMPLETE!"
	@echo "🚀 GPU functions used: convolution + interpolation"

# Clean build artifacts, executables, libraries, and temporary files
clean-all:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(EXECUTABLE)
	rm -f *.o *.a
	rm -f $(OUTPUT_DIR)/feat*.ppm $(OUTPUT_DIR)/features.ft $(OUTPUT_DIR)/features.txt
	@echo "✅ Clean complete!"
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(EXECUTABLE)
	rm -f *.o *.a
	@echo "✅ Clean complete!"

# Force fetch from remote (replaces git pull --force)
pull:
	@echo "📥 Force fetching from remote..."
	git fetch --all
	git reset --hard origin/$(shell git rev-parse --abbrev-ref HEAD)
	@echo "✅ Force pull complete!"

# Force push to remote (works for both local and Colab)
push:
	@echo "🧹 Cleaning build files before push..."
	$(MAKE) clean
	@echo "📤 Force pushing to repository..."
	@echo "📁 Using environment variables..."
	@if [ -z "$(GIT_USERNAME)" ] || [ -z "$(GIT_PASSWORD)" ] || [ -z "$(GIT_REPO)" ]; then \
		echo "❌ Missing git credentials!"; \
		echo "Set these environment variables:"; \
		echo "  export GIT_USERNAME=your_username"; \
		echo "  export GIT_PASSWORD=your_token"; \
		echo "  export GIT_REPO=username/repo"; \
		echo "  make push"; \
		echo ""; \
		echo "For Google Colab:"; \
		echo "  !export GIT_USERNAME=your_username"; \
		echo "  !export GIT_PASSWORD=your_token"; \
		echo "  !export GIT_REPO=username/repo"; \
		echo "  !make push"; \
		exit 1; \
	fi
	git add .
	git commit -m "update" || echo "No changes to commit"
	@echo "🔐 Using username and password authentication..."
	git push https://$(GIT_USERNAME):$(GIT_PASSWORD)@github.com/$(GIT_REPO) $(shell git rev-parse --abbrev-ref HEAD)
	@echo "✅ Force push complete!"

# Show help
help:
	@echo "KLT GPU-Accelerated Makefile"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  gpu     - Build GPU-accelerated version (same as compile.py)"
	@echo "  run     - Run GPU example (same as run.py)"
	@echo "  clean   - Clean build artifacts, executables, libraries, and temp files"
	@echo "  pull    - Force fetch from remote"
	@echo "  push   - Force push to remote (works for both local and Colab)"
	@echo "  help   - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make gpu    # Build everything with GPU acceleration"
	@echo "  make run    # Run the GPU-accelerated KLT algorithm"
	@echo "  make clean  # Clean all build files"
	@echo "  make pull   # Force fetch from remote"
	@echo "  make push  # Force push to remote"
	@echo ""
	@echo "Git Authentication Setup:"
	@echo "  Local:"
	@echo "    export GIT_USERNAME=your_username"
	@echo "    export GIT_PASSWORD=your_token"
	@echo "    export GIT_REPO=username/repo"
	@echo "    make push"
	@echo ""
	@echo "  Google Colab:"
	@echo "    !export GIT_USERNAME=your_username"
	@echo "    !export GIT_PASSWORD=your_token"
	@echo "    !export GIT_REPO=username/repo"
	@echo "    !make push"

# Phony targets
.PHONY: all gpu run clean pull push help
