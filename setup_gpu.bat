@echo off
REM KLT CUDA Setup Script for University GPU Environment (Windows)
REM This script helps configure the Makefile for your specific GPU

echo 🚀 KLT CUDA Setup for University GPU
echo =====================================
echo.

REM Check if nvidia-smi is available
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 📊 Detecting GPU information...
    echo ==============================
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits
    echo.
    
    REM Extract compute capability (simplified for Windows)
    for /f "tokens=2 delims=," %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits') do set COMPUTE_CAP=%%i
    echo Detected compute capability: %COMPUTE_CAP%
    
    REM Determine CUDA architecture
    if "%COMPUTE_CAP%"=="5.0" set CUDA_ARCH=sm_50
    if "%COMPUTE_CAP%"=="5.1" set CUDA_ARCH=sm_50
    if "%COMPUTE_CAP%"=="5.2" set CUDA_ARCH=sm_50
    if "%COMPUTE_CAP%"=="6.0" set CUDA_ARCH=sm_60
    if "%COMPUTE_CAP%"=="6.1" set CUDA_ARCH=sm_60
    if "%COMPUTE_CAP%"=="7.0" set CUDA_ARCH=sm_70
    if "%COMPUTE_CAP%"=="7.5" set CUDA_ARCH=sm_75
    if "%COMPUTE_CAP%"=="8.0" set CUDA_ARCH=sm_80
    if "%COMPUTE_CAP%"=="8.6" set CUDA_ARCH=sm_86
    if "%COMPUTE_CAP%"=="8.9" set CUDA_ARCH=sm_89
    
    if "%CUDA_ARCH%"=="" (
        echo ⚠️  Unknown compute capability: %COMPUTE_CAP%
        echo → Using default: sm_80 (Ampere)
        set CUDA_ARCH=sm_80
    ) else (
        echo → Using CUDA architecture: %CUDA_ARCH%
    )
) else (
    echo ⚠️  nvidia-smi not found. Using default CUDA architecture.
    set CUDA_ARCH=sm_80
)

echo.
echo 🔧 Configuring Makefile...
echo ==========================

REM Create a simple Makefile for Windows
echo # Makefile for KLT CUDA Implementation > Makefile
echo # Auto-configured for University GPU Environment >> Makefile
echo. >> Makefile
echo # Compiler settings >> Makefile
echo CC = gcc >> Makefile
echo NVCC = nvcc >> Makefile
echo AR = ar >> Makefile
echo. >> Makefile
echo # CUDA architecture (auto-detected) >> Makefile
echo CUDA_ARCH = -arch=%CUDA_ARCH% >> Makefile
echo. >> Makefile
echo # Compiler flags >> Makefile
echo CFLAGS = -O3 -Wall -DNDEBUG >> Makefile
echo CUDAFLAGS = -O3 -std=c++11 $(CUDA_ARCH) -Xcompiler -fPIC >> Makefile
echo INCLUDES = -I./include >> Makefile
echo LIBRARIES = -lm >> Makefile
echo. >> Makefile
echo # Directories >> Makefile
echo SRC_DIR = src >> Makefile
echo BUILD_DIR = build >> Makefile
echo INCLUDE_DIR = include >> Makefile
echo INPUT_DIR = input >> Makefile
echo OUTPUT_DIR = output >> Makefile
echo. >> Makefile
echo # Source files >> Makefile
echo CPU_SOURCES = $(SRC_DIR)/convolve.c \ >> Makefile
echo               $(SRC_DIR)/error.c \ >> Makefile
echo               $(SRC_DIR)/pnmio.c \ >> Makefile
echo               $(SRC_DIR)/pyramid.c \ >> Makefile
echo               $(SRC_DIR)/selectGoodFeatures.c \ >> Makefile
echo               $(SRC_DIR)/storeFeatures.c \ >> Makefile
echo               $(SRC_DIR)/trackFeatures.c \ >> Makefile
echo               $(SRC_DIR)/klt.c \ >> Makefile
echo               $(SRC_DIR)/klt_util.c \ >> Makefile
echo               $(SRC_DIR)/writeFeatures.c >> Makefile
echo. >> Makefile
echo CUDA_SOURCES = $(SRC_DIR)/convolve_cuda.cu >> Makefile
echo. >> Makefile
echo # Object files >> Makefile
echo CPU_OBJECTS = $(CPU_SOURCES:$(SRC_DIR)/%%.c=$(BUILD_DIR)/%%.o) >> Makefile
echo. >> Makefile
echo # Targets >> Makefile
echo LIBRARY = $(BUILD_DIR)/libklt.a >> Makefile
echo CUDA_PROGRAM = convolve_cuda >> Makefile
echo EXAMPLES_BIN = example1 example2 example3 example4 example5 >> Makefile
echo. >> Makefile
echo # Default target >> Makefile
echo all: setup $(LIBRARY) $(CUDA_PROGRAM) $(EXAMPLES_BIN) >> Makefile
echo. >> Makefile
echo # Create necessary directories >> Makefile
echo setup: >> Makefile
echo 	@echo "Setting up build environment..." >> Makefile
echo 	@mkdir -p $(BUILD_DIR) >> Makefile
echo 	@mkdir -p $(OUTPUT_DIR) >> Makefile
echo 	@echo "✅ Directories created" >> Makefile
echo. >> Makefile
echo # Compile CPU source files >> Makefile
echo $(BUILD_DIR)/%%.o: $(SRC_DIR)/%%.c >> Makefile
echo 	@echo "Compiling $<..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ >> Makefile
echo. >> Makefile
echo # Create static library >> Makefile
echo $(LIBRARY): $(CPU_OBJECTS) >> Makefile
echo 	@echo "Creating static library..." >> Makefile
echo 	$(AR) rcs $@ $^ >> Makefile
echo 	@echo "✅ Library created: $@" >> Makefile
echo. >> Makefile
echo # Compile CUDA convolution program >> Makefile
echo $(CUDA_PROGRAM): $(SRC_DIR)/convolve_cuda.cu >> Makefile
echo 	@echo "Compiling CUDA convolution program..." >> Makefile
echo 	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) >> Makefile
echo 	@echo "✅ CUDA program created: $@" >> Makefile
echo. >> Makefile
echo # Compile example programs >> Makefile
echo example1: $(LIBRARY) >> Makefile
echo 	@echo "Compiling example1..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_DIR)/$@.c -L$(BUILD_DIR) -lklt $(LIBRARIES) >> Makefile
echo. >> Makefile
echo example2: $(LIBRARY) >> Makefile
echo 	@echo "Compiling example2..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_DIR)/$@.c -L$(BUILD_DIR) -lklt $(LIBRARIES) >> Makefile
echo. >> Makefile
echo example3: $(LIBRARY) >> Makefile
echo 	@echo "Compiling example3..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_DIR)/$@.c -L$(BUILD_DIR) -lklt $(LIBRARIES) >> Makefile
echo. >> Makefile
echo example4: $(LIBRARY) >> Makefile
echo 	@echo "Compiling example4..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_DIR)/$@.c -L$(BUILD_DIR) -lklt $(LIBRARIES) >> Makefile
echo. >> Makefile
echo example5: $(LIBRARY) >> Makefile
echo 	@echo "Compiling example5..." >> Makefile
echo 	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_DIR)/$@.c -L$(BUILD_DIR) -lklt $(LIBRARIES) >> Makefile
echo. >> Makefile
echo # Test CUDA program >> Makefile
echo test-cuda: $(CUDA_PROGRAM) >> Makefile
echo 	@echo "Testing CUDA convolution..." >> Makefile
echo 	@echo "==========================" >> Makefile
echo 	./$(CUDA_PROGRAM) >> Makefile
echo. >> Makefile
echo # Test complete KLT algorithm >> Makefile
echo test-klt: example3 >> Makefile
echo 	@echo "Testing complete KLT algorithm..." >> Makefile
echo 	@echo "================================" >> Makefile
echo 	./example3 >> Makefile
echo. >> Makefile
echo # Run performance comparison >> Makefile
echo benchmark: $(CUDA_PROGRAM) example3 >> Makefile
echo 	@echo "Running performance benchmark..." >> Makefile
echo 	@echo "==============================" >> Makefile
echo 	@echo "CUDA Convolution Test:" >> Makefile
echo 	@time ./$(CUDA_PROGRAM) >> Makefile
echo 	@echo "" >> Makefile
echo 	@echo "Complete KLT Algorithm:" >> Makefile
echo 	@time ./example3 >> Makefile
echo. >> Makefile
echo # Clean build files >> Makefile
echo clean: >> Makefile
echo 	@echo "Cleaning build files..." >> Makefile
echo 	rm -rf $(BUILD_DIR) >> Makefile
echo 	rm -f $(CUDA_PROGRAM) $(EXAMPLES_BIN) >> Makefile
echo 	rm -f $(OUTPUT_DIR)/*.ppm $(OUTPUT_DIR)/*.txt $(OUTPUT_DIR)/*.ft >> Makefile
echo 	@echo "✅ Cleaned" >> Makefile
echo. >> Makefile
echo # Show help >> Makefile
echo help: >> Makefile
echo 	@echo "KLT CUDA Makefile Help" >> Makefile
echo 	@echo "======================" >> Makefile
echo 	@echo "" >> Makefile
echo 	@echo "Available targets:" >> Makefile
echo 	@echo "  all          - Build everything (default)" >> Makefile
echo 	@echo "  $(CUDA_PROGRAM) - Build CUDA convolution program" >> Makefile
echo 	@echo "  example1-5   - Build individual example programs" >> Makefile
echo 	@echo "  test-cuda    - Test CUDA convolution" >> Makefile
echo 	@echo "  test-klt     - Test complete KLT algorithm" >> Makefile
echo 	@echo "  benchmark   - Run performance comparison" >> Makefile
echo 	@echo "  clean        - Remove build files" >> Makefile
echo 	@echo "  help         - Show this help" >> Makefile
echo. >> Makefile
echo .PHONY: all setup test-cuda test-klt benchmark clean help >> Makefile

echo ✅ Makefile configured for %CUDA_ARCH%
echo.
echo 🎯 Ready to build! Run:
echo   make all        # Build everything
echo   make test-cuda  # Test CUDA convolution
echo   make test-klt   # Test complete KLT algorithm
echo   make benchmark  # Run performance comparison
echo.
echo 📚 For more options, run: make help
echo.
echo Note: This script creates a basic Makefile for Windows.
echo For full functionality, use the Linux/Unix version with WSL or Cygwin.
