######################################################################
# Compiler Setup
CC      = gcc
NVCC    = nvcc

######################################################################
# Directories
SRC_CORE      = src/core
SRC_FEATURES  = src/features
SRC_IO        = src/io
INCLUDE_DIR   = include
EXAMPLES_DIR  = examples
BUILD_DIR     = build
TOOLS_DIR     = tools
DATA_DIR      = data/new/frames/frames
OUTPUT_DIR    = output
PROFILE_DIR   = profiles

OUTPUT_CPU    = $(OUTPUT_DIR)/cpu
OUTPUT_GPU    = $(OUTPUT_DIR)/gpu
FRAMES_CPU    = $(OUTPUT_CPU)/frames
FRAMES_GPU    = $(OUTPUT_GPU)/frames


######################################################################
# Flags
ARCH          = sm_86
FLAG1         = -DNDEBUG
CFLAGS        = $(FLAG1) -I$(INCLUDE_DIR)
GPUFLAGS      = -Xcompiler "-fPIC" -I$(INCLUDE_DIR) -arch=$(ARCH)

LIB           = -L/usr/local/lib -L/usr/lib

# CPU object files
OBJS_CPU      = $(BUILD_DIR)/convolve.o $(BUILD_DIR)/pyramid.o \
                 $(BUILD_DIR)/klt.o $(BUILD_DIR)/klt_util.o \
                 $(BUILD_DIR)/selectGoodFeatures.o $(BUILD_DIR)/storeFeatures.o \
                 $(BUILD_DIR)/trackFeatures.o $(BUILD_DIR)/writeFeatures.o \
                 $(BUILD_DIR)/error.o $(BUILD_DIR)/pnmio.o

# GPU object files  
OBJS_GPU      = $(BUILD_DIR)/convolve_cuda.o $(BUILD_DIR)/pyramid.o \
                 $(BUILD_DIR)/klt.o $(BUILD_DIR)/klt_util.o \
                 $(BUILD_DIR)/selectGoodFeatures.o $(BUILD_DIR)/storeFeatures.o \
                 $(BUILD_DIR)/trackFeatures.o $(BUILD_DIR)/writeFeatures.o \
                 $(BUILD_DIR)/error.o $(BUILD_DIR)/pnmio.o

######################################################################
# Default build
all: $(BUILD_DIR) $(OUTPUT_CPU) $(OUTPUT_GPU) $(PROFILE_DIR) lib cpu gpu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUTPUT_CPU):
	mkdir -p $(FRAMES_CPU)

$(OUTPUT_GPU):
	mkdir -p $(FRAMES_GPU)

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)/cpu $(PROFILE_DIR)/gpu

######################################################################
# Compile object files
$(BUILD_DIR)/%.o: $(SRC_CORE)/%.c
	$(CC) -c $(CFLAGS) $< -o $@
$(BUILD_DIR)/%.o: $(SRC_CORE)/%.cu
	$(NVCC) -c $(GPUFLAGS) $< -o $@
$(BUILD_DIR)/%.o: $(SRC_FEATURES)/%.c
	$(CC) -c $(CFLAGS) $< -o $@
$(BUILD_DIR)/%.o: $(SRC_IO)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

# Specific rules for GPU build
$(BUILD_DIR)/convolve_cuda.o: $(SRC_CORE)/convolve_cuda.cu
	$(NVCC) -c $(GPUFLAGS) $< -o $@

######################################################################
# Build library
lib: lib-cpu

lib-cpu:
	@mkdir -p $(BUILD_DIR)
	$(MAKE) $(OBJS_CPU)
	rm -f libklt.a
	ar ruv libklt.a $(OBJS_CPU)
	@echo "✅ CPU Library built: libklt.a"

lib-gpu:
	@mkdir -p $(BUILD_DIR)
	$(MAKE) $(OBJS_GPU)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(OBJS_GPU)
	@echo "✅ GPU Library built: libklt_gpu.a"

######################################################################
# CPU build & run
cpu: lib-cpu $(OUTPUT_CPU)
	@echo "🚀 Building CPU version..."
	$(CC) -O3 $(CFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_CPU)/"' \
		-o main_cpu $(EXAMPLES_DIR)/main_cpu.c -L. -lklt $(LIB) -lm
	@echo "✅ Running CPU version..."
	./main_cpu
	@echo "🎬 Creating CPU video..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		ffmpeg -y -framerate 30 -pattern_type glob -i "$(FRAMES_CPU)/feat*.ppm" \
			-c:v libx264 -pix_fmt yuv420p $(OUTPUT_CPU)/video.mp4; \
		echo "🎞️ CPU video created at $(OUTPUT_CPU)/video.mp4"; \
	else \
		echo "⚠️  ffmpeg not found - skipping video creation"; \
		echo "📁 CPU frames saved in $(FRAMES_CPU)/"; \
	fi

######################################################################
# GPU build & run
gpu: lib-gpu $(OUTPUT_GPU)
	@echo "⚡ Building GPU version..."
	$(NVCC) -O3 $(GPUFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_GPU)/"' \
		-o main_gpu $(EXAMPLES_DIR)/main_gpu.c -L. -lklt_gpu $(LIB) -lm
	@echo "✅ Running GPU version..."
	./main_gpu
	@echo "🎬 Creating GPU video..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		ffmpeg -y -framerate 30 -pattern_type glob -i "$(FRAMES_GPU)/feat*.ppm" \
			-c:v libx264 -pix_fmt yuv420p $(OUTPUT_GPU)/video.mp4; \
		echo "🎞️ GPU video created at $(OUTPUT_GPU)/video.mp4"; \
	else \
		echo "⚠️  ffmpeg not found - skipping video creation"; \
		echo "📁 GPU frames saved in $(FRAMES_GPU)/"; \
	fi

######################################################################
# Compare both
compare: cpu gpu
	@echo "🆚 Comparison done — CPU and GPU outputs ready!"
	@echo "📂 CPU: $(OUTPUT_CPU)/video.mp4"
	@echo "📂 GPU: $(OUTPUT_GPU)/video.mp4"

######################################################################
# Profiling CPU
cpu-profile: clean lib-cpu $(OUTPUT_CPU)
	@echo "📊 Profiling CPU version..."
	$(CC) -pg -O3 $(CFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_CPU)/"' \
		-o main_cpu $(EXAMPLES_DIR)/main_cpu.c -L. -lklt $(LIB) -lm
	./main_cpu
	$(eval PROFILE_TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	$(eval CPU_PROF_DIR := $(PROFILE_DIR)/cpu/test_$(PROFILE_TIMESTAMP))
	mkdir -p $(CPU_PROF_DIR)
	mv gmon.out $(CPU_PROF_DIR)/
	gprof ./main_cpu $(CPU_PROF_DIR)/gmon.out > $(CPU_PROF_DIR)/profile.txt
	gprof ./main_cpu $(CPU_PROF_DIR)/gmon.out | python3 $(TOOLS_DIR)/gprof2dot.py -s -o $(CPU_PROF_DIR)/profile.dot
	dot -Tpdf $(CPU_PROF_DIR)/profile.dot -o $(CPU_PROF_DIR)/profile.pdf
	@echo "✅ CPU profiling complete: $(CPU_PROF_DIR)/profile.pdf"

######################################################################
# Profiling GPU
gpu-profile: clean lib-gpu $(OUTPUT_GPU)
	@echo "📊 Profiling GPU version..."
	$(NVCC) -pg -O3 $(GPUFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_GPU)/"' \
		-o main_gpu $(EXAMPLES_DIR)/main_gpu.c -L. -lklt_gpu $(LIB) -lm
	./main_gpu
	$(eval PROFILE_TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	$(eval GPU_PROF_DIR := $(PROFILE_DIR)/gpu/test_$(PROFILE_TIMESTAMP))
	mkdir -p $(GPU_PROF_DIR)
	mv gmon.out $(GPU_PROF_DIR)/
	gprof ./main_gpu $(GPU_PROF_DIR)/gmon.out > $(GPU_PROF_DIR)/profile.txt
	gprof ./main_gpu $(GPU_PROF_DIR)/gmon.out | python3 $(TOOLS_DIR)/gprof2dot.py -s -o $(GPU_PROF_DIR)/profile.dot
	dot -Tpdf $(GPU_PROF_DIR)/profile.dot -o $(GPU_PROF_DIR)/profile.pdf
	@echo "✅ GPU profiling complete: $(GPU_PROF_DIR)/profile.pdf"

######################################################################
# Cleaning
clean:
	@echo "🧹 Cleaning build files and outputs..."
	rm -f $(BUILD_DIR)/*.o *.a main_cpu main_gpu *.tar *.tar.gz libklt.a libklt_gpu.a \
	      feat*.ppm features.ft features.txt
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "✓ Cleaned build and output directories."

clean-all:
	@echo "🔥 Deep cleaning everything..."
	rm -f $(BUILD_DIR)/*.o *.a main_cpu main_gpu *.tar *.tar.gz libklt.a libklt_gpu.a \
	      feat*.ppm features.ft features.txt
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR) $(PROFILE_DIR)
	@echo "✓ All cleaned up — fresh start!"

######################################################################
# Help
help:
	@echo "========================================================="
	@echo "KLT Feature Tracker Makefile"
	@echo "---------------------------------------------------------"
	@echo "make cpu           → Build and run CPU version"
	@echo "make gpu           → Build and run GPU version"
	@echo "make compare       → Run both CPU + GPU"
	@echo "make cpu-profile   → Profile CPU version"
	@echo "make gpu-profile   → Profile GPU version"
	@echo "make clean         → Clean build and outputs"
	@echo "make clean-all     → Clean everything (incl. profiles)"
	@echo "========================================================="
