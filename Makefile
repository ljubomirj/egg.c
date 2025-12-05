.DEFAULT_GOAL := all

# Build type: debug (default) or release
BUILD ?= debug
BUILD_SUFFIX := .$(BUILD)

# Build directory (all objects and binaries go here)
BUILD_DIR ?= ./build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
override CC = clang
override CXX = clang++
BLOCKS_FLAGS := -fblocks
else ifeq ($(UNAME_S),Linux)
CC ?= gcc
CXX ?= g++
BLOCKS_FLAGS :=
else
CC ?= cc
CXX ?= c++
BLOCKS_FLAGS :=
endif

ifeq ($(BUILD),release)
OPTFLAGS := -O3 -DNDEBUG
else
OPTFLAGS := -O0 -g -DDEBUG
endif

# Generate and include dependency files so changes in headers trigger rebuilds
DEPFLAGS = -MMD -MP -MF $(BUILD_DIR)/$*.d
-include $(wildcard $(BUILD_DIR)/*.d)

EGG_SRC := full_trained_egg.c
EGG_CPU_MULTI_SRC := full_trained_egg-cpumulti.c
EGG_GPU_MULTI_SRC := full_trained_egg-gpumulti.c
EGG_GPU_METAL_SRC := full_trained_egg-gpu-metal.mm
EGG_GPU_METAL_OPTIMIZED_SRC := full_trained_egg-gpu-metal-optimized.mm
EGG_GPU_OPTIMIZED_SRC := full_trained_egg-gpu-optimized.c
EGG_GPU_METAL_OBJ := $(BUILD_DIR)/full_trained_egg-gpumulti-metal$(BUILD_SUFFIX).o
EGG_GPU_METAL_MM_OBJ := $(BUILD_DIR)/full_trained_egg-gpu-metal$(BUILD_SUFFIX).o
EGG_GPU_METAL_OPTIMIZED_OBJ := $(BUILD_DIR)/full_trained_egg-gpu-metal-optimized$(BUILD_SUFFIX).o
EGG_GPU_OPTIMIZED_OBJ := $(BUILD_DIR)/full_trained_egg-gpu-optimized$(BUILD_SUFFIX).o

.PHONY: all clean gpu-targets

all: $(BUILD_DIR)/egg$(BUILD_SUFFIX) $(BUILD_DIR)/egg-cpu-linux-amd64$(BUILD_SUFFIX) $(BUILD_DIR)/egg-cpu-macos-arm64$(BUILD_SUFFIX) $(BUILD_DIR)/egg-cpumulti$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-macos-metal$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-optimized$(BUILD_SUFFIX)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/egg$(BUILD_SUFFIX): $(EGG_SRC) | $(BUILD_DIR)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Building..."
	$(CC) $(OPTFLAGS) -std=c11 $(BLOCKS_FLAGS) $(DEPFLAGS) $(EGG_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS."
endif

$(BUILD_DIR)/egg-cpu-macos-arm64$(BUILD_SUFFIX): $(EGG_CPU_MULTI_SRC) | $(BUILD_DIR)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Building..."
	$(CC) $(OPTFLAGS) -DEGG_FORCE_NEON $(BLOCKS_FLAGS) $(DEPFLAGS) $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS, while this is $(UNAME_S)"
endif

$(BUILD_DIR)/egg-cpu-linux-amd64$(BUILD_SUFFIX): $(EGG_CPU_MULTI_SRC) | $(BUILD_DIR)
ifeq ($(UNAME_S),Linux)
	@echo "==> $@ Building..."
	$(CXX) $(OPTFLAGS) -mavx2 -mfma -fopenmp -DEGG_FORCE_AVX2 $(BLOCKS_FLAGS) $(DEPFLAGS) $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on Linux."
endif

$(BUILD_DIR)/egg-cpumulti$(BUILD_SUFFIX): $(EGG_CPU_MULTI_SRC) | $(BUILD_DIR)
	@echo "==> $@ Portable CPU build building..."
	$(CC) $(OPTFLAGS) $(BLOCKS_FLAGS) $(DEPFLAGS) $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"

GPU_STUB_FLAGS := $(OPTFLAGS) $(EGG_GPU_MULTI_SRC)

$(BUILD_DIR)/egg-gpu-macos-metal$(BUILD_SUFFIX): $(EGG_GPU_MULTI_SRC) $(EGG_GPU_METAL_SRC) | $(BUILD_DIR)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Metal training build building..."
	$(CC) $(OPTFLAGS) $(BLOCKS_FLAGS) $(DEPFLAGS) -c -DEGG_BUILD_METAL -DEGG_USE_METAL \
		$(EGG_GPU_MULTI_SRC) -o $(EGG_GPU_METAL_OBJ)
	$(CXX) $(OPTFLAGS) $(DEPFLAGS) -fobjc-arc -std=c++17 -c $(EGG_GPU_METAL_SRC) -o $(EGG_GPU_METAL_MM_OBJ)
	$(CXX) $(OPTFLAGS) -fobjc-arc -std=c++17 \
		$(EGG_GPU_METAL_OBJ) $(EGG_GPU_METAL_MM_OBJ) \
		-framework Metal -framework Foundation -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS, while this is $(UNAME_S)"
endif

$(BUILD_DIR)/egg-gpu-linux-rocm$(BUILD_SUFFIX): $(EGG_GPU_MULTI_SRC) | $(BUILD_DIR)
	@echo "==> $@ ROCm stub building..."
	$(CC) $(GPU_STUB_FLAGS) $(DEPFLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_ROCM -o $@
	@echo "[$@ done]"

$(BUILD_DIR)/egg-gpu-linux-cuda$(BUILD_SUFFIX): $(EGG_GPU_MULTI_SRC) | $(BUILD_DIR)
	@echo "==> $@ CUDA stub building..."
	$(CC) $(GPU_STUB_FLAGS) $(DEPFLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_CUDA -o $@
	@echo "[$@ done]"

$(BUILD_DIR)/egg-gpu-linux-vulcan$(BUILD_SUFFIX): $(EGG_GPU_MULTI_SRC) | $(BUILD_DIR)
	@echo "==> $@ Vulkan stub building..."
	$(CC) $(GPU_STUB_FLAGS) $(DEPFLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_VULKAN -o $@
	@echo "[$@ done]"

$(BUILD_DIR)/egg-gpu-optimized$(BUILD_SUFFIX): $(EGG_GPU_OPTIMIZED_SRC) $(EGG_GPU_METAL_OPTIMIZED_SRC) | $(BUILD_DIR)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Optimized Metal GPU build building..."
	$(CC) $(OPTFLAGS) $(BLOCKS_FLAGS) $(DEPFLAGS) -c $(EGG_GPU_OPTIMIZED_SRC) -o $(EGG_GPU_OPTIMIZED_OBJ)
	$(CXX) $(OPTFLAGS) $(DEPFLAGS) -fobjc-arc -std=c++17 -c $(EGG_GPU_METAL_OPTIMIZED_SRC) -o $(EGG_GPU_METAL_OPTIMIZED_OBJ)
	$(CXX) $(OPTFLAGS) -fobjc-arc -std=c++17 \
		$(EGG_GPU_OPTIMIZED_OBJ) $(EGG_GPU_METAL_OPTIMIZED_OBJ) \
		-framework Metal -framework Foundation -framework CoreFoundation -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS, while this is $(UNAME_S)"
endif

gpu-targets: $(BUILD_DIR)/egg-gpu-macos-metal$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-linux-rocm$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-linux-cuda$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-linux-vulcan$(BUILD_SUFFIX) $(BUILD_DIR)/egg-gpu-optimized$(BUILD_SUFFIX)

clean:
	rm -rf $(BUILD_DIR)
