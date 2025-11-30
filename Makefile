UNAME_S := $(shell uname -s)
CC ?= cc
CXX ?= c++

EGG_SRC := full_trained_egg.c
EGG_CPU_MULTI_SRC := full_trained_egg-cpumulti.c
EGG_GPU_MULTI_SRC := full_trained_egg-gpumulti.c
EGG_GPU_METAL_SRC := full_trained_egg-gpu-metal.mm
EGG_GPU_METAL_OBJ := full_trained_egg-gpumulti-metal.o
EGG_GPU_METAL_MM_OBJ := full_trained_egg-gpu-metal.o

.PHONY: all egg egg-linux-amd64 egg-macos-arm64 egg-cpumulti \
	egg-gpumetal egg-gpurocm egg-gpucuda \
	egg-gpuvulcan clean gpu-targets

all: egg egg-linux-amd64 egg-macos-arm64 egg-cpumulti egg-gpumetal 

egg: $(EGG_SRC)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Building..."
	$(CC) -O3 -std=c11 $(EGG_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS."
endif

egg-macos-arm64: $(EGG_CPU_MULTI_SRC)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Building..."
	$(CC) -O3 -DEGG_FORCE_NEON $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS, while this is $(UNAME_S)"
endif

egg-linux-amd64: $(EGG_CPU_MULTI_SRC)
ifeq ($(UNAME_S),Linux)
	@echo "==> $@ Building..."
	$(CXX) -O3 -mavx2 -mfma -fopenmp -DEGG_FORCE_AVX2 $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on Linux."
endif

egg-cpumulti: $(EGG_CPU_MULTI_SRC)
	@echo "==> $@ Portable CPU build building..."
	$(CC) -O3 $(EGG_CPU_MULTI_SRC) -lm -o $@
	@echo "[$@ done]"

GPU_STUB_FLAGS := -O2 $(EGG_GPU_MULTI_SRC)

egg-gpumetal: $(EGG_GPU_MULTI_SRC) $(EGG_GPU_METAL_SRC)
ifeq ($(UNAME_S),Darwin)
	@echo "==> $@ Metal training build building..."
	$(CC) -O2 -c -DEGG_BUILD_METAL \
		$(EGG_GPU_MULTI_SRC) -o $(EGG_GPU_METAL_OBJ)
	$(CXX) -O2 -fobjc-arc -std=c++17 -c $(EGG_GPU_METAL_SRC) -o $(EGG_GPU_METAL_MM_OBJ)
	$(CXX) -O2 -fobjc-arc -std=c++17 \
		$(EGG_GPU_METAL_OBJ) $(EGG_GPU_METAL_MM_OBJ) \
		-framework Metal -framework Foundation -o $@
	@echo "[$@ done]"
else
	@echo "Target $@ is only supported on macOS, while this is $(UNAME_S)"
endif

egg-gpurocm: $(EGG_GPU_MULTI_SRC)
	@echo "==> $@ ROCm stub building..."
	$(CC) $(GPU_STUB_FLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_ROCM -o $@
	@echo "[$@ done]"

egg-gpucuda: $(EGG_GPU_MULTI_SRC)
	@echo "==> $@ CUDA stub building..."
	$(CC) $(GPU_STUB_FLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_CUDA -o $@
	@echo "[$@ done]"

egg-gpuvulcan: $(EGG_GPU_MULTI_SRC)
	@echo "==> $@ Vulkan stub building..."
	$(CC) $(GPU_STUB_FLAGS) -DEGG_GPU_BACKEND=EGG_GPU_BACKEND_VULKAN -o $@
	@echo "[$@ done]"

gpu-targets: egg-gpumetal egg-gpurocm egg-gpucuda egg-gpuvulcan

clean:
	rm -f egg egg-linux-amd64 egg-macos-arm64 egg-cpumulti egg-gpumetal \
		egg-gpurocm egg-gpucuda egg-gpuvulcan \
		$(EGG_GPU_METAL_OBJ) $(EGG_GPU_METAL_MM_OBJ)

