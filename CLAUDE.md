# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EGGROLL in C is a minimalist, dependency-free implementation of the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) algorithm in pure C. It demonstrates **integer-only training** of a language model using Evolution Strategies with low-rank perturbations instead of backpropagation.

## Build Commands

All binaries and object files are built in the `./build` directory to keep the source tree clean.

### Quick Build (macOS only)
```bash
make                           # Builds all supported targets for current platform
```

### Individual Targets

**CPU Targets:**
```bash
make                           # Builds: build/egg.debug (or .release with BUILD=release)
                              #         build/egg-cpu-macos-arm64.debug
                              #         build/egg-cpu-linux-amd64.debug
                              #         build/egg-cpumulti.debug
```

**GPU Targets:**
```bash
make                           # Also builds: build/egg-gpu-macos-metal.debug
```

**Build Types:**
```bash
make BUILD=debug              # Debug build (default) - binaries have .debug suffix
make BUILD=release            # Release build - binaries have .release suffix
```

**Cleanup:**
```bash
make clean                    # Remove entire build/ directory
```

### Manual Compilation Examples

**macOS ARM64 (NEON):**
```bash
clang -O3 -DEGG_FORCE_NEON full_trained_egg-cpumulti.c -lm -o egg-cpu-macos-arm64
```

**Linux x86_64 (AVX2 + OpenMP):**
```bash
g++ -O3 -mavx2 -mfma -fopenmp -DEGG_FORCE_AVX2 full_trained_egg-cpumulti.c -lm -o egg-cpu-linux-amd64
```

**Portable (scalar fallback):**
```bash
clang -O3 full_trained_egg-cpumulti.c -lm -o egg-cpumulti
```

## Running Training

```bash
./build/egg-cpu-macos-arm64.release      # Best performance on Apple Silicon
./build/egg-cpu-linux-amd64.release      # Best performance on Linux x86_64
./build/egg-gpu-macos-metal.release      # Metal GPU (optimized - should be competitive with CPU)
```

**Note:** The GPU Metal version now uses a kernel that computes xB on GPU, with only one sync per timestep. Performance should be competitive with CPU versions.

### Environment Variables

```bash
EGG_DISABLE_GPU=1 ./build/egg-gpu-macos-metal.release   # Force CPU path even in GPU build
```

## Architecture

### Key Components

**Entry Points:**
- `full_trained_egg.c` - Original reference implementation (macOS only, ARM NEON)
- `full_trained_egg-cpumulti.c` - Multi-arch CPU implementation with SIMD abstraction
- `full_trained_egg-gpumulti.c` - GPU multi-platform entry point
- `full_trained_egg-gpu-metal.mm` - Metal runtime implementation (Objective-C++)

**Configuration:**
- `egg_config.h` - Hyperparameters (HIDDEN_DIM, N_LAYERS, SEQ_LEN, POPULATION_SIZE, etc.)

**GPU Headers:**
- `egg_gpu_metal.h` - Metal C API declarations

### SIMD Abstraction Layer

The multi-arch CPU implementation (`full_trained_egg-cpumulti.c`) uses preprocessor flags to select SIMD backend:

- `EGG_SIMD_IMPL=1` → ARM NEON (`arm_neon.h`)
- `EGG_SIMD_IMPL=2` → x86 AVX2 (`immintrin.h`)
- `EGG_SIMD_IMPL=0` → Scalar fallback (portable)

**Auto-detection hierarchy:**
1. Explicit flags: `EGG_FORCE_NEON`, `EGG_FORCE_AVX2`, `EGG_FORCE_SCALAR`
2. Compiler defines: `__ARM_NEON`, `__AVX2__`
3. Fallback: scalar

### Parallelization

**macOS:** Uses Grand Central Dispatch (`dispatch_apply`)
**Linux:** Uses OpenMP (`#pragma omp parallel for`) when compiled with `-fopenmp`
**Fallback:** Single-threaded loop

The population loop evaluates pairs of perturbations in parallel across available CPU cores.

### Data Loading

**Compressed input (preferred):**
- Checks for `input.txt.zst` first
- Uses `popen("zstd -dc ...")` to stream decompression
- Full dataset loaded into memory once at startup

**Fallback to plain text:**
- If no `.zst` file exists, loads `input.txt` directly

**Memory layout:** Entire dataset resides in RAM; workers index into shared buffer (no further file I/O during training).

### Model Structure

```c
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];  // Wz, Wr, Wh, Uh
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM];                // bf, bh
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM*4)]; // Expand, Project
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];                // LN1, LN2
    int8_t ln_out[HIDDEN_DIM];
} EggModel;
```

### Training Loop

**Evolution Strategy with low-rank perturbations:**

1. Generate rank-1 noise: `A[i]` and `B[i]` for each population member
2. For each perturbation pair `(+ΔW, -ΔW)` where `ΔW = A ⊗ B`:
   - Run forward pass on different data streams
   - Compute losses `L+` and `L-`
   - Record fitness: `+1` if `L+ < L-`, `-1` if `L- < L+`, `0` if equal
3. Update weights: for each `W[r,c]`, compute vote = `Σ fitness[i] * A[r,i] * B[c,i]`
   - If `vote > UPDATE_THRESHOLD` and `W < MAX_VAL`: increment
   - If `vote < -UPDATE_THRESHOLD` and `W > MIN_VAL`: decrement

**No backpropagation, no gradient computation.** All math is `int8` weights/activations with `int32` accumulation.

### Metal GPU Backend

**Current status:** Functional but significantly slower than CPU (~5 min/step vs ~10 sec/step on M2 Max).

**Implemented:**
- GPU-resident model weights (single Metal buffer)
- Batched matmul operations (`egg_matmul_perturbed_kernel`)
- Batched weight updates (`egg_update_matrix_kernel`)
- Batch API to reduce synchronization overhead

**Bottlenecks:**
- Many small kernel launches (hundreds per forward pass)
- CPU↔GPU transfers for intermediate activations
- Element-wise operations (layer norm, GRU gates) still on CPU
- GPU kernel launch overhead dominates execution time

**Why CPU is faster:**
- NEON SIMD processes operations efficiently in single pass
- No GPU overhead or memory transfers
- Well-suited for this workload pattern

**Recommendation:** Use `egg-cpu-macos-arm64` for training on Apple Silicon. The Metal path is experimental and requires significant restructuring (kernel fusion, larger workgroups, operation batching, GPU-resident activations) to be competitive.

## Configuration

Edit `egg_config.h` to modify hyperparameters:

```c
#define VOCAB_SIZE 256        // Byte-level tokenization
#define HIDDEN_DIM 512        // Model width
#define N_LAYERS 4            // Number of layers
#define SEQ_LEN 256           // Sequence length (reduce for faster experiments)
#define POPULATION_SIZE 32    // Number of perturbations (reduce for faster experiments)
#define UPDATE_THRESHOLD 160  // Votes needed to flip a weight
```

**For quick experiments:**
- Reduce `SEQ_LEN` (e.g., 64, 128, 256)
- Reduce `POPULATION_SIZE` (e.g., 16, 32)
- Use smaller dataset (e.g., `wikitext_combined-head-35111.txt.zst`)

## Git Setup

This repository has a nested git structure:
- `~/LJ-RL-Reinforcement_Learning/.git` - Private local repository
- `~/LJ-RL-Reinforcement_Learning/egg.c/.git` - Public GitHub repository

**GitHub remotes (SSH):**
```bash
origin     git@github.com:ljubomirj/egg.c.git      # Personal fork
origin-org git@github.com:d0rc/egg.c.git           # Original upstream
```

## Code Invariants

1. **Keep `full_trained_egg.c` as reference:** Never modify the original implementation; it serves as the "gold standard" for correctness validation.

2. **SIMD abstraction:** When adding CPU optimizations to `full_trained_egg-cpumulti.c`, provide implementations for all three backends (NEON, AVX2, scalar) or use portable code.

3. **Integer-only arithmetic:** All training math uses `int8` weights/activations with `int32` accumulation. No floating-point operations in the training loop.

4. **Reproducible RNG:** Noise generation uses deterministic seeds (`step_seed + p_idx`) for reproducibility across platforms.

5. **Metal compilation:** The Metal backend uses Objective-C++ (`.mm` files) linked with C code via `extern "C"` declarations.

## Testing and Validation

**Compare CPU variants:**
```bash
# Run both versions on same small dataset
./egg-cpu-macos-arm64
./egg-cpumulti
# Loss curves should be qualitatively similar (not bit-identical due to SIMD differences)
```

**Verify GPU correctness:**
```bash
# Compare GPU vs CPU on tiny dataset
./egg-cpu-macos-arm64  # Reference
EGG_DISABLE_GPU=0 ./egg-gpu-macos-metal  # GPU path
# Check that losses trend similarly
```

## References

- **Original JAX Implementation:** [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg)
- **Paper & Project:** [EGGROLL Website](https://eshyperscale.github.io/)
