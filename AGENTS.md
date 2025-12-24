# Repository Guidelines

## Project history

This repo is a scratchpad for personal educating by hacking trial & error of myself Ljubomir - LJ, all based based on and using the original egg.c and the original EGG and nano-egg repos.

git@github.com:ljubomirj/egg.c.git

This repo is a clone of the egg.c official repo

git@github.com:d0rc/egg.c.git

That repository implements egg - the ML training algorithm as descibed in

Evolution Strategies at the Hyperscale
https://eshyperscale.github.io/
https://www.alphaxiv.org/abs/2511.16652
https://github.com/ESHyperscale/HyperscaleES
https://github.com/ESHyperscale/nano-egg

## Project Structure & Key Files
- `full_trained_egg.c`: original single-file ARM NEON CPU reference (GRU-based RNN).
- `full_trained_egg-multi.c`: legacy multi-arch CPU (NEON/AVX2/scalar) with in-file config; GRU RNN.
- `full_trained_egg-cpumulti.c`: portable CPU path with SIMD autodetect and zstd-aware dataset loading; main place for CPU changes.
- `full_trained_egg-gpumulti.c`: GPU-capable entry point (Metal when enabled) with CPU fallback/parity hooks; GRU RNN.
- `full_trained_egg-gpu-metal.mm`: Metal backend for gpumulti (kernels, buffers, fused GRU/MLP/head).
- `full_trained_egg-gpu-optimized.c`: experimental Metal wrapper; uses optimized pair-eval kernels when `EGG_USE_OPTIMIZED_METAL=1`, else CPU.
- `full_trained_egg-gpu-metal-optimized.mm`: optimized Metal pair-at-a-time implementation (fused GRU; ignores POPULATION_SIZE).
- `full_cuda_train_egg.cu`: CUDA/HIP GRU RNN training (ES) with Thrust/CUB; HIP shims for AMD.
- `full_cuda_train_egg_transformer.cu`: CUDA transformer ES trainer with pos embeddings and checkpointing.
- `full_cuda_train_egg_transformer_adam.cu`: CUDA/HIP transformer with AdamW, RoPE, extra bias/MLP init, checkpoint/resume.
- `full_cuda_train_transformer_adam_mgpu.cu`: multi-GPU transformer Adam variant with extra compile-time flags (NTT embeddings, tokenizer, logging).
- `d-eggs/`: distributed transformer training (coordinator/worker), custom protocol, chunked population, ternary packing; has its own `Makefile` and `include/config.h`.
- `egg_config.h`: hyperparameters for the C GRU CPU/GPU paths; supports tiny debug overrides via `-DEGG_TINY_DEBUG`.
- Binaries live in repo root; build suffix indicates configuration (see below).

## Build, Run, and Debug
- Default build is debug: `make` or `make BUILD=debug` → targets like `egg-gpu-macos-metal.debug`.
- Release build: `make BUILD=release` → `*.release` binaries with `-DNDEBUG`.
- Tiny parity config (fast): `make BUILD=debug CFLAGS_EXTRA=-DEGG_TINY_DEBUG`.
- Helpful env flags at runtime:
  - `EGG_COMPARE_CPU=1` compares GPU vs CPU logits/loss.
  - `EGG_DISABLE_GPU_UPDATE=1` runs GPU forward but CPU weight updates.
  - `EGG_DISABLE_GPU_GRU=1` forces CPU GRU path.
  - `EGG_PARITY_TRACE=1` prints per-layer diff snapshots (debug builds).
  - `EGG_DISABLE_GPU=1` forces CPU even on GPU binaries.

## Coding Style & Naming
- Target boring, readable C/C++; favor clarity over cleverness.
- Use `const &` for read-only heavy params; use pointers for outputs/IO.
- Member vars prefixed `m`, inputs `i`, outputs `o`, IO `io`; public APIs UpperCamelCase, private snake_case.
- Pointers as `T *p`; references `T & v`.
- Keep integer-only math in training path; avoid floats in kernels/updates.
- Maintain SIMD parity: if changing CPU kernels, cover NEON, AVX2, and scalar.

## Testing & Parity Checks
- No formal test suite; rely on parity runs:
  - CPU vs GPU: `EGG_COMPARE_CPU=1 ./egg-gpu-macos-metal.debug`.
  - Tiny config for quick repro: add `CFLAGS_EXTRA=-DEGG_TINY_DEBUG`.
- For divergence hunts, combine `EGG_PARITY_TRACE=1` and `EGG_DISABLE_GPU_UPDATE=1`.

## Commit & PR Practice
- Commits: short imperative subject (e.g., “Fix Metal layer norm scaling”); keep logical changes grouped.
- Before PR: note build tested (`BUILD=debug|release`), key flags used, and datasets/seeds if relevant.
- Include parity evidence (log snippets) when touching GPU or SIMD code; mention any new env flags.

## Security & Config Tips
- Inputs are local text/zst files; no network fetch in training loop.
- Avoid writing to arbitrary paths; temp artifacts live in `/tmp/egg_*`.
- Do not write do not change README.LJ - it's owner-maintained personal notes by myself, LJ. Do read though to see the history of changes.

## Overview

EGGROLL in C is a minimalist, dependency-free implementation of the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) algorithm in pure C. It demonstrates **integer-only training** of a language model using Evolution Strategies with low-rank perturbations instead of backpropagation.

## Build Commands

### Quick Build (macOS only)
```bash
make                           # Builds all supported targets for current platform
```

### Individual Targets

**CPU Targets:**
```bash
make egg                       # Original single-binary CPU (macOS only)
make egg-cpu-macos-arm64      # Multi-arch NEON-optimized (macOS ARM64)
make egg-cpu-linux-amd64      # Multi-arch AVX2-optimized with OpenMP (Linux x86_64)
make egg-cpumulti             # Portable fallback (any platform, scalar)
```

**GPU Targets:**
```bash
make egg-gpu-macos-metal      # Metal backend (macOS ARM64)
make egg-gpu-linux-rocm       # ROCm stub (AMD GPU, Linux)
make egg-gpu-linux-cuda       # CUDA stub (NVIDIA GPU, Linux)
make egg-gpu-linux-vulcan     # Vulkan stub (cross-platform)
make egg-train-gpu-linux-rocm # HIP/ROCm GRU trainer (full_cuda_train_egg.cu)
make egg-train-transformer-gpu-linux-rocm # HIP/ROCm transformer Adam trainer
```

**Distributed (`d-eggs`) targets:**
```bash
make -C d-eggs                # coordinator, worker, print_arch
```

**Cleanup:**
```bash
make clean                     # Remove all binaries and intermediate objects
```

## Running Training

```bash
./egg-cpu-macos-arm64.release  # Best performance on Apple Silicon
./egg-cpu-linux-amd64.release  # Best performance on Linux x86_64
./egg-gpu-macos-metal.debug    # Metal GPU (currently slower than CPU), buggy
```

**Distributed (`d-eggs`) training:**
```bash
./d-eggs/coordinator --save-dir ./checkpoints
./d-eggs/worker 127.0.0.1
```

### Environment Variables

```bash
EGG_DISABLE_GPU=1 ./egg-gpu-macos-metal   # Force CPU path even in GPU build
EGG_USE_OPTIMIZED_METAL=1 ./build/egg-gpu-macos-metal-optimized.debug
```

## Architecture

### Key Components

**Entry Points:**
- `full_trained_egg.c` - Original ARM NEON single-file reference (GRU RNN).
- `full_trained_egg-multi.c` - Legacy CPU multi-arch (NEON/AVX2/scalar).
- `full_trained_egg-cpumulti.c` - Current CPU SIMD multi-arch (uses `egg_config.h`).
- `full_trained_egg-gpumulti.c` - GPU-capable entry (Metal when enabled).
- `full_trained_egg-gpu-metal.mm` - Metal runtime implementation (Objective-C++).
- `full_trained_egg-gpu-optimized.c` + `full_trained_egg-gpu-metal-optimized.mm` - Experimental optimized Metal pair-eval path.
- `full_cuda_train_egg.cu` - CUDA/HIP GRU trainer.
- `full_cuda_train_egg_transformer.cu` - CUDA transformer ES trainer.
- `full_cuda_train_egg_transformer_adam.cu` - CUDA/HIP transformer AdamW + RoPE trainer.
- `full_cuda_train_transformer_adam_mgpu.cu` - Multi-GPU transformer Adam variant.
- `d-eggs/src/coordinator.cpp` + `d-eggs/src/worker.cu` + `d-eggs/src/kernels.cu` - Distributed transformer training (coordinator/worker).

**Configuration:**
- `egg_config.h` - Hyperparameters for GRU CPU/GPU paths (HIDDEN_DIM, N_LAYERS, SEQ_LEN, POPULATION_SIZE, etc.).
- `d-eggs/include/config.h` - Distributed transformer configuration (HIDDEN_DIM, N_LAYERS, POPULATION_SIZE/CHUNK_SIZE, sampling, etc.).

**GPU Headers:**
- `egg_gpu_metal.h` - Metal C API declarations
- `egg_gpu_metal_optimized.h` - Optimized Metal pair-eval API used by the experimental path

### Model Variants
- GRU-based RNN: `full_trained_egg*.c`, `full_trained_egg-gpumulti.c`, `full_cuda_train_egg.cu`.
- Transformer ES (CUDA): `full_cuda_train_egg_transformer.cu`.
- Transformer AdamW + RoPE (CUDA/HIP): `full_cuda_train_egg_transformer_adam.cu`.
- Transformer Adam multi-GPU (CUDA): `full_cuda_train_transformer_adam_mgpu.cu`.
- Distributed transformer (CUDA): `d-eggs/` (coordinator/worker + kernels).
- RWKV/RWK.V: no implementation found in current tree; all RNN paths here are GRU.

**GRU RNN Details (baseline egg):**
- Single gated GRU variant: one gate `ft` drives both reset and update (no separate reset gate).
- Gate scaling is integer: `gate = (ft + 127) / 256`, then `gated_past = gate * h`.
- Candidate `ht = W2*x + W3*gated_past + b1`, state update `h = h + gate * (ht - h)`.
- No explicit sigmoid/tanh; values are clipped to int8 range.
- Per-layer L1-style layer norm (`sum(abs(x))/HIDDEN_DIM`) before GRU; residual add after GRU; MLP block follows.

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

**Distributed (`d-eggs`) inputs:** workers expect `input.bin` and `decoding.bin` in their working directory (coordinator does not need them).
