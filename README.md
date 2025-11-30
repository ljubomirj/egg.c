# EGGROLL in C

A minimalist, dependency-free implementation of the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) algorithm in pure C.

This project demonstrates **integer-only training** of a language model directly on the CPU (optimized for Apple Silicon/M-series chips), completely bypassing the need for GPUs, floating-point arithmetic, or heavy ML frameworks like PyTorch or JAX.

## Key Features

*   **Pure C**: Zero external dependencies (uses standard libraries + `dispatch` for threading).
*   **Apple Silicon Optimized**: Vectorized operations using ARM NEON intrinsics and parallelized via Grand Central Dispatch (GCD).
*   **Integer Only**: Operates entirely on `int8` weights/activations with `int32` accumulation. No float math in the training loop.
*   **Gradient Free**: Uses Evolution Strategies (ES) with low-rank perturbations instead of backpropagation.

## Quick Start

### 1. Prepare Data
Ensure you have a text dataset named `input.txt` in the current directory.

### 2. Compile
```bash
clang -O3 full_trained_egg.c -o egg
```

### 3. Run
```bash
./egg
```

![Training Output](_imgs_/egg_train.jpeg)

## Configuration

![Configuration](_imgs_/egg_config.jpeg)

## CPU vs GPU Builds

This repo currently has two main training entry points:

- `full_trained_egg.c`: the original single-binary CPU reference implementation.
- `full_trained_egg-cpumulti.c`: the multi-core, cross-arch CPU variant used by the `egg-macos-arm64`, `egg-linux-amd64`, and `egg-cpumulti` targets.

The GPU-oriented entry point is:

- `full_trained_egg-gpumulti.c`: for now this simply reuses the `full_trained_egg-cpumulti.c` training loop, but can be compiled with `EGG_BUILD_METAL` to call into a Metal backend on Apple Silicon.

### Metal Backend (Apple Silicon)

The Metal runtime is implemented in `full_trained_egg-gpu-metal.mm` and wired through a small C API in `egg_gpu_metal.h`. When you build:

```bash
make egg-gpumetal
```

the build system:

- Compiles `full_trained_egg-gpumulti.c` with `-DEGG_BUILD_METAL`, which pulls in the CPU training loop and enables optional GPU paths (`EGG_USE_METAL`) for certain kernels.
- Compiles and links the Objective-C++ Metal runtime, which:
  - Creates a Metal device and command queue on the default GPU (e.g. Apple M2 Max).
  - Builds a small Metal library containing:
    - A rank-1 matmul kernel (`egg_matmul_perturbed_kernel`) used for the noisy matrix–vector multiplies in the GRU/MLP.
    - An update kernel (`egg_update_matrix_kernel`) that computes ES “votes” and flips weights accordingly.
  - Manages a set of persistent GPU buffers for inputs, weights, outputs, noise, and update parameters.

At runtime:

- `matmul_perturbed`:
  - Generates rank-1 noise on the CPU (to stay consistent with the reference).
  - Calls `egg_gpu_matmul_perturbed` when Metal is available:
    - Copies the input vector, weight matrix and noise slice into shared GPU buffers.
    - Dispatches `egg_matmul_perturbed_kernel` across all output rows.
    - Copies the resulting `out` vector back to host memory.
  - Falls back to the scalar/NEON/AVX2 C implementation if the GPU path is disabled or fails.

- `update_matrix`:
  - Precomputes the transposed noise buffers `A_T` and `B_T` exactly as in the CPU version.
  - Calls `egg_gpu_update_matrix`, which:
    - Copies `W`, `A_T`, and `B_T` into Metal buffers.
    - Launches `egg_update_matrix_kernel`, where each GPU thread computes the vote for a single `(row, col)` weight and applies the `UPDATE_THRESHOLD` rule.
    - Copies updated weights back to the host model.
  - Falls back to the CPU loop if the GPU path is unavailable.

This design keeps the math identical to the CPU path, but it is **not yet optimized for maximum performance**:

- We still transfer full weight matrices for every matmul and every update step.
- The GRU gating logic, layer norms, RNG, and sampling remain on the CPU.
- A single training “step” can involve tens of thousands of small GPU launches plus large CPU↔GPU copies, so the first step on a full 500MB+ dataset can take a long time to complete.

### Practical Notes for Metal Training

- For **quick experiments** on Apple Silicon, it is strongly recommended to:
  - Reduce `SEQ_LEN` and `POPULATION_SIZE` in `egg_config.h`.
  - Optionally train on a truncated `input.txt` (e.g. a few MB) to see steps advancing more quickly.
- For **long runs**, keep in mind the current Metal backend is correctness-oriented; to make it truly faster than `egg-macos-arm64`, the next steps are:
  - Keep model weights live on the GPU instead of copying them for each matmul.
  - Batch more work into each kernel launch (e.g. multiple gates/layers/timesteps at once).
  - Gradually move GRU/LN glue and population evaluation onto the device.

## References

* **Original JAX Implementation**: [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg)
* **Original Paper & Project**: [EGGROLL Website](https://eshyperscale.github.io/)
