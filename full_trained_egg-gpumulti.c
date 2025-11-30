#if defined(EGG_BUILD_METAL)
#define EGG_USE_METAL 1
#include "full_trained_egg-cpumulti.c"
#else
#include <stdio.h>
#include <stdlib.h>

typedef enum {
    EGG_GPU_BACKEND_PLACEHOLDER = 0,
    EGG_GPU_BACKEND_ROCM,
    EGG_GPU_BACKEND_CUDA,
    EGG_GPU_BACKEND_VULKAN
} EggGpuBackend;

#ifndef EGG_GPU_BACKEND
#define EGG_GPU_BACKEND EGG_GPU_BACKEND_PLACEHOLDER
#endif

static const char *egg_gpu_backend_name(int backend) {
    switch (backend) {
        case EGG_GPU_BACKEND_ROCM: return "ROCm";
        case EGG_GPU_BACKEND_CUDA: return "CUDA";
        case EGG_GPU_BACKEND_VULKAN: return "Vulkan";
        default: return "Generic GPU";
    }
}

static void egg_gpu_print_todo(const char *backend_name) {
    fprintf(stderr,
        "[EGG GPU] Backend '%s' is not implemented yet.\n"
        "TODO: port core kernels (RNN, matmul, LN, population update) to %s.\n"
        "Suggested steps:\n"
        "  1. Implement device memory layout + host staging.\n"
        "  2. Add kernel launch wrappers (HIP/CUDA/Vulkan compute).\n"
        "  3. Reuse common host utilities from full_trained_egg-cpumulti.c.\n"
        "  4. Validate parity with CPU training loop.\n",
        backend_name, backend_name, backend_name);
}

int main(void) {
    const char *backend_name = egg_gpu_backend_name(EGG_GPU_BACKEND);
    egg_gpu_print_todo(backend_name);
    return EXIT_FAILURE;
}
#endif

