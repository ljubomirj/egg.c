#ifndef EGG_GPU_METAL_H
#define EGG_GPU_METAL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool egg_gpu_metal_init(void);
void egg_gpu_metal_shutdown(void);

bool egg_gpu_matmul_perturbed(
    const int8_t *input,
    const int8_t *weights,
    int8_t *output,
    const int8_t *noise_a,
    int rows,
    int cols,
    int shift,
    int noise_sign,
    int32_t xB
);

#ifdef __cplusplus
}
#endif

#endif // EGG_GPU_METAL_H

