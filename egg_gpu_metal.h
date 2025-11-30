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

bool egg_gpu_update_matrix(
    int8_t *weights,
    const int8_t *A_T,
    const int8_t *B_T,
    int rows,
    int cols,
    int pairs
);

// Bind host-side model weights into a single persistent Metal buffer.
// Sizes are in bytes and should match the lengths of the corresponding arrays.
bool egg_gpu_bind_model_weights(
    const int8_t *embedding,  size_t embedding_size,
    const int8_t *gru_weights, size_t gru_size,
    const int8_t *mlp_weights, size_t mlp_size,
    const int8_t *head,       size_t head_size
);

#ifdef __cplusplus
}
#endif

#endif // EGG_GPU_METAL_H

