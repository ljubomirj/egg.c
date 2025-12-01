#ifndef EGG_GPU_METAL_H
#define EGG_GPU_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include "egg_config.h"

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

// GPU matmul that works directly with GPU buffers (no CPU copies)
// input_gpu, weights_gpu, output_gpu are GPU buffers (void*)
// noise_a is CPU array (needs to be uploaded)
// Returns true if successful
bool egg_gpu_matmul_perturbed_gpu(
    void *input_gpu,
    const int8_t *weights,  // Can be CPU pointer or GPU offset into modelBuffer
    void *output_gpu,
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

// Batch API: Begin/end batching to reduce synchronization overhead
// Operations encoded between begin/end are batched into a single command buffer
void egg_gpu_batch_begin(void);
void egg_gpu_batch_end(void);  // Commits and waits for all batched operations
void egg_gpu_batch_flush(void); // Commits current batch but doesn't wait (for async execution)

// GPU-resident buffer management for activations
// These functions manage GPU buffers that stay on GPU (no CPU-GPU transfers)
typedef struct {
    void *gpu_x;           // GPU buffer for x (HIDDEN_DIM)
    void *gpu_buf1;        // GPU buffer for buf1 (HIDDEN_DIM * 4)
    void *gpu_buf2;        // GPU buffer for buf2 (HIDDEN_DIM)
    void *gpu_residual;    // GPU buffer for residual (HIDDEN_DIM)
    void *gpu_ft;          // GPU buffer for ft (HIDDEN_DIM)
    void *gpu_ht;          // GPU buffer for ht (HIDDEN_DIM)
    void *gpu_gated_past;  // GPU buffer for gated_past (HIDDEN_DIM)
    void *gpu_rnn_state[N_LAYERS]; // GPU buffers for RNN state h[l] (HIDDEN_DIM each)
} EggGpuActivationBuffers;

// Allocate GPU-resident activation buffers (stays on GPU, no CPU access)
EggGpuActivationBuffers* egg_gpu_alloc_activation_buffers(void);
void egg_gpu_free_activation_buffers(EggGpuActivationBuffers *bufs);

// Element-wise operations on GPU (work on GPU buffers, no CPU access)
// These encode into the current batch encoder
bool egg_gpu_clipped_add(void *gpu_a, void *gpu_b, void *gpu_out, int count);
bool egg_gpu_clipped_add_scalar(void *gpu_a, int scalar, void *gpu_out, int count);
bool egg_gpu_clipped_add_three(void *gpu_a, void *gpu_b, void *gpu_c, void *gpu_out, int count); // a + b + c with clipping
bool egg_gpu_layer_norm(void *gpu_x, const int8_t *ln_weights_host, void *gpu_out, int dim);
bool egg_gpu_gru_gate(void *gpu_ft, void *gpu_h, void *gpu_out, int count);
bool egg_gpu_gru_state_update(void *gpu_h, void *gpu_ft, void *gpu_ht, void *gpu_out, int count);
bool egg_gpu_dot_product(void *gpu_a, void *gpu_b, int32_t *result, int count); // Compute dot product on GPU, result written to *result

// Copy data to/from GPU buffers (only when necessary)
bool egg_gpu_copy_to_buffer(void *gpu_buffer, const void *cpu_data, size_t bytes);
bool egg_gpu_copy_from_buffer(void *cpu_data, void *gpu_buffer, size_t bytes);

// Temporary buffer helpers (Shared mode for easy CPU access)
void* egg_gpu_alloc_temp_buffer(size_t bytes);
void egg_gpu_free_temp_buffer(void *gpu_buffer);
void* egg_gpu_get_buffer_contents(void *gpu_buffer); // Get pointer to buffer contents (for Shared buffers)

#ifdef __cplusplus
}
#endif

#endif // EGG_GPU_METAL_H

