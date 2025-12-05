#ifndef EGG_GPU_METAL_OPTIMIZED_H
#define EGG_GPU_METAL_OPTIMIZED_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the optimized Metal GPU implementation
// Returns true on success
bool egg_gpu_metal_optimized_init(void);

// Run forward pass for entire population
// model: Pointer to model on CPU (will be copied to GPU)
// tokens: Input token sequence (length = seq_len)
// seq_len: Length of sequence
// step_seed: Random seed for this step
// losses_out: Output array of size POPULATION_SIZE for losses
// Returns true on success
bool egg_gpu_metal_optimized_forward_pass(
    const void *model,
    const uint8_t *tokens,
    const int seq_len,
    const uint32_t step_seed,
    int *losses_out
);

// Update model weights based on fitness evaluations
// model: Model to update (both CPU and GPU versions)
// pair_fitnesses: Fitness array of size POPULATION_SIZE/2
// step_seed: Random seed for noise generation
// Returns true on success
bool egg_gpu_metal_optimized_update_weights(
    void *model,
    const int *pair_fitnesses,
    const uint32_t step_seed
);

// Cleanup and shutdown
void egg_gpu_metal_optimized_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // EGG_GPU_METAL_OPTIMIZED_H