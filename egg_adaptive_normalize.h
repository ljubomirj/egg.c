#ifndef EGG_ADAPTIVE_NORMALIZE_H
#define EGG_ADAPTIVE_NORMALIZE_H

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#if defined(__HIPCC__)
// HIP's *_sync intrinsics use a 64-bit mask on AMD; map them to the non-sync variants
// since this code always assumes full-warp participation (warp size 32).
#undef __shfl_down_sync
#define __shfl_down_sync(mask, var, delta) __shfl_down((var), (delta), 32)
#endif

// Helper for Adaptive QKV Normalization
// Normalizes values layer-wise (across all warps/heads)
template <int N_WARPS>
__device__ __forceinline__ int8_t adaptive_qkv_normalize(
    int32_t val, 
    int tid, 
    int32_t *warp_maxs // Shared memory scratchpad (needs N_WARPS integers)
) {
    // 1. Calculate absolute value
    int32_t abs_v = abs(val);

    // 2. Warp-level reduction to find max(abs_v) in this warp
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        int32_t other = __shfl_down_sync(mask, abs_v, offset);
        if (other > abs_v) abs_v = other;
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Write warp max to shared memory
    if (lane_id == 0) {
        warp_maxs[warp_id] = abs_v;
    }
    
    // 3. Synchronize to ensure all warps have written their maxes
    __syncthreads();

    // 4. Layer-level max reduction (Per-Layer Scaling)
    int32_t layer_max = 0;
    #pragma unroll
    for (int i = 0; i < N_WARPS; i++) {
        if (warp_maxs[i] > layer_max) layer_max = warp_maxs[i];
    }

    // 5. Scaling
    float scale = 127.0f / (float)(layer_max + 1e-9f);
    
    float scaled_f = (float)val * scale;
    int32_t scaled = (int32_t)roundf(scaled_f);
    
    // 6. Clip to int8 range
    if (scaled > 127) scaled = 127;
    if (scaled < -127) scaled = -127;
    
    return (int8_t)scaled;
}

// Helper for Adaptive Layer Normalization
// Normalizes values across the entire layer dimension
template <int N_WARPS>
__device__ __forceinline__ int8_t adaptive_layer_normalize(
    int32_t val, 
    int tid, 
    int32_t *warp_maxs 
) {
    // 1. Calculate absolute value
    int32_t abs_v = abs(val);

    // 2. Warp-level reduction to find max(abs_v) in this warp
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        int32_t other = __shfl_down_sync(mask, abs_v, offset);
        if (other > abs_v) abs_v = other;
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Write warp max to shared memory
    if (lane_id == 0) {
        warp_maxs[warp_id] = abs_v;
    }
    
    // 3. Synchronize to ensure all warps have written their maxes
    __syncthreads();

    // 4. Layer-level max reduction
    int32_t layer_max = 0;
    #pragma unroll
    for (int i = 0; i < N_WARPS; i++) {
        if (warp_maxs[i] > layer_max) layer_max = warp_maxs[i];
    }

    // 5. Scaling
    float scale = 127.0f / (float)(layer_max + 1e-9f);
    
    float scaled_f = (float)val * scale;
    int32_t scaled = (int32_t)roundf(scaled_f);
    
    // 6. Clip to int8 range
    if (scaled > 127) scaled = 127;
    if (scaled < -127) scaled = -127;
    
    return (int8_t)scaled;
}

#endif // EGG_ADAPTIVE_NORMALIZE_H
