#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "egg_config.h"
#include "egg_gpu_metal.h"

typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t shift;
    int32_t noise_sign;
    int32_t xB;
    int32_t pad[3];
} EggMetalMatmulParams;

typedef struct {
    bool ready;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> matmulPipeline;
    id<MTLComputePipelineState> updatePipeline;
    id<MTLComputePipelineState> clippedAddPipeline;
    id<MTLComputePipelineState> clippedAddScalarPipeline;
    id<MTLComputePipelineState> clippedAddThreePipeline;
    id<MTLComputePipelineState> layerNormPipeline;
    id<MTLComputePipelineState> gruGatePipeline;
    id<MTLComputePipelineState> gruStateUpdatePipeline;
    id<MTLComputePipelineState> dotProductPipeline;
    id<MTLBuffer> inputBuffer;
    NSUInteger inputCapacity;
    id<MTLBuffer> weightBuffer;
    NSUInteger weightCapacity;
    id<MTLBuffer> outputBuffer;
    NSUInteger outputCapacity;
    id<MTLBuffer> noiseBuffer;
    NSUInteger noiseCapacity;
    id<MTLBuffer> paramsBuffer;
    id<MTLBuffer> updateParamsBuffer;
    // Persistent model weights buffer (all large matrices live here).
    id<MTLBuffer> modelBuffer;
    NSUInteger modelCapacity;
    struct {
        const int8_t *embeddingHost;
        NSUInteger    embeddingOffset;
        NSUInteger    embeddingSize;
        const int8_t *gruHost;
        NSUInteger    gruOffset;
        NSUInteger    gruSize;
        const int8_t *mlpHost;
        NSUInteger    mlpOffset;
        NSUInteger    mlpSize;
        const int8_t *headHost;
        NSUInteger    headOffset;
        NSUInteger    headSize;
        bool          bound;
    } modelLayout;
    // Temporary buffers for ES update (noise and per-call weight tiles)
    id<MTLBuffer> updateWeightsBuffer;
    NSUInteger    updateWeightsCapacity;
    // Temporary noise/vote buffers for update_matrix
    id<MTLBuffer> updateABuffer;
    NSUInteger    updateACapacity;
    id<MTLBuffer> updateBBuffer;
    NSUInteger    updateBCapacity;
    dispatch_semaphore_t lock;
} EggMetalContext;

static EggMetalContext g_ctx = {};

// Pending read structure
struct EggMetalPendingRead {
    id<MTLBuffer> buffer;
    void *hostPtr;
    size_t size;
};

// Thread-local batching state
typedef struct {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> encoder;
    bool inBatch;
    int operationCount;
    // Track outputs that need to be read back after batch execution (dynamic array)
    struct EggMetalPendingRead *pendingReads;
    int pendingReadCount;
    int pendingReadCapacity;
} EggMetalBatchState;

static pthread_key_t g_batchKey;
static pthread_once_t g_batchKeyOnce = PTHREAD_ONCE_INIT;

static void EggMetalBatchKeyDestructor(void *value) {
    if (value) {
        EggMetalBatchState *state = (EggMetalBatchState *)value;
        if (state->encoder) {
            [state->encoder endEncoding];
            state->encoder = nil;
        }
        if (state->commandBuffer) {
            state->commandBuffer = nil;
        }
        if (state->pendingReads) {
            free(state->pendingReads);
            state->pendingReads = NULL;
        }
        free(state);
    }
}

static void EggMetalBatchKeyInit(void) {
    pthread_key_create(&g_batchKey, EggMetalBatchKeyDestructor);
}

static EggMetalBatchState *EggMetalGetBatchState(void) {
    pthread_once(&g_batchKeyOnce, EggMetalBatchKeyInit);
    EggMetalBatchState *state = (EggMetalBatchState *)pthread_getspecific(g_batchKey);
    if (!state) {
        state = (EggMetalBatchState *)calloc(1, sizeof(EggMetalBatchState));
        state->pendingReadCapacity = 8192;  // Initial capacity - large enough for most forward passes
        state->pendingReads = (struct EggMetalPendingRead *)calloc(state->pendingReadCapacity, sizeof(struct EggMetalPendingRead));
        if (!state->pendingReads) {
            fprintf(stderr, "[EGG METAL] Error: failed to allocate initial pending reads array.\n");
            free(state);
            return NULL;
        }
        pthread_setspecific(g_batchKey, state);
    }
    return state;
}

typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t pairs;
    int32_t strideA;
    int32_t strideB;
    int32_t threshold;
} EggMetalUpdateParams;

static NSString *EggMetalShaderSource(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "#define FIXED_POINT %d\n"
         "#define SIGMA_SHIFT %d\n"
         "#define MAX_VAL %d\n"
         "#define MIN_VAL %d\n"
         "struct EggMetalMatmulParams {\n"
         "  int rows;\n"
         "  int cols;\n"
         "  int shift;\n"
         "  int noise_sign;\n"
         "  int xB;\n"
         "};\n"
         "struct EggMetalUpdateParams {\n"
         "  int rows;\n"
         "  int cols;\n"
         "  int pairs;\n"
         "  int strideA;\n"
         "  int strideB;\n"
         "  int threshold;\n"
         "};\n"
         "kernel void egg_matmul_perturbed_kernel(\n"
         "    device const char *input [[buffer(0)]],\n"
         "    device const char *weights [[buffer(1)]],\n"
         "    device char *output [[buffer(2)]],\n"
         "    device const char *noiseA [[buffer(3)]],\n"
         "    constant EggMetalMatmulParams &params [[buffer(4)]],\n"
         "    uint gid [[thread_position_in_grid]]) {\n"
         "  if (gid >= params.rows) { return; }\n"
         "  const device char *row = weights + gid * params.cols;\n"
         "  int acc = 0;\n"
         "  for (int c = 0; c < params.cols; ++c) {\n"
         "    acc += int(input[c]) * int(row[c]);\n"
         "  }\n"
         "  if (params.noise_sign != 0) {\n"
         "    int noise = (params.xB * int(noiseA[gid])) * params.noise_sign;\n"
         "    acc += (noise >> (FIXED_POINT + SIGMA_SHIFT));\n"
         "  }\n"
         "  int res = acc >> params.shift;\n"
         "  res = min(res, MAX_VAL);\n"
         "  res = max(res, MIN_VAL);\n"
         "  output[gid] = char(res);\n"
         "}\n"
         "kernel void egg_update_matrix_kernel(\n"
         "    device char *weights [[buffer(0)]],\n"
         "    device const char *A_T [[buffer(1)]],\n"
         "    device const char *B_T [[buffer(2)]],\n"
         "    constant EggMetalUpdateParams &params [[buffer(3)]],\n"
         "    uint gid [[thread_position_in_grid]]) {\n"
         "  int total = params.rows * params.cols;\n"
         "  if (gid >= total) { return; }\n"
         "  int r = gid / params.cols;\n"
         "  int c = gid %% params.cols;\n"
         "  const device char *a_ptr = A_T + r * params.strideA;\n"
         "  const device char *b_ptr = B_T + c * params.strideB;\n"
         "  int vote = 0;\n"
         "  for (int p = 0; p < params.pairs; ++p) {\n"
         "    vote += int(a_ptr[p]) * int(b_ptr[p]);\n"
         "  }\n"
         "  device char *w_ptr = weights + gid;\n"
         "  if (vote > params.threshold && *w_ptr < MAX_VAL) {\n"
         "    (*w_ptr)++;\n"
         "  } else if (vote < -params.threshold && *w_ptr > MIN_VAL) {\n"
         "    (*w_ptr)--;\n"
         "  }\n"
         "}\n"
         "kernel void egg_clipped_add_kernel(\n"
         "    device const char *a [[buffer(0)]],\n"
         "    device const char *b [[buffer(1)]],\n"
         "    device char *out [[buffer(2)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int res = int(a[gid]) + int(b[gid]);\n"
         "  res = min(res, MAX_VAL);\n"
         "  res = max(res, MIN_VAL);\n"
         "  out[gid] = char(res);\n"
         "}\n"
         "kernel void egg_clipped_add_scalar_kernel(\n"
         "    device const char *a [[buffer(0)]],\n"
         "    constant int *scalar [[buffer(1)]],\n"
         "    device char *out [[buffer(2)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int res = int(a[gid]) + *scalar;\n"
         "  res = min(res, MAX_VAL);\n"
         "  res = max(res, MIN_VAL);\n"
         "  out[gid] = char(res);\n"
         "}\n"
         "kernel void egg_layer_norm_kernel(\n"
         "    device const char *x [[buffer(0)]],\n"
         "    device const char *w [[buffer(1)]],\n"
         "    device char *out [[buffer(2)]],\n"
         "    constant int *dim [[buffer(3)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  // Compute sum of absolute values\n"
         "  int sum = 0;\n"
         "  for (uint i = 0; i < *dim; ++i) {\n"
         "    int val = int(x[i]);\n"
         "    sum += (val < 0) ? -val : val;\n"
         "  }\n"
         "  if (sum == 0) sum = 1;\n"
         "  int mean = sum / int(*dim);\n"
         "  // Normalize and scale\n"
         "  int val = int(x[gid]);\n"
         "  int normalized = (val << 8) / mean;\n"
         "  int scaled = (normalized * int(w[gid])) >> 8;\n"
         "  scaled = min(scaled, MAX_VAL);\n"
         "  scaled = max(scaled, MIN_VAL);\n"
         "  out[gid] = char(scaled);\n"
         "}\n"
         "kernel void egg_gru_gate_kernel(\n"
         "    device const char *ft [[buffer(0)]],\n"
         "    device const char *h [[buffer(1)]],\n"
         "    device char *out [[buffer(2)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int ft_val = int(ft[gid]) + 127;\n"
         "  int h_val = int(h[gid]);\n"
         "  int res = (ft_val * h_val) >> 8;\n"
         "  res = min(res, MAX_VAL);\n"
         "  res = max(res, MIN_VAL);\n"
         "  out[gid] = char(res);\n"
         "}\n"
         "kernel void egg_gru_state_update_kernel(\n"
         "    device char *h [[buffer(0)]],\n"
         "    device const char *ft [[buffer(1)]],\n"
         "    device const char *ht [[buffer(2)]],\n"
         "    device char *out [[buffer(3)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int ft_val = int(ft[gid]) + 127;\n"
         "  int ht_val = int(ht[gid]);\n"
         "  int h_old = int(h[gid]);\n"
         "  int update = (ft_val * (ht_val - h_old)) >> 8;\n"
         "  int h_new = h_old + update;\n"
         "  h_new = min(h_new, MAX_VAL);\n"
         "  h_new = max(h_new, MIN_VAL);\n"
         "  h[gid] = char(h_new);\n"
         "  out[gid] = char(h_new);\n"
         "}\n"
         "kernel void egg_clipped_add_three_kernel(\n"
         "    device const char *a [[buffer(0)]],\n"
         "    device const char *b [[buffer(1)]],\n"
         "    device const char *c [[buffer(2)]],\n"
         "    device char *out [[buffer(3)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int res = int(a[gid]) + int(b[gid]) + int(c[gid]);\n"
         "  res = min(res, MAX_VAL);\n"
         "  res = max(res, MIN_VAL);\n"
         "  out[gid] = char(res);\n"
         "}\n"
         "kernel void egg_dot_product_kernel(\n"
         "    device const char *a [[buffer(0)]],\n"
         "    device const char *b [[buffer(1)]],\n"
         "    device atomic_int *result [[buffer(2)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  if (gid >= count) return;\n"
         "  int prod = int(a[gid]) * int(b[gid]);\n"
         "  atomic_fetch_add_explicit(result, prod, memory_order_relaxed);\n"
         "}\n"
         "kernel void egg_dot_product_reduce_kernel(\n"
         "    device const char *a [[buffer(0)]],\n"
         "    device const char *b [[buffer(1)]],\n"
         "    device int *partial_sums [[buffer(2)]],\n"
         "    threadgroup int *shared_data [[threadgroup(0)]],\n"
         "    uint gid [[thread_position_in_grid]],\n"
         "    uint tid [[thread_index_in_threadgroup]],\n"
         "    uint count [[threads_per_grid]]) {\n"
         "  uint group_size = 256;\n"
         "  uint group_id = gid / group_size;\n"
         "  uint local_id = gid %% group_size;\n"
         "  \n"
         "  int sum = 0;\n"
         "  for (uint i = local_id; i < count; i += group_size) {\n"
         "    sum += int(a[i]) * int(b[i]);\n"
         "  }\n"
         "  shared_data[tid] = sum;\n"
         "  \n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  \n"
         "  for (uint s = group_size / 2; s > 0; s >>= 1) {\n"
         "    if (tid < s) {\n"
         "      shared_data[tid] += shared_data[tid + s];\n"
         "    }\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  }\n"
         "  \n"
         "  if (tid == 0) {\n"
         "    partial_sums[group_id] = shared_data[0];\n"
         "  }\n"
         "}\n",
        FIXED_POINT,
        SIGMA_SHIFT,
        MAX_VAL,
        MIN_VAL];
}

static id<MTLBuffer> EggEnsureBuffer(id<MTLDevice> device,
                                     id<MTLBuffer> __strong *buffer,
                                     NSUInteger *capacity,
                                     NSUInteger required) {
    if (required == 0) required = 1;
    if (*buffer && *capacity >= required) return *buffer;
    NSUInteger newCapacity = (required + 0xFF) & ~0xFF;
    // Use Shared mode for buffers that need CPU access (input/output)
    // Private mode would be faster but requires blit encoders for CPU-GPU transfers
    *buffer = [device newBufferWithLength:newCapacity options:MTLResourceStorageModeShared];
    *capacity = newCapacity;
    return *buffer;
}

static void EggLogError(const char *label, NSError *error) {
    if (!error) return;
    fprintf(stderr, "[EGG METAL] %s failed: %s\n", label, error.localizedDescription.UTF8String);
}

static bool EggResolveModelWeightOffset(const int8_t *hostPtr, NSUInteger *offsetOut) {
    if (!g_ctx.modelLayout.bound) return false;
    auto &ml = g_ctx.modelLayout;
    if (ml.embeddingHost && hostPtr >= ml.embeddingHost && hostPtr < ml.embeddingHost + ml.embeddingSize) {
        *offsetOut = ml.embeddingOffset + (NSUInteger)(hostPtr - ml.embeddingHost);
        return true;
    }
    if (ml.gruHost && hostPtr >= ml.gruHost && hostPtr < ml.gruHost + ml.gruSize) {
        *offsetOut = ml.gruOffset + (NSUInteger)(hostPtr - ml.gruHost);
        return true;
    }
    if (ml.mlpHost && hostPtr >= ml.mlpHost && hostPtr < ml.mlpHost + ml.mlpSize) {
        *offsetOut = ml.mlpOffset + (NSUInteger)(hostPtr - ml.mlpHost);
        return true;
    }
    if (ml.headHost && hostPtr >= ml.headHost && hostPtr < ml.headHost + ml.headSize) {
        *offsetOut = ml.headOffset + (NSUInteger)(hostPtr - ml.headHost);
        return true;
    }
    return false;
}

extern "C" bool egg_gpu_metal_init(void) {
    @autoreleasepool {
        if (g_ctx.ready) return true;
        g_ctx.device = MTLCreateSystemDefaultDevice();
        if (!g_ctx.device) {
            fprintf(stderr, "[EGG METAL] No compatible GPU device found.\n");
            return false;
        }
        g_ctx.queue = [g_ctx.device newCommandQueue];
        if (!g_ctx.queue) {
            fprintf(stderr, "[EGG METAL] Failed to create command queue.\n");
            g_ctx.device = nil;
            return false;
        }

        NSError *error = nil;
        NSString *source = EggMetalShaderSource();
        id<MTLLibrary> library = [g_ctx.device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            EggLogError("newLibraryWithSource", error);
            g_ctx.queue = nil;
            g_ctx.device = nil;
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"egg_matmul_perturbed_kernel"];
        if (!function) {
            fprintf(stderr, "[EGG METAL] Failed to create kernel function.\n");
            g_ctx.queue = nil;
            g_ctx.device = nil;
            return false;
        }

        g_ctx.matmulPipeline = [g_ctx.device newComputePipelineStateWithFunction:function error:&error];
        if (!g_ctx.matmulPipeline) {
            EggLogError("newComputePipelineStateWithFunction", error);
            g_ctx.queue = nil;
            g_ctx.device = nil;
            return false;
        }

        id<MTLFunction> updateFunction = [library newFunctionWithName:@"egg_update_matrix_kernel"];
        if (!updateFunction) {
            fprintf(stderr, "[EGG METAL] Failed to create update kernel function.\n");
            g_ctx.queue = nil;
            g_ctx.device = nil;
            g_ctx.matmulPipeline = nil;
            return false;
        }

        g_ctx.updatePipeline = [g_ctx.device newComputePipelineStateWithFunction:updateFunction error:&error];
        if (!g_ctx.updatePipeline) {
            EggLogError("newComputePipelineStateWithFunction(update)", error);
            g_ctx.queue = nil;
            g_ctx.device = nil;
            g_ctx.matmulPipeline = nil;
            return false;
        }

        // Initialize element-wise operation pipelines
        id<MTLFunction> clippedAddFunc = [library newFunctionWithName:@"egg_clipped_add_kernel"];
        if (clippedAddFunc) {
            g_ctx.clippedAddPipeline = [g_ctx.device newComputePipelineStateWithFunction:clippedAddFunc error:&error];
            if (!g_ctx.clippedAddPipeline) {
                EggLogError("newComputePipelineStateWithFunction(clipped_add)", error);
            }
        }

        id<MTLFunction> clippedAddScalarFunc = [library newFunctionWithName:@"egg_clipped_add_scalar_kernel"];
        if (clippedAddScalarFunc) {
            g_ctx.clippedAddScalarPipeline = [g_ctx.device newComputePipelineStateWithFunction:clippedAddScalarFunc error:&error];
            if (!g_ctx.clippedAddScalarPipeline) {
                EggLogError("newComputePipelineStateWithFunction(clipped_add_scalar)", error);
            }
        }

        id<MTLFunction> layerNormFunc = [library newFunctionWithName:@"egg_layer_norm_kernel"];
        if (layerNormFunc) {
            g_ctx.layerNormPipeline = [g_ctx.device newComputePipelineStateWithFunction:layerNormFunc error:&error];
            if (!g_ctx.layerNormPipeline) {
                EggLogError("newComputePipelineStateWithFunction(layer_norm)", error);
            }
        }

        id<MTLFunction> gruGateFunc = [library newFunctionWithName:@"egg_gru_gate_kernel"];
        if (gruGateFunc) {
            g_ctx.gruGatePipeline = [g_ctx.device newComputePipelineStateWithFunction:gruGateFunc error:&error];
            if (!g_ctx.gruGatePipeline) {
                EggLogError("newComputePipelineStateWithFunction(gru_gate)", error);
            }
        }

        id<MTLFunction> gruStateUpdateFunc = [library newFunctionWithName:@"egg_gru_state_update_kernel"];
        if (gruStateUpdateFunc) {
            g_ctx.gruStateUpdatePipeline = [g_ctx.device newComputePipelineStateWithFunction:gruStateUpdateFunc error:&error];
            if (!g_ctx.gruStateUpdatePipeline) {
                EggLogError("newComputePipelineStateWithFunction(gru_state_update)", error);
            }
        }

        id<MTLFunction> clippedAddThreeFunc = [library newFunctionWithName:@"egg_clipped_add_three_kernel"];
        if (clippedAddThreeFunc) {
            g_ctx.clippedAddThreePipeline = [g_ctx.device newComputePipelineStateWithFunction:clippedAddThreeFunc error:&error];
            if (!g_ctx.clippedAddThreePipeline) {
                EggLogError("newComputePipelineStateWithFunction(clipped_add_three)", error);
            }
        }

        id<MTLFunction> dotProductFunc = [library newFunctionWithName:@"egg_dot_product_kernel"];
        if (dotProductFunc) {
            g_ctx.dotProductPipeline = [g_ctx.device newComputePipelineStateWithFunction:dotProductFunc error:&error];
            if (!g_ctx.dotProductPipeline) {
                EggLogError("newComputePipelineStateWithFunction(dot_product)", error);
            }
        }

        g_ctx.lock = dispatch_semaphore_create(1);
        g_ctx.ready = true;
        fprintf(stdout, "[EGG METAL] Initialized on device: %s\n", g_ctx.device.name.UTF8String);
        return true;
    }
}

extern "C" bool egg_gpu_bind_model_weights(
    const int8_t *embedding,  size_t embedding_size,
    const int8_t *gru_weights, size_t gru_size,
    const int8_t *mlp_weights, size_t mlp_size,
    const int8_t *head,       size_t head_size
) {
    if (!g_ctx.ready) {
        if (!egg_gpu_metal_init()) {
            return false;
        }
    }

    if (!embedding || !gru_weights || !mlp_weights || !head) {
        fprintf(stderr, "[EGG METAL] Invalid model pointers in egg_gpu_bind_model_weights.\n");
        return false;
    }

    NSUInteger embBytes = (NSUInteger)embedding_size;
    NSUInteger gruBytes = (NSUInteger)gru_size;
    NSUInteger mlpBytes = (NSUInteger)mlp_size;
    NSUInteger headBytes = (NSUInteger)head_size;
    NSUInteger totalBytes = embBytes + gruBytes + mlpBytes + headBytes;
    if (totalBytes == 0) {
        fprintf(stderr, "[EGG METAL] Zero-sized model in egg_gpu_bind_model_weights.\n");
        return false;
    }

    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    @autoreleasepool {
        // Ensure model buffer is large enough.
        id<MTLBuffer> modelBuffer = g_ctx.modelBuffer;
        if (!modelBuffer || g_ctx.modelCapacity < totalBytes) {
            g_ctx.modelBuffer = [g_ctx.device newBufferWithLength:totalBytes
                                                          options:MTLResourceStorageModeShared];
            if (!g_ctx.modelBuffer) {
                fprintf(stderr, "[EGG METAL] Failed to allocate model buffer (%zu bytes).\n",
                        (size_t)totalBytes);
                g_ctx.modelCapacity = 0;
                dispatch_semaphore_signal(g_ctx.lock);
                return false;
            }
            g_ctx.modelCapacity = totalBytes;
        }

        uint8_t *base = static_cast<uint8_t *>([g_ctx.modelBuffer contents]);
        if (!base) {
            fprintf(stderr, "[EGG METAL] modelBuffer has no contents.\n");
            dispatch_semaphore_signal(g_ctx.lock);
            return false;
        }

        // Lay out sub-ranges.
        NSUInteger off = 0;
        g_ctx.modelLayout.embeddingHost   = embedding;
        g_ctx.modelLayout.embeddingOffset = off;
        g_ctx.modelLayout.embeddingSize   = embBytes;
        memcpy(base + off, embedding, embBytes);
        off += embBytes;

        g_ctx.modelLayout.gruHost   = gru_weights;
        g_ctx.modelLayout.gruOffset = off;
        g_ctx.modelLayout.gruSize   = gruBytes;
        memcpy(base + off, gru_weights, gruBytes);
        off += gruBytes;

        g_ctx.modelLayout.mlpHost   = mlp_weights;
        g_ctx.modelLayout.mlpOffset = off;
        g_ctx.modelLayout.mlpSize   = mlpBytes;
        memcpy(base + off, mlp_weights, mlpBytes);
        off += mlpBytes;

        g_ctx.modelLayout.headHost   = head;
        g_ctx.modelLayout.headOffset = off;
        g_ctx.modelLayout.headSize   = headBytes;
        memcpy(base + off, head, headBytes);

        g_ctx.modelLayout.bound = true;
    }
    dispatch_semaphore_signal(g_ctx.lock);
    fprintf(stdout, "[EGG METAL] Bound model weights into GPU buffer (%zu bytes).\n",
            (size_t)g_ctx.modelCapacity);
    return true;
}

extern "C" void egg_gpu_metal_shutdown(void) {
    g_ctx.ready = false;
    g_ctx.inputBuffer = nil;
    g_ctx.weightBuffer = nil;
    g_ctx.outputBuffer = nil;
    g_ctx.noiseBuffer = nil;
    g_ctx.paramsBuffer = nil;
    g_ctx.updateParamsBuffer = nil;
    g_ctx.modelBuffer = nil;
    g_ctx.modelCapacity = 0;
    g_ctx.modelLayout.embeddingHost = nullptr;
    g_ctx.modelLayout.embeddingOffset = 0;
    g_ctx.modelLayout.embeddingSize = 0;
    g_ctx.modelLayout.gruHost = nullptr;
    g_ctx.modelLayout.gruOffset = 0;
    g_ctx.modelLayout.gruSize = 0;
    g_ctx.modelLayout.mlpHost = nullptr;
    g_ctx.modelLayout.mlpOffset = 0;
    g_ctx.modelLayout.mlpSize = 0;
    g_ctx.modelLayout.headHost = nullptr;
    g_ctx.modelLayout.headOffset = 0;
    g_ctx.modelLayout.headSize = 0;
    g_ctx.modelLayout.bound = false;
    g_ctx.updateWeightsBuffer = nil;
    g_ctx.updateWeightsCapacity = 0;
    g_ctx.updateABuffer = nil;
    g_ctx.updateBBuffer = nil;
    g_ctx.updateACapacity = 0;
    g_ctx.updateBCapacity = 0;
    g_ctx.matmulPipeline = nil;
    g_ctx.updatePipeline = nil;
    g_ctx.queue = nil;
    g_ctx.device = nil;
    g_ctx.inputCapacity = 0;
    g_ctx.weightCapacity = 0;
    g_ctx.outputCapacity = 0;
    g_ctx.noiseCapacity = 0;
    g_ctx.lock = nil;
}

extern "C" void egg_gpu_batch_begin(void) {
    if (!g_ctx.ready) return;
    EggMetalBatchState *state = EggMetalGetBatchState();
    if (state->inBatch) {
        fprintf(stderr, "[EGG METAL] Warning: batch_begin called while already in batch.\n");
        return;
    }
    state->commandBuffer = [g_ctx.queue commandBuffer];
    state->encoder = [state->commandBuffer computeCommandEncoder];
    state->inBatch = true;
    state->operationCount = 0;
    state->pendingReadCount = 0;
    // Ensure capacity is sufficient (will grow if needed)
    if (!state->pendingReads) {
        state->pendingReadCapacity = 8192;  // Start with larger capacity for large forward passes
        state->pendingReads = (struct EggMetalPendingRead *)calloc(state->pendingReadCapacity, sizeof(struct EggMetalPendingRead));
        if (!state->pendingReads) {
            fprintf(stderr, "[EGG METAL] Error: failed to allocate pending reads array.\n");
            state->inBatch = false;
            return;
        }
    }
}

extern "C" void egg_gpu_batch_flush(void) {
    if (!g_ctx.ready) return;
    EggMetalBatchState *state = EggMetalGetBatchState();
    if (!state->inBatch || !state->encoder) return;
    
    [state->encoder endEncoding];
    state->encoder = nil;
    if (state->commandBuffer) {
        [state->commandBuffer commit];
        state->commandBuffer = nil;
    }
    state->inBatch = false;
    state->operationCount = 0;
}

extern "C" void egg_gpu_batch_end(void) {
    if (!g_ctx.ready) return;
    EggMetalBatchState *state = EggMetalGetBatchState();
    if (!state->inBatch) return;
    
    if (!state->encoder || !state->commandBuffer) {
        state->inBatch = false;
        return;
    }
    
    [state->encoder endEncoding];
    state->encoder = nil;
    
    // For Shared buffers, we can read immediately without waiting
    // Only wait if we have Private buffers that need read-back
    bool needsSync = false;
    for (int i = 0; i < state->pendingReadCount; i++) {
        if (state->pendingReads[i].buffer) {
            id<MTLBuffer> buf = state->pendingReads[i].buffer;
            if ([buf storageMode] == MTLStorageModePrivate) {
                needsSync = true;
                break;
            }
        }
    }
    
    if (needsSync) {
        // Only wait if we have Private buffers
        dispatch_semaphore_t readSem = dispatch_semaphore_create(0);
        [state->commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            for (int i = 0; i < state->pendingReadCount; i++) {
                if (state->pendingReads[i].buffer && state->pendingReads[i].hostPtr) {
                    id<MTLBuffer> buf = state->pendingReads[i].buffer;
                    if ([buf storageMode] == MTLStorageModePrivate) {
                        memcpy(state->pendingReads[i].hostPtr,
                               [state->pendingReads[i].buffer contents],
                               state->pendingReads[i].size);
                    }
                }
            }
            dispatch_semaphore_signal(readSem);
        }];
        [state->commandBuffer commit];
        dispatch_semaphore_wait(readSem, DISPATCH_TIME_FOREVER);
    } else {
        // For Shared buffers, commit async and read immediately
        [state->commandBuffer commit];
        // Read back immediately (Shared buffers are CPU-visible)
        for (int i = 0; i < state->pendingReadCount; i++) {
            if (state->pendingReads[i].buffer && state->pendingReads[i].hostPtr) {
                id<MTLBuffer> buf = state->pendingReads[i].buffer;
                if ([buf storageMode] == MTLStorageModeShared) {
                    memcpy(state->pendingReads[i].hostPtr,
                           [buf contents],
                           state->pendingReads[i].size);
                }
            }
        }
    }
    
    state->commandBuffer = nil;
    state->inBatch = false;
    state->operationCount = 0;
    state->pendingReadCount = 0;
}

extern "C" bool egg_gpu_matmul_perturbed(
    const int8_t *input,
    const int8_t *weights,
    int8_t *output,
    const int8_t *noise_a,
    int rows,
    int cols,
    int shift,
    int noise_sign,
    int32_t xB
) {
    if (!g_ctx.ready) return false;
    if (rows <= 0 || cols <= 0) return false;

    // Lock only for buffer allocation (shared state)
    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    NSUInteger inputBytes  = (NSUInteger)cols;
    NSUInteger outputBytes = (NSUInteger)rows;

    id<MTLBuffer> inputBuffer  = EggEnsureBuffer(g_ctx.device, &g_ctx.inputBuffer, &g_ctx.inputCapacity, inputBytes);
    id<MTLBuffer> outputBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.outputBuffer, &g_ctx.outputCapacity, outputBytes);
    id<MTLBuffer> noiseBuffer  = EggEnsureBuffer(g_ctx.device, &g_ctx.noiseBuffer, &g_ctx.noiseCapacity, outputBytes);
    id<MTLBuffer> paramsBuffer = g_ctx.paramsBuffer;
    if (!paramsBuffer) {
        paramsBuffer = [g_ctx.device newBufferWithLength:sizeof(EggMetalMatmulParams)
                                                options:MTLResourceStorageModeShared];
        g_ctx.paramsBuffer = paramsBuffer;
    }

    // Prefer using the persistent model buffer when possible.
    NSUInteger wOffset = 0;
    bool useModelBuffer = (g_ctx.modelBuffer && EggResolveModelWeightOffset(weights, &wOffset));
    id<MTLBuffer> weightBuffer = nil;
    NSUInteger weightBytes = 0;
    if (!useModelBuffer) {
        weightBytes = (NSUInteger)rows * (NSUInteger)cols;
        weightBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.weightBuffer, &g_ctx.weightCapacity, weightBytes);
    }

    if (!inputBuffer || !outputBuffer || !noiseBuffer || !paramsBuffer || (!useModelBuffer && !weightBuffer)) {
        dispatch_semaphore_signal(g_ctx.lock);
        fprintf(stderr, "[EGG METAL] Failed to allocate buffers for matmul.\n");
        return false;
    }

    // Copy input data while holding lock
    memcpy([inputBuffer contents], input, inputBytes);
    memcpy([noiseBuffer contents], noise_a, outputBytes);
    if (!useModelBuffer) {
        memcpy([weightBuffer contents], weights, weightBytes);
    }

    EggMetalMatmulParams params = {};
    params.rows = rows;
    params.cols = cols;
    params.shift = shift;
    params.noise_sign = noise_sign;
    params.xB = xB;
    memcpy([paramsBuffer contents], &params, sizeof(EggMetalMatmulParams));

    // Release lock before GPU work (Metal queues are thread-safe)
    dispatch_semaphore_signal(g_ctx.lock);

    // GPU work can proceed without lock
    @autoreleasepool {
        EggMetalBatchState *batchState = EggMetalGetBatchState();
        bool useBatch = batchState->inBatch && batchState->encoder != nil;
        
        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        
        if (useBatch) {
            // Use batch encoder
            encoder = batchState->encoder;
            commandBuffer = batchState->commandBuffer;
            batchState->operationCount++;
        } else {
            // Create new command buffer for immediate execution
            commandBuffer = [g_ctx.queue commandBuffer];
            encoder = [commandBuffer computeCommandEncoder];
            if (!encoder) {
                fprintf(stderr, "[EGG METAL] Failed to create compute encoder.\n");
                return false;
            }
        }

        [encoder setComputePipelineState:g_ctx.matmulPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        if (useModelBuffer) {
            [encoder setBuffer:g_ctx.modelBuffer offset:wOffset atIndex:1];
        } else {
            [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        }
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:noiseBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        NSUInteger maxThreads = g_ctx.matmulPipeline.maxTotalThreadsPerThreadgroup;
        if (maxThreads < 1) maxThreads = 1;
        NSUInteger threadsPerGroup = maxThreads < 256 ? maxThreads : 256;

        MTLSize gridSize = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        
        if (!useBatch) {
            // Immediate execution: commit and wait
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            // Read results (buffers are still valid)
            memcpy(output, [outputBuffer contents], outputBytes);
        } else {
            // Batched: register output for read-back after batch_end
            // Safety check: ensure pendingReads is allocated
            if (!batchState->pendingReads) {
                batchState->pendingReadCapacity = 8192;
                batchState->pendingReads = (struct EggMetalPendingRead *)calloc(batchState->pendingReadCapacity, sizeof(struct EggMetalPendingRead));
                if (!batchState->pendingReads) {
                    fprintf(stderr, "[EGG METAL] Error: failed to allocate pending reads array in matmul.\n");
                    return false;
                }
            }
            // Grow array if needed
            if (batchState->pendingReadCount >= batchState->pendingReadCapacity) {
                int newCapacity = batchState->pendingReadCapacity * 2;
                struct EggMetalPendingRead *newReads = (struct EggMetalPendingRead *)realloc(
                    batchState->pendingReads, 
                    newCapacity * sizeof(struct EggMetalPendingRead)
                );
                if (!newReads) {
                    fprintf(stderr, "[EGG METAL] Error: failed to grow pending reads array.\n");
                    return false;
                }
                batchState->pendingReads = newReads;
                batchState->pendingReadCapacity = newCapacity;
            }
            batchState->pendingReads[batchState->pendingReadCount].buffer = outputBuffer;
            batchState->pendingReads[batchState->pendingReadCount].hostPtr = output;
            batchState->pendingReads[batchState->pendingReadCount].size = outputBytes;
            batchState->pendingReadCount++;
        }
    }
    return true;
}

// GPU matmul that works directly with GPU buffers (no CPU copies for input/output)
extern "C" bool egg_gpu_matmul_perturbed_gpu(
    void *input_gpu,
    const int8_t *weights,
    void *output_gpu,
    const int8_t *noise_a,
    int rows,
    int cols,
    int shift,
    int noise_sign,
    int32_t xB
) {
    if (!g_ctx.ready) return false;
    if (rows <= 0 || cols <= 0) return false;
    
    id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)input_gpu;
    id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)output_gpu;
    
    if (!inputBuffer || !outputBuffer) return false;
    if ([inputBuffer length] < (NSUInteger)cols || [outputBuffer length] < (NSUInteger)rows) return false;
    
    // Lock only for buffer allocation
    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    NSUInteger outputBytes = (NSUInteger)rows;
    
    id<MTLBuffer> noiseBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.noiseBuffer, &g_ctx.noiseCapacity, outputBytes);
    id<MTLBuffer> paramsBuffer = g_ctx.paramsBuffer;
    if (!paramsBuffer) {
        paramsBuffer = [g_ctx.device newBufferWithLength:sizeof(EggMetalMatmulParams)
                                                options:MTLResourceStorageModeShared];
        g_ctx.paramsBuffer = paramsBuffer;
    }
    
    // Prefer using the persistent model buffer when possible
    NSUInteger wOffset = 0;
    bool useModelBuffer = (g_ctx.modelBuffer && EggResolveModelWeightOffset(weights, &wOffset));
    id<MTLBuffer> weightBuffer = nil;
    NSUInteger weightBytes = 0;
    if (!useModelBuffer) {
        weightBytes = (NSUInteger)rows * (NSUInteger)cols;
        weightBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.weightBuffer, &g_ctx.weightCapacity, weightBytes);
    }
    
    if (!noiseBuffer || !paramsBuffer || (!useModelBuffer && !weightBuffer)) {
        dispatch_semaphore_signal(g_ctx.lock);
        return false;
    }
    
    // Copy noise and weights (if needed) while holding lock
    memcpy([noiseBuffer contents], noise_a, outputBytes);
    if (!useModelBuffer) {
        memcpy([weightBuffer contents], weights, weightBytes);
    }
    
    EggMetalMatmulParams params = {};
    params.rows = rows;
    params.cols = cols;
    params.shift = shift;
    params.noise_sign = noise_sign;
    params.xB = xB;
    memcpy([paramsBuffer contents], &params, sizeof(EggMetalMatmulParams));
    
    dispatch_semaphore_signal(g_ctx.lock);
    
    // GPU work
    @autoreleasepool {
        EggMetalBatchState *batchState = EggMetalGetBatchState();
        bool useBatch = batchState->inBatch && batchState->encoder != nil;
        
        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        
        if (useBatch) {
            encoder = batchState->encoder;
            commandBuffer = batchState->commandBuffer;
            batchState->operationCount++;
        } else {
            commandBuffer = [g_ctx.queue commandBuffer];
            encoder = [commandBuffer computeCommandEncoder];
            if (!encoder) return false;
        }
        
        [encoder setComputePipelineState:g_ctx.matmulPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        if (useModelBuffer) {
            [encoder setBuffer:g_ctx.modelBuffer offset:wOffset atIndex:1];
        } else {
            [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        }
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:noiseBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
        
        NSUInteger maxThreads = g_ctx.matmulPipeline.maxTotalThreadsPerThreadgroup;
        if (maxThreads < 1) maxThreads = 1;
        NSUInteger threadsPerGroup = maxThreads < 256 ? maxThreads : 256;
        
        MTLSize gridSize = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        
        if (!useBatch) {
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        // In batch mode, output stays on GPU - no read-back needed
    }
    return true;
}

extern "C" bool egg_gpu_update_matrix(
    int8_t *weights,
    const int8_t *A_T,
    const int8_t *B_T,
    int rows,
    int cols,
    int pairs
) {
    if (!g_ctx.ready || !g_ctx.updatePipeline) return false;
    if (rows <= 0 || cols <= 0 || pairs <= 0) return false;

    // Lock only for buffer allocation (shared state)
    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    NSUInteger weightBytes = (NSUInteger)rows * (NSUInteger)cols;
    NSUInteger stridePairs = MAX_POP_PAIRS;
    NSUInteger aBytes = (NSUInteger)rows * stridePairs;
    NSUInteger bBytes = (NSUInteger)cols * stridePairs;

    id<MTLBuffer> weightBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.updateWeightsBuffer, &g_ctx.updateWeightsCapacity, weightBytes);
    id<MTLBuffer> aBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.updateABuffer, &g_ctx.updateACapacity, aBytes);
    id<MTLBuffer> bBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.updateBBuffer, &g_ctx.updateBCapacity, bBytes);
    if (!g_ctx.updateParamsBuffer) {
        g_ctx.updateParamsBuffer = [g_ctx.device newBufferWithLength:sizeof(EggMetalUpdateParams)
                                                             options:MTLResourceStorageModeShared];
    }

    if (!weightBuffer || !aBuffer || !bBuffer || !g_ctx.updateParamsBuffer) {
        dispatch_semaphore_signal(g_ctx.lock);
        fprintf(stderr, "[EGG METAL] Failed to allocate buffers for update_matrix.\n");
        return false;
    }

    // Copy input data while holding lock
    memcpy([weightBuffer contents], weights, weightBytes);
    memcpy([aBuffer contents], A_T, aBytes);
    memcpy([bBuffer contents], B_T, bBytes);

    EggMetalUpdateParams params = {};
    params.rows = rows;
    params.cols = cols;
    params.pairs = pairs;
    params.strideA = (int32_t)stridePairs;
    params.strideB = (int32_t)stridePairs;
    params.threshold = UPDATE_THRESHOLD;
    memcpy([g_ctx.updateParamsBuffer contents], &params, sizeof(EggMetalUpdateParams));

    // Release lock before GPU work (Metal queues are thread-safe)
    dispatch_semaphore_signal(g_ctx.lock);

    // GPU work can proceed without lock
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [g_ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            fprintf(stderr, "[EGG METAL] Failed to create compute encoder for update.\n");
            return false;
        }

        [encoder setComputePipelineState:g_ctx.updatePipeline];
        [encoder setBuffer:weightBuffer offset:0 atIndex:0];
        [encoder setBuffer:aBuffer offset:0 atIndex:1];
        [encoder setBuffer:bBuffer offset:0 atIndex:2];
        [encoder setBuffer:g_ctx.updateParamsBuffer offset:0 atIndex:3];

        NSUInteger total = (NSUInteger)rows * (NSUInteger)cols;
        NSUInteger maxThreads = g_ctx.updatePipeline.maxTotalThreadsPerThreadgroup;
        if (maxThreads < 1) maxThreads = 1;
        NSUInteger threadsPerGroup = maxThreads < 256 ? maxThreads : 256;
        MTLSize gridSize = MTLSizeMake(total, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        
        // Wait for completion - waitUntilCompleted is optimized by Metal runtime
        [commandBuffer waitUntilCompleted];

        // Read results (buffers are still valid)
        memcpy(weights, [weightBuffer contents], weightBytes);

        // Keep the persistent model buffer in sync if this matrix is part of it.
        // Re-acquire lock for model buffer update (shared state)
        dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
        if (g_ctx.modelBuffer && g_ctx.modelLayout.bound) {
            NSUInteger wOffset = 0;
            if (EggResolveModelWeightOffset(weights, &wOffset)) {
                uint8_t *base = static_cast<uint8_t *>([g_ctx.modelBuffer contents]);
                if (base && wOffset + weightBytes <= g_ctx.modelCapacity) {
                    memcpy(base + wOffset, [weightBuffer contents], weightBytes);
                }
            }
        }
        dispatch_semaphore_signal(g_ctx.lock);
    }
    return true;
}


// GPU-resident activation buffer management
struct EggGpuActivationBuffersImpl {
    id<MTLBuffer> gpu_x;
    id<MTLBuffer> gpu_buf1;
    id<MTLBuffer> gpu_buf2;
    id<MTLBuffer> gpu_residual;
    id<MTLBuffer> gpu_ft;
    id<MTLBuffer> gpu_ht;
    id<MTLBuffer> gpu_gated_past;
    id<MTLBuffer> gpu_rnn_state[N_LAYERS];
};

extern "C" EggGpuActivationBuffers* egg_gpu_alloc_activation_buffers(void) {
    if (!g_ctx.ready) return NULL;
    
    EggGpuActivationBuffersImpl *impl = (EggGpuActivationBuffersImpl *)calloc(1, sizeof(EggGpuActivationBuffersImpl));
    if (!impl) return NULL;
    
    NSUInteger hiddenDim = HIDDEN_DIM;
    NSUInteger buf1Size = HIDDEN_DIM * 4;
    
    impl->gpu_x = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    impl->gpu_buf1 = [g_ctx.device newBufferWithLength:buf1Size options:MTLResourceStorageModePrivate];
    impl->gpu_buf2 = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    impl->gpu_residual = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    impl->gpu_ft = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    impl->gpu_ht = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    impl->gpu_gated_past = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
    
    // Allocate RNN state buffers for each layer
    for (int l = 0; l < N_LAYERS; l++) {
        impl->gpu_rnn_state[l] = [g_ctx.device newBufferWithLength:hiddenDim options:MTLResourceStorageModePrivate];
        if (!impl->gpu_rnn_state[l]) {
            egg_gpu_free_activation_buffers((EggGpuActivationBuffers*)impl);
            return NULL;
        }
    }
    
    if (!impl->gpu_x || !impl->gpu_buf1 || !impl->gpu_buf2 || !impl->gpu_residual ||
        !impl->gpu_ft || !impl->gpu_ht || !impl->gpu_gated_past) {
        egg_gpu_free_activation_buffers((EggGpuActivationBuffers*)impl);
        return NULL;
    }
    
    return (EggGpuActivationBuffers*)impl;
}

extern "C" void egg_gpu_free_activation_buffers(EggGpuActivationBuffers *bufs) {
    if (!bufs) return;
    EggGpuActivationBuffersImpl *impl = (EggGpuActivationBuffersImpl *)bufs;
    free(impl);
}

extern "C" bool egg_gpu_clipped_add(void *gpu_a, void *gpu_b, void *gpu_out, int count) {
    if (!g_ctx.ready || !g_ctx.clippedAddPipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)gpu_a;
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)gpu_b;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    [batchState->encoder setComputePipelineState:g_ctx.clippedAddPipeline];
    [batchState->encoder setBuffer:bufA offset:0 atIndex:0];
    [batchState->encoder setBuffer:bufB offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:2];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.clippedAddPipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_layer_norm(void *gpu_x, const int8_t *ln_weights_host, void *gpu_out, int dim) {
    if (!g_ctx.ready || !g_ctx.layerNormPipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    static id<MTLBuffer> lnWeightsBuffer = nil;
    static NSUInteger lnWeightsCapacity = 0;
    NSUInteger weightBytes = (NSUInteger)dim;
    id<MTLBuffer> weightBuf = EggEnsureBuffer(g_ctx.device, &lnWeightsBuffer, &lnWeightsCapacity, weightBytes);
    if (!weightBuf) return false;
    
    memcpy([weightBuf contents], ln_weights_host, weightBytes);
    
    id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)gpu_x;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    static id<MTLBuffer> dimBuffer = nil;
    if (!dimBuffer) {
        dimBuffer = [g_ctx.device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    }
    memcpy([dimBuffer contents], &dim, sizeof(int));
    
    [batchState->encoder setComputePipelineState:g_ctx.layerNormPipeline];
    [batchState->encoder setBuffer:bufX offset:0 atIndex:0];
    [batchState->encoder setBuffer:weightBuf offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:2];
    [batchState->encoder setBuffer:dimBuffer offset:0 atIndex:3];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)dim, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.layerNormPipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_gru_gate(void *gpu_ft, void *gpu_h, void *gpu_out, int count) {
    if (!g_ctx.ready || !g_ctx.gruGatePipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufFt = (__bridge id<MTLBuffer>)gpu_ft;
    id<MTLBuffer> bufH = (__bridge id<MTLBuffer>)gpu_h;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    [batchState->encoder setComputePipelineState:g_ctx.gruGatePipeline];
    [batchState->encoder setBuffer:bufFt offset:0 atIndex:0];
    [batchState->encoder setBuffer:bufH offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:2];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.gruGatePipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_gru_state_update(void *gpu_h, void *gpu_ft, void *gpu_ht, void *gpu_out, int count) {
    if (!g_ctx.ready || !g_ctx.gruStateUpdatePipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufH = (__bridge id<MTLBuffer>)gpu_h;
    id<MTLBuffer> bufFt = (__bridge id<MTLBuffer>)gpu_ft;
    id<MTLBuffer> bufHt = (__bridge id<MTLBuffer>)gpu_ht;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    [batchState->encoder setComputePipelineState:g_ctx.gruStateUpdatePipeline];
    [batchState->encoder setBuffer:bufH offset:0 atIndex:0];
    [batchState->encoder setBuffer:bufFt offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufHt offset:0 atIndex:2];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:3];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.gruStateUpdatePipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_copy_to_buffer(void *gpu_buffer, const void *cpu_data, size_t bytes) {
    if (!g_ctx.ready) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpu_buffer;
    if (!buf || [buf length] < bytes) return false;
    
    // For Shared buffers, use direct memcpy (faster and doesn't conflict with compute encoder)
    if ([buf storageMode] == MTLStorageModeShared) {
        memcpy([buf contents], cpu_data, bytes);
        return true;
    }
    
    // For Private buffers, need blit encoder but only if not in batch
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch) {
        // Not in batch, can create our own command buffer
        id<MTLCommandBuffer> cmdBuf = [g_ctx.queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];
        if (blitEncoder) {
            id<MTLBuffer> tempBuf = [g_ctx.device newBufferWithBytes:cpu_data length:bytes options:MTLResourceStorageModeShared];
            [blitEncoder copyFromBuffer:tempBuf sourceOffset:0 toBuffer:buf destinationOffset:0 size:bytes];
            [blitEncoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            return true;
        }
    }
    
    // Can't copy to Private buffer while in batch - return false
    return false;
}

extern "C" bool egg_gpu_clipped_add_scalar(void *gpu_a, int scalar, void *gpu_out, int count) {
    if (!g_ctx.ready || !g_ctx.clippedAddScalarPipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)gpu_a;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    static id<MTLBuffer> scalarBuffer = nil;
    if (!scalarBuffer) {
        scalarBuffer = [g_ctx.device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    }
    memcpy([scalarBuffer contents], &scalar, sizeof(int));
    
    [batchState->encoder setComputePipelineState:g_ctx.clippedAddScalarPipeline];
    [batchState->encoder setBuffer:bufA offset:0 atIndex:0];
    [batchState->encoder setBuffer:scalarBuffer offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:2];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.clippedAddScalarPipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_clipped_add_three(void *gpu_a, void *gpu_b, void *gpu_c, void *gpu_out, int count) {
    if (!g_ctx.ready || !g_ctx.clippedAddThreePipeline) return false;
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)gpu_a;
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)gpu_b;
    id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)gpu_c;
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    [batchState->encoder setComputePipelineState:g_ctx.clippedAddThreePipeline];
    [batchState->encoder setBuffer:bufA offset:0 atIndex:0];
    [batchState->encoder setBuffer:bufB offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufC offset:0 atIndex:2];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:3];
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.clippedAddThreePipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}

extern "C" bool egg_gpu_dot_product(void *gpu_a, void *gpu_b, int32_t *result, int count) {
    if (!g_ctx.ready || !g_ctx.dotProductPipeline) return false;
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)gpu_a;
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)gpu_b;
    
    // Create atomic result buffer (Shared so we can read it)
    // Use static buffer - batches are single-threaded so this is safe
    static id<MTLBuffer> resultBuffer = nil;
    if (!resultBuffer) {
        resultBuffer = [g_ctx.device newBufferWithLength:sizeof(int32_t) options:MTLResourceStorageModeShared];
    }
    if (!resultBuffer) return false;
    
    // Initialize to zero
    int32_t zero = 0;
    memcpy([resultBuffer contents], &zero, sizeof(int32_t));
    
    // Use batch encoder if available
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    bool useBatch = (batchState->inBatch && batchState->encoder);
    
    if (useBatch) {
        // In batch mode - encode into batch, result will be available after batch_end
        [batchState->encoder setComputePipelineState:g_ctx.dotProductPipeline];
        [batchState->encoder setBuffer:bufA offset:0 atIndex:0];
        [batchState->encoder setBuffer:bufB offset:0 atIndex:1];
        [batchState->encoder setBuffer:resultBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
        NSUInteger threadsPerGroup = g_ctx.dotProductPipeline.maxTotalThreadsPerThreadgroup;
        if (threadsPerGroup < 1) threadsPerGroup = 1;
        if (threadsPerGroup > 256) threadsPerGroup = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        
        [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        
        // Store result buffer pointer for later read-back
        // We'll read it after batch_end, but for now return 0 and caller should handle
        // Actually, we need to compute this synchronously for matmul to work
        // So we'll compute on CPU for now when in batch mode
        return false; // Fall back to CPU computation
    } else {
        // Not in batch - can compute immediately
        id<MTLCommandBuffer> cmdBuf = [g_ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (!encoder) return false;
        
        [encoder setComputePipelineState:g_ctx.dotProductPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:resultBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
        NSUInteger threadsPerGroup = g_ctx.dotProductPipeline.maxTotalThreadsPerThreadgroup;
        if (threadsPerGroup < 1) threadsPerGroup = 1;
        if (threadsPerGroup > 256) threadsPerGroup = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(result, [resultBuffer contents], sizeof(int32_t));
        return true;
    }
}

extern "C" bool egg_gpu_copy_from_buffer(void *cpu_data, void *gpu_buffer, size_t bytes) {
    if (!g_ctx.ready) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpu_buffer;
    if (!buf || [buf length] < bytes) return false;
    
    if ([buf storageMode] == MTLStorageModeShared) {
        memcpy(cpu_data, [buf contents], bytes);
        return true;
    }
    return false;
}

// Helper: Copy embedding row to GPU buffer
extern "C" bool egg_gpu_copy_embedding_row(void *gpu_buffer, const int8_t *embedding_host, int vocab_idx) {
    if (!g_ctx.ready) return false;
    const int8_t *row = &embedding_host[vocab_idx * HIDDEN_DIM];
    return egg_gpu_copy_to_buffer(gpu_buffer, row, HIDDEN_DIM);
}

// Helper: Allocate temporary GPU buffer (Shared mode for easy CPU access)
extern "C" void* egg_gpu_alloc_temp_buffer(size_t bytes) {
    if (!g_ctx.ready) return NULL;
    id<MTLBuffer> buf = [g_ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;  // Transfer ownership to caller
}

// Helper: Free temporary GPU buffer
extern "C" void egg_gpu_free_temp_buffer(void *gpu_buffer) {
    if (gpu_buffer) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)gpu_buffer;
        // ARC will release it
    }
}

// Helper: Get pointer to buffer contents (for Shared buffers only)
extern "C" void* egg_gpu_get_buffer_contents(void *gpu_buffer) {
    if (!gpu_buffer) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpu_buffer;
    if ([buf storageMode] != MTLStorageModeShared) return NULL;
    return [buf contents];
}

// Helper: Copy bias and add to GPU buffer (for GRU biases)
extern "C" bool egg_gpu_add_bias(void *gpu_a, void *gpu_b, const int8_t *bias_host, void *gpu_out, int count) {
    // First add a + b, then add bias
    if (!egg_gpu_clipped_add(gpu_a, gpu_b, gpu_out, count)) return false;
    
    // Upload bias to temporary buffer and add
    static id<MTLBuffer> biasBuffer = nil;
    static NSUInteger biasCapacity = 0;
    id<MTLBuffer> biasBuf = EggEnsureBuffer(g_ctx.device, &biasBuffer, &biasCapacity, (NSUInteger)count);
    if (!biasBuf) return false;
    memcpy([biasBuf contents], bias_host, count);
    
    EggMetalBatchState *batchState = EggMetalGetBatchState();
    if (!batchState->inBatch || !batchState->encoder) return false;
    
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)gpu_out;
    
    [batchState->encoder setComputePipelineState:g_ctx.clippedAddPipeline];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:0];
    [batchState->encoder setBuffer:biasBuf offset:0 atIndex:1];
    [batchState->encoder setBuffer:bufOut offset:0 atIndex:2];  // In-place
    
    MTLSize gridSize = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threadsPerGroup = g_ctx.clippedAddPipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup < 1) threadsPerGroup = 1;
    if (threadsPerGroup > 256) threadsPerGroup = 256;
    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    
    [batchState->encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    return true;
}
