#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <string.h>

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
    id<MTLBuffer> updateWeightsBuffer;
    NSUInteger updateWeightsCapacity;
    id<MTLBuffer> updateABuffer;
    NSUInteger updateACapacity;
    id<MTLBuffer> updateBBuffer;
    NSUInteger updateBCapacity;
    dispatch_semaphore_t lock;
} EggMetalContext;

static EggMetalContext g_ctx = {};

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
    *buffer = [device newBufferWithLength:newCapacity options:MTLResourceStorageModeShared];
    *capacity = newCapacity;
    return *buffer;
}

static void EggLogError(const char *label, NSError *error) {
    if (!error) return;
    fprintf(stderr, "[EGG METAL] %s failed: %s\n", label, error.localizedDescription.UTF8String);
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

        g_ctx.lock = dispatch_semaphore_create(1);
        g_ctx.ready = true;
        fprintf(stdout, "[EGG METAL] Initialized on device: %s\n", g_ctx.device.name.UTF8String);
        return true;
    }
}

extern "C" void egg_gpu_metal_shutdown(void) {
    g_ctx.ready = false;
    g_ctx.inputBuffer = nil;
    g_ctx.weightBuffer = nil;
    g_ctx.outputBuffer = nil;
    g_ctx.noiseBuffer = nil;
    g_ctx.paramsBuffer = nil;
    g_ctx.updateParamsBuffer = nil;
    g_ctx.updateWeightsBuffer = nil;
    g_ctx.updateABuffer = nil;
    g_ctx.updateBBuffer = nil;
    g_ctx.updateWeightsCapacity = 0;
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

    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    @autoreleasepool {
        NSUInteger inputBytes = (NSUInteger)cols;
        NSUInteger weightBytes = (NSUInteger)rows * (NSUInteger)cols;
        NSUInteger outputBytes = (NSUInteger)rows;

        id<MTLBuffer> inputBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.inputBuffer, &g_ctx.inputCapacity, inputBytes);
        id<MTLBuffer> weightBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.weightBuffer, &g_ctx.weightCapacity, weightBytes);
        id<MTLBuffer> outputBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.outputBuffer, &g_ctx.outputCapacity, outputBytes);
        id<MTLBuffer> noiseBuffer = EggEnsureBuffer(g_ctx.device, &g_ctx.noiseBuffer, &g_ctx.noiseCapacity, outputBytes);
        id<MTLBuffer> paramsBuffer = g_ctx.paramsBuffer;
        if (!paramsBuffer) {
            paramsBuffer = [g_ctx.device newBufferWithLength:sizeof(EggMetalMatmulParams)
                                                    options:MTLResourceStorageModeShared];
            g_ctx.paramsBuffer = paramsBuffer;
        }

        if (!inputBuffer || !weightBuffer || !outputBuffer || !noiseBuffer || !paramsBuffer) {
            dispatch_semaphore_signal(g_ctx.lock);
            fprintf(stderr, "[EGG METAL] Failed to allocate buffers for matmul.\n");
            return false;
        }

        memcpy([inputBuffer contents], input, inputBytes);
        memcpy([weightBuffer contents], weights, weightBytes);
        memcpy([noiseBuffer contents], noise_a, outputBytes);

        EggMetalMatmulParams params = {};
        params.rows = rows;
        params.cols = cols;
        params.shift = shift;
        params.noise_sign = noise_sign;
        params.xB = xB;
        memcpy([paramsBuffer contents], &params, sizeof(EggMetalMatmulParams));

        id<MTLCommandBuffer> commandBuffer = [g_ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            dispatch_semaphore_signal(g_ctx.lock);
            fprintf(stderr, "[EGG METAL] Failed to create compute encoder.\n");
            return false;
        }

        [encoder setComputePipelineState:g_ctx.matmulPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:noiseBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        NSUInteger maxThreads = g_ctx.matmulPipeline.maxTotalThreadsPerThreadgroup;
        if (maxThreads < 1) maxThreads = 1;
        NSUInteger threadsPerGroup = maxThreads < 256 ? maxThreads : 256;

        MTLSize gridSize = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(output, [outputBuffer contents], outputBytes);
    }
    dispatch_semaphore_signal(g_ctx.lock);
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

    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);
    @autoreleasepool {
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

        id<MTLCommandBuffer> commandBuffer = [g_ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            dispatch_semaphore_signal(g_ctx.lock);
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
        [commandBuffer waitUntilCompleted];

        memcpy(weights, [weightBuffer contents], weightBytes);
    }
    dispatch_semaphore_signal(g_ctx.lock);
    return true;
}

