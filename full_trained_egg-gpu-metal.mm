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
    id<MTLBuffer> inputBuffer;
    NSUInteger inputCapacity;
    id<MTLBuffer> weightBuffer;
    NSUInteger weightCapacity;
    id<MTLBuffer> outputBuffer;
    NSUInteger outputCapacity;
    id<MTLBuffer> noiseBuffer;
    NSUInteger noiseCapacity;
    id<MTLBuffer> paramsBuffer;
    dispatch_semaphore_t lock;
} EggMetalContext;

static EggMetalContext g_ctx = {};

static NSString *EggMetalMatmulSource(void) {
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
        NSString *source = EggMetalMatmulSource();
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
    g_ctx.matmulPipeline = nil;
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

