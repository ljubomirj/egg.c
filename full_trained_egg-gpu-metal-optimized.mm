#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "egg_config.h"
#include "egg_gpu_metal.h"
#include "egg_gpu_metal_optimized.h"

// Define missing constants
#define SIGMA_SHIFT_VECTOR (SIGMA_SHIFT - 2)

// Forward declare EggModel structure
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM];
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// Optimized Metal implementation with fused kernels and population batching

// Configuration for optimized implementation
#define OPTIMIZED_MAX_POPULATION_SIZE 256
#define OPTIMIZED_THREADGROUP_SIZE 256
#define OPTIMIZED_WARPS_PER_THREADGROUP 8
#define OPTIMIZED_SHARED_MEM_SIZE (HIDDEN_DIM * 4 * OPTIMIZED_WARPS_PER_THREADGROUP)

typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t shift;
    int32_t noise_sign;
    int32_t xB;
    int32_t seed;
    int32_t layer_idx;
    int32_t pad[1];
} OptimizedMetalParams;

typedef struct {
    bool ready;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    // Kernel functions
    id<MTLFunction> embeddingAndGRUFusedFunc;
    id<MTLFunction> mlpFusedFunc;
    id<MTLFunction> headFusedFunc;
    id<MTLFunction> updateMatrixFunc;

    // Pipeline states
    id<MTLComputePipelineState> embeddingAndGRUFused;
    id<MTLComputePipelineState> mlpFused;
    id<MTLComputePipelineState> headFused;
    id<MTLComputePipelineState> updateMatrixKernel;

    // Large buffer for all model weights (GPU-resident)
    id<MTLBuffer> modelBuffer;
    size_t modelBufferSize;

    // Population batch buffer
    id<MTLBuffer> populationBuffer;
    size_t populationBufferSize;

    // Temporary workspace buffer
    id<MTLBuffer> workspaceBuffer;
    size_t workspaceBufferSize;

    // Results buffer
    id<MTLBuffer> resultsBuffer;
    size_t resultsBufferSize;

    dispatch_semaphore_t lock;
} OptimizedMetalContext;

static OptimizedMetalContext g_opt_ctx = {};

// Optimized Metal shader source with fused kernels
static NSString *OptimizedMetalShaderSource(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "#define FIXED_POINT %d\n"
         "#define SIGMA_SHIFT %d\n"
         "#define SIGMA_SHIFT_VECTOR %d\n"
         "#define MAX_VAL %d\n"
         "#define MIN_VAL %d\n"
         "#define HIDDEN_DIM %d\n"
         "#define N_LAYERS %d\n"
         "#define VOCAB_SIZE %d\n"
         "#define POPULATION_SIZE %d\n"
         "#define SIGMA_SHIFT_VECTOR %d\n"
         "\n"
         // Parameter structure passed to kernels
         "struct OptimizedMetalParams {\n"
         "    int32_t rows;\n"
         "    int32_t cols;\n"
         "    int32_t shift;\n"
         "    int32_t noise_sign;\n"
         "    int32_t xB;\n"
         "    int32_t seed;\n"
         "    int32_t layer_idx;\n"
         "    int32_t pad[1];\n"
         "};\n"
         "\n"
         // Fast hash-based RNG
         "inline uint hash_rng(uint s, uint idx) {\n"
         "  uint x = s + idx * 0x9e3779b9u;\n"
         "  x ^= x >> 16; x *= 0x85ebca6b;\n"
         "  x ^= x >> 13; x *= 0xc2b2ae35;\n"
         "  x ^= x >> 16;\n"
         "  return x;\n"
         "}\n"
         "\n"
         "inline char noise_from_hash(uint s, uint idx) {\n"
         "  uint r = hash_rng(s, idx);\n"
         "  return (char)((r & 1 ? 1 : -1) * ((r >> 1) & 31));\n"
         "}\n"
         "\n"
         // SIMD group reduction for efficient parallel reduction
         "inline int simd_reduce_sum(threadgroup int *shared, int value, uint lane, uint simd_size, uint tid) {\n"
         "  // First reduce within simd\n"
         "  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {\n"
         "    value += simd_shuffle_down(value, offset);\n"
         "  }\n"
         "  \n"
         "  // First lane writes to shared memory\n"
         "  if (lane == 0) {\n"
         "    shared[ tid / simd_size ] = value;\n"
         "  }\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  \n"
         "  // Reduce across simds (simplified)\n"
         "  if (tid < 32) {\n"
         "    value = 0;\n"
         "    for (uint i = tid; i < 256; i += 32) {\n"
         "      value += shared[i];\n"
         "    }\n"
         "    shared[0] = value;\n"
         "  }\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  \n"
         "  return shared[0];\n"
         "}\n"
         "\n"
         // Optimized matmul with perturbation
         "inline int perturbed_matmul(\n"
         "    device const char *input,\n"
         "    device const char *weights,\n"
         "    const int rows,\n"
         "    const int cols,\n"
         "    const int seed,\n"
         "    const int noise_sign,\n"
         "    const uint idx,\n"
         "    const int shift\n"
         ") {\n"
         "  int acc = 0;\n"
         "  for (int c = 0; c < cols; ++c) {\n"
         "    char w = weights[c];\n"
         "    char x = input[c];\n"
         "    acc += int(x) * int(w);\n"
         "  }\n"
         "  \n"
         "  if (noise_sign != 0) {\n"
         "    // Compute rank-1 perturbation efficiently\n"
         "    char a = noise_from_hash(seed, idx);\n"
         "    int xB = 0;\n"
         "    for (int c = 0; c < cols; ++c) {\n"
         "      char b = noise_from_hash(seed + HIDDEN_DIM, c);\n"
         "      xB += int(input[c]) * int(b);\n"
         "    }\n"
         "    \n"
         "    int noise = (xB * int(a)) * noise_sign;\n"
         "    acc += (noise >> (FIXED_POINT + SIGMA_SHIFT));\n"
         "  }\n"
         "  \n"
         "  return acc >> shift;\n"
         "}\n"
         "\n"
         // Layer normalization optimized for GPU
         "inline void layer_norm(\n"
         "    device const char *input,\n"
         "    device const char *weights,\n"
         "    device char *output,\n"
         "    threadgroup int *shared_mem,\n"
         "    const uint tid,\n"
         "    const uint simd_size\n"
         ") {\n"
         "  // Compute sum of absolute values\n"
         "  int local_sum = 0;\n"
         "  for (uint i = tid; i < HIDDEN_DIM; i += simd_size) {\n"
         "    local_sum += abs(int(input[i]));\n"
         "  }\n"
         "  \n"
         "  int total_sum = simd_reduce_sum(shared_mem, local_sum, tid %% 32, 32, tid);\n"
         "  \n"
         "  int mean = (total_sum > 0) ? (total_sum / HIDDEN_DIM) : 1;\n"
         "  \n"
         "  // Apply normalization\n"
         "  for (uint i = tid; i < HIDDEN_DIM; i += simd_size) {\n"
         "    int val = (int(input[i]) * int(weights[i])) / mean;\n"
         "    output[i] = char(clamp(val, MIN_VAL, MAX_VAL));\n"
         "  }\n"
         "}\n"
         "\n"
         "// Fused kernel: Embedding lookup + GRU forward pass for one layer\n"
         "kernel void fused_embedding_gru(\n"
         "    device const char *model_data [[buffer(0)]],\n"
         "    device const uint8_t *tokens [[buffer(1)]],\n"
         "    device char *population_data [[buffer(2)]],\n"
         "    constant struct OptimizedMetalParams &params [[buffer(3)]],\n"
         "    threadgroup int *shared_mem [[threadgroup(0)]],\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    uint2 tgid [[threadgroup_position_in_grid]]\n"
         ") {\n"
         "  uint pop_idx = tgid.y;  // Which population member\n"
         "  uint layer = params.layer_idx;\n"
         "  uint seq_pos = tid.x;\n"
         "  \n"
         "  if (seq_pos >= HIDDEN_DIM) return;\n"
         "  \n"
         "  // Each threadgroup handles one population member and one sequence position\n"
         "  uint population_offset = pop_idx * (N_LAYERS * HIDDEN_DIM * 3); // Space for x, h, workspace\n"
         "  uint layer_offset = layer * HIDDEN_DIM;\n"
         "  \n"
         "  device char *x = population_data + population_offset + layer_offset;\n"
         "  device char *h = population_data + population_offset + (N_LAYERS * HIDDEN_DIM) + layer_offset;\n"
         "  device char *workspace = population_data + population_offset + (2 * N_LAYERS * HIDDEN_DIM) + layer_offset;\n"
         "  \n"
         "  // Get token for this timestep\n"
         "  uint8_t token = tokens[0]; // Single token for inference, or vary for training\n"
         "  \n"
         "  // Embedding lookup (first layer only)\n"
         "  if (layer == 0) {\n"
         "    device const char *embedding = model_data + token * HIDDEN_DIM;\n"
         "    char emb_val = embedding[seq_pos];\n"
         "    \n"
         "    // Add perturbation\n"
         "    char noise_a = noise_from_hash(params.seed, pop_idx * HIDDEN_DIM + seq_pos);\n"
         "    char noise_b = noise_from_hash(params.seed + HIDDEN_DIM, seq_pos);\n"
         "    int perturb = ((int)noise_a * (int)noise_b * params.noise_sign) >> (FIXED_POINT + SIGMA_SHIFT);\n"
         "    \n"
         "    x[seq_pos] = char(clamp((int)emb_val + perturb, MIN_VAL, MAX_VAL));\n"
         "  }\n"
         "  \n"
         "  // Layer normalization\n"
         "  device const char *ln_weights = model_data + VOCAB_SIZE * HIDDEN_DIM + layer * 2 * HIDDEN_DIM;\n"
         "  layer_norm(x, ln_weights, workspace, shared_mem, seq_pos, 32);\n"
         "  \n"
         "  // GRU computation (4 gates: z, r, zh, rh)\n"
         "  device const char *gru_weights = model_data + VOCAB_SIZE * HIDDEN_DIM + N_LAYERS * 2 * HIDDEN_DIM + layer * 4 * HIDDEN_DIM * HIDDEN_DIM;\n"
         "  device const char *gru_biases = model_data + VOCAB_SIZE * HIDDEN_DIM + N_LAYERS * 2 * HIDDEN_DIM + N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM + layer * 2 * HIDDEN_DIM;\n"
         "  \n"
         "  // Compute all 4 gates in parallel\n"
         "  int gate_vals[4] = {0};\n"
         "  for (int gate = 0; gate < 4; gate++) {\n"
         "    device const char *gate_weights = gru_weights + gate * HIDDEN_DIM * HIDDEN_DIM;\n"
         "    gate_vals[gate] = perturbed_matmul(\n"
         "        workspace, gate_weights, HIDDEN_DIM, HIDDEN_DIM,\n"
         "        params.seed + layer * 100 + gate + 1, params.noise_sign,\n"
         "        pop_idx * HIDDEN_DIM + seq_pos, 8\n"
         "    );\n"
         "  }\n"
         "  \n"
         "  // Apply sigmoid-like activation and compute new state\n"
         "  int h_prev = (int)h[seq_pos];\n"
         "  int z_val = gate_vals[0] + gru_biases[0 * HIDDEN_DIM + seq_pos];\n"
         "  int r_val = gate_vals[1] + gru_biases[1 * HIDDEN_DIM + seq_pos];\n"
         "  int zh_val = gate_vals[2];\n"
         "  int rh_val = gate_vals[3];\n"
         "  \n"
         "  // Simplified GRU update\n"
         "  int z_act = clamp((z_val + 128) >> 8, 0, 255);  // Sigmoid approximation\n"
         "  int r_act = clamp((r_val + 128) >> 8, 0, 255);\n"
         "  \n"
         "  int h_candidate = clamp(((h_prev * r_act) >> 8) + zh_val, MIN_VAL, MAX_VAL);\n"
         "  int h_new = h_prev + (((h_candidate - h_prev) * (255 - z_act)) >> 8);\n"
         "  \n"
         "  h[seq_pos] = char(h_new);\n"
         "  x[seq_pos] = char(h_new);\n"
         "}\n"
         "\n"
         "// Fused MLP kernel: Layer norm + Expand + Project + Residual\n"
         "kernel void fused_mlp(\n"
         "    device const char *model_data [[buffer(0)]],\n"
         "    device char *population_data [[buffer(1)]],\n"
         "    constant struct OptimizedMetalParams &params [[buffer(2)]],\n"
         "    threadgroup int *shared_mem [[threadgroup(0)]],\n"
         "    uint2 tid [[thread_position_in_grid]],\n"
         "    uint2 tgid [[threadgroup_position_in_grid]]\n"
         ") {\n"
         "  uint pop_idx = tgid.y;\n"
         "  uint layer = params.layer_idx;\n"
         "  uint seq_pos = tid.x;\n"
         "  \n"
         "  if (seq_pos >= HIDDEN_DIM) return;\n"
         "  \n"
         "  uint population_offset = pop_idx * (N_LAYERS * HIDDEN_DIM * 3);\n"
         "  uint layer_offset = layer * HIDDEN_DIM;\n"
         "  \n"
         "  device char *x = population_data + population_offset + layer_offset;\n"
         "  device char *workspace = population_data + population_offset + (2 * N_LAYERS * HIDDEN_DIM) + layer_offset;\n"
         "  \n"
         "  // Store residual\n"
         "  char residual = x[seq_pos];\n"
         "  \n"
         "  // Layer norm\n"
         "  device const char *ln_weights = model_data + VOCAB_SIZE * HIDDEN_DIM + N_LAYERS * 2 * HIDDEN_DIM + layer * 2 * HIDDEN_DIM;\n"
         "  layer_norm(x, ln_weights, workspace, shared_mem, seq_pos, 32);\n"
         "  \n"
         "  // MLP Expand: HIDDEN_DIM -> 4*HIDDEN_DIM\n"
         "  device const char *mlp_expand_weights = model_data + VOCAB_SIZE * HIDDEN_DIM + N_LAYERS * 2 * HIDDEN_DIM + N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM + N_LAYERS * 2 * HIDDEN_DIM + layer * 2 * HIDDEN_DIM * HIDDEN_DIM * 4;\n"
         "  \n"
         "  threadgroup char expanded_shared[32 * 4]; // Each thread contributes 4 values\n"
         "  \n"
         "  // Each thread computes 4 elements of the expanded vector\n"
         "  for (int i = 0; i < 4; i++) {\n"
         "    int acc = 0;\n"
         "    for (int j = 0; j < HIDDEN_DIM; j++) {\n"
         "      acc += (int)workspace[j] * (int)mlp_expand_weights[j * (HIDDEN_DIM * 4) + seq_pos * 4 + i];\n"
         "    }\n"
         "    expanded_shared[(seq_pos %% 32) * 4 + i] = char(clamp(acc >> 8, MIN_VAL, MAX_VAL));\n"
         "  }\n"
         "  \n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  \n"
         "  // MLP Project: 4*HIDDEN_DIM -> HIDDEN_DIM\n"
         "  device const char *mlp_proj_weights = mlp_expand_weights + HIDDEN_DIM * (HIDDEN_DIM * 4);\n"
         "  \n"
         "  int proj_acc = 0;\n"
         "  for (int i = 0; i < 4; i++) {\n"
         "    char expanded_val = expanded_shared[(seq_pos %% 32) + i * 32];\n"
         "    proj_acc += (int)expanded_val * (int)mlp_proj_weights[seq_pos * 4 + i];\n"
         "  }\n"
         "  \n"
         "  // Add residual and store result\n"
         "  int result = clamp((proj_acc >> 8) + (int)residual, MIN_VAL, MAX_VAL);\n"
         "  x[seq_pos] = char(result);\n"
         "}\n"
         "\n"
         "// Optimized matrix update kernel\n"
         "kernel void update_matrix_optimized(\n"
         "    device char *weights [[buffer(0)]],\n"
         "    device const char *A_T [[buffer(1)]],\n"
         "    device const char *B_T [[buffer(2)]],\n"
         "    device const int *fitnesses [[buffer(3)]],\n"
         "    constant struct OptimizedMetalParams &params [[buffer(4)]],\n"
         "    uint2 tid [[thread_position_in_grid]]\n"
         ") {\n"
         "  uint row = tid.x;\n"
         "  uint col = tid.y;\n"
         "  \n"
         "  if (row >= params.rows || col >= params.cols) return;\n"
         "  \n"
         "  // Compute vote for this weight\n"
         "  int vote = 0;\n"
         "  for (int p = 0; p < POPULATION_SIZE / 2; p++) {\n"
         "    int fitness = fitnesses[p];\n"
         "    if (fitness != 0) {\n"
         "      char a_val = A_T[row * (POPULATION_SIZE / 2) + p];\n"
         "      char b_val = B_T[col * (POPULATION_SIZE / 2) + p];\n"
         "      vote += (int)a_val * (int)b_val * fitness;\n"
         "    }\n"
         "  }\n"
         "  \n"
         "  // Update weight if vote exceeds threshold\n"
         "  uint weight_idx = row * params.cols + col;\n"
         "  char current_weight = weights[weight_idx];\n"
         "  \n"
         "  const int UPDATE_THRESHOLD = 160;\n"
         "  \n"
         "  if (vote > UPDATE_THRESHOLD && current_weight < MAX_VAL) {\n"
         "    weights[weight_idx]++;\n"
         "  } else if (vote < -UPDATE_THRESHOLD && current_weight > MIN_VAL) {\n"
         "    weights[weight_idx]--;\n"
         "  }\n"
         "}\n"
         "\n"
         "// Compute cross-entropy loss for vocabulary\n"
         "kernel void compute_loss(\n"
         "    device const char *logits [[buffer(0)]],\n"
         "    device const uint8_t *targets [[buffer(1)]],\n"
         "    device int *losses [[buffer(2)]],\n"
         "    uint tid [[thread_position_in_grid]]\n"
         ") {\n"
         "  if (tid >= POPULATION_SIZE) return;\n"
         "  \n"
         "  device const char *my_logits = logits + tid * VOCAB_SIZE;\n"
         "  uint8_t target = targets[tid];\n"
         "  \n"
         "  // Find max for numerical stability\n"
         "  int max_val = -128;\n"
         "  for (int i = 0; i < VOCAB_SIZE; i++) {\n"
         "    if (my_logits[i] > max_val) max_val = my_logits[i];\n"
         "  }\n"
         "  \n"
         "  // Compute sum of exp\n"
         "  int sum_exp = 0;\n"
         "  for (int i = 0; i < VOCAB_SIZE; i++) {\n"
         "    int diff = my_logits[i] - max_val;\n"
         "    if (diff > 0) sum_exp += (1 << diff);\n"
         "  }\n"
         "  \n"
         "  // Compute loss: -log(softmax(target))\n"
         "  int target_logit = my_logits[target];\n"
         "  int loss = max_val - target_logit + (sum_exp > 0 ? (int)log2((float)sum_exp) : 0);\n"
         "  \n"
         "  losses[tid] = loss;\n"
         "}\n",
        FIXED_POINT,
        SIGMA_SHIFT,
        SIGMA_SHIFT_VECTOR,
        MAX_VAL,
        MIN_VAL,
        HIDDEN_DIM,
        N_LAYERS,
        VOCAB_SIZE,
        POPULATION_SIZE,
        SIGMA_SHIFT_VECTOR
    ];
}

// Initialize optimized Metal context
bool egg_gpu_metal_optimized_init(void) {
    if (g_opt_ctx.ready) return true;

    g_opt_ctx.device = MTLCreateSystemDefaultDevice();
    if (!g_opt_ctx.device) {
        fprintf(stderr, "[EGG METAL OPT] No Metal device found\n");
        return false;
    }

    g_opt_ctx.queue = [g_opt_ctx.device newCommandQueue];
    g_opt_ctx.lock = dispatch_semaphore_create(1);

    // Compile optimized shaders
    NSError *error = nil;
    NSString *shaderSource = OptimizedMetalShaderSource();

    id<MTLLibrary> library = [g_opt_ctx.device newLibraryWithSource:shaderSource
                                                             options:nil
                                                               error:&error];
    if (!library) {
        fprintf(stderr, "[EGG METAL OPT] Failed to compile shaders: %s\n",
                error.localizedDescription.UTF8String);
        return false;
    }

    // Create functions
    g_opt_ctx.embeddingAndGRUFusedFunc = [library newFunctionWithName:@"fused_embedding_gru"];
    g_opt_ctx.mlpFusedFunc = [library newFunctionWithName:@"fused_mlp"];
    g_opt_ctx.headFusedFunc = [library newFunctionWithName:@"compute_loss"];
    g_opt_ctx.updateMatrixFunc = [library newFunctionWithName:@"update_matrix_optimized"];

    if (!g_opt_ctx.embeddingAndGRUFusedFunc || !g_opt_ctx.mlpFusedFunc || !g_opt_ctx.updateMatrixFunc) {
        fprintf(stderr, "[EGG METAL OPT] Failed to create kernel functions\n");
        return false;
    }

    // Create compute pipeline states
    g_opt_ctx.embeddingAndGRUFused = [g_opt_ctx.device newComputePipelineStateWithFunction:g_opt_ctx.embeddingAndGRUFusedFunc
                                                                                       error:&error];
    g_opt_ctx.mlpFused = [g_opt_ctx.device newComputePipelineStateWithFunction:g_opt_ctx.mlpFusedFunc
                                                                           error:&error];
    g_opt_ctx.updateMatrixKernel = [g_opt_ctx.device newComputePipelineStateWithFunction:g_opt_ctx.updateMatrixFunc
                                                                                  error:&error];

    if (!g_opt_ctx.embeddingAndGRUFused || !g_opt_ctx.mlpFused || !g_opt_ctx.updateMatrixKernel) {
        fprintf(stderr, "[EGG METAL OPT] Failed to create pipeline states: %s\n",
                error.localizedDescription.UTF8String);
        return false;
    }

    // Allocate buffers
    g_opt_ctx.modelBufferSize = VOCAB_SIZE * HIDDEN_DIM + // embedding
                               N_LAYERS * 2 * HIDDEN_DIM + // layer norm weights
                               N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM + // GRU weights
                               N_LAYERS * 2 * HIDDEN_DIM + // GRU biases
                               N_LAYERS * 2 * HIDDEN_DIM * HIDDEN_DIM * 4 + // MLP weights
                               HIDDEN_DIM * VOCAB_SIZE + // head
                               HIDDEN_DIM; // ln_out
    g_opt_ctx.modelBuffer = [g_opt_ctx.device newBufferWithLength:g_opt_ctx.modelBufferSize
                                                           options:MTLResourceStorageModeShared];

    g_opt_ctx.populationBufferSize = POPULATION_SIZE * N_LAYERS * HIDDEN_DIM * 3; // x, h, workspace
    g_opt_ctx.populationBuffer = [g_opt_ctx.device newBufferWithLength:g_opt_ctx.populationBufferSize
                                                               options:MTLResourceStorageModeShared];

    g_opt_ctx.workspaceBufferSize = HIDDEN_DIM * 4; // For MLP expansion
    g_opt_ctx.workspaceBuffer = [g_opt_ctx.device newBufferWithLength:g_opt_ctx.workspaceBufferSize
                                                                options:MTLResourceStorageModeShared];

    g_opt_ctx.resultsBufferSize = POPULATION_SIZE * sizeof(int); // Loss values
    g_opt_ctx.resultsBuffer = [g_opt_ctx.device newBufferWithLength:g_opt_ctx.resultsBufferSize
                                                              options:MTLResourceStorageModeShared];

    if (!g_opt_ctx.modelBuffer || !g_opt_ctx.populationBuffer || !g_opt_ctx.resultsBuffer) {
        fprintf(stderr, "[EGG METAL OPT] Failed to allocate buffers\n");
        return false;
    }

    printf("[EGG METAL OPT] Initialized optimized implementation\n");
    printf("  Model buffer: %.2f MB\n", g_opt_ctx.modelBufferSize / (1024.0 * 1024.0));
    printf("  Population buffer: %.2f MB\n", g_opt_ctx.populationBufferSize / (1024.0 * 1024.0));

    g_opt_ctx.ready = true;
    return true;
}

// Optimized forward pass for entire population
extern "C" bool egg_gpu_metal_optimized_forward_pass(
    const void *model,
    const uint8_t *tokens,
    const int seq_len,
    const uint32_t step_seed,
    int *losses_out
) {
    if (!g_opt_ctx.ready) {
        if (!egg_gpu_metal_optimized_init()) {
            return false;
        }
    }

    dispatch_semaphore_wait(g_opt_ctx.lock, DISPATCH_TIME_FOREVER);

    // Copy model to GPU (only if changed)
    memcpy([g_opt_ctx.modelBuffer contents], model, g_opt_ctx.modelBufferSize);

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [g_opt_ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Setup params
        OptimizedMetalParams params;
        params.seed = step_seed;
        params.shift = 8;
        params.noise_sign = 1; // Will alternate for population pairs

        // Process each sequence position
        for (int t = 0; t < seq_len; t++) {
            // Layer 0: Embedding lookup
            params.layer_idx = 0;
            [encoder setComputePipelineState:g_opt_ctx.embeddingAndGRUFused];
            [encoder setBuffer:g_opt_ctx.modelBuffer offset:0 atIndex:0];
            [encoder setBytes:&tokens[t] length:1 atIndex:1];
            [encoder setBuffer:g_opt_ctx.populationBuffer offset:0 atIndex:2];
            [encoder setBytes:&params length:sizeof(params) atIndex:3];

            MTLSize gridSize = MTLSizeMake(HIDDEN_DIM, POPULATION_SIZE, 1);
            MTLSize threadgroupSize = MTLSizeMake(32, 8, 1); // 32 threads per simd, 8 simds per group

            [encoder dispatchThreadgroups:MTLSizeMake((gridSize.width + 31) / 32,
                                                     (gridSize.height + 7) / 8, 1)
                      threadsPerThreadgroup:threadgroupSize];

            // Remaining layers
            for (int l = 1; l < N_LAYERS; l++) {
                params.layer_idx = l;

                // GRU layer
                [encoder setComputePipelineState:g_opt_ctx.embeddingAndGRUFused];
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 31) / 32,
                                                         (POPULATION_SIZE + 7) / 8, 1)
                          threadsPerThreadgroup:threadgroupSize];

                // MLP layer
                [encoder setComputePipelineState:g_opt_ctx.mlpFused];
                [encoder setBuffer:g_opt_ctx.populationBuffer offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 31) / 32,
                                                         (POPULATION_SIZE + 7) / 8, 1)
                          threadsPerThreadgroup:threadgroupSize];
            }
        }

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Compute losses
        // This would need a separate head computation kernel
        // For now, return dummy losses
        for (int i = 0; i < POPULATION_SIZE; i++) {
            losses_out[i] = 1000 + (step_seed % 100);
        }
    }

    dispatch_semaphore_signal(g_opt_ctx.lock);
    return true;
}

// Optimized matrix update
extern "C" bool egg_gpu_metal_optimized_update_weights(
    void *model,
    const int *pair_fitnesses,
    const uint32_t step_seed
) {
    if (!g_opt_ctx.ready) return false;

    // This would implement the optimized weight update
    // For now, just return true
    return true;
}

// Cleanup
extern "C" void egg_gpu_metal_optimized_shutdown(void) {
    if (!g_opt_ctx.ready) return;

    g_opt_ctx.modelBuffer = nil;
    g_opt_ctx.populationBuffer = nil;
    g_opt_ctx.workspaceBuffer = nil;
    g_opt_ctx.resultsBuffer = nil;
    g_opt_ctx.embeddingAndGRUFused = nil;
    g_opt_ctx.mlpFused = nil;
    g_opt_ctx.headFused = nil;
    g_opt_ctx.updateMatrixKernel = nil;
    g_opt_ctx.queue = nil;
    g_opt_ctx.device = nil;

    g_opt_ctx.ready = false;
    printf("[EGG METAL OPT] Shutdown complete\n");
}