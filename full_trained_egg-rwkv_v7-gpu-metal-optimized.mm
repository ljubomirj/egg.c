#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "egg_config.h"
#include "egg_gpu_metal_optimized.h"

// NOTE: This “optimized” path is a lean, pair-at-a-time Metal rewrite that
// keeps the full model resident on the GPU, fuses the GRU math into a single
// kernel, and accumulates losses on-GPU.  It intentionally evaluates the
// +/- perturbation pair together (POPULATION_SIZE is ignored here) because the
// current caller only passes space for two losses.

typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM];
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; // unused for now
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];  // [layer][0]=GRU LN
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

typedef struct {
    bool ready;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> gruPipeline;
    id<MTLComputePipelineState> headPipeline;
    id<MTLBuffer> modelBuffer;
    id<MTLBuffer> xBuffer;       // 2 * HIDDEN_DIM bytes
    id<MTLBuffer> hBuffer;       // N_LAYERS * 2 * HIDDEN_DIM bytes
    id<MTLBuffer> lossBuffer;    // 2 * int32
    id<MTLBuffer> tokenBuffer;   // seq_len+1 tokens (filled per call)
    dispatch_semaphore_t lock;
} MetalOptContext;

static MetalOptContext g_ctx = {};

// ------------------------------------------------------------ //
// Metal shader source
// ------------------------------------------------------------ //
static NSString *MetalOptShader(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\\n"
         "using namespace metal;\\n"
         "#define HIDDEN_DIM %d\\n"
         "#define VOCAB_SIZE %d\\n"
         "#define FIXED_POINT %d\\n"
         "#define SIGMA_SHIFT %d\\n"
         "#define MAX_VAL 127\\n"
         "#define MIN_VAL -127\\n"
         "struct EggModel {\\n"
         "  char embedding[VOCAB_SIZE * HIDDEN_DIM];\\n"
         "  char gru_weights[%d][4][HIDDEN_DIM * HIDDEN_DIM];\\n"
         "  char gru_biases[%d][2][HIDDEN_DIM];\\n"
         "  char mlp_weights[%d][2][HIDDEN_DIM * (HIDDEN_DIM * 4)];\\n"
         "  char head[HIDDEN_DIM * VOCAB_SIZE];\\n"
         "  char ln_weights[%d][2][HIDDEN_DIM];\\n"
         "  char ln_out[HIDDEN_DIM];\\n"
         "};\\n"
         "struct GruParams { uint layer; uint seed; };\\n"
         "struct HeadParams { uint seed; uint target; };\\n"
         "inline int clamp_int(int v) { return v > MAX_VAL ? MAX_VAL : (v < MIN_VAL ? MIN_VAL : v); }\\n"
         "inline int dot_noise(const device char *w, const device char *inp, uint tid, uint ltid, threadgroup int *scratch, int noise_sign, uint seedA, uint seedB) {\\n"
         "  // xB reduction\\n"
         "  int partial = 0;\\n"
         "  for (uint c = ltid; c < HIDDEN_DIM; c += HIDDEN_DIM) {\\n"
         "    char nb = noise_from_hash(seedB, c);\\n"
         "    partial += int(inp[c]) * int(nb);\\n"
         "  }\\n"
         "  scratch[ltid] = partial;\\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\\n"
         "  for (uint s = HIDDEN_DIM/2; s>0; s>>=1) {\\n"
         "    if (ltid < s) scratch[ltid] += scratch[ltid + s];\\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);\\n"
         "  }\\n"
         "  int xB = scratch[0];\\n"
         "  int acc = 0;\\n"
         "  const device char *row = w + tid * HIDDEN_DIM;\\n"
         "  for (uint c = 0; c < HIDDEN_DIM; ++c) acc += int(inp[c]) * int(row[c]);\\n"
         "  int noise = (xB * int(noise_from_hash(seedA, tid))) * noise_sign;\\n"
         "  acc += noise >> (FIXED_POINT + SIGMA_SHIFT);\\n"
         "  return acc >> 8;\\n"
         "}\\n"
         "inline uint hash_rng(uint s, uint idx) {\\n"
         "  uint x = s + idx * 0x9e3779b9u;\\n"
         "  x ^= x >> 16; x *= 0x85ebca6b;\\n"
         "  x ^= x >> 13; x *= 0xc2b2ae35;\\n"
         "  x ^= x >> 16;\\n"
         "  return x;\\n"
         "}\\n"
         "inline char noise_from_hash(uint s, uint idx) {\\n"
         "  uint r = hash_rng(s, idx);\\n"
         "  return (char)((r & 1 ? 1 : -1) * ((r >> 1) & 31));\\n"
         "}\\n"
         "kernel void gru_pair_kernel(\\n"
         "    device const EggModel *model [[buffer(0)]],\\n"
         "    device char *x_buf [[buffer(1)]],\\n"
         "    device char *h_buf [[buffer(2)]],\\n"
         "    constant GruParams &p [[buffer(3)]],\\n"
         "    uint tid [[thread_position_in_grid]],\\n"
         "    uint ltid [[thread_index_in_threadgroup]]) {\\n"
         "  if (tid >= HIDDEN_DIM) return;\\n"
         "  threadgroup int sum0[HIDDEN_DIM];\\n"
         "  threadgroup int sum1[HIDDEN_DIM];\\n"
         "  // layer norm mean (L1) for each pop\\n"
         "  int v0 = int(x_buf[tid]);\\n"
         "  int v1 = int(x_buf[HIDDEN_DIM + tid]);\\n"
         "  sum0[ltid] = v0 < 0 ? -v0 : v0;\\n"
         "  sum1[ltid] = v1 < 0 ? -v1 : v1;\\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\\n"
         "  for (uint s = HIDDEN_DIM / 2; s > 0; s >>= 1) {\\n"
         "    if (ltid < s) { sum0[ltid] += sum0[ltid + s]; sum1[ltid] += sum1[ltid + s]; }\\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);\\n"
         "  }\\n"
         "  int mean0 = (ltid == 0) ? sum0[0] / HIDDEN_DIM : sum0[0];\\n"
         "  int mean1 = (ltid == 0) ? sum1[0] / HIDDEN_DIM : sum1[0];\\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\\n"
         "  if (mean0 == 0) mean0 = 1;\\n"
         "  if (mean1 == 0) mean1 = 1;\\n"
         "  // load ln weights\\n"
         "  uint l = p.layer;\\n"
         "  int ln0 = int(model->ln_weights[l][0][tid]);\\n"
         "  // normalize\\n"
         "  int xn0 = clamp_int((v0 * ln0) / mean0);\\n"
         "  int xn1 = clamp_int((v1 * ln0) / mean1);\\n"
         "  // pop 0 (+1)\\n"
         "  device const char *W0 = model->gru_weights[l][0];\\n"
         "  device const char *W1 = model->gru_weights[l][1];\\n"
         "  device const char *W2 = model->gru_weights[l][2];\\n"
         "  device const char *W3 = model->gru_weights[l][3];\\n"
         "  const device char *h0 = h_buf + (l * 2 + 0) * HIDDEN_DIM;\\n"
         "  const device char *h1 = h_buf + (l * 2 + 1) * HIDDEN_DIM;\\n"
         "  int ft0 = dot_noise(W0, x_buf, tid, ltid, sum0, +1, p.seed + 1, p.seed + 11) +\\n"
         "            dot_noise(W1, h0, tid, ltid, sum0, +1, p.seed + 2, p.seed + 12) + int(model->gru_biases[l][0][tid]);\\n"
         "  int ft1 = dot_noise(W0, x_buf + HIDDEN_DIM, tid, ltid, sum0, -1, p.seed + 1, p.seed + 11) +\\n"
         "            dot_noise(W1, h1, tid, ltid, sum0, -1, p.seed + 2, p.seed + 12) + int(model->gru_biases[l][0][tid]);\\n"
         "  ft0 = clamp_int(ft0); ft1 = clamp_int(ft1);\\n"
         "  int gp0 = clamp_int(((ft0 + 127) * int(h0[tid])) >> 8);\\n"
         "  int gp1 = clamp_int(((ft1 + 127) * int(h1[tid])) >> 8);\\n"
         "  int ht0 = dot_noise(W2, x_buf, tid, ltid, sum0, +1, p.seed + 3, p.seed + 13) +\\n"
         "            dot_noise(W3, (device char*)h0, tid, ltid, sum0, +1, p.seed + 4, p.seed + 14) + int(model->gru_biases[l][1][tid]);\\n"
         "  int ht1 = dot_noise(W2, x_buf + HIDDEN_DIM, tid, ltid, sum0, -1, p.seed + 3, p.seed + 13) +\\n"
         "            dot_noise(W3, (device char*)h1, tid, ltid, sum0, -1, p.seed + 4, p.seed + 14) + int(model->gru_biases[l][1][tid]);\\n"
         "  ht0 = clamp_int(ht0); ht1 = clamp_int(ht1);\\n"
         "  int h_old0 = int(h0[tid]);\\n"
         "  int h_old1 = int(h1[tid]);\\n"
         "  int h_new0 = h_old0 + (((ft0 + 127) * (ht0 - h_old0)) >> 8);\\n"
         "  int h_new1 = h_old1 + (((ft1 + 127) * (ht1 - h_old1)) >> 8);\\n"
         "  h_new0 = clamp_int(h_new0); h_new1 = clamp_int(h_new1);\\n"
         "  ((device char*)h0)[tid] = char(h_new0);\\n"
         "  ((device char*)h1)[tid] = char(h_new1);\\n"
         "  // residual add\\n"
         "  int x_out0 = clamp_int(h_new0 + v0);\\n"
         "  int x_out1 = clamp_int(h_new1 + v1);\\n"
         "  x_buf[tid] = char(x_out0);\\n"
         "  x_buf[HIDDEN_DIM + tid] = char(x_out1);\\n"
         "}\\n"
         "kernel void head_pair_kernel(\\n"
         "    device const EggModel *model [[buffer(0)]],\\n"
         "    device const char *x_buf [[buffer(1)]],\\n"
         "    device atomic_int *losses [[buffer(2)]],\\n"
         "    constant HeadParams &hp [[buffer(3)]],\\n"
         "    uint tid [[thread_position_in_grid]],\\n"
         "    uint ltid [[thread_index_in_threadgroup]]) {\\n"
         "  if (ltid >= 2) return; // only two pops\\n"
         "  const device char *x = x_buf + ltid * HIDDEN_DIM;\\n"
         "  // layer norm using ln_out\\n"
         "  int sum = 0;\\n"
         "  for (uint i = 0; i < HIDDEN_DIM; ++i) { int v = int(x[i]); sum += (v < 0) ? -v : v; }\\n"
         "  int mean = sum / HIDDEN_DIM; if (mean == 0) mean = 1;\\n"
         "  float logits[VOCAB_SIZE];\\n"
         "  for (uint v = 0; v < VOCAB_SIZE; ++v) {\\n"
         "    const device char *w = model->head + v * HIDDEN_DIM;\\n"
         "    int acc = 0;\\n"
         "    for (uint i = 0; i < HIDDEN_DIM; ++i) {\\n"
         "      int xv = (int(x[i]) * int(model->ln_out[i])) / mean;\\n"
         "      acc += xv * int(w[i]);\\n"
         "    }\\n"
         "    // perturb head\\n"
         "    int xB = 0;\\n"
         "    for (uint i = 0; i < HIDDEN_DIM; ++i) xB += int(x[i]) * int(noise_from_hash(hp.seed + 200, i));\\n"
         "    int noise = (xB * int(noise_from_hash(hp.seed + 100, v))) * (ltid == 0 ? 1 : -1);\\n"
         "    acc += noise >> (FIXED_POINT + SIGMA_SHIFT);\\n"
         "    logits[v] = float(acc) / 8.0f;\\n"
         "  }\\n"
         "  float maxv = logits[0]; for (uint v = 1; v < VOCAB_SIZE; ++v) maxv = fmax(maxv, logits[v]);\\n"
         "  float sum_exp = 0.f; for (uint v = 0; v < VOCAB_SIZE; ++v) sum_exp += exp(logits[v] - maxv);\\n"
         "  float log_sum = maxv + log(sum_exp);\\n"
         "  float loss = log_sum - logits[hp.target];\\n"
         "  atomic_fetch_add_explicit(&losses[ltid], int(loss * float(1 << FIXED_POINT)), memory_order_relaxed);\\n"
         "}\\n",
        HIDDEN_DIM, VOCAB_SIZE, FIXED_POINT, SIGMA_SHIFT,
        N_LAYERS, N_LAYERS, N_LAYERS, N_LAYERS];
}

// ------------------------------------------------------------ //
// Context init / shutdown
// ------------------------------------------------------------ //
bool egg_gpu_metal_optimized_init(void) {
    if (g_ctx.ready) return true;

    g_ctx.device = MTLCreateSystemDefaultDevice();
    if (!g_ctx.device) {
        fprintf(stderr, "[EGG OPT] No Metal device available.\n");
        return false;
    }
    g_ctx.queue = [g_ctx.device newCommandQueue];
    if (!g_ctx.queue) {
        fprintf(stderr, "[EGG OPT] Failed to create command queue.\n");
        return false;
    }

    NSError *err = nil;
    NSString *src = MetalOptShader();
    id<MTLLibrary> lib = [g_ctx.device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "[EGG OPT] Shader compile failed: %s\n", err.localizedDescription.UTF8String);
        return false;
    }

    g_ctx.gruPipeline = [g_ctx.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gru_pair_kernel"]
                                                                   error:&err];
    if (!g_ctx.gruPipeline) {
        fprintf(stderr, "[EGG OPT] GRU pipeline failed: %s\n", err.localizedDescription.UTF8String);
        return false;
    }
    g_ctx.headPipeline = [g_ctx.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"head_pair_kernel"]
                                                                    error:&err];
    if (!g_ctx.headPipeline) {
        fprintf(stderr, "[EGG OPT] Head pipeline failed: %s\n", err.localizedDescription.UTF8String);
        return false;
    }

    g_ctx.modelBuffer = [g_ctx.device newBufferWithLength:sizeof(EggModel) options:MTLResourceStorageModeShared];
    g_ctx.xBuffer     = [g_ctx.device newBufferWithLength:HIDDEN_DIM * 2 options:MTLResourceStorageModeShared];
    g_ctx.hBuffer     = [g_ctx.device newBufferWithLength:N_LAYERS * HIDDEN_DIM * 2 options:MTLResourceStorageModeShared];
    g_ctx.lossBuffer  = [g_ctx.device newBufferWithLength:sizeof(int32_t) * 2 options:MTLResourceStorageModeShared];
    g_ctx.tokenBuffer = [g_ctx.device newBufferWithLength:SEQ_LEN + 1 options:MTLResourceStorageModeShared];
    g_ctx.lock = dispatch_semaphore_create(1);

    if (!g_ctx.modelBuffer || !g_ctx.xBuffer || !g_ctx.hBuffer || !g_ctx.lossBuffer || !g_ctx.tokenBuffer) {
        fprintf(stderr, "[EGG OPT] Buffer allocation failed.\n");
        return false;
    }

    g_ctx.ready = true;
    fprintf(stdout, "[EGG OPT] Metal pair evaluator ready on %s\n", g_ctx.device.name.UTF8String);
    return true;
}

// ------------------------------------------------------------ //
// Forward pass for a +/- pair (two perturbations)
// ------------------------------------------------------------ //
extern "C" bool egg_gpu_metal_optimized_forward_pass(
    const void *model,
    const uint8_t *tokens,
    const int seq_len,
    const uint32_t step_seed,
    int *losses_out)
{
    if (!g_ctx.ready && !egg_gpu_metal_optimized_init()) return false;
    if (seq_len <= 0 || !model || !tokens || !losses_out) return false;

    dispatch_semaphore_wait(g_ctx.lock, DISPATCH_TIME_FOREVER);

    // Copy model & tokens
    memcpy([g_ctx.modelBuffer contents], model, sizeof(EggModel));
    memcpy([g_ctx.tokenBuffer contents], tokens, seq_len + 1); // need next-token target
    memset([g_ctx.hBuffer contents], 0, N_LAYERS * HIDDEN_DIM * 2);
    int32_t *loss_host = (int32_t *)[g_ctx.lossBuffer contents];
    loss_host[0] = loss_host[1] = 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_ctx.queue commandBuffer];

        // Per-timestep work
        for (int t = 0; t < seq_len; ++t) {
            uint8_t tok = tokens[t];
            // CPU-side embedding copy (shared buffer → GPU visible)
            const EggModel *mhost = (const EggModel *)model;
            memcpy([g_ctx.xBuffer contents], &mhost->embedding[tok * HIDDEN_DIM], HIDDEN_DIM);
            memcpy((uint8_t *)[g_ctx.xBuffer contents] + HIDDEN_DIM, &mhost->embedding[tok * HIDDEN_DIM], HIDDEN_DIM);

            // GRU layers
            for (uint l = 0; l < N_LAYERS; ++l) {
                GruParams params = {l, step_seed + l * 100 + t};
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:g_ctx.gruPipeline];
                [enc setBuffer:g_ctx.modelBuffer offset:0 atIndex:0];
                [enc setBuffer:g_ctx.xBuffer offset:0 atIndex:1];
                [enc setBuffer:g_ctx.hBuffer offset:0 atIndex:2];
                [enc setBytes:&params length:sizeof(params) atIndex:3];
                MTLSize grid = MTLSizeMake(HIDDEN_DIM, 1, 1);
                NSUInteger tg = g_ctx.gruPipeline.maxTotalThreadsPerThreadgroup;
                if (tg > HIDDEN_DIM) tg = HIDDEN_DIM;
                MTLSize tgs = MTLSizeMake(tg, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
                [enc endEncoding];
            }

            // Head + loss
            HeadParams hp = {step_seed + 999 + (uint)t, tokens[t + 1]};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_ctx.headPipeline];
            [enc setBuffer:g_ctx.modelBuffer offset:0 atIndex:0];
            [enc setBuffer:g_ctx.xBuffer offset:0 atIndex:1];
            [enc setBuffer:g_ctx.lossBuffer offset:0 atIndex:2];
            [enc setBytes:&hp length:sizeof(hp) atIndex:3];
            MTLSize grid = MTLSizeMake(2, 1, 1); // two populations
            MTLSize tgs = MTLSizeMake(32, 1, 1); // plenty for small work
            [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
    }

    // Copy losses back
    loss_host = (int32_t *)[g_ctx.lossBuffer contents];
    losses_out[0] = loss_host[0];
    losses_out[1] = loss_host[1];

    dispatch_semaphore_signal(g_ctx.lock);
    return true;
}

// ------------------------------------------------------------ //
// Update weights – not yet optimized (CPU update is still used)
// ------------------------------------------------------------ //
extern "C" bool egg_gpu_metal_optimized_update_weights(
    void *model,
    const int *pair_fitnesses,
    const uint32_t step_seed)
{
    (void)model; (void)pair_fitnesses; (void)step_seed;
    // TODO: GPU ES update path.
    return true;
}

// ------------------------------------------------------------ //
void egg_gpu_metal_optimized_shutdown(void) {
    if (!g_ctx.ready) return;
    g_ctx.modelBuffer = nil;
    g_ctx.xBuffer = nil;
    g_ctx.hBuffer = nil;
    g_ctx.lossBuffer = nil;
    g_ctx.tokenBuffer = nil;
    g_ctx.gruPipeline = nil;
    g_ctx.headPipeline = nil;
    g_ctx.queue = nil;
    g_ctx.device = nil;
    g_ctx.ready = false;
    fprintf(stdout, "[EGG OPT] Metal shutdown complete\n");
}
