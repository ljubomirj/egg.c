#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>
#include <unistd.h>
#include <assert.h>

#if defined(EGG_USE_METAL)
#include "egg_gpu_metal.h"
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(EGG_FORCE_SCALAR)
#define EGG_SIMD_IMPL 0
#elif defined(EGG_FORCE_NEON)
#define EGG_SIMD_IMPL 1
#elif defined(EGG_FORCE_AVX2)
#define EGG_SIMD_IMPL 2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define EGG_SIMD_IMPL 1
#elif defined(__AVX2__)
#define EGG_SIMD_IMPL 2
#else
#define EGG_SIMD_IMPL 0
#endif

#if EGG_SIMD_IMPL == 1
#include <arm_neon.h>
#elif EGG_SIMD_IMPL == 2
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#elif defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#define EGG_ALIGNED16 __declspec(align(16))
#else
#define EGG_ALIGNED16 __attribute__((aligned(16)))
#endif

#include "egg_config.h"

// --- Lookup Tables [cite: 998-1000] ---
int32_t EXP2_TABLE[256];

// --- Data Structure ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// --- Recurrent State ---
typedef struct {
    int8_t h[N_LAYERS][HIDDEN_DIM];
} RecurrentState;

// --- Model Parameters Struct ---
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; // 0: bf, 1: bh
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; // 0: Expand, 1: Project
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM]; // 0: LN1, 1: LN2
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// --- Helper Functions ---

// Forward Declaration
int32_t compute_loss(int8_t *logits, uint8_t target);

// CPU reference forward (used for GPU parity checks)
static void forward_pass_cpu_reference(
    EggModel *model, 
    const uint8_t *inputs, 
    const uint8_t *targets,
    int seq_len, 
    int8_t *logits_out, 
    int32_t *accumulated_loss,
    uint32_t step_seed, 
    int noise_sign,
    RecurrentState *rnn_state
);

// CPU reference forward for t==0 snapshot capture (parity tracing)
static void forward_pass_cpu_reference_snap(
    EggModel *model,
    const uint8_t *inputs,
    int seq_len,
    uint32_t step_seed,
    int noise_sign,
    RecurrentState *rnn_state,
    int8_t snap_h[N_LAYERS][HIDDEN_DIM]  // captures h after each layer at t==0
);

static bool ends_with_ignore_case(const char *str, const char *suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return false;
    const char *str_ptr = str + (str_len - suffix_len);
    for (size_t i = 0; i < suffix_len; i++) {
        if (tolower((unsigned char)str_ptr[i]) != tolower((unsigned char)suffix[i])) {
            return false;
        }
    }
    return true;
}

static bool has_zstd_extension(const char *filename) {
    return ends_with_ignore_case(filename, ".zst") ||
           ends_with_ignore_case(filename, ".zstd");
}

static FILE *open_dataset_stream(const char *filename, bool *is_pipe, char *resolved_out, size_t resolved_size) {
    *is_pipe = has_zstd_extension(filename);

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
    const char *actual_path = filename;

    // If it's a compressed file, resolve to canonical path first (zstd refuses symlinks)
    if (*is_pipe) {
        char resolved[PATH_MAX];
        const char *canonical = realpath(filename, resolved);
        if (!canonical) {
            fprintf(stderr, "Error: failed to resolve canonical path for '%s'.\n", filename);
            return NULL;
        }
        actual_path = canonical;
        // Copy resolved path to caller if requested
        if (resolved_out && resolved_size > 0) {
            strncpy(resolved_out, actual_path, resolved_size - 1);
            resolved_out[resolved_size - 1] = '\0';
        }
    } else {
        // For non-compressed files, optionally resolve for display purposes
        if (resolved_out && resolved_size > 0) {
            char resolved[PATH_MAX];
            const char *canonical = realpath(filename, resolved);
            if (canonical) {
                strncpy(resolved_out, canonical, resolved_size - 1);
                resolved_out[resolved_size - 1] = '\0';
            } else {
                strncpy(resolved_out, filename, resolved_size - 1);
                resolved_out[resolved_size - 1] = '\0';
            }
        }
        return fopen(filename, "rb");
    }

    // Now actual_path is the canonical path (guaranteed non-symlink)
    // Build command with proper shell escaping
    char cmd[PATH_MAX + 64];
    // Escape single quotes by replacing ' with '\'' (end quote, escaped quote, start quote)
    char escaped[PATH_MAX * 2];
    const char *src = actual_path;
    char *dst = escaped;
    while (*src && (dst - escaped) < (int)sizeof(escaped) - 4) {
        if (*src == '\'') {
            *dst++ = '\'';
            *dst++ = '\\';
            *dst++ = '\'';
            *dst++ = '\'';
        } else {
            *dst++ = *src;
        }
        src++;
    }
    *dst = '\0';
    
    int written = snprintf(cmd, sizeof(cmd), "zstd -dc -- '%s'", escaped);
    if (written < 0 || written >= (int)sizeof(cmd)) {
        fprintf(stderr, "Error: command too long for filename '%s'.\n", actual_path);
        return NULL;
    }

    FILE *pipe = popen(cmd, "r");
    if (!pipe) fprintf(stderr, "Error: failed to launch '%s'.\n", cmd);
    return pipe;
}

static void close_dataset_stream(FILE *stream, bool is_pipe) {
    if (!stream) return;
    if (is_pipe) pclose(stream);
    else fclose(stream);
}

void init_tables() {
    for(int i=0; i<256; i++) 
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// Simple RNG helpers for scalar usage
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_val(uint32_t *rng) {
    uint32_t r = xorshift32(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

// Noise generation shared across architectures
static inline void gen_noise_vector(uint32_t *rng, int8_t *out, int count) {
    for (int i = 0; i < count; i += 16) {
        for (int j = 0; j < 16; j++) {
             if(i+j < count) out[i+j] = gen_noise_val(rng);
        }
    }
}

static inline int32_t dot_product_i8(const int8_t *a, const int8_t *b, int len) {
#if EGG_SIMD_IMPL == 1
    int32x4_t acc_v = vdupq_n_s32(0);
    int i = 0;
    for (; i <= len - 16; i += 16) {
        int8x16_t va = vld1q_s8(&a[i]);
        int8x16_t vb = vld1q_s8(&b[i]);
        int16x8_t mul_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t mul_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc_v = vpadalq_s16(acc_v, mul_lo);
        acc_v = vpadalq_s16(acc_v, mul_hi);
    }
    int32_t sum = vaddvq_s32(acc_v);
    for (; i < len; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#elif EGG_SIMD_IMPL == 2
    int i = 0;
    __m256i acc = _mm256_setzero_si256();
    for (; i <= len - 32; i += 32) {
        __m128i a_low = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_low = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i a_low16 = _mm256_cvtepi8_epi16(a_low);
        __m256i b_low16 = _mm256_cvtepi8_epi16(b_low);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_low16, b_low16));

        __m128i a_high = _mm_loadu_si128((const __m128i*)(a + i + 16));
        __m128i b_high = _mm_loadu_si128((const __m128i*)(b + i + 16));
        __m256i a_high16 = _mm256_cvtepi8_epi16(a_high);
        __m256i b_high16 = _mm256_cvtepi8_epi16(b_high);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_high16, b_high16));
    }
    if (i <= len - 16) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i va16 = _mm256_cvtepi8_epi16(va);
        __m256i vb16 = _mm256_cvtepi8_epi16(vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va16, vb16));
        i += 16;
    }
    __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    acc128 = _mm_hadd_epi32(acc128, acc128);
    acc128 = _mm_hadd_epi32(acc128, acc128);
    int32_t sum = _mm_cvtsi128_si32(acc128);
    for (; i < len; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#else
    int32_t sum = 0;
    for (int i = 0; i < len; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#endif
}

static inline int32_t sum_abs_i8(const int8_t *x, int len) {
#if EGG_SIMD_IMPL == 1
    int32x4_t sum_v = vdupq_n_s32(0);
    int i = 0;
    for (; i <= len - 16; i += 16) {
        int8x16_t xv = vld1q_s8(&x[i]);
        int8x16_t abs_xv = vabsq_s8(xv);
        uint16x8_t s1 = vpaddlq_u8(vreinterpretq_u8_s8(abs_xv));
        uint32x4_t s2 = vpaddlq_u16(s1);
        sum_v = vaddq_s32(sum_v, vreinterpretq_s32_u32(s2));
    }
    int32_t sum = vaddvq_s32(sum_v);
    for (; i < len; ++i) sum += (x[i] < 0) ? -x[i] : x[i];
    return sum;
#elif EGG_SIMD_IMPL == 2
    int i = 0;
    __m256i acc = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();
    for (; i <= len - 32; i += 32) {
        __m256i xv = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i sad = _mm256_sad_epu8(xv, zero);
        acc = _mm256_add_epi64(acc, sad);
    }
    __m128i acc_low = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    int64_t sum = (int64_t)_mm_cvtsi128_si64(acc_low);
    sum += (int64_t)_mm_extract_epi64(acc_low, 1);
    if (i <= len - 16) {
        __m128i xv = _mm_loadu_si128((const __m128i*)(x + i));
        __m128i sad = _mm_sad_epu8(xv, _mm_setzero_si128());
        sum += (int64_t)_mm_cvtsi128_si64(sad);
        sum += (int64_t)_mm_extract_epi64(sad, 1);
        i += 16;
    }
    for (; i < len; ++i) sum += (x[i] < 0) ? -x[i] : x[i];
    return (int32_t)sum;
#else
    int32_t sum = 0;
    for (int i = 0; i < len; ++i) sum += (x[i] < 0) ? -x[i] : x[i];
    return sum;
#endif
}

// --- The Core "Rank-1 Perturbed Matrix Mul" ---
void matmul_perturbed(
    const int8_t *in, const int8_t *w, int8_t *out, 
    int rows, int cols, 
    uint32_t layer_seed, int noise_sign,
    int shift
) {
    int8_t noiseA[rows];
    int8_t noiseB[cols];

    uint32_t rng = layer_seed;
    gen_noise_vector(&rng, noiseA, rows);
    gen_noise_vector(&rng, noiseB, cols);

    int32_t xB = dot_product_i8(in, noiseB, cols);

#if defined(EGG_USE_METAL)
    if (egg_gpu_matmul_perturbed(
            in, w, out,
            noiseA, rows, cols,
            shift, noise_sign, xB, 0)) {
        return;
    }
#endif

    for(int r=0; r<rows; r++) {
        const int8_t *w_row = &w[r * cols];
        int32_t acc = dot_product_i8(in, w_row, cols);
        if (noise_sign != 0) {
            int32_t noise = (xB * (int32_t)noiseA[r]) * noise_sign;
            acc += (noise >> (FIXED_POINT + SIGMA_SHIFT));
        }
        
        int32_t res = acc >> shift;
        if(res > MAX_VAL) out[r] = MAX_VAL;
        else if(res < MIN_VAL) out[r] = MIN_VAL;
        else out[r] = (int8_t)res;
    }
}

// --- Layer Norm [cite: 965] ---
void egg_ln(const int8_t *x, const int8_t *w, int8_t *out) {
    int32_t sum = sum_abs_i8(x, HIDDEN_DIM);
    if(sum == 0) sum = 1;
    int32_t mean = sum / HIDDEN_DIM;
    if(mean == 0) mean = 1;
    
    // Apply normalization
    for(int i=0; i < HIDDEN_DIM; i++) {
        int32_t val = ((int32_t)x[i] * w[i]) / mean;
        if(val > MAX_VAL) out[i] = MAX_VAL;
        else if(val < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)val;
    }
}

// --- Forward Pass (With Noise Injection) [cite: 915] ---
void forward_pass(
    EggModel *model, 
    const uint8_t *inputs, 
    const uint8_t *targets,
    int seq_len, 
    int8_t *logits_out, 
    int32_t *accumulated_loss,
    uint32_t step_seed, 
    int noise_sign,
    RecurrentState *rnn_state
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM * 4], buf2[HIDDEN_DIM];
    RecurrentState initial_state_copy;
#if defined(EGG_USE_METAL)
    bool gpu_used = false;
    bool compare_cpu = false;
        const char *cmp_env = getenv("EGG_COMPARE_CPU");
        if (cmp_env && cmp_env[0] == '1') compare_cpu = true;
        bool parity_trace = false;
        const char *trace_env = getenv("EGG_PARITY_TRACE");
        if (trace_env && trace_env[0] == '1') parity_trace = true;
        static bool gpu_gru_disabled = false;
        static bool gpu_gru_checked = false;
        if (!gpu_gru_checked) {
            const char *env = getenv("EGG_DISABLE_GPU_GRU");
            gpu_gru_disabled = (env && env[0] == '1');
            gpu_gru_checked = true;
        }
        int8_t gpu_snap_h[N_LAYERS][HIDDEN_DIM];
        bool gpu_snap_taken = false;
        if (compare_cpu) {
            if (rnn_state) initial_state_copy = *rnn_state;
            else memset(&initial_state_copy, 0, sizeof(initial_state_copy));
    }
#endif
    
    // If no state provided, use a temporary zeroed one (stateless mode)
    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }
    
    if(accumulated_loss) *accumulated_loss = 0;

#if defined(EGG_USE_METAL)
    // Begin batching GPU operations for this forward pass
    // Set EGG_DISABLE_GPU=1 to force CPU path even when Metal is available
    static bool gpu_disabled = false;
    static bool gpu_disabled_checked = false;
    if (!gpu_disabled_checked) {
        const char *env = getenv("EGG_DISABLE_GPU");
        gpu_disabled = (env && env[0] == '1');
        gpu_disabled_checked = true;
    }
    if (!gpu_disabled) {
        egg_gpu_batch_begin();
    }

#endif

    for(int t=0; t<seq_len; t++) {
        // Embedding
#if defined(EGG_USE_METAL)
        // Try GPU path: use GPU buffers throughout
        // Thread-local scratch buffers so parallel population workers don't trample
        // each other's Metal allocations.
        static _Thread_local void *gpu_embedding_buffer = NULL;
        static _Thread_local void *gpu_work_buffers[10] = {NULL}; // Reusable buffers (need more for MLP)
        static _Thread_local void *gpu_mlp_large_buffer = NULL; // For HIDDEN_DIM * 4
        // Note: gpu_noise_buffers removed - using hash-based noise generation instead!
        static _Thread_local void *gpu_h_state_buffers[N_LAYERS] = {NULL}; // One buffer per layer for h
        static _Thread_local bool buffers_allocated = false;

        if (!buffers_allocated) {
            // Allocate reusable GPU buffers once
            gpu_embedding_buffer = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            for (int i = 0; i < 10; i++) {
                gpu_work_buffers[i] = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            }
            gpu_mlp_large_buffer = egg_gpu_alloc_temp_buffer(HIDDEN_DIM * 4);
            // No noise buffer allocations needed - using hash-based generation!
            for (int l = 0; l < N_LAYERS; l++) {
                gpu_h_state_buffers[l] = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            }
            buffers_allocated = (gpu_embedding_buffer != NULL && gpu_mlp_large_buffer != NULL);
            for (int i = 0; i < 10; i++) {
                if (!gpu_work_buffers[i]) buffers_allocated = false;
            }
            // No noise buffer validation needed!
            for (int l = 0; l < N_LAYERS; l++) {
                if (!gpu_h_state_buffers[l]) buffers_allocated = false;
            }
#ifdef DEBUG
            if (buffers_allocated) {
                for (int l = 0; l < N_LAYERS; l++) {
                    assert(gpu_h_state_buffers[l] != NULL);
                    for (int k = 0; k < l; k++) {
                        assert(gpu_h_state_buffers[l] != gpu_h_state_buffers[k]);
                    }
                }
            }
#endif
        }
        
        if (!gpu_disabled && !gpu_gru_disabled && buffers_allocated && gpu_embedding_buffer) {
            // Copy embedding row to GPU
            egg_gpu_copy_to_buffer(gpu_embedding_buffer, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);
            
            // Keep CPU copy for xB computations (avoid GPU->CPU copies)
            int8_t x_cpu[HIDDEN_DIM];
            memcpy(x_cpu, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);
            
            // Use GPU buffer as x for this timestep
            void *gpu_x = gpu_embedding_buffer;
            void *gpu_residual = gpu_work_buffers[0];
            void *gpu_buf1 = gpu_work_buffers[1];
            void *gpu_buf2 = gpu_work_buffers[2];
            void *gpu_ft = gpu_work_buffers[3];
            void *gpu_ht = gpu_work_buffers[4];
            void *gpu_gated_past = gpu_work_buffers[5];
            void *gpu_h_state[N_LAYERS];
            for (int l = 0; l < N_LAYERS; l++) {
                gpu_h_state[l] = gpu_h_state_buffers[l]; // Dedicated buffer per layer
            }
            
            // Copy RNN state to GPU at start of timestep
            for (int l = 0; l < N_LAYERS; l++) {
                if (rnn_state) {
                    egg_gpu_copy_to_buffer(gpu_h_state[l], rnn_state->h[l], HIDDEN_DIM);
                } else {
                    static int8_t zero[HIDDEN_DIM] = {0};
                    egg_gpu_copy_to_buffer(gpu_h_state[l], zero, HIDDEN_DIM);
                }
            }
            
            // Layers - all on GPU
            for(int l=0; l<N_LAYERS; l++) {
                uint32_t l_seed = step_seed + (l * 100);
                
                // -- GRU --
                // Copy x to residual (on GPU)
                // For Shared buffers, we can just memcpy the contents
                void *gpu_x_contents_residual = egg_gpu_get_buffer_contents(gpu_x);
                void *gpu_residual_contents = egg_gpu_get_buffer_contents(gpu_residual);
                if (gpu_x_contents_residual && gpu_residual_contents) {
                    memcpy(gpu_residual_contents, gpu_x_contents_residual, HIDDEN_DIM);
                }
                
                // Layer norm on GPU
                egg_gpu_layer_norm(gpu_x, model->ln_weights[l][0], gpu_x, HIDDEN_DIM);

                // Matmuls using hash-based noise (no arrays, no sync needed!)
                uint32_t seed1_a = l_seed+1;
                uint32_t seed1_b = l_seed+1 + 0x10000; // Separate seed for B noise

                // Use new matmul with hash-based noise
                egg_gpu_matmul_noiseb(gpu_x, model->gru_weights[l][0], gpu_buf1,
                                      seed1_a, seed1_b, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, 0);

                uint32_t seed2_a = l_seed+2;
                uint32_t seed2_b = l_seed+2 + 0x10000;

                // Use new matmul for h_state too
                egg_gpu_matmul_noiseb(gpu_h_state[l], model->gru_weights[l][1], gpu_buf2,
                                      seed2_a, seed2_b, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, 0);
                
                // Element-wise ops on GPU: buf1 + buf2 + bias -> ft
                void *gpu_bias1 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_bias1) {
                    egg_gpu_copy_to_buffer(gpu_bias1, model->gru_biases[l][0], HIDDEN_DIM);
                    egg_gpu_clipped_add_three(gpu_buf1, gpu_buf2, gpu_bias1, gpu_ft, HIDDEN_DIM);
                    egg_gpu_free_temp_buffer(gpu_bias1);
                }
                
                // GRU gate on GPU: ft * h -> gated_past
                egg_gpu_gru_gate(gpu_ft, gpu_h_state[l], gpu_gated_past, HIDDEN_DIM);
                
                // More matmuls - use hash-based noise
                uint32_t seed3_a = l_seed+3;
                uint32_t seed3_b = l_seed+3 + 0x10000;

                egg_gpu_matmul_noiseb(gpu_x, model->gru_weights[l][2], gpu_buf1,
                                      seed3_a, seed3_b, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, 0);

                uint32_t seed4_a = l_seed+4;
                uint32_t seed4_b = l_seed+4 + 0x10000;

                egg_gpu_matmul_noiseb(gpu_gated_past, model->gru_weights[l][3], gpu_buf2,
                                      seed4_a, seed4_b, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, 0);
                
                // buf1 + buf2 + bias -> ht
                void *gpu_bias2 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_bias2) {
                    egg_gpu_copy_to_buffer(gpu_bias2, model->gru_biases[l][1], HIDDEN_DIM);
                    egg_gpu_clipped_add_three(gpu_buf1, gpu_buf2, gpu_bias2, gpu_ht, HIDDEN_DIM);
                    egg_gpu_free_temp_buffer(gpu_bias2);
                }
                
                // State update on GPU
                egg_gpu_gru_state_update(gpu_h_state[l], gpu_ft, gpu_ht, gpu_x, HIDDEN_DIM);
                
                // Residual add on GPU
                egg_gpu_clipped_add(gpu_x, gpu_residual, gpu_x, HIDDEN_DIM);
                
                // MLP on GPU - use GPU copy kernel for residual to avoid sync
                {
                    void *gpu_mlp_residual = gpu_residual; // Reuse buffer

                    // Use GPU copy for residual (no sync needed)
                    egg_gpu_clipped_add_scalar(gpu_x, 0, gpu_mlp_residual, HIDDEN_DIM);

                    // Layer norm on GPU
                    egg_gpu_layer_norm(gpu_x, model->ln_weights[l][1], gpu_x, HIDDEN_DIM);

                    // Expand matmul - use hash-based noise
                    uint32_t seed5_a = l_seed+5;
                    uint32_t seed5_b = l_seed+5 + 0x10000;

                    egg_gpu_matmul_noiseb(gpu_x, model->mlp_weights[l][0], gpu_mlp_large_buffer,
                                          seed5_a, seed5_b, HIDDEN_DIM * 4, HIDDEN_DIM, 8, noise_sign, 0);

                    // Project matmul - use hash-based noise
                    uint32_t seed6_a = l_seed+6;
                    uint32_t seed6_b = l_seed+6 + 0x10000;

                    egg_gpu_matmul_noiseb(gpu_mlp_large_buffer, model->mlp_weights[l][1], gpu_x,
                                          seed6_a, seed6_b, HIDDEN_DIM, HIDDEN_DIM * 4, 9, noise_sign, 0);

                    // Residual add on GPU
                    egg_gpu_clipped_add(gpu_x, gpu_mlp_residual, gpu_x, HIDDEN_DIM);
                }
            }
            
            // Final Head - on GPU
            static _Thread_local void *gpu_logits_buffer = NULL;
            if (!gpu_logits_buffer) {
                gpu_logits_buffer = egg_gpu_alloc_temp_buffer(VOCAB_SIZE);
            }
            
            if (gpu_logits_buffer) {
                // Layer norm on GPU (no sync needed - will sync at end)
                egg_gpu_layer_norm(gpu_x, model->ln_out, gpu_x, HIDDEN_DIM);

                // Head matmul - use hash-based noise
                uint32_t seed_head_a = step_seed+999;
                uint32_t seed_head_b = step_seed+999 + 0x10000;

                egg_gpu_matmul_noiseb(gpu_x, model->head, gpu_logits_buffer,
                                      seed_head_a, seed_head_b, VOCAB_SIZE, HIDDEN_DIM, 8, noise_sign, 0);

                // Single sync at end of timestep before reading all results
                egg_gpu_batch_sync();

                // Copy head logits for this timestep to logits_out for loss/compare
                void *gpu_logits_contents = egg_gpu_get_buffer_contents(gpu_logits_buffer);
                if (gpu_logits_contents) {
                    memcpy(logits_out, gpu_logits_contents, VOCAB_SIZE);
                }
            }

            // Copy RNN state back from GPU (synced above)
            if (rnn_state) {
                for (int l = 0; l < N_LAYERS; l++) {
                    egg_gpu_copy_from_buffer(rnn_state->h[l], gpu_h_state[l], HIDDEN_DIM);
                    if (compare_cpu && parity_trace && t == 0) {
                        egg_gpu_copy_from_buffer(gpu_snap_h[l], gpu_h_state[l], HIDDEN_DIM);
                        gpu_snap_taken = true;
                    }
                }
            }

            // Copy final x back to CPU for loss computation
            egg_gpu_copy_from_buffer(x, gpu_x, HIDDEN_DIM);
            
            if(targets && accumulated_loss) {
                *accumulated_loss += compute_loss(logits_out, targets[t]);
            }
            
            // Skip CPU path - continue to next timestep
            gpu_used = true;
            continue;
        }
        // If GPU buffers failed to allocate, fall through to CPU path
#endif
        
        // CPU path (fallback or when GPU not available)
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        // Layers
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            // -- GRU --
            memcpy(residual, x, HIDDEN_DIM);
#if defined(EGG_USE_METAL)
            // Use GPU for layer norm
            void *gpu_x = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_residual = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_buf1 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_buf2 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_ft = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_ht = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_gated_past = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            void *gpu_h = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
            
            if (gpu_x && gpu_residual && gpu_buf1 && gpu_buf2 && gpu_ft && gpu_ht && gpu_gated_past && gpu_h) {
                // Copy inputs to GPU
                egg_gpu_copy_to_buffer(gpu_x, x, HIDDEN_DIM);
                egg_gpu_copy_to_buffer(gpu_residual, residual, HIDDEN_DIM);
                egg_gpu_copy_to_buffer(gpu_h, rnn_state->h[l], HIDDEN_DIM);
                
                // Layer norm on GPU
                egg_gpu_layer_norm(gpu_x, model->ln_weights[l][0], gpu_x, HIDDEN_DIM);
                
                // Matmuls (output to GPU buffers)
                int8_t noiseA1[HIDDEN_DIM], noiseB1[HIDDEN_DIM];
                uint32_t rng1 = l_seed+1;
                gen_noise_vector(&rng1, noiseA1, HIDDEN_DIM);
                gen_noise_vector(&rng1, noiseB1, HIDDEN_DIM);
                
                // Upload noiseB to GPU and compute xB on GPU
                void *gpu_noiseB1 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_noiseB1) {
                    egg_gpu_copy_to_buffer(gpu_noiseB1, noiseB1, HIDDEN_DIM);
                    int32_t xB1 = 0;
                    if (egg_gpu_dot_product(gpu_x, gpu_noiseB1, &xB1, HIDDEN_DIM)) {
                        int8_t *buf1_ptr = (int8_t*)egg_gpu_get_buffer_contents(gpu_buf1);
                        if (buf1_ptr) {
                            // Copy x to CPU for matmul (matmul still needs CPU input for now)
                            int8_t x_cpu[HIDDEN_DIM];
                            egg_gpu_copy_from_buffer(x_cpu, gpu_x, HIDDEN_DIM);
                            egg_gpu_matmul_perturbed(x_cpu, model->gru_weights[l][0], buf1_ptr,
                                                     noiseA1, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, xB1, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB1);
                }
                
                int8_t noiseA2[HIDDEN_DIM], noiseB2[HIDDEN_DIM];
                uint32_t rng2 = l_seed+2;
                gen_noise_vector(&rng2, noiseA2, HIDDEN_DIM);
                gen_noise_vector(&rng2, noiseB2, HIDDEN_DIM);
                
                // Upload noiseB2 to GPU and compute xB2 on GPU
                void *gpu_noiseB2 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_noiseB2) {
                    egg_gpu_copy_to_buffer(gpu_noiseB2, noiseB2, HIDDEN_DIM);
                    int32_t xB2 = 0;
                    if (egg_gpu_dot_product(gpu_h, gpu_noiseB2, &xB2, HIDDEN_DIM)) {
                        int8_t *buf2_ptr = (int8_t*)egg_gpu_get_buffer_contents(gpu_buf2);
                        if (buf2_ptr) {
                            // Copy h to CPU for matmul
                            int8_t h_cpu[HIDDEN_DIM];
                            egg_gpu_copy_from_buffer(h_cpu, gpu_h, HIDDEN_DIM);
                            egg_gpu_matmul_perturbed(h_cpu, model->gru_weights[l][1], buf2_ptr,
                                                     noiseA2, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, xB2, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB2);
                }
                
                // Element-wise ops on GPU: buf1 + buf2 + bias -> ft
                void *gpu_bias1 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_bias1) {
                    egg_gpu_copy_to_buffer(gpu_bias1, model->gru_biases[l][0], HIDDEN_DIM);
                    egg_gpu_clipped_add_three(gpu_buf1, gpu_buf2, gpu_bias1, gpu_ft, HIDDEN_DIM);
                    egg_gpu_free_temp_buffer(gpu_bias1);
                }
                
                // GRU gate on GPU: ft * h -> gated_past
                egg_gpu_gru_gate(gpu_ft, gpu_h, gpu_gated_past, HIDDEN_DIM);
                
                // More matmuls...
                int8_t noiseA3[HIDDEN_DIM], noiseB3[HIDDEN_DIM];
                uint32_t rng3 = l_seed+3;
                gen_noise_vector(&rng3, noiseA3, HIDDEN_DIM);
                gen_noise_vector(&rng3, noiseB3, HIDDEN_DIM);
                
                // Compute xB3 on GPU
                void *gpu_noiseB3 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_noiseB3) {
                    egg_gpu_copy_to_buffer(gpu_noiseB3, noiseB3, HIDDEN_DIM);
                    int32_t xB3 = 0;
                    if (egg_gpu_dot_product(gpu_x, gpu_noiseB3, &xB3, HIDDEN_DIM)) {
                        int8_t *buf1_ptr2 = (int8_t*)egg_gpu_get_buffer_contents(gpu_buf1);
                        if (buf1_ptr2) {
                            int8_t x_cpu2[HIDDEN_DIM];
                            egg_gpu_copy_from_buffer(x_cpu2, gpu_x, HIDDEN_DIM);
                            egg_gpu_matmul_perturbed(x_cpu2, model->gru_weights[l][2], buf1_ptr2,
                                                     noiseA3, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, xB3, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB3);
                }
                
                int8_t noiseA4[HIDDEN_DIM], noiseB4[HIDDEN_DIM];
                uint32_t rng4 = l_seed+4;
                gen_noise_vector(&rng4, noiseA4, HIDDEN_DIM);
                gen_noise_vector(&rng4, noiseB4, HIDDEN_DIM);
                
                // Compute xB4 on GPU
                void *gpu_noiseB4 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_noiseB4) {
                    egg_gpu_copy_to_buffer(gpu_noiseB4, noiseB4, HIDDEN_DIM);
                    int32_t xB4 = 0;
                    if (egg_gpu_dot_product(gpu_gated_past, gpu_noiseB4, &xB4, HIDDEN_DIM)) {
                        int8_t *buf2_ptr2 = (int8_t*)egg_gpu_get_buffer_contents(gpu_buf2);
                        if (buf2_ptr2) {
                            int8_t gated_cpu[HIDDEN_DIM];
                            egg_gpu_copy_from_buffer(gated_cpu, gpu_gated_past, HIDDEN_DIM);
                            egg_gpu_matmul_perturbed(gated_cpu, model->gru_weights[l][3], buf2_ptr2,
                                                     noiseA4, HIDDEN_DIM, HIDDEN_DIM, 8, noise_sign, xB4, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB4);
                }
                
                // buf1 + buf2 + bias -> ht
                void *gpu_bias2 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_bias2) {
                    egg_gpu_copy_to_buffer(gpu_bias2, model->gru_biases[l][1], HIDDEN_DIM);
                    egg_gpu_clipped_add_three(gpu_buf1, gpu_buf2, gpu_bias2, gpu_ht, HIDDEN_DIM);
                    egg_gpu_free_temp_buffer(gpu_bias2);
                }
                
                // State update on GPU
                egg_gpu_gru_state_update(gpu_h, gpu_ft, gpu_ht, gpu_x, HIDDEN_DIM);
                
                // Residual add on GPU
                egg_gpu_clipped_add(gpu_x, gpu_residual, gpu_x, HIDDEN_DIM);
                
                // Copy final x back to CPU
                egg_gpu_copy_from_buffer(x, gpu_x, HIDDEN_DIM);
                egg_gpu_copy_from_buffer(rnn_state->h[l], gpu_h, HIDDEN_DIM);
                
                // Cleanup
                egg_gpu_free_temp_buffer(gpu_x);
                egg_gpu_free_temp_buffer(gpu_residual);
                egg_gpu_free_temp_buffer(gpu_buf1);
                egg_gpu_free_temp_buffer(gpu_buf2);
                egg_gpu_free_temp_buffer(gpu_ft);
                egg_gpu_free_temp_buffer(gpu_ht);
                egg_gpu_free_temp_buffer(gpu_gated_past);
                egg_gpu_free_temp_buffer(gpu_h);
                
                // Skip CPU path
                goto skip_cpu_gru;
            }
            
            // Fallback: free any allocated buffers
            if (gpu_x) egg_gpu_free_temp_buffer(gpu_x);
            if (gpu_residual) egg_gpu_free_temp_buffer(gpu_residual);
            if (gpu_buf1) egg_gpu_free_temp_buffer(gpu_buf1);
            if (gpu_buf2) egg_gpu_free_temp_buffer(gpu_buf2);
            if (gpu_ft) egg_gpu_free_temp_buffer(gpu_ft);
            if (gpu_ht) egg_gpu_free_temp_buffer(gpu_ht);
            if (gpu_gated_past) egg_gpu_free_temp_buffer(gpu_gated_past);
            if (gpu_h) egg_gpu_free_temp_buffer(gpu_h);
#endif
            
            // CPU path (fallback)
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+1, noise_sign, 8);
            matmul_perturbed(rnn_state->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+2, noise_sign, 8);
            
            int8_t ft[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * rnn_state->h[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+3, noise_sign, 8);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+4, noise_sign, 8);
            
            int8_t ht[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            // State Update
            for(int i=0; i<HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - rnn_state->h[l][i])) >> 8;
                rnn_state->h[l][i] = clipped_add(rnn_state->h[l][i], update);
                x[i] = rnn_state->h[l][i]; // Output is new state
            }
            
            // Residual Add
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
            
#if defined(EGG_USE_METAL)
skip_cpu_gru:
#endif

            // -- MLP --
#if defined(EGG_USE_METAL)
            // GPU path for MLP
            {
                void *gpu_mlp_residual = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                void *gpu_mlp_buf1 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM * 4);
                
                if (gpu_mlp_residual && gpu_mlp_buf1) {
                // Copy residual
                egg_gpu_copy_to_buffer(gpu_mlp_residual, x, HIDDEN_DIM);
                
                // Layer norm on GPU
                egg_gpu_layer_norm(gpu_x, model->ln_weights[l][1], gpu_x, HIDDEN_DIM);
                
                // Expand matmul: Hidden -> 4*Hidden
                int8_t noiseA5[HIDDEN_DIM * 4], noiseB5[HIDDEN_DIM];
                uint32_t rng5 = l_seed+5;
                gen_noise_vector(&rng5, noiseA5, HIDDEN_DIM * 4);
                gen_noise_vector(&rng5, noiseB5, HIDDEN_DIM);
                
                void *gpu_noiseB5 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM);
                if (gpu_noiseB5) {
                    egg_gpu_copy_to_buffer(gpu_noiseB5, noiseB5, HIDDEN_DIM);
                    int32_t xB5 = 0;
                    if (egg_gpu_dot_product(gpu_x, gpu_noiseB5, &xB5, HIDDEN_DIM)) {
                        int8_t *mlp_buf1_ptr = (int8_t*)egg_gpu_get_buffer_contents(gpu_mlp_buf1);
                        if (mlp_buf1_ptr) {
                            int8_t x_mlp_cpu[HIDDEN_DIM];
                            egg_gpu_copy_from_buffer(x_mlp_cpu, gpu_x, HIDDEN_DIM);
                            egg_gpu_matmul_perturbed(x_mlp_cpu, model->mlp_weights[l][0], mlp_buf1_ptr,
                                                     noiseA5, HIDDEN_DIM * 4, HIDDEN_DIM, 8, noise_sign, xB5, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB5);
                }
                
                // Project matmul: 4*Hidden -> Hidden
                int8_t noiseA6[HIDDEN_DIM], noiseB6[HIDDEN_DIM * 4];
                uint32_t rng6 = l_seed+6;
                gen_noise_vector(&rng6, noiseA6, HIDDEN_DIM);
                gen_noise_vector(&rng6, noiseB6, HIDDEN_DIM * 4);
                
                void *gpu_noiseB6 = egg_gpu_alloc_temp_buffer(HIDDEN_DIM * 4);
                if (gpu_noiseB6) {
                    egg_gpu_copy_to_buffer(gpu_noiseB6, noiseB6, HIDDEN_DIM * 4);
                    int32_t xB6 = 0;
                    if (egg_gpu_dot_product(gpu_mlp_buf1, gpu_noiseB6, &xB6, HIDDEN_DIM * 4)) {
                        int8_t *x_final_ptr = (int8_t*)egg_gpu_get_buffer_contents(gpu_x);
                        if (x_final_ptr) {
                            int8_t buf1_mlp_cpu[HIDDEN_DIM * 4];
                            egg_gpu_copy_from_buffer(buf1_mlp_cpu, gpu_mlp_buf1, HIDDEN_DIM * 4);
                            egg_gpu_matmul_perturbed(buf1_mlp_cpu, model->mlp_weights[l][1], x_final_ptr,
                                                     noiseA6, HIDDEN_DIM, HIDDEN_DIM * 4, 9, noise_sign, xB6, 0);
                        }
                    }
                    egg_gpu_free_temp_buffer(gpu_noiseB6);
                }
                
                // Residual add on GPU
                egg_gpu_clipped_add(gpu_x, gpu_mlp_residual, gpu_x, HIDDEN_DIM);
                
                // Copy final x back to CPU
                egg_gpu_copy_from_buffer(x, gpu_x, HIDDEN_DIM);
                
                    egg_gpu_free_temp_buffer(gpu_mlp_residual);
                    egg_gpu_free_temp_buffer(gpu_mlp_buf1);
                } else {
                    // Fallback: free buffers
                    if (gpu_mlp_residual) egg_gpu_free_temp_buffer(gpu_mlp_residual);
                    if (gpu_mlp_buf1) egg_gpu_free_temp_buffer(gpu_mlp_buf1);
                    
                    // CPU path (fallback)
                    memcpy(residual, x, HIDDEN_DIM);
                    egg_ln(x, model->ln_weights[l][1], x);
                    
                    // Expand: Hidden -> 4*Hidden
                    // Cols = HIDDEN_DIM (256). Sqrt(256)=16. Shift = 4+4=8.
                    matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, noise_sign, 8);
                    
                    // Project: 4*Hidden -> Hidden
                    // Cols = HIDDEN_DIM*4 (1024). Sqrt(1024)=32. Shift = 4+5=9.
                    matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, noise_sign, 9);

                    for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
                }
            }
#else
            // CPU path
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);
            
            // Expand: Hidden -> 4*Hidden
            // Cols = HIDDEN_DIM (256). Sqrt(256)=16. Shift = 4+4=8.
            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, noise_sign, 8);
            
            // Project: 4*Hidden -> Hidden
            // Cols = HIDDEN_DIM*4 (1024). Sqrt(1024)=32. Shift = 4+5=9.
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, noise_sign, 9);

            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
#endif
        }

        // Final Head
        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, noise_sign, 8);
        
        if(targets && accumulated_loss) {
            *accumulated_loss += compute_loss(logits_out, targets[t]);
        }
    }

#if defined(EGG_USE_METAL)
    // End batching: commit all GPU operations and wait for results
    if (!gpu_disabled) {
        egg_gpu_batch_end();
    }
    // If compate to reference cpu implementation mode
    if (gpu_used && compare_cpu && targets) {
        int8_t cpu_logits[VOCAB_SIZE];
        int32_t cpu_loss = 0;
        RecurrentState cpu_state = initial_state_copy; // start from pre-GPU state
        forward_pass_cpu_reference(model, inputs, targets, seq_len,
                                   cpu_logits, &cpu_loss, step_seed, noise_sign, &cpu_state);
        int max_diff = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            int d = (int)logits_out[i] - (int)cpu_logits[i];
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        }
        fprintf(stderr, "[EGG METAL][COMPARE] loss_gpu=%d loss_cpu=%d max_logit_diff=%d\n",
                (int)(accumulated_loss ? *accumulated_loss : 0), (int)cpu_loss, max_diff);

        const char *trace_env = getenv("EGG_PARITY_TRACE");
        if (trace_env && trace_env[0] == '1') {
            // Compare per-layer hidden states and head input
            int max_h_diff = 0;
            int max_head_diff = 0;
            for (int l = 0; l < N_LAYERS; l++) {
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    int d = (int)rnn_state->h[l][i] - (int)cpu_state.h[l][i];
                    if (d < 0) d = -d;
                    if (d > max_h_diff) max_h_diff = d;
                }
            }
            for (int i = 0; i < HIDDEN_DIM; i++) {
                int d = (int)rnn_state->h[N_LAYERS-1][i] - (int)cpu_state.h[N_LAYERS-1][i];
                if (d < 0) d = -d;
                if (d > max_head_diff) max_head_diff = d;
            }
            fprintf(stderr, "[EGG METAL][TRACE] max_h_diff=%d max_head_input_diff=%d\n",
                    max_h_diff, max_head_diff);

            if (gpu_snap_taken) {
                int8_t cpu_snap[N_LAYERS][HIDDEN_DIM];
                RecurrentState snap_state = initial_state_copy;
                forward_pass_cpu_reference_snap(
                    model, inputs, seq_len, step_seed, noise_sign, &snap_state, cpu_snap);
                for (int l = 0; l < N_LAYERS; l++) {
                    int layer_max = 0;
                    for (int i = 0; i < HIDDEN_DIM; i++) {
                        int d = (int)gpu_snap_h[l][i] - (int)cpu_snap[l][i];
                        if (d < 0) d = -d;
                        if (d > layer_max) layer_max = d;
                    }
                    if (layer_max) {
                        fprintf(stderr, "[EGG METAL][TRACE] layer %d h_diff=%d\n", l, layer_max);
                    }
                }
            }
        }
        const char *dump_env = getenv("EGG_DUMP_T0");
        if (dump_env && dump_env[0] == '1') {
            FILE *f = fopen("/tmp/egg_cpu_logits.bin", "wb");
            if (f) {
                fwrite(cpu_logits, 1, VOCAB_SIZE, f);
                fclose(f);
            }
            f = fopen("/tmp/egg_cpu_x_head.bin", "wb");
            if (f) {
                fwrite(cpu_state.h[N_LAYERS-1], 1, HIDDEN_DIM, f); // head input after last ln
                fclose(f);
            }
        }
    }
#endif
}

// --- CPU-only reference forward (no GPU, no batching, for debugging parity) ---
static void forward_pass_cpu_reference(
    EggModel *model,
    const uint8_t *inputs,
    const uint8_t *targets,
    int seq_len,
    int8_t *logits_out,
    int32_t *accumulated_loss,
    uint32_t step_seed,
    int noise_sign,
    RecurrentState *rnn_state
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM * 4], buf2[HIDDEN_DIM];

    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }
    if (accumulated_loss) *accumulated_loss = 0;

    for (int t = 0; t < seq_len; t++) {
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed + 1, noise_sign, 8);
            matmul_perturbed(rnn_state->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 2, noise_sign, 8);

            int8_t ft[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * rnn_state->h[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed + 3, noise_sign, 8);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 4, noise_sign, 8);

            int8_t ht[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            for (int i = 0; i < HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - rnn_state->h[l][i])) >> 8;
                rnn_state->h[l][i] = clipped_add(rnn_state->h[l][i], update);
                x[i] = rnn_state->h[l][i];
            }

            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);

            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed + 5, noise_sign, 8);
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed + 6, noise_sign, 9);

            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // Optional dump before ln_out (t==0)
        const char *dump_env = getenv("EGG_DUMP_T0");
        if (dump_env && dump_env[0] == '1' && t == 0) {
            FILE *f = fopen("/tmp/egg_cpu_x_before_lnout.bin", "wb");
            if (f) { fwrite(x, 1, HIDDEN_DIM, f); fclose(f); }
        }

        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed + 999, noise_sign, 8);

        if (targets && accumulated_loss) {
            *accumulated_loss += compute_loss(logits_out, targets[t]);
        }
    }
}

// CPU reference forward that captures per-layer h after each layer for t==0 (parity tracing)
static void forward_pass_cpu_reference_snap(
    EggModel *model,
    const uint8_t *inputs,
    int seq_len,
    uint32_t step_seed,
    int noise_sign,
    RecurrentState *rnn_state,
    int8_t snap_h[N_LAYERS][HIDDEN_DIM]
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM * 4], buf2[HIDDEN_DIM];

    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }

    for (int t = 0; t < seq_len; t++) {
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed + 1, noise_sign, 8);
            matmul_perturbed(rnn_state->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 2, noise_sign, 8);

            int8_t ft[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * rnn_state->h[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed + 3, noise_sign, 8);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 4, noise_sign, 8);

            int8_t ht[HIDDEN_DIM];
            for (int i = 0; i < HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            for (int i = 0; i < HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - rnn_state->h[l][i])) >> 8;
                rnn_state->h[l][i] = clipped_add(rnn_state->h[l][i], update);
                x[i] = rnn_state->h[l][i];
            }

            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);

            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed + 5, noise_sign, 8);
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed + 6, noise_sign, 9);

            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            if (t == 0) {
                memcpy(snap_h[l], rnn_state->h[l], HIDDEN_DIM);
            }
        }

        // We only need snapshots at t==0; no further work required
        if (t == 0) break;
    }
}

static inline void evaluate_population_pair(
    EggModel *model,
    const Dataset *ds,
    long start_idx,
    int *pair_fitnesses,
    RecurrentState *pop_states,
    uint32_t step_seed,
    size_t p_idx
) {
    uint32_t p_seed = step_seed + (uint32_t)p_idx;
    int8_t local_logits[VOCAB_SIZE];
    int32_t loss_pos = 0;
    int32_t loss_neg = 0;

    long stride = ds->length / (POPULATION_SIZE / 2);
    long stream_idx = (start_idx + (p_idx * stride)) % (ds->length - SEQ_LEN);

    forward_pass(model, &ds->data[stream_idx], &ds->data[stream_idx+1], SEQ_LEN, local_logits, &loss_pos, p_seed, 1, &pop_states[p_idx*2]);
    forward_pass(model, &ds->data[stream_idx], &ds->data[stream_idx+1], SEQ_LEN, local_logits, &loss_neg, p_seed, -1, &pop_states[p_idx*2+1]);

    if (loss_pos < loss_neg) pair_fitnesses[p_idx] = 1;
    else if (loss_neg < loss_pos) pair_fitnesses[p_idx] = -1;
    else pair_fitnesses[p_idx] = 0;
}

#if defined(__APPLE__)
typedef struct {
    EggModel *model;
    const Dataset *ds;
    long start_idx;
    int *pair_fitnesses;
    RecurrentState *pop_states;
    uint32_t step_seed;
} EggDispatchContext;

static void egg_dispatch_apply(void *ctx_void, size_t p_idx) {
    EggDispatchContext *ctx = (EggDispatchContext *)ctx_void;
    evaluate_population_pair(
        ctx->model,
        ctx->ds,
        ctx->start_idx,
        ctx->pair_fitnesses,
        ctx->pop_states,
        ctx->step_seed,
        p_idx
    );
}
#endif

// --- Helper for Loss Calculation ---
static inline int get_msb(uint32_t n) {
    int pos = 0;
    if (n >= 1<<16) { n >>= 16; pos += 16; }
    if (n >= 1<<8)  { n >>= 8;  pos += 8; }
    if (n >= 1<<4)  { n >>= 4;  pos += 4; }
    if (n >= 1<<2)  { n >>= 2;  pos += 2; }
    if (n >= 1<<1)  {           pos += 1; }
    return pos;
}

int32_t log2_fixed(int32_t x) {
    if (x <= 0) return 0;
    int k = get_msb(x);
    int32_t fraction;
    if (k >= 4) {
        fraction = (x - (1 << k)) >> (k - 4);
    } else {
        fraction = (x - (1 << k)) << (4 - k);
    }
    return (k << 4) + fraction - 64;
}

int32_t compute_loss(int8_t *logits, uint8_t target) {
    int32_t sum_exp = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        sum_exp += EXP2_TABLE[idx < 0 ? 0 : (idx > 255 ? 255 : idx)];
    }
    int32_t log_sum = log2_fixed(sum_exp);
    int32_t target_logit = (int32_t)logits[target] + 128;
    return log_sum - target_logit;
}

void update_matrix(
    int8_t *W, int rows, int cols, 
    uint32_t seed, 
    const int *fitnesses, 
    int pop_size
) {
    // Optimized "Inverted Loop" Strategy:
    // 1. Pre-compute and transpose noise vectors into compact buffers.
    // 2. Single pass over weights to compute votes (dot product over population) and update.
    // This eliminates the large 'votes' memset and repeated sweeps.

    // Transposed buffers: [index][population_pair_index]
    // A_T stores A * f directly.
    static EGG_ALIGNED16 int8_t A_T[MAX_MATRIX_DIM][MAX_POP_PAIRS];
    static EGG_ALIGNED16 int8_t B_T[MAX_MATRIX_DIM][MAX_POP_PAIRS];
    
    int pairs = pop_size / 2;
    if (pairs > MAX_POP_PAIRS) pairs = MAX_POP_PAIRS;

    // 1. Pre-compute Noise and Transpose
    for(int p=0; p<pairs; p++) {
        int f = fitnesses[p];
        uint32_t rng = seed + p;

        int8_t A_temp[MAX_MATRIX_DIM];
        int8_t B_temp[MAX_MATRIX_DIM];
        
        gen_noise_vector(&rng, A_temp, rows);
        gen_noise_vector(&rng, B_temp, cols);

        if (f == 0) {
            for(int r=0; r<rows; r++) A_T[r][p] = 0;
        } else {
            for(int r=0; r<rows; r++) A_T[r][p] = (int8_t)(A_temp[r] * f);
            for(int c=0; c<cols; c++) B_T[c][p] = B_temp[c];
        }
    }

    bool gpu_handled = false;
#if defined(EGG_USE_METAL)
    // Optional escape hatch: force CPU updates even when Metal is available.
    static bool gpu_update_disabled = false;
    static bool gpu_update_checked = false;
    if (!gpu_update_checked) {
        const char *env = getenv("EGG_DISABLE_GPU_UPDATE");
        gpu_update_disabled = (env && env[0] == '1');
        gpu_update_checked = true;
    }
    if (!gpu_update_disabled) {
        gpu_handled = egg_gpu_update_matrix(
            W,
            (const int8_t *)A_T,
            (const int8_t *)B_T,
            rows,
            cols,
            pairs
        );
    }
#endif
    if (gpu_handled) return;

    // 2. Compute Votes and Update Weights (Single Pass)
    for(int r=0; r<rows; r++) {
        int8_t *w_row = &W[r * cols];
        int8_t *a_ptr = A_T[r];

        for(int c=0; c<cols; c++) {
            int8_t *b_ptr = B_T[c];
            int32_t vote = dot_product_i8(a_ptr, b_ptr, pairs);

            // Apply Update
            if(vote > UPDATE_THRESHOLD && w_row[c] < MAX_VAL) w_row[c]++;
            else if(vote < -UPDATE_THRESHOLD && w_row[c] > MIN_VAL) w_row[c]--;
        }
    }
}

// --- Sampling Helper ---
#define COLOR_GREEN "\033[32m"
#define COLOR_CYAN  "\033[36m"
#define COLOR_RESET "\033[0m"

int sample_logits(int8_t *logits) {
    int32_t probs[VOCAB_SIZE];
    int32_t sum = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        probs[i] = EXP2_TABLE[idx];
        sum += probs[i];
    }
    if(sum == 0) return 0;
    int32_t r = rand() % sum;
    int32_t acc = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        acc += probs[i];
        if(r < acc) return i;
    }
    return VOCAB_SIZE - 1;
}

void sample_model(EggModel *model, const uint8_t *seed_text, int seed_len, int gen_len) {
    int8_t logits[VOCAB_SIZE];
    RecurrentState state;
    memset(&state, 0, sizeof(state));

    printf(COLOR_GREEN);
    int input_token = 0;
    
    for(int t=0; t < seed_len + gen_len; t++) {
        if (t < seed_len) {
            input_token = seed_text[t];
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        } else {
            if (t == seed_len) printf(COLOR_CYAN);
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        }

        // Infer Only - No Noise
        forward_pass(model, (uint8_t*)&input_token, NULL, 1, logits, NULL, 0, 0, &state);
        // Note: forward_pass handles state update internally for seq_len=1
        
        if (t >= seed_len - 1) {
            input_token = sample_logits(logits);
        }
    }
    printf(COLOR_RESET "\n");
}

Dataset load_data(const char *filename, char *resolved_path_out, size_t resolved_size) {
    bool is_pipe = false;
    FILE *f = open_dataset_stream(filename, &is_pipe, resolved_path_out, resolved_size);
    if (!f) {
        fprintf(stderr, "Error: failed to open dataset '%s'.\n", filename);
        exit(1);
    }

    size_t capacity = 1 << 20;
    uint8_t *data = (uint8_t*)malloc(capacity);
    if (!data) {
        fprintf(stderr, "Error: failed to allocate initial buffer for dataset.\n");
        close_dataset_stream(f, is_pipe);
        exit(1);
    }

    size_t len = 0;
    while (1) {
        if (len == capacity) {
            size_t new_capacity = capacity * 2;
            uint8_t *tmp = (uint8_t*)realloc(data, new_capacity);
            if (!tmp) {
                fprintf(stderr, "Error: realloc failed while reading dataset.\n");
                free(data);
                close_dataset_stream(f, is_pipe);
                exit(1);
            }
            data = tmp;
            capacity = new_capacity;
        }

        size_t space = capacity - len;
        size_t bytes_read = fread(data + len, 1, space, f);
        if (bytes_read == 0) {
            if (feof(f)) break;
            fprintf(stderr, "Error: failed to read from dataset '%s'.\n", filename);
            free(data);
            close_dataset_stream(f, is_pipe);
            exit(1);
        }
        len += bytes_read;
    }

    close_dataset_stream(f, is_pipe);
    return (Dataset){data, (long)len};
}

int main() {
    srand(time(NULL));
    init_tables();

    // Prefer compressed input if it exists
    const char *dataset_path = "input.txt";
    if (access("input.txt.zst", F_OK) == 0) {
        dataset_path = "input.txt.zst";
    }
    char resolved_path[PATH_MAX];
    Dataset ds = load_data(dataset_path, resolved_path, sizeof(resolved_path));
    printf("Loaded dataset: %ld bytes from %s\n", ds.length, resolved_path);
    fflush(stdout);
    
    EggModel *model = NULL;
    int pm_status = posix_memalign((void**)&model, 16, sizeof(EggModel));
    if (pm_status != 0 || !model) {
        fprintf(stderr, "Error: posix_memalign failed (code %d).\n", pm_status);
        free(ds.data);
        exit(1);
    }
    memset(model, 0, sizeof(EggModel));

#if defined(EGG_USE_METAL)
    if (!egg_gpu_metal_init()) {
        fprintf(stderr, "Error: failed to initialize Metal backend.\n");
        free(ds.data);
        free(model);
        exit(1);
    }

    // Bind model weights into a persistent Metal buffer so we avoid
    // re-uploading large matrices on every kernel launch.
    size_t embedding_size = (size_t)VOCAB_SIZE * (size_t)HIDDEN_DIM;
    size_t gru_size       = (size_t)N_LAYERS * 4u * (size_t)HIDDEN_DIM * (size_t)HIDDEN_DIM;
    size_t mlp_size       = (size_t)N_LAYERS * 2u * (size_t)HIDDEN_DIM * (size_t)(HIDDEN_DIM * 4);
    size_t head_size      = (size_t)HIDDEN_DIM * (size_t)VOCAB_SIZE;

    if (!egg_gpu_bind_model_weights(
            model->embedding,                   embedding_size,
            &model->gru_weights[0][0][0],      gru_size,
            &model->mlp_weights[0][0][0],      mlp_size,
            model->head,                       head_size
        )) {
        fprintf(stderr, "Error: failed to bind model weights to Metal.\n");
        egg_gpu_metal_shutdown();
        free(ds.data);
        free(model);
        exit(1);
    }
#endif

    uint32_t init_rng = 42;
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_val(&init_rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) model->head[i] = gen_noise_val(&init_rng);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) {
                model->gru_weights[l][g][i] = gen_noise_val(&init_rng);
            }
        }
        for(int m=0; m<2; m++) {
             // Utilize the full allocated space: HIDDEN * (HIDDEN * 4)
             for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) {
                 model->mlp_weights[l][m][i] = gen_noise_val(&init_rng);
             }
        }
    }

    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    }

    int *pair_fitnesses = (int*)malloc((POPULATION_SIZE/2) * sizeof(int));
    int8_t *logits = (int8_t*)malloc(VOCAB_SIZE);
    
    // Persistent States
    RecurrentState *pop_states = (RecurrentState*)aligned_alloc(16, POPULATION_SIZE * sizeof(RecurrentState));
    RecurrentState main_state;
    memset(pop_states, 0, POPULATION_SIZE * sizeof(RecurrentState));
    memset(&main_state, 0, sizeof(RecurrentState));

    time_t now = time(NULL);
    char time_buf[32] = "unknown";
    struct tm *tm_now = localtime(&now);
    if (tm_now) {
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_now);
    }
    printf("Starting EGGROLL Training (Stateful + Optimized) at datetime %s\n", time_buf);
    fflush(stdout);
    
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    long total_tokens = 0;
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step < max_steps; step++) {
        // Use a more robust seed mixing to avoid correlations if time(NULL) doesn't change
        uint32_t step_seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        if(step % 10 == 0) {
            sample_model(model, &ds.data[start_idx], 30, 30);
            
            int32_t loss_val = 0;
            forward_pass(model, &ds.data[start_idx], &ds.data[start_idx+1], SEQ_LEN, logits, &loss_val, step_seed, 0, &main_state);
            
            struct timespec current_time;
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double elapsed_sec = (current_time.tv_sec - start_time.tv_sec) + 
                                 (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            double tps = (elapsed_sec > 0) ? (double)total_tokens / elapsed_sec : 0.0;

            /* Format current wall-clock time */
            time_t now = time(NULL);
            char time_buf[32] = "unknown";
            struct tm *tm_now = localtime(&now);
            if (tm_now) {
                strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_now);
            }

            // Loss is accumulated fixed point. Divide by SEQ_LEN to get average per token, then by 2^FIXED_POINT
            printf("Step %ld/%ld | Loss: %.4f | Tok/s: %.2f at datetime %s\n",
                   step, max_steps,
                   (double)loss_val / (SEQ_LEN * (1 << FIXED_POINT)),
                   tps,
                   time_buf);
            fflush(stdout);
        }

#if defined(__APPLE__)
        EggDispatchContext ctx = {
            model,
            &ds,
            start_idx,
            pair_fitnesses,
            pop_states,
            step_seed
        };
        dispatch_apply_f(
            POPULATION_SIZE / 2,
            dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
            &ctx,
            egg_dispatch_apply
        );
#elif defined(_OPENMP)
        {
            size_t egg_pairs = POPULATION_SIZE / 2;
#pragma omp parallel for schedule(static)
            for (size_t p_idx = 0; p_idx < egg_pairs; ++p_idx) {
                evaluate_population_pair(model, &ds, start_idx, pair_fitnesses, pop_states, step_seed, p_idx);
            }
        }
#else
        for (size_t p_idx = 0; p_idx < POPULATION_SIZE / 2; ++p_idx) {
            evaluate_population_pair(model, &ds, start_idx, pair_fitnesses, pop_states, step_seed, p_idx);
        }
#endif

        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);
            update_matrix(model->gru_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed+1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed+2, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][2], HIDDEN_DIM, HIDDEN_DIM, l_seed+3, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][3], HIDDEN_DIM, HIDDEN_DIM, l_seed+4, pair_fitnesses, POPULATION_SIZE);
            
            update_matrix(model->mlp_weights[l][0], HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, pair_fitnesses, POPULATION_SIZE);
        }
        update_matrix(model->head, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, pair_fitnesses, POPULATION_SIZE);

        total_tokens += SEQ_LEN;
    }

    printf("Training Done.\n");
    free(ds.data); free(model); free(logits); free(pair_fitnesses);
    free(pop_states);
#if defined(EGG_USE_METAL)
    egg_gpu_metal_shutdown();
#endif
    return 0;
}
