#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>

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

static FILE *open_dataset_stream(const char *filename, bool *is_pipe) {
    *is_pipe = has_zstd_extension(filename);
    if (!*is_pipe) return fopen(filename, "rb");

    if (strchr(filename, '\'')) {
        fprintf(stderr, "Error: filename '%s' contains a single quote; cannot safely stream.\n", filename);
        return NULL;
    }

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
    char cmd[PATH_MAX + 32];
    int written = snprintf(cmd, sizeof(cmd), "zstd -dc -- '%s'", filename);
    if (written < 0 || written >= (int)sizeof(cmd)) {
        fprintf(stderr, "Error: command too long for filename '%s'.\n", filename);
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
            shift, noise_sign, xB)) {
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
    
    // If no state provided, use a temporary zeroed one (stateless mode)
    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }
    
    if(accumulated_loss) *accumulated_loss = 0;

    for(int t=0; t<seq_len; t++) {
        // Embedding
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        // Layers
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            // -- GRU --
            memcpy(residual, x, HIDDEN_DIM);
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

            // -- MLP --
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

        // Final Head
        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, noise_sign, 8);
        
        if(targets && accumulated_loss) {
            *accumulated_loss += compute_loss(logits_out, targets[t]);
        }
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
    gpu_handled = egg_gpu_update_matrix(
        W,
        (const int8_t *)A_T,
        (const int8_t *)B_T,
        rows,
        cols,
        pairs
    );
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

Dataset load_data(const char *filename) {
    bool is_pipe = false;
    FILE *f = open_dataset_stream(filename, &is_pipe);
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
    Dataset ds = load_data("input.txt");
    printf("Loaded dataset: %ld bytes\n", ds.length);

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

    printf("Starting EGGROLL Training (Stateful + Optimized)...\n");
    
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
            // Loss is accumulated fixed point. Divide by SEQ_LEN to get average per token, then by 2^FIXED_POINT
            printf("Step %ld/%ld | Loss: %.4f | Tok/s: %.2f\n", step, max_steps, (double)loss_val / (SEQ_LEN * (1 << FIXED_POINT)), tps);
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
