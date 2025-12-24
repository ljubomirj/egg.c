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

// --- RWKV-v7 Configuration ---
#ifndef RWKV_HEAD_SIZE
#define RWKV_HEAD_SIZE 64
#endif
#define RWKV_N_HEAD (HIDDEN_DIM / RWKV_HEAD_SIZE)

#ifndef RWKV_MIX_DIM
#define RWKV_MIX_DIM (HIDDEN_DIM / 4)
#endif

#ifndef COUPLE_RANK
#define COUPLE_RANK (HIDDEN_DIM / 4)
#endif

#define LOSS_X_NUM 1
#define LOSS_X_DEN 5
#define VEC_NOISE_SHIFT 4
#define RWKV_MATMUL_SHIFT 8

#define SEED_OFF_XR 1
#define SEED_OFF_XW 2
#define SEED_OFF_XK 3
#define SEED_OFF_XV 4
#define SEED_OFF_XA 5
#define SEED_OFF_XG 6
#define SEED_OFF_W0 7
#define SEED_OFF_W1 8
#define SEED_OFF_W2 9
#define SEED_OFF_A0 10
#define SEED_OFF_A1 11
#define SEED_OFF_A2 12
#define SEED_OFF_V0 13
#define SEED_OFF_V1 14
#define SEED_OFF_V2 15
#define SEED_OFF_G1 16
#define SEED_OFF_G2 17
#define SEED_OFF_KK 18
#define SEED_OFF_KA 19
#define SEED_OFF_RK 20
#define SEED_OFF_R 21
#define SEED_OFF_K 22
#define SEED_OFF_V 23
#define SEED_OFF_O 24
#define SEED_OFF_FFN_XK 25
#define SEED_OFF_FFN_K 26
#define SEED_OFF_FFN_V 27

#define SEED_OFF_EMB 200
#define SEED_OFF_HEAD_X 900
#define SEED_OFF_HEAD_Y 901
#define SEED_OFF_COUPLE_A 902
#define SEED_OFF_COUPLE_B 903

#if (HIDDEN_DIM % RWKV_HEAD_SIZE) != 0
#error "HIDDEN_DIM must be divisible by RWKV_HEAD_SIZE"
#endif

// --- Lookup Tables [cite: 998-1000] ---
int32_t EXP2_TABLE[256];
static int8_t SIGMOID_TABLE[256];
static int8_t TANH_TABLE[256];
static int8_t DECAY_TABLE[128];

// --- Data Structure ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// --- Recurrent State ---
typedef struct {
    int8_t x_prev[N_LAYERS][HIDDEN_DIM];
    int8_t x_prev_ffn[N_LAYERS][HIDDEN_DIM];
    int32_t wkv_state[N_LAYERS][RWKV_N_HEAD][RWKV_HEAD_SIZE][RWKV_HEAD_SIZE];
} RecurrentState;

// --- Model Parameters Struct ---
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t ln0_weight[HIDDEN_DIM];
    int8_t ln0_bias[HIDDEN_DIM];

    int8_t ln1_weight[N_LAYERS][HIDDEN_DIM];
    int8_t ln1_bias[N_LAYERS][HIDDEN_DIM];
    int8_t ln2_weight[N_LAYERS][HIDDEN_DIM];
    int8_t ln2_bias[N_LAYERS][HIDDEN_DIM];

    int8_t x_r[N_LAYERS][HIDDEN_DIM];
    int8_t x_w[N_LAYERS][HIDDEN_DIM];
    int8_t x_k[N_LAYERS][HIDDEN_DIM];
    int8_t x_v[N_LAYERS][HIDDEN_DIM];
    int8_t x_a[N_LAYERS][HIDDEN_DIM];
    int8_t x_g[N_LAYERS][HIDDEN_DIM];

    int8_t w0[N_LAYERS][HIDDEN_DIM];
    int8_t w1[N_LAYERS][RWKV_MIX_DIM * HIDDEN_DIM];
    int8_t w2[N_LAYERS][HIDDEN_DIM * RWKV_MIX_DIM];

    int8_t a0[N_LAYERS][HIDDEN_DIM];
    int8_t a1[N_LAYERS][RWKV_MIX_DIM * HIDDEN_DIM];
    int8_t a2[N_LAYERS][HIDDEN_DIM * RWKV_MIX_DIM];

    int8_t v0[N_LAYERS][HIDDEN_DIM];
    int8_t v1[N_LAYERS][RWKV_MIX_DIM * HIDDEN_DIM];
    int8_t v2[N_LAYERS][HIDDEN_DIM * RWKV_MIX_DIM];

    int8_t g1[N_LAYERS][RWKV_MIX_DIM * HIDDEN_DIM];
    int8_t g2[N_LAYERS][HIDDEN_DIM * RWKV_MIX_DIM];

    int8_t k_k[N_LAYERS][HIDDEN_DIM];
    int8_t k_a[N_LAYERS][HIDDEN_DIM];
    int8_t r_k[N_LAYERS][HIDDEN_DIM];

    int8_t key[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t value[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t receptance[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t output[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];

    int8_t ln_x_weight[N_LAYERS][HIDDEN_DIM];
    int8_t ln_x_bias[N_LAYERS][HIDDEN_DIM];

    int8_t ffn_xk[N_LAYERS][HIDDEN_DIM];
    int8_t ffn_key[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t ffn_value[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];

    int8_t ln_out_weight[HIDDEN_DIM];
    int8_t ln_out_bias[HIDDEN_DIM];
    int8_t head_x[HIDDEN_DIM * VOCAB_SIZE];
    int8_t head_y[HIDDEN_DIM * VOCAB_SIZE];

    int8_t couple_a[VOCAB_SIZE * COUPLE_RANK];
    int8_t couple_b[COUPLE_RANK * HIDDEN_DIM];
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
    for(int i=0; i<256; i++) {
        double x = (double)(i - 128) / (double)(1 << FIXED_POINT);
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
        double s = 1.0 / (1.0 + exp(-x));
        double t = tanh(x);
        int s_q = (int)lrint(s * 127.0);
        int t_q = (int)lrint(t * 127.0);
        if (s_q < 0) s_q = 0;
        if (s_q > 127) s_q = 127;
        if (t_q < -127) t_q = -127;
        if (t_q > 127) t_q = 127;
        SIGMOID_TABLE[i] = (int8_t)s_q;
        TANH_TABLE[i] = (int8_t)t_q;
    }
    for (int i = 0; i < 128; i++) {
        double s = (double)i / 127.0;
        double w = exp(-0.606531 * s);
        int w_q = (int)lrint(w * 127.0);
        if (w_q < 0) w_q = 0;
        if (w_q > 127) w_q = 127;
        DECAY_TABLE[i] = (int8_t)w_q;
    }
}

int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

static inline int8_t clamp_i8(int32_t v) {
    if (v > MAX_VAL) return MAX_VAL;
    if (v < MIN_VAL) return MIN_VAL;
    return (int8_t)v;
}

static inline int8_t sigmoid_i8(int32_t v) {
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return SIGMOID_TABLE[v + 128];
}

static inline int8_t tanh_i8(int32_t v) {
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return TANH_TABLE[v + 128];
}

static inline int8_t decay_from_sigmoid(int8_t s) {
    if (s < 0) s = 0;
    if (s > 127) s = 127;
    return DECAY_TABLE[s];
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

// --- Layer Norm (L1, int8) ---
static void egg_ln_affine_i8(const int8_t *x, const int8_t *w, const int8_t *b, int8_t *out, int len) {
    int32_t sum = sum_abs_i8(x, len);
    if (sum == 0) sum = 1;
    int32_t mean = sum / len;
    if (mean == 0) mean = 1;

    for (int i = 0; i < len; i++) {
        int32_t val = ((int32_t)x[i] * w[i]) / mean;
        val += b ? b[i] : 0;
        out[i] = clamp_i8(val);
    }
}

// --- Group Norm (per head, int32 input -> int8 output) ---
static void egg_group_norm_i32(const int32_t *x, const int8_t *w, const int8_t *b, int8_t *out) {
    for (int h = 0; h < RWKV_N_HEAD; h++) {
        const int base = h * RWKV_HEAD_SIZE;
        int64_t sum = 0;
        for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
            int32_t v = x[base + i];
            sum += (v < 0) ? -(int64_t)v : (int64_t)v;
        }
        if (sum == 0) sum = 1;
        int32_t mean = (int32_t)(sum / RWKV_HEAD_SIZE);
        if (mean == 0) mean = 1;
        for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
            int32_t val = (int32_t)(x[base + i] / mean);
            val = (val * w[base + i]) + (b ? b[base + i] : 0);
            out[base + i] = clamp_i8(val);
        }
    }
}

static void mix_vectors(const int8_t *x, const int8_t *x_prev, const int8_t *mix, int8_t *out) {
    for (int i = 0; i < HIDDEN_DIM; i++) {
        int32_t diff = (int32_t)x_prev[i] - (int32_t)x[i];
        int32_t v = (int32_t)x[i] + ((diff * (int32_t)mix[i]) >> 7);
        out[i] = clamp_i8(v);
    }
}

static void vector_with_noise(const int8_t *w, int len, uint32_t seed, int noise_sign, int8_t *out) {
    if (noise_sign == 0) {
        memcpy(out, w, (size_t)len);
        return;
    }
    uint32_t rng = seed;
    int8_t a = gen_noise_val(&rng);
    int8_t b[MAX_MATRIX_DIM];
    gen_noise_vector(&rng, b, len);
    for (int i = 0; i < len; i++) {
        int32_t n = (int32_t)a * (int32_t)b[i];
        int32_t v = (int32_t)w[i] + ((noise_sign * n) >> VEC_NOISE_SHIFT);
        out[i] = clamp_i8(v);
    }
}

// --- Forward Pass (RWKV-v7, With Noise Injection) ---
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
    int8_t x[HIDDEN_DIM];
    int8_t x_ln0[HIDDEN_DIM];
    int8_t x_ln1[HIDDEN_DIM];
    int8_t x_ln2[HIDDEN_DIM];

    int8_t xr[HIDDEN_DIM], xw[HIDDEN_DIM], xk[HIDDEN_DIM], xv[HIDDEN_DIM], xa[HIDDEN_DIM], xg[HIDDEN_DIM];
    int8_t mix_r[HIDDEN_DIM], mix_w[HIDDEN_DIM], mix_k[HIDDEN_DIM], mix_v[HIDDEN_DIM], mix_a[HIDDEN_DIM], mix_g[HIDDEN_DIM];
    int8_t w0_vec[HIDDEN_DIM], a0_vec[HIDDEN_DIM], v0_vec[HIDDEN_DIM];
    int8_t k_k_vec[HIDDEN_DIM], k_a_vec[HIDDEN_DIM], r_k_vec[HIDDEN_DIM];
    int8_t w1_out[RWKV_MIX_DIM], w1_tanh[RWKV_MIX_DIM];
    int8_t a1_out[RWKV_MIX_DIM], v1_out[RWKV_MIX_DIM], g1_out[RWKV_MIX_DIM];

    int8_t r[HIDDEN_DIM], k[HIDDEN_DIM], v[HIDDEN_DIM];
    int8_t a_vec[HIDDEN_DIM], g_vec[HIDDEN_DIM];
    int8_t w_decay[HIDDEN_DIM];
    int8_t kk[HIDDEN_DIM], kk_norm[HIDDEN_DIM], vvec[HIDDEN_DIM];
    int8_t v_gate[HIDDEN_DIM];
    int8_t v_first[HIDDEN_DIM];
    int8_t ffn_k[HIDDEN_DIM], ffn_out[HIDDEN_DIM];

    int32_t out_state[HIDDEN_DIM];
    int8_t out_norm[HIDDEN_DIM];
    int8_t time_out[HIDDEN_DIM];

    int8_t logits_x[VOCAB_SIZE];
    int8_t logits_y[VOCAB_SIZE];
    int8_t z[COUPLE_RANK];
    int8_t y_couple[VOCAB_SIZE];

    // If no state provided, use a temporary zeroed one (stateless mode)
    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }

    if (accumulated_loss) *accumulated_loss = 0;

    for (int t = 0; t < seq_len; t++) {
        const int8_t *emb = &model->embedding[inputs[t] * HIDDEN_DIM];
        memcpy(x, emb, HIDDEN_DIM);
        egg_ln_affine_i8(x, model->ln0_weight, model->ln0_bias, x_ln0, HIDDEN_DIM);
        memcpy(x, x_ln0, HIDDEN_DIM);

        bool v_first_valid = false;

        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (uint32_t)(l * 1000);

            egg_ln_affine_i8(x, model->ln1_weight[l], model->ln1_bias[l], x_ln1, HIDDEN_DIM);

            vector_with_noise(model->x_r[l], HIDDEN_DIM, l_seed + SEED_OFF_XR, noise_sign, mix_r);
            vector_with_noise(model->x_w[l], HIDDEN_DIM, l_seed + SEED_OFF_XW, noise_sign, mix_w);
            vector_with_noise(model->x_k[l], HIDDEN_DIM, l_seed + SEED_OFF_XK, noise_sign, mix_k);
            vector_with_noise(model->x_v[l], HIDDEN_DIM, l_seed + SEED_OFF_XV, noise_sign, mix_v);
            vector_with_noise(model->x_a[l], HIDDEN_DIM, l_seed + SEED_OFF_XA, noise_sign, mix_a);
            vector_with_noise(model->x_g[l], HIDDEN_DIM, l_seed + SEED_OFF_XG, noise_sign, mix_g);

            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_r, xr);
            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_w, xw);
            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_k, xk);
            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_v, xv);
            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_a, xa);
            mix_vectors(x_ln1, rnn_state->x_prev[l], mix_g, xg);
            memcpy(rnn_state->x_prev[l], x_ln1, HIDDEN_DIM);

            matmul_perturbed(xr, model->receptance[l], r, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_R, noise_sign, RWKV_MATMUL_SHIFT);
            matmul_perturbed(xk, model->key[l], k, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_K, noise_sign, RWKV_MATMUL_SHIFT);
            matmul_perturbed(xv, model->value[l], v, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_V, noise_sign, RWKV_MATMUL_SHIFT);

            matmul_perturbed(xw, model->w1[l], w1_out, RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_W1, noise_sign, RWKV_MATMUL_SHIFT);
            for (int i = 0; i < RWKV_MIX_DIM; i++) w1_tanh[i] = tanh_i8(w1_out[i]);
            matmul_perturbed(w1_tanh, model->w2[l], w_decay, HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_W2, noise_sign, RWKV_MATMUL_SHIFT);
            vector_with_noise(model->w0[l], HIDDEN_DIM, l_seed + SEED_OFF_W0, noise_sign, w0_vec);
            for (int i = 0; i < HIDDEN_DIM; i++) {
                int8_t s = sigmoid_i8((int32_t)w_decay[i] + (int32_t)w0_vec[i]);
                w_decay[i] = decay_from_sigmoid(s);
            }

            matmul_perturbed(xa, model->a1[l], a1_out, RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_A1, noise_sign, RWKV_MATMUL_SHIFT);
            matmul_perturbed(a1_out, model->a2[l], a_vec, HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_A2, noise_sign, RWKV_MATMUL_SHIFT);
            vector_with_noise(model->a0[l], HIDDEN_DIM, l_seed + SEED_OFF_A0, noise_sign, a0_vec);
            for (int i = 0; i < HIDDEN_DIM; i++) a_vec[i] = sigmoid_i8((int32_t)a_vec[i] + (int32_t)a0_vec[i]);

            matmul_perturbed(xg, model->g1[l], g1_out, RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_G1, noise_sign, RWKV_MATMUL_SHIFT);
            for (int i = 0; i < RWKV_MIX_DIM; i++) g1_out[i] = sigmoid_i8(g1_out[i]);
            matmul_perturbed(g1_out, model->g2[l], g_vec, HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_G2, noise_sign, RWKV_MATMUL_SHIFT);

            if (!v_first_valid) {
                memcpy(v_first, v, HIDDEN_DIM);
                v_first_valid = true;
            } else {
                matmul_perturbed(xv, model->v1[l], v1_out, RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_V1, noise_sign, RWKV_MATMUL_SHIFT);
                matmul_perturbed(v1_out, model->v2[l], v_gate, HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_V2, noise_sign, RWKV_MATMUL_SHIFT);
                vector_with_noise(model->v0[l], HIDDEN_DIM, l_seed + SEED_OFF_V0, noise_sign, v0_vec);
                for (int i = 0; i < HIDDEN_DIM; i++) v_gate[i] = sigmoid_i8((int32_t)v_gate[i] + (int32_t)v0_vec[i]);
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    int32_t dv = ((int32_t)v_first[i] - (int32_t)v[i]) * (int32_t)v_gate[i];
                    v[i] = clamp_i8((int32_t)v[i] + (dv >> 7));
                }
            }

            vector_with_noise(model->k_k[l], HIDDEN_DIM, l_seed + SEED_OFF_KK, noise_sign, k_k_vec);
            vector_with_noise(model->k_a[l], HIDDEN_DIM, l_seed + SEED_OFF_KA, noise_sign, k_a_vec);
            vector_with_noise(model->r_k[l], HIDDEN_DIM, l_seed + SEED_OFF_RK, noise_sign, r_k_vec);

            for (int i = 0; i < HIDDEN_DIM; i++) {
                kk[i] = clamp_i8(((int32_t)k[i] * (int32_t)k_k_vec[i]) >> 7);
            }

            for (int h = 0; h < RWKV_N_HEAD; h++) {
                int base = h * RWKV_HEAD_SIZE;
                int32_t sum = 0;
                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    int8_t vkk = kk[base + i];
                    sum += (vkk < 0) ? -vkk : vkk;
                }
                if (sum == 0) sum = 1;
                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    int idx = base + i;
                    kk_norm[idx] = (int8_t)(((int32_t)kk[idx] * 127) / sum);
                }
            }

            for (int i = 0; i < HIDDEN_DIM; i++) {
                int32_t delta = ((int32_t)a_vec[i] - 127) * (int32_t)k_a_vec[i];
                delta >>= 7;
                k[i] = clamp_i8((int32_t)k[i] + (((int32_t)k[i] * delta) >> 7));
                vvec[i] = clamp_i8(((int32_t)kk_norm[i] * (int32_t)a_vec[i]) >> 7);
            }

            memset(out_state, 0, sizeof(out_state));
            for (int h = 0; h < RWKV_N_HEAD; h++) {
                int base = h * RWKV_HEAD_SIZE;
                int32_t *state = &rnn_state->wkv_state[l][h][0][0];
                int32_t tmp[RWKV_HEAD_SIZE];
                for (int i = 0; i < RWKV_HEAD_SIZE; i++) tmp[i] = 0;

                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        int idx = i * RWKV_HEAD_SIZE + j;
                        int32_t s = state[idx];
                        s = (s * (int32_t)w_decay[base + j]) >> 7;
                        state[idx] = s;
                        tmp[i] += (s * (int32_t)(-kk_norm[base + j])) >> 7;
                    }
                }

                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        int idx = i * RWKV_HEAD_SIZE + j;
                        state[idx] += (tmp[i] * (int32_t)vvec[base + j]) >> 7;
                        state[idx] += ((int32_t)v[base + i] * (int32_t)k[base + j]) >> 7;
                    }
                }

                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    int32_t acc = 0;
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        acc += (state[i * RWKV_HEAD_SIZE + j] * (int32_t)r[base + j]) >> 7;
                    }
                    out_state[base + i] = acc;
                }
            }

            egg_group_norm_i32(out_state, model->ln_x_weight[l], model->ln_x_bias[l], out_norm);

            for (int h = 0; h < RWKV_N_HEAD; h++) {
                int base = h * RWKV_HEAD_SIZE;
                int64_t dot = 0;
                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    int idx = base + i;
                    dot += (int32_t)r[idx] * (int32_t)k[idx] * (int32_t)r_k_vec[idx];
                }
                int32_t dot_scaled = (int32_t)(dot >> 14);
                for (int i = 0; i < RWKV_HEAD_SIZE; i++) {
                    int idx = base + i;
                    out_norm[idx] = clamp_i8((int32_t)out_norm[idx] + ((dot_scaled * (int32_t)v[idx]) >> 7));
                }
            }

            for (int i = 0; i < HIDDEN_DIM; i++) {
                out_norm[i] = clamp_i8(((int32_t)out_norm[i] * (int32_t)g_vec[i]) >> 7);
            }

            matmul_perturbed(out_norm, model->output[l], time_out, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_O, noise_sign, RWKV_MATMUL_SHIFT);
            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], time_out[i]);

            egg_ln_affine_i8(x, model->ln2_weight[l], model->ln2_bias[l], x_ln2, HIDDEN_DIM);
            vector_with_noise(model->ffn_xk[l], HIDDEN_DIM, l_seed + SEED_OFF_FFN_XK, noise_sign, mix_k);
            mix_vectors(x_ln2, rnn_state->x_prev_ffn[l], mix_k, xk);
            memcpy(rnn_state->x_prev_ffn[l], x_ln2, HIDDEN_DIM);

            matmul_perturbed(xk, model->ffn_key[l], ffn_k, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_FFN_K, noise_sign, RWKV_MATMUL_SHIFT);
            for (int i = 0; i < HIDDEN_DIM; i++) {
                int32_t vffn = ffn_k[i];
                if (vffn < 0) vffn = 0;
                vffn = (vffn * vffn) >> 7;
                ffn_k[i] = clamp_i8(vffn);
            }
            matmul_perturbed(ffn_k, model->ffn_value[l], ffn_out, HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_FFN_V, noise_sign, RWKV_MATMUL_SHIFT);
            for (int i = 0; i < HIDDEN_DIM; i++) x[i] = clipped_add(x[i], ffn_out[i]);
        }

        egg_ln_affine_i8(x, model->ln_out_weight, model->ln_out_bias, x_ln0, HIDDEN_DIM);
        matmul_perturbed(x_ln0, model->head_x, logits_x, VOCAB_SIZE, HIDDEN_DIM, step_seed + SEED_OFF_HEAD_X, noise_sign, RWKV_MATMUL_SHIFT);
        matmul_perturbed(x_ln0, model->head_y, logits_y, VOCAB_SIZE, HIDDEN_DIM, step_seed + SEED_OFF_HEAD_Y, noise_sign, RWKV_MATMUL_SHIFT);

        matmul_perturbed(emb, model->couple_b, z, COUPLE_RANK, HIDDEN_DIM, step_seed + SEED_OFF_COUPLE_B, noise_sign, RWKV_MATMUL_SHIFT);
        matmul_perturbed(z, model->couple_a, y_couple, VOCAB_SIZE, COUPLE_RANK, step_seed + SEED_OFF_COUPLE_A, noise_sign, RWKV_MATMUL_SHIFT);
        for (int i = 0; i < VOCAB_SIZE; i++) logits_y[i] = clipped_add(logits_y[i], y_couple[i]);

        if (targets && accumulated_loss) {
            int32_t loss_y = compute_loss(logits_y, targets[t]);
            int32_t loss_x = compute_loss(logits_x, inputs[t]);
            *accumulated_loss += loss_y + (loss_x * LOSS_X_NUM) / LOSS_X_DEN;
        }

        if (logits_out) memcpy(logits_out, logits_y, VOCAB_SIZE);
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

    uint32_t init_rng = 42;
    for (int i = 0; i < VOCAB_SIZE * HIDDEN_DIM; i++) model->embedding[i] = gen_noise_val(&init_rng);

    for (int i = 0; i < HIDDEN_DIM; i++) {
        model->ln0_weight[i] = 16;
        model->ln0_bias[i] = 0;
        model->ln_out_weight[i] = 16;
        model->ln_out_bias[i] = 0;
    }

    for (int i = 0; i < HIDDEN_DIM * VOCAB_SIZE; i++) {
        model->head_x[i] = gen_noise_val(&init_rng);
        model->head_y[i] = gen_noise_val(&init_rng);
    }

    for (int i = 0; i < VOCAB_SIZE * COUPLE_RANK; i++) model->couple_a[i] = gen_noise_val(&init_rng);
    for (int i = 0; i < COUPLE_RANK * HIDDEN_DIM; i++) model->couple_b[i] = gen_noise_val(&init_rng);

    for (int l = 0; l < N_LAYERS; l++) {
        for (int i = 0; i < HIDDEN_DIM; i++) {
            model->ln1_weight[l][i] = 16;
            model->ln1_bias[l][i] = 0;
            model->ln2_weight[l][i] = 16;
            model->ln2_bias[l][i] = 0;
            model->ln_x_weight[l][i] = 16;
            model->ln_x_bias[l][i] = 0;

            model->x_r[l][i] = 0;
            model->x_w[l][i] = 0;
            model->x_k[l][i] = 0;
            model->x_v[l][i] = 0;
            model->x_a[l][i] = 0;
            model->x_g[l][i] = 0;

            model->w0[l][i] = 0;
            model->a0[l][i] = 0;
            model->v0[l][i] = 0;

            model->k_k[l][i] = 127;
            model->k_a[l][i] = 0;
            model->r_k[l][i] = 127;

            model->ffn_xk[l][i] = 0;
        }

        for (int i = 0; i < RWKV_MIX_DIM * HIDDEN_DIM; i++) {
            model->w1[l][i] = gen_noise_val(&init_rng);
            model->a1[l][i] = gen_noise_val(&init_rng);
            model->v1[l][i] = gen_noise_val(&init_rng);
            model->g1[l][i] = gen_noise_val(&init_rng);
        }
        for (int i = 0; i < HIDDEN_DIM * RWKV_MIX_DIM; i++) {
            model->w2[l][i] = gen_noise_val(&init_rng);
            model->a2[l][i] = gen_noise_val(&init_rng);
            model->v2[l][i] = gen_noise_val(&init_rng);
            model->g2[l][i] = gen_noise_val(&init_rng);
        }

        for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) {
            model->key[l][i] = gen_noise_val(&init_rng);
            model->value[l][i] = gen_noise_val(&init_rng);
            model->receptance[l][i] = gen_noise_val(&init_rng);
            model->output[l][i] = gen_noise_val(&init_rng);
            model->ffn_key[l][i] = gen_noise_val(&init_rng);
            model->ffn_value[l][i] = gen_noise_val(&init_rng);
        }
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

        update_matrix(model->embedding, VOCAB_SIZE, HIDDEN_DIM, step_seed + SEED_OFF_EMB, pair_fitnesses, POPULATION_SIZE);

        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (uint32_t)(l * 1000);

            update_matrix(model->x_r[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XR, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->x_w[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XW, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->x_k[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XK, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->x_v[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XV, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->x_a[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XA, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->x_g[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_XG, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->w0[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_W0, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->a0[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_A0, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->v0[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_V0, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->w1[l], RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_W1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->w2[l], HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_W2, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->a1[l], RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_A1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->a2[l], HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_A2, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->v1[l], RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_V1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->v2[l], HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_V2, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->g1[l], RWKV_MIX_DIM, HIDDEN_DIM, l_seed + SEED_OFF_G1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->g2[l], HIDDEN_DIM, RWKV_MIX_DIM, l_seed + SEED_OFF_G2, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->k_k[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_KK, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->k_a[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_KA, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->r_k[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_RK, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->receptance[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_R, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->key[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_K, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->value[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_V, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->output[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_O, pair_fitnesses, POPULATION_SIZE);

            update_matrix(model->ffn_xk[l], 1, HIDDEN_DIM, l_seed + SEED_OFF_FFN_XK, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->ffn_key[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_FFN_K, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->ffn_value[l], HIDDEN_DIM, HIDDEN_DIM, l_seed + SEED_OFF_FFN_V, pair_fitnesses, POPULATION_SIZE);
        }

        update_matrix(model->head_x, VOCAB_SIZE, HIDDEN_DIM, step_seed + SEED_OFF_HEAD_X, pair_fitnesses, POPULATION_SIZE);
        update_matrix(model->head_y, VOCAB_SIZE, HIDDEN_DIM, step_seed + SEED_OFF_HEAD_Y, pair_fitnesses, POPULATION_SIZE);
        update_matrix(model->couple_a, VOCAB_SIZE, COUPLE_RANK, step_seed + SEED_OFF_COUPLE_A, pair_fitnesses, POPULATION_SIZE);
        update_matrix(model->couple_b, COUPLE_RANK, HIDDEN_DIM, step_seed + SEED_OFF_COUPLE_B, pair_fitnesses, POPULATION_SIZE);

        total_tokens += SEQ_LEN;
    }

    printf("Training Done.\n");
    free(ds.data); free(model); free(logits); free(pair_fitnesses);
    free(pop_states);
    return 0;
}
