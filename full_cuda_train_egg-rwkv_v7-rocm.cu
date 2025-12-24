#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#endif
#include <signal.h>
#include <unistd.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

static void format_local_datetime(char *out, size_t out_size) {
    if (!out || out_size == 0) return;
    time_t now = time(NULL);
    struct tm tm_now;
    if (!localtime_r(&now, &tm_now)) {
        out[0] = '\0';
        return;
    }
    strftime(out, out_size, "%Y-%m-%d %H:%M:%S", &tm_now);
}

static const char *build_name(void) {
#if defined(NDEBUG)
    return "release";
#elif defined(DEBUG)
    return "debug";
#else
    return "unknown";
#endif
}

// --- Configuration ---
#define SM_CORES 128
#define WARP_SIZE 32
#define BLOCK_THREADS 256
#define BATCH WARP_SIZE
#define VOCAB_SIZE 256
#define HIDDEN_DIM (SM_CORES * 1)
#define N_LAYERS 4
#define SEQ_LEN 256
#define POPULATION_SIZE (SM_CORES * 512)
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define SIGMA_SHIFT_VECTOR (SIGMA_SHIFT - 2)
#define MAX_VAL 127
#define MIN_VAL -127

#define RWKV_HEAD_SIZE 64
#define RWKV_N_HEAD (HIDDEN_DIM / RWKV_HEAD_SIZE)
#define RWKV_MIX_DIM (HIDDEN_DIM / 4)
#define COUPLE_RANK (HIDDEN_DIM / 4)
#define RWKV_Z_DIM ((RWKV_MIX_DIM > COUPLE_RANK) ? RWKV_MIX_DIM : COUPLE_RANK)

#define LOSS_X_NUM 1
#define LOSS_X_DEN 5
#define VEC_NOISE_SHIFT 4
#define RWKV_MATMUL_SHIFT 8

#define SHARED_STRIDE (HIDDEN_DIM * 17 + RWKV_Z_DIM)
#define MAX_STRIDE 8

#if (HIDDEN_DIM % RWKV_HEAD_SIZE) != 0
#error "HIDDEN_DIM must be divisible by RWKV_HEAD_SIZE"
#endif

// --- Seed Offsets ---
#define SEED_OFF_EMB 200
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
#define SEED_OFF_HEAD_X 900
#define SEED_OFF_HEAD_Y 901
#define SEED_OFF_COUPLE_A 902
#define SEED_OFF_COUPLE_B 903

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#if defined(__HIPCC__)
// Minimal CUDA runtime API shims for building this .cu file with hipcc on AMD.
// Keep the rest of the code unchanged for easier upstream rebases.
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyToSymbol(symbol, src, count) hipMemcpyToSymbol(HIP_SYMBOL(symbol), (src), (count), 0, hipMemcpyHostToDevice)
#define cudaMemcpyFromSymbol(dst, symbol, count, offset, kind) hipMemcpyFromSymbol((dst), HIP_SYMBOL(symbol), (count), (offset), (kind))

// CUDA uses __shfl_sync; HIP on AMD uses __shfl with an explicit width.
#ifndef __shfl_sync
#define __shfl_sync(mask, var, src_lane) __shfl((var), (src_lane), WARP_SIZE)
#endif
#endif

// --- Data Structures ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

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

// Global Tables
__constant__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];
__constant__ int8_t d_SIGMOID_TABLE[256];
__constant__ int8_t d_TANH_TABLE[256];
__constant__ int8_t d_DECAY_TABLE[128];
int8_t h_SIGMOID_TABLE[256];
int8_t h_TANH_TABLE[256];
int8_t h_DECAY_TABLE[128];

__device__ int32_t d_debug_updates[2]; // 0: Inc, 1: Dec

// --- Helpers (Host) ---

int get_update_threshold(double loss) {
    if (loss > 5.0) return 1000;
    if (loss > 4.0) return 5000;
    if (loss > 3.8) return 30000;
    if (loss > 3.6) return 60000;
    if (loss > 3.4) return 90000;
    if (loss > 3.2) return 120000;
    if (loss > 3.0) return 150000;
    if (loss > 1.0) return 270000;
    return 400000; 
}

double get_time_diff_ms(struct timespec start, struct timespec end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + 
           ((end.tv_nsec - start.tv_nsec) / 1e6);
}

void init_tables() {
    for (int i = 0; i < 256; i++) {
        double x = (double)(i - 128) / (double)(1 << FIXED_POINT);
        h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
        double s = 1.0 / (1.0 + exp(-x));
        double t = tanh(x);
        int s_q = (int)lrint(s * 127.0);
        int t_q = (int)lrint(t * 127.0);
        if (s_q < 0) s_q = 0;
        if (s_q > 127) s_q = 127;
        if (t_q < -127) t_q = -127;
        if (t_q > 127) t_q = 127;
        h_SIGMOID_TABLE[i] = (int8_t)s_q;
        h_TANH_TABLE[i] = (int8_t)t_q;
    }
    for (int i = 0; i < 128; i++) {
        double s = (double)i / 127.0;
        double w = exp(-0.606531 * s);
        int w_q = (int)lrint(w * 127.0);
        if (w_q < 0) w_q = 0;
        if (w_q > 127) w_q = 127;
        h_DECAY_TABLE[i] = (int8_t)w_q;
    }
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_host(uint32_t *rng) {
    uint32_t r = xorshift32_host(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) {
        for(int c=0; c<cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void init_model(EggModel *model) {
    uint32_t rng = 42;
    EggModel *temp = (EggModel*)calloc(1, sizeof(EggModel));
    if (!temp) { printf("Failed to allocate temp model\n"); exit(1); }

    for (int i = 0; i < VOCAB_SIZE * HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    transpose_matrix(model->embedding, temp->embedding, VOCAB_SIZE, HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM * VOCAB_SIZE; i++) {
        temp->head_x[i] = gen_noise_host(&rng);
        temp->head_y[i] = gen_noise_host(&rng);
    }
    transpose_matrix(model->head_x, temp->head_x, VOCAB_SIZE, HIDDEN_DIM);
    transpose_matrix(model->head_y, temp->head_y, VOCAB_SIZE, HIDDEN_DIM);

    for (int i = 0; i < VOCAB_SIZE * COUPLE_RANK; i++) temp->couple_a[i] = gen_noise_host(&rng);
    for (int i = 0; i < COUPLE_RANK * HIDDEN_DIM; i++) temp->couple_b[i] = gen_noise_host(&rng);
    transpose_matrix(model->couple_a, temp->couple_a, VOCAB_SIZE, COUPLE_RANK);
    transpose_matrix(model->couple_b, temp->couple_b, COUPLE_RANK, HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM; i++) {
        model->ln0_weight[i] = 16;
        model->ln0_bias[i] = 0;
        model->ln_out_weight[i] = 16;
        model->ln_out_bias[i] = 0;
    }

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
            temp->w1[l][i] = gen_noise_host(&rng);
            temp->a1[l][i] = gen_noise_host(&rng);
            temp->v1[l][i] = gen_noise_host(&rng);
            temp->g1[l][i] = gen_noise_host(&rng);
        }
        transpose_matrix(model->w1[l], temp->w1[l], RWKV_MIX_DIM, HIDDEN_DIM);
        transpose_matrix(model->a1[l], temp->a1[l], RWKV_MIX_DIM, HIDDEN_DIM);
        transpose_matrix(model->v1[l], temp->v1[l], RWKV_MIX_DIM, HIDDEN_DIM);
        transpose_matrix(model->g1[l], temp->g1[l], RWKV_MIX_DIM, HIDDEN_DIM);

        for (int i = 0; i < HIDDEN_DIM * RWKV_MIX_DIM; i++) {
            temp->w2[l][i] = gen_noise_host(&rng);
            temp->a2[l][i] = gen_noise_host(&rng);
            temp->v2[l][i] = gen_noise_host(&rng);
            temp->g2[l][i] = gen_noise_host(&rng);
        }
        transpose_matrix(model->w2[l], temp->w2[l], HIDDEN_DIM, RWKV_MIX_DIM);
        transpose_matrix(model->a2[l], temp->a2[l], HIDDEN_DIM, RWKV_MIX_DIM);
        transpose_matrix(model->v2[l], temp->v2[l], HIDDEN_DIM, RWKV_MIX_DIM);
        transpose_matrix(model->g2[l], temp->g2[l], HIDDEN_DIM, RWKV_MIX_DIM);

        for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) {
            temp->key[l][i] = gen_noise_host(&rng);
            temp->value[l][i] = gen_noise_host(&rng);
            temp->receptance[l][i] = gen_noise_host(&rng);
            temp->output[l][i] = gen_noise_host(&rng);
            temp->ffn_key[l][i] = gen_noise_host(&rng);
            temp->ffn_value[l][i] = gen_noise_host(&rng);
        }
        transpose_matrix(model->key[l], temp->key[l], HIDDEN_DIM, HIDDEN_DIM);
        transpose_matrix(model->value[l], temp->value[l], HIDDEN_DIM, HIDDEN_DIM);
        transpose_matrix(model->receptance[l], temp->receptance[l], HIDDEN_DIM, HIDDEN_DIM);
        transpose_matrix(model->output[l], temp->output[l], HIDDEN_DIM, HIDDEN_DIM);
        transpose_matrix(model->ffn_key[l], temp->ffn_key[l], HIDDEN_DIM, HIDDEN_DIM);
        transpose_matrix(model->ffn_value[l], temp->ffn_value[l], HIDDEN_DIM, HIDDEN_DIM);
    }

    free(temp);
}


// --- DEVICE ---
#define KERNEL_LOOP(idx, limit) for(int idx = threadIdx.x; idx < (limit); idx += blockDim.x)

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; 
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__device__ __forceinline__ int8_t clip(long long a) {
    return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a);
}

__device__ __forceinline__ int8_t sigmoid_i8(int32_t v) {
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return d_SIGMOID_TABLE[v + 128];
}

__device__ __forceinline__ int8_t tanh_i8(int32_t v) {
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return d_TANH_TABLE[v + 128];
}

__device__ __forceinline__ int8_t decay_from_sigmoid(int8_t s) {
    if (s < 0) s = 0;
    if (s > 127) s = 127;
    return d_DECAY_TABLE[s];
}

__device__ __forceinline__ long long warpBroadcast(long long val, int src_lane);

template <typename WarpReduceT>
__device__ void matmul_perturbed_warp(
    const int8_t *x,
    const int8_t *W,
    int rows,
    int cols,
    uint32_t seed_a,
    uint32_t seed_b,
    int ns,
    int8_t *out,
    typename WarpReduceT::TempStorage *temp_storage,
    int lane_id
) {
    long long partial = 0;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        int8_t b = noise_from_hash(seed_b, c);
        partial += (long long)x[c] * b;
    }
    long long xB = WarpReduceT(*temp_storage).Sum(partial);
    xB = warpBroadcast(xB, 0);

    for (int r = lane_id; r < rows; r += WARP_SIZE) {
        long long acc = 0;
        for (int c = 0; c < cols; c++) {
            acc += (long long)x[c] * W[c * rows + r];
        }
        if (ns != 0) {
            int8_t a = noise_from_hash(seed_a, r);
            acc += ((xB * (long long)a) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        }
        out[r] = clip(acc >> RWKV_MATMUL_SHIFT);
    }
}

__device__ void vector_with_noise_warp(
    const int8_t *base,
    int len,
    uint32_t seed,
    int ns,
    int8_t *out,
    int lane_id
) {
    for (int i = lane_id; i < len; i += WARP_SIZE) {
        int8_t a = noise_from_hash(seed, i);
        long long perturb = ((long long)a * ns) >> VEC_NOISE_SHIFT;
        out[i] = clip((long long)base[i] + perturb);
    }
}

// Helper to broadcast 64-bit value from a lane to all threads in warp
__device__ __forceinline__ long long warpBroadcast(long long val, int src_lane) {
    int lo = __shfl_sync(0xFFFFFFFF, (int)val, src_lane);
    int hi = __shfl_sync(0xFFFFFFFF, (int)(val >> 32), src_lane);
    return ((long long)hi << 32) | (unsigned int)lo;
}

extern __shared__ int8_t s_mem[]; 

#if 0
__global__ void generate_sequence_kernel(
    const EggModel * __restrict__ model,
    uint32_t seed,
    const uint8_t *seed_text,
    int seed_len,
    int gen_len,
    uint8_t *output
) {
    int8_t *s_ptr = s_mem;
    
    // CUB definitions
    typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce;
    typedef cub::BlockScan<long long, BLOCK_THREADS> BlockScan;
    
    // Shared storage for CUB. Using a union to save memory.
    __shared__ union {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    __shared__ long long shared_sum; // For broadcasting reduction result

    // Local hidden state
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    KERNEL_LOOP(i, HIDDEN_DIM) {
        for(int l=0; l<N_LAYERS; l++) h_local[l][(i - threadIdx.x)/blockDim.x] = 0;
    }

    int total_len = seed_len + gen_len;
    uint8_t current_token = 0; 
    
    for(int t=0; t < total_len; t++) {
        if (t < seed_len) current_token = seed_text[t];
        
        // 1. Embed
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = model->embedding[i * VOCAB_SIZE + current_token];
        __syncthreads();
        
        // 2. Layers
        for(int l=0; l<N_LAYERS; l++) {
             // LN 1
             long long local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             
             long long sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
             // Broadcast result to all threads
             if (threadIdx.x == 0) shared_sum = sum;
             __syncthreads();
             sum = shared_sum;

             if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][0][i]) / mean);
             __syncthreads();
             
             // Copy H to Shared (Buf 1)
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[HIDDEN_DIM + i] = h_local[l][(i - threadIdx.x)/blockDim.x];
             __syncthreads();
             
             // GRU Phase 1
             const int8_t *w0 = &model->gru_weights[l][0][0];
             const int8_t *w1 = &model->gru_weights[l][1][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w0[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 s_ptr[3*HIDDEN_DIM + i] = acc0 >> 8;
                 s_ptr[2*HIDDEN_DIM + i] = acc1 >> 8;
             }
             __syncthreads();
             
             // Gates
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 int8_t b1 = s_ptr[3*HIDDEN_DIM + i];
                 int8_t b2 = s_ptr[2*HIDDEN_DIM + i];
                 int8_t ft = clip((long long)b1 + b2 + model->gru_biases[l][0][i]);
                 
                 int8_t h_val = s_ptr[HIDDEN_DIM + i];
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 s_ptr[3*HIDDEN_DIM + i] = ft;
                 s_ptr[2*HIDDEN_DIM + i] = gated;
             }
             __syncthreads();
             
             // GRU Phase 2
             const int8_t *w2 = &model->gru_weights[l][2][0];
             const int8_t *w3 = &model->gru_weights[l][3][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w2[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                 }
                 int8_t ft = s_ptr[3*HIDDEN_DIM + i];
                 int8_t ht = clip((acc0 >> 8) + (acc1 >> 8) + model->gru_biases[l][1][i]);
                 
                 int8_t h_curr = h_local[l][(i - threadIdx.x)/blockDim.x];
                 int32_t diff = ht - h_curr;
                 int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                 h_curr = clip(h_curr + update);
                 h_local[l][(i - threadIdx.x)/blockDim.x] = h_curr;
                 
                 s_ptr[i] = clip((long long)h_curr + s_ptr[i]);
             }
             __syncthreads();
             
             // Save MLP Residual
             int8_t mlp_resid[MAX_STRIDE];
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 mlp_resid[(i - threadIdx.x)/blockDim.x] = s_ptr[i];
             }
             
             // MLP LN
             local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
             if (threadIdx.x == 0) shared_sum = sum;
             __syncthreads();
             sum = shared_sum;

             if(!sum) sum=1; mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][1][i]) / mean);
             __syncthreads();
             
             // MLP Expand
             int8_t exp_res[MAX_STRIDE][4];
             const int8_t *w_exp = &model->mlp_weights[l][0][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     long long acc = 0;
                     for(int k=0; k<HIDDEN_DIM; k++) {
                         acc += (long long)s_ptr[k] * w_exp[k*(4*HIDDEN_DIM) + (i + sub*HIDDEN_DIM)];
                     }
                     exp_res[(i - threadIdx.x)/blockDim.x][sub] = clip(acc >> 8);
                 }
             }
             __syncthreads();
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     s_ptr[i + sub*HIDDEN_DIM] = exp_res[(i - threadIdx.x)/blockDim.x][sub];
                 }
             }
             __syncthreads();
             
             // MLP Project
             const int8_t *w_proj = &model->mlp_weights[l][1][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc = 0;
                 for(int k=0; k<(4*HIDDEN_DIM); k++) {
                     acc += (long long)s_ptr[k] * w_proj[k*HIDDEN_DIM + i];
                 }
                 s_ptr[i] = clip((acc >> 9) + mlp_resid[(i - threadIdx.x)/blockDim.x]);
             }
             __syncthreads();
        }
        
        // 3. Head
        long long local_sum = 0;
        KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
        long long sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
        if (threadIdx.x == 0) shared_sum = sum;
        __syncthreads();
        sum = shared_sum;        

        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_out[i]) / mean);
        __syncthreads();
        
        // Logits
        const int8_t *w_head = &model->head[0];
        KERNEL_LOOP(i, VOCAB_SIZE) {
            long long acc = 0;
            for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)s_ptr[k] * w_head[k*VOCAB_SIZE + i]; 
            }
            s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
        }
        __syncthreads();
        
        // 4. Sample
        if (t >= seed_len) {
            long long val = 0;
            KERNEL_LOOP(i, VOCAB_SIZE) {
                int idx = (int32_t)s_ptr[HIDDEN_DIM + i] + 128;
                idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                val = d_EXP2_TABLE[idx];
            }
            // Block Reduce to get Sum
            long long sum_exp = BlockReduce(temp_storage.reduce).Sum(val);
            if (threadIdx.x == 0) shared_sum = sum_exp;
            __syncthreads();
            sum_exp = shared_sum; 
            
            uint32_t s = seed + t * 222;
            uint32_t r_val = hash_rng(s, 0);
            long long r_threshold = (sum_exp > 0) ? (r_val % sum_exp) : 0;
            
            // Parallel Selection using BlockScan
            long long prefix_sum;
            long long block_aggregate; 
            BlockScan(temp_storage.scan).InclusiveSum(val, prefix_sum, block_aggregate);
            
            __shared__ int selected_token;
            if (threadIdx.x == 0) selected_token = VOCAB_SIZE - 1; // Default
            __syncthreads();
            
            if (prefix_sum > r_threshold) {
                long long prev_sum = prefix_sum - val;
                if (prev_sum <= r_threshold) {
                    // Logic implies this thread holds the target range
                    // Note: If VOCAB_SIZE > BLOCK_THREADS, we need mapping.
                    // Assuming VOCAB_SIZE (256) <= BLOCK_THREADS (256) as per config
                    atomicMin(&selected_token, threadIdx.x);
                }
            }
            __syncthreads();
            
            current_token = (uint8_t)selected_token;
            output[t - seed_len] = current_token;
        }
    }
}
#endif

__global__ void generate_sequence_kernel(
    const EggModel * __restrict__ model,
    uint32_t seed,
    const uint8_t *seed_text,
    int seed_len,
    int gen_len,
    uint8_t *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gen_len) output[idx] = 0;
}

#if 0
__global__ void __launch_bounds__(BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ pop_states, 
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int p_idx = blockIdx.x * (BLOCK_THREADS / WARP_SIZE) + warp_id;
    
    if (p_idx >= POPULATION_SIZE) return;

    int8_t *my_s_ptr = &s_mem[warp_id * SHARED_STRIDE];

    // CUB WarpReduce
    typedef cub::WarpReduce<long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_THREADS / WARP_SIZE];
    
    // Load State
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE;
        for(int l=0; l<N_LAYERS; l++) {
             h_local[l][sub] = pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
        }
    }
    
    long long my_loss = 0;
    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;

    for (int t = 0; t < SEQ_LEN; t++) {
        __syncwarp();
        
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        int8_t a_token = noise_from_hash(seed_emb, input_token);
        
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int8_t base = model->embedding[i * VOCAB_SIZE + input_token];
            int8_t b_val = noise_from_hash(seed_emb + HIDDEN_DIM, i);
            long long perturb = ((long long)a_token * b_val * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            my_s_ptr[i] = clip((long long)base + perturb);
        }
        __syncwarp();
        
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed = (step_seed + pair_idx) + (l * 100);

            // LN 1
            long long local_sum = 0;
            int8_t gru_resid[MAX_STRIDE];

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t val = my_s_ptr[i];
                gru_resid[i / WARP_SIZE] = val;
                local_sum += abs((long long)val);
            }
            long long sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
            sum = warpBroadcast(sum, 0); // Broadcast to all lanes

            if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t w = model->ln_weights[l][0][i];
                int8_t a = noise_from_hash(seed + SEED_OFF_LN_W1, i);
                long long perturb = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + perturb)) / mean);
            }

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 my_s_ptr[HIDDEN_DIM + i] = h_local[l][i / WARP_SIZE];
            }
            __syncwarp();

            // Rank-1 Precalc
            long long ls_1 = 0, ls_2 = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b1 = noise_from_hash(seed + SEED_OFF_GRU_M1 + HIDDEN_DIM, i);
                ls_1 += (long long)my_s_ptr[i] * b1;
                
                int8_t b2 = noise_from_hash(seed + SEED_OFF_GRU_M2 + HIDDEN_DIM, i);
                ls_2 += (long long)my_s_ptr[HIDDEN_DIM + i] * b2;
            }
            long long xB_m1 = WarpReduce(temp_storage[warp_id]).Sum(ls_1);
            xB_m1 = warpBroadcast(xB_m1, 0);
            
            long long xB_m2 = WarpReduce(temp_storage[warp_id]).Sum(ls_2);
            xB_m2 = warpBroadcast(xB_m2, 0);

            // MatMul 1 & 2 + Gates
            const int8_t *w0 = &model->gru_weights[l][0][0];
            const int8_t *w1 = &model->gru_weights[l][1][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long dot1 = 0, dot2 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w0[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 
                 int8_t a1 = noise_from_hash(seed + SEED_OFF_GRU_M1, i);
                 if(ns!=0) dot1 += ((xB_m1 * (long long)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a2 = noise_from_hash(seed + SEED_OFF_GRU_M2, i);
                 if(ns!=0) dot2 += ((xB_m2 * (long long)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t b_gate = model->gru_biases[l][0][i];
                 int8_t a_gate = noise_from_hash(seed + SEED_OFF_GRU_B1, i);
                 long long p_gate = ((long long)a_gate * ns) >> SIGMA_SHIFT_VECTOR;

                 int8_t ft = clip((dot1 >> 8) + (dot2 >> 8) + (b_gate + p_gate));
                 int8_t h_val = my_s_ptr[HIDDEN_DIM + i];
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 my_s_ptr[2*HIDDEN_DIM + i] = gated; 
                 my_s_ptr[3*HIDDEN_DIM + i] = ft;    
            }
            __syncwarp();
            
            // Rank-1 M3 & M4
            long long ls_3=0, ls_4=0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b3 = noise_from_hash(seed + SEED_OFF_GRU_M3 + HIDDEN_DIM, i);
                ls_3 += (long long)my_s_ptr[i] * b3;
                
                int8_t b4 = noise_from_hash(seed + SEED_OFF_GRU_M4 + HIDDEN_DIM, i);
                ls_4 += (long long)my_s_ptr[2*HIDDEN_DIM + i] * b4;
            }
            long long xB_m3 = WarpReduce(temp_storage[warp_id]).Sum(ls_3);
            xB_m3 = warpBroadcast(xB_m3, 0);

            long long xB_m4 = WarpReduce(temp_storage[warp_id]).Sum(ls_4);
            xB_m4 = warpBroadcast(xB_m4, 0);
            
            // Phase 2
            const int8_t *w2 = &model->gru_weights[l][2][0];
            const int8_t *w3 = &model->gru_weights[l][3][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long dot1 = 0, dot2 = 0;
                for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w2[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                }
                
                int8_t a3 = noise_from_hash(seed + SEED_OFF_GRU_M3, i);
                if(ns!=0) dot1 += ((xB_m3 * (long long)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t a4 = noise_from_hash(seed + SEED_OFF_GRU_M4, i);
                if(ns!=0) dot2 += ((xB_m4 * (long long)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t b_ht = model->gru_biases[l][1][i];
                int8_t a_ht = noise_from_hash(seed + SEED_OFF_GRU_B2, i);
                long long p_ht = ((long long)a_ht * ns) >> SIGMA_SHIFT_VECTOR;

                int8_t ht = clip((dot1 >> 8) + (dot2 >> 8) + (b_ht + p_ht));
                
                int8_t ft = my_s_ptr[3*HIDDEN_DIM + i];
                int8_t h_curr = h_local[l][i / WARP_SIZE];
                int32_t diff = ht - h_curr;
                int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                h_curr = clip(h_curr + update);
                h_local[l][i / WARP_SIZE] = h_curr;
                
                my_s_ptr[i] = clip((long long)h_curr + gru_resid[i / WARP_SIZE]); 
            }
            __syncwarp(); 
            
            // Pre-LN Resid
            int8_t mlp_resid[MAX_STRIDE];
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                mlp_resid[i / WARP_SIZE] = my_s_ptr[i];
            }

            // MLP LN 2
            long long local_mlp_sum = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                local_mlp_sum += abs((long long)my_s_ptr[i]);
            }
            long long sum_mlp = WarpReduce(temp_storage[warp_id]).Sum(local_mlp_sum);
            sum_mlp = warpBroadcast(sum_mlp, 0);

            if(!sum_mlp) sum_mlp=1; long long mean_mlp = sum_mlp/HIDDEN_DIM; if(!mean_mlp) mean_mlp=1;

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t w = model->ln_weights[l][1][i];
                int8_t a = noise_from_hash(seed + SEED_OFF_LN_W2, i);
                long long p = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + p)) / mean_mlp);
            }
            __syncwarp(); 
            
            // Expand
            int8_t mlp_res[MAX_STRIDE][4];
            
            uint32_t seed_exp = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
            long long ls_exp = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b_val = noise_from_hash(seed_exp + SHARED_STRIDE, i); 
                ls_exp += (long long)my_s_ptr[i] * b_val;
            }
            long long xB_mlp1 = WarpReduce(temp_storage[warp_id]).Sum(ls_exp);
            xB_mlp1 = warpBroadcast(xB_mlp1, 0);

            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                      int out_idx = i + sub * HIDDEN_DIM;
                      long long acc_mlp = 0;
                      
                      for(int k=0; k<HIDDEN_DIM; k++) {
                          acc_mlp += (long long)my_s_ptr[k] * w_mlp1[k*(4*HIDDEN_DIM) + out_idx];
                      }
                      
                      int8_t a_val = noise_from_hash(seed_exp, out_idx);
                      if(ns!=0) acc_mlp += ((xB_mlp1 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                      mlp_res[i / WARP_SIZE][sub] = clip(acc_mlp >> 8);
                 }
             }
             
             // Write Expand to Shared
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     my_s_ptr[i + sub*HIDDEN_DIM] = mlp_res[i / WARP_SIZE][sub];
                 }
             }
             __syncwarp();
             
             // xB for M2
             uint32_t seed_proj = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
             long long ls_proj = 0;
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     int in_idx = i + sub*HIDDEN_DIM;
                     int8_t b_val = noise_from_hash(seed_proj + HIDDEN_DIM, in_idx);
                     ls_proj += (long long)my_s_ptr[in_idx] * b_val;
                 }
             }
             long long xB_mlp2 = WarpReduce(temp_storage[warp_id]).Sum(ls_proj);
             xB_mlp2 = warpBroadcast(xB_mlp2, 0);
             
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long acc_proj = 0;
                 for(int k=0; k<HIDDEN_DIM*4; k++) {
                     acc_proj += (long long)my_s_ptr[k] * w_mlp2[k*HIDDEN_DIM + i];
                 }
                 int8_t a_val = noise_from_hash(seed_proj, i);
                 if(ns!=0) acc_proj += ((xB_mlp2 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int32_t res = acc_proj >> 8; 
                 my_s_ptr[i] = clip(res + mlp_resid[i / WARP_SIZE]);
             }
             __syncwarp();
        } 
        
        // 3. Head LN
        long long local_sum = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            local_sum += abs((long long)my_s_ptr[i]);
        }
        long long sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
        sum = warpBroadcast(sum, 0);
        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        uint32_t seed_ln_out = (step_seed+pair_idx);
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             int8_t w = model->ln_out[i];
             int8_t a = noise_from_hash(seed_ln_out + SEED_OFF_LN_OUT, i);
             long long p = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
             my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + p)) / mean);
        }
        __syncwarp();
        
        // 4. Head Dense & Loss
         uint32_t seed_head = (step_seed+pair_idx) + SEED_OFF_HEAD;
         long long ls_head = 0;
         for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             int8_t b_val = noise_from_hash(seed_head + VOCAB_SIZE, i); 
             ls_head += (long long)my_s_ptr[i] * b_val;
         }
         long long xB_head = WarpReduce(temp_storage[warp_id]).Sum(ls_head);
         xB_head = warpBroadcast(xB_head, 0);
         
         const int8_t *w_head = &model->head[0];
         
         for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
             long long acc = 0;
             for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)my_s_ptr[k] * w_head[k*VOCAB_SIZE + i];
             }
            
             int8_t a_val = noise_from_hash(seed_head, i);
             if(ns!=0) acc += ((xB_head * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
             
             my_s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
         }
         __syncwarp();
         
          // 5. Softmax Loss
          uint8_t target_token = dataset[stream_pos + t + 1];
          
          long long local_exp = 0;
          for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
              int idx = (int32_t)my_s_ptr[HIDDEN_DIM + i] + 128;
              idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
              local_exp += d_EXP2_TABLE[idx];
          }
          long long sum_exp = WarpReduce(temp_storage[warp_id]).Sum(local_exp);
          sum_exp = warpBroadcast(sum_exp, 0);
          
          if(lane_id == 0) {
             long long log_sum = 0;
             long long x = sum_exp;
             if (x > 0) {
                 int pos = 0;
                 while(x >= 65536) { x >>= 16; pos += 16; }
                 if (x >= 256)  { x >>= 8;  pos += 8; }
                 if (x >= 16)   { x >>= 4;  pos += 4; }
                 if (x >= 4)    { x >>= 2;  pos += 2; }
                 if (x >= 2)    {           pos += 1; }
                 
                 long long fraction = (pos>=4) ? (sum_exp-(1LL<<pos))>>(pos-4) : (sum_exp-(1LL<<pos))<<(4-pos);
                 log_sum = (pos<<4) + fraction - 64;
             }
             
             int32_t target_l = (int32_t)my_s_ptr[HIDDEN_DIM + target_token] + 128;
             my_loss += (log_sum - target_l);
          }
    } 
    
    if (lane_id == 0) {
        accum_loss[p_idx] = (int32_t)my_loss;
    }
    
    // Store State
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE; 
        for(int l=0; l<N_LAYERS; l++) {
             pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = h_local[l][sub];
        }
    }
}
#endif

__global__ void __launch_bounds__(BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ state_x,
    int8_t * __restrict__ state_ffn,
    int32_t * __restrict__ state_wkv,
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int p_idx = blockIdx.x * (BLOCK_THREADS / WARP_SIZE) + warp_id;
    if (p_idx >= POPULATION_SIZE) return;

    int8_t *base = &s_mem[warp_id * SHARED_STRIDE];
    int8_t *x = base;
    int8_t *xr = x + HIDDEN_DIM;
    int8_t *xk = xr + HIDDEN_DIM;
    int8_t *xv = xk + HIDDEN_DIM;
    int8_t *xw = xv + HIDDEN_DIM;
    int8_t *xa = xw + HIDDEN_DIM;
    int8_t *xg = xa + HIDDEN_DIM;
    int8_t *r = xg + HIDDEN_DIM;
    int8_t *k = r + HIDDEN_DIM;
    int8_t *v = k + HIDDEN_DIM;
    int8_t *w_decay = v + HIDDEN_DIM;
    int8_t *a_vec = w_decay + HIDDEN_DIM;
    int8_t *kk_norm = a_vec + HIDDEN_DIM;
    int8_t *vvec = kk_norm + HIDDEN_DIM;
    int8_t *g_vec = vvec + HIDDEN_DIM;
    int8_t *v_first = g_vec + HIDDEN_DIM;
    int8_t *emb_cache = v_first + HIDDEN_DIM;
    int8_t *z = emb_cache + HIDDEN_DIM;

    typedef cub::WarpReduce<long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_THREADS / WARP_SIZE];
    __shared__ int32_t head_sum[BLOCK_THREADS / WARP_SIZE][RWKV_N_HEAD];
    __shared__ int32_t target_logit_shared[BLOCK_THREADS / WARP_SIZE];
    __shared__ int32_t out_state_shared[BLOCK_THREADS / WARP_SIZE][HIDDEN_DIM];

    int8_t x_prev_local[N_LAYERS][MAX_STRIDE];
    int8_t x_prev_ffn_local[N_LAYERS][MAX_STRIDE];
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE;
        for (int l = 0; l < N_LAYERS; l++) {
            x_prev_local[l][sub] = state_x[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
            x_prev_ffn_local[l][sub] = state_ffn[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
        }
    }

    int32_t *my_wkv = state_wkv + p_idx * (N_LAYERS * RWKV_N_HEAD * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE);

    long long my_loss = 0;
    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;

    for (int t = 0; t < SEQ_LEN; t++) {
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        int8_t a_token = noise_from_hash(seed_emb, input_token);

        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int8_t base_val = model->embedding[i * VOCAB_SIZE + input_token];
            int8_t b_val = noise_from_hash(seed_emb + HIDDEN_DIM, i);
            long long perturb = ((long long)a_token * b_val * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            x[i] = clip((long long)base_val + perturb);
            emb_cache[i] = x[i];
        }
        __syncwarp();

        long long local_sum = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            long long vabs = (long long)x[i];
            local_sum += (vabs < 0) ? -vabs : vabs;
        }
        long long sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
        sum = warpBroadcast(sum, 0);
        if (!sum) sum = 1;
        long long mean = sum / HIDDEN_DIM;
        if (!mean) mean = 1;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int32_t val = ((long long)x[i] * model->ln0_weight[i]) / mean + model->ln0_bias[i];
            x[i] = clip(val);
        }
        __syncwarp();

        bool v_first_set = false;

        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = (step_seed + pair_idx) + (l * 1000);

            local_sum = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long vabs = (long long)x[i];
                local_sum += (vabs < 0) ? -vabs : vabs;
            }
            sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
            sum = warpBroadcast(sum, 0);
            if (!sum) sum = 1;
            mean = sum / HIDDEN_DIM;
            if (!mean) mean = 1;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int32_t val = ((long long)x[i] * model->ln1_weight[l][i]) / mean + model->ln1_bias[l][i];
                x[i] = clip(val);
            }
            __syncwarp();

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int sub = i / WARP_SIZE;
                int8_t x_i = x[i];
                int8_t prev = x_prev_local[l][sub];
                int32_t diff = (int32_t)prev - (int32_t)x_i;

                int8_t mix_r = clip((long long)model->x_r[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XR, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t mix_w = clip((long long)model->x_w[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XW, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t mix_k = clip((long long)model->x_k[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XK, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t mix_v = clip((long long)model->x_v[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XV, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t mix_a = clip((long long)model->x_a[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XA, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t mix_g = clip((long long)model->x_g[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_XG, i) * ns) >> VEC_NOISE_SHIFT));

                xr[i] = clip((long long)x_i + ((diff * (int32_t)mix_r) >> 7));
                xw[i] = clip((long long)x_i + ((diff * (int32_t)mix_w) >> 7));
                xk[i] = clip((long long)x_i + ((diff * (int32_t)mix_k) >> 7));
                xv[i] = clip((long long)x_i + ((diff * (int32_t)mix_v) >> 7));
                xa[i] = clip((long long)x_i + ((diff * (int32_t)mix_a) >> 7));
                xg[i] = clip((long long)x_i + ((diff * (int32_t)mix_g) >> 7));

                x_prev_local[l][sub] = x_i;
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(xr, model->receptance[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_R, l_seed + SEED_OFF_R + HIDDEN_DIM, ns,
                                              r, &temp_storage[warp_id], lane_id);
            matmul_perturbed_warp<WarpReduce>(xk, model->key[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_K, l_seed + SEED_OFF_K + HIDDEN_DIM, ns,
                                              k, &temp_storage[warp_id], lane_id);
            matmul_perturbed_warp<WarpReduce>(xv, model->value[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_V, l_seed + SEED_OFF_V + HIDDEN_DIM, ns,
                                              v, &temp_storage[warp_id], lane_id);
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(xw, model->w1[l], RWKV_MIX_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_W1, l_seed + SEED_OFF_W1 + HIDDEN_DIM, ns,
                                              z, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < RWKV_MIX_DIM; i += WARP_SIZE) z[i] = tanh_i8(z[i]);
            __syncwarp();
            matmul_perturbed_warp<WarpReduce>(z, model->w2[l], HIDDEN_DIM, RWKV_MIX_DIM,
                                              l_seed + SEED_OFF_W2, l_seed + SEED_OFF_W2 + RWKV_MIX_DIM, ns,
                                              w_decay, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t w0 = clip((long long)model->w0[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_W0, i) * ns) >> VEC_NOISE_SHIFT));
                int8_t s = sigmoid_i8((int32_t)w_decay[i] + w0);
                w_decay[i] = decay_from_sigmoid(s);
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(xa, model->a1[l], RWKV_MIX_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_A1, l_seed + SEED_OFF_A1 + HIDDEN_DIM, ns,
                                              z, &temp_storage[warp_id], lane_id);
            __syncwarp();
            matmul_perturbed_warp<WarpReduce>(z, model->a2[l], HIDDEN_DIM, RWKV_MIX_DIM,
                                              l_seed + SEED_OFF_A2, l_seed + SEED_OFF_A2 + RWKV_MIX_DIM, ns,
                                              a_vec, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t a0 = clip((long long)model->a0[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_A0, i) * ns) >> VEC_NOISE_SHIFT));
                a_vec[i] = sigmoid_i8((int32_t)a_vec[i] + a0);
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(xg, model->g1[l], RWKV_MIX_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_G1, l_seed + SEED_OFF_G1 + HIDDEN_DIM, ns,
                                              z, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < RWKV_MIX_DIM; i += WARP_SIZE) z[i] = sigmoid_i8(z[i]);
            __syncwarp();
            matmul_perturbed_warp<WarpReduce>(z, model->g2[l], HIDDEN_DIM, RWKV_MIX_DIM,
                                              l_seed + SEED_OFF_G2, l_seed + SEED_OFF_G2 + RWKV_MIX_DIM, ns,
                                              g_vec, &temp_storage[warp_id], lane_id);
            __syncwarp();

            if (!v_first_set) {
                for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) v_first[i] = v[i];
            } else {
                matmul_perturbed_warp<WarpReduce>(xv, model->v1[l], RWKV_MIX_DIM, HIDDEN_DIM,
                                                  l_seed + SEED_OFF_V1, l_seed + SEED_OFF_V1 + HIDDEN_DIM, ns,
                                                  z, &temp_storage[warp_id], lane_id);
                __syncwarp();
                matmul_perturbed_warp<WarpReduce>(z, model->v2[l], HIDDEN_DIM, RWKV_MIX_DIM,
                                                  l_seed + SEED_OFF_V2, l_seed + SEED_OFF_V2 + RWKV_MIX_DIM, ns,
                                                  xw, &temp_storage[warp_id], lane_id);
                __syncwarp();
                for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                    int8_t v0 = clip((long long)model->v0[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_V0, i) * ns) >> VEC_NOISE_SHIFT));
                    int8_t gate = sigmoid_i8((int32_t)xw[i] + v0);
                    int32_t dv = ((int32_t)v_first[i] - (int32_t)v[i]) * (int32_t)gate;
                    v[i] = clip((int32_t)v[i] + (dv >> 7));
                }
            }
            __syncwarp();
            v_first_set = true;

            vector_with_noise_warp(model->k_k[l], HIDDEN_DIM, l_seed + SEED_OFF_KK, ns, xr, lane_id);
            vector_with_noise_warp(model->k_a[l], HIDDEN_DIM, l_seed + SEED_OFF_KA, ns, xk, lane_id);
            vector_with_noise_warp(model->r_k[l], HIDDEN_DIM, l_seed + SEED_OFF_RK, ns, xv, lane_id);
            __syncwarp();

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                kk_norm[i] = clip(((int32_t)k[i] * (int32_t)xr[i]) >> 7);
            }
            __syncwarp();

            if (lane_id < RWKV_N_HEAD) head_sum[warp_id][lane_id] = 0;
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long vabs = (long long)kk_norm[i];
                atomicAdd(&head_sum[warp_id][i / RWKV_HEAD_SIZE], (int32_t)((vabs < 0) ? -vabs : vabs));
            }
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int32_t s = head_sum[warp_id][i / RWKV_HEAD_SIZE];
                if (!s) s = 1;
                kk_norm[i] = (int8_t)(((int32_t)kk_norm[i] * 127) / s);
            }
            __syncwarp();

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int32_t delta = ((int32_t)a_vec[i] - 127) * (int32_t)xk[i];
                delta >>= 7;
                k[i] = clip((int32_t)k[i] + (((int32_t)k[i] * delta) >> 7));
                vvec[i] = clip(((int32_t)kk_norm[i] * (int32_t)a_vec[i]) >> 7);
            }
            __syncwarp();

            int32_t *layer_state = my_wkv + l * (RWKV_N_HEAD * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE);
            int32_t *out_state = out_state_shared[warp_id];
            for (int h = 0; h < RWKV_N_HEAD; h++) {
                int base = h * RWKV_HEAD_SIZE;
                int32_t *state = layer_state + h * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
                for (int i = lane_id; i < RWKV_HEAD_SIZE; i += WARP_SIZE) {
                    int32_t tmp = 0;
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        int idx = i * RWKV_HEAD_SIZE + j;
                        int32_t s = state[idx];
                        s = (s * (int32_t)w_decay[base + j]) >> 7;
                        state[idx] = s;
                        tmp += (s * (int32_t)(-kk_norm[base + j])) >> 7;
                    }
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        int idx = i * RWKV_HEAD_SIZE + j;
                        state[idx] += (tmp * (int32_t)vvec[base + j]) >> 7;
                        state[idx] += ((int32_t)v[base + i] * (int32_t)k[base + j]) >> 7;
                    }
                    int32_t acc = 0;
                    for (int j = 0; j < RWKV_HEAD_SIZE; j++) {
                        acc += (state[i * RWKV_HEAD_SIZE + j] * (int32_t)r[base + j]) >> 7;
                    }
                    out_state[base + i] = acc;
                }
            }
            __syncwarp();

            if (lane_id < RWKV_N_HEAD) head_sum[warp_id][lane_id] = 0;
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long vabs = (long long)out_state[i];
                atomicAdd(&head_sum[warp_id][i / RWKV_HEAD_SIZE], (int32_t)((vabs < 0) ? -vabs : vabs));
            }
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int head = i / RWKV_HEAD_SIZE;
                int32_t mean_h = head_sum[warp_id][head] / RWKV_HEAD_SIZE;
                if (!mean_h) mean_h = 1;
                int32_t val = (int32_t)(out_state[i] / mean_h);
                val = val * (int32_t)model->ln_x_weight[l][i] + model->ln_x_bias[l][i];
                a_vec[i] = clip(val);
            }
            __syncwarp();

            for (int h = 0; h < RWKV_N_HEAD; h++) {
                int base = h * RWKV_HEAD_SIZE;
                long long partial = 0;
                for (int i = lane_id; i < RWKV_HEAD_SIZE; i += WARP_SIZE) {
                    int idx = base + i;
                    partial += (long long)r[idx] * (long long)k[idx] * (long long)xv[idx];
                }
                long long dot = WarpReduce(temp_storage[warp_id]).Sum(partial);
                dot = warpBroadcast(dot, 0);
                int32_t dot_scaled = (int32_t)(dot >> 14);
                for (int i = lane_id; i < RWKV_HEAD_SIZE; i += WARP_SIZE) {
                    int idx = base + i;
                    a_vec[idx] = clip((int32_t)a_vec[idx] + ((dot_scaled * (int32_t)v[idx]) >> 7));
                }
                __syncwarp();
            }

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                a_vec[i] = clip(((int32_t)a_vec[i] * (int32_t)g_vec[i]) >> 7);
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(a_vec, model->output[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_O, l_seed + SEED_OFF_O + HIDDEN_DIM, ns,
                                              w_decay, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                x[i] = clip((int32_t)x[i] + w_decay[i]);
            }
            __syncwarp();

            local_sum = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long vabs = (long long)x[i];
                local_sum += (vabs < 0) ? -vabs : vabs;
            }
            sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
            sum = warpBroadcast(sum, 0);
            if (!sum) sum = 1;
            mean = sum / HIDDEN_DIM;
            if (!mean) mean = 1;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int32_t val = ((long long)x[i] * model->ln2_weight[l][i]) / mean + model->ln2_bias[l][i];
                x[i] = clip(val);
            }
            __syncwarp();

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int sub = i / WARP_SIZE;
                int8_t prev = x_prev_ffn_local[l][sub];
                int32_t diff = (int32_t)prev - (int32_t)x[i];
                int8_t mix = clip((long long)model->ffn_xk[l][i] + (((long long)noise_from_hash(l_seed + SEED_OFF_FFN_XK, i) * ns) >> VEC_NOISE_SHIFT));
                xk[i] = clip((long long)x[i] + ((diff * (int32_t)mix) >> 7));
                x_prev_ffn_local[l][sub] = x[i];
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(xk, model->ffn_key[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_FFN_K, l_seed + SEED_OFF_FFN_K + HIDDEN_DIM, ns,
                                              r, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int32_t val = r[i];
                if (val < 0) val = 0;
                val = (val * val) >> 7;
                r[i] = clip(val);
            }
            __syncwarp();

            matmul_perturbed_warp<WarpReduce>(r, model->ffn_value[l], HIDDEN_DIM, HIDDEN_DIM,
                                              l_seed + SEED_OFF_FFN_V, l_seed + SEED_OFF_FFN_V + HIDDEN_DIM, ns,
                                              w_decay, &temp_storage[warp_id], lane_id);
            __syncwarp();
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                x[i] = clip((int32_t)x[i] + w_decay[i]);
            }
            __syncwarp();
        }

        local_sum = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            long long vabs = (long long)x[i];
            local_sum += (vabs < 0) ? -vabs : vabs;
        }
        sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
        sum = warpBroadcast(sum, 0);
        if (!sum) sum = 1;
        mean = sum / HIDDEN_DIM;
        if (!mean) mean = 1;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int32_t val = ((long long)x[i] * model->ln_out_weight[i]) / mean + model->ln_out_bias[i];
            x[i] = clip(val);
        }
        __syncwarp();

        matmul_perturbed_warp<WarpReduce>(emb_cache, model->couple_b, COUPLE_RANK, HIDDEN_DIM,
                                          (step_seed + pair_idx) + SEED_OFF_COUPLE_B,
                                          (step_seed + pair_idx) + SEED_OFF_COUPLE_B + HIDDEN_DIM,
                                          ns, z, &temp_storage[warp_id], lane_id);
        __syncwarp();

        uint8_t target_y = dataset[stream_pos + t + 1];
        uint32_t seed_head_y = (step_seed + pair_idx) + SEED_OFF_HEAD_Y;
        uint32_t seed_couple_a = (step_seed + pair_idx) + SEED_OFF_COUPLE_A;

        long long partial_head = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            partial_head += (long long)x[i] * noise_from_hash(seed_head_y + HIDDEN_DIM, i);
        }
        long long xB_head = WarpReduce(temp_storage[warp_id]).Sum(partial_head);
        xB_head = warpBroadcast(xB_head, 0);

        long long partial_couple = 0;
        for (int i = lane_id; i < COUPLE_RANK; i += WARP_SIZE) {
            partial_couple += (long long)z[i] * noise_from_hash(seed_couple_a + COUPLE_RANK, i);
        }
        long long xB_couple = WarpReduce(temp_storage[warp_id]).Sum(partial_couple);
        xB_couple = warpBroadcast(xB_couple, 0);

        long long local_exp = 0;
        for (int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
            long long acc = 0;
            for (int k_idx = 0; k_idx < HIDDEN_DIM; k_idx++) {
                acc += (long long)x[k_idx] * model->head_y[k_idx * VOCAB_SIZE + i];
            }
            if (ns != 0) acc += ((xB_head * (long long)noise_from_hash(seed_head_y, i)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);

            long long acc_couple = 0;
            for (int k_idx = 0; k_idx < COUPLE_RANK; k_idx++) {
                acc_couple += (long long)z[k_idx] * model->couple_a[k_idx * VOCAB_SIZE + i];
            }
            if (ns != 0) acc_couple += ((xB_couple * (long long)noise_from_hash(seed_couple_a, i)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);

            int8_t logit_i = clip((acc >> RWKV_MATMUL_SHIFT) + (acc_couple >> RWKV_MATMUL_SHIFT));
            int idx = (int32_t)logit_i + 128;
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            local_exp += d_EXP2_TABLE[idx];
            if (i == target_y) target_logit_shared[warp_id] = (int32_t)logit_i + 128;
        }
        long long sum_exp = WarpReduce(temp_storage[warp_id]).Sum(local_exp);
        sum_exp = warpBroadcast(sum_exp, 0);
        __syncwarp();

        if (lane_id == 0) {
            long long log_sum = 0;
            long long xsum = sum_exp;
            if (xsum > 0) {
                int pos = 0;
                while (xsum >= 65536) { xsum >>= 16; pos += 16; }
                if (xsum >= 256)  { xsum >>= 8;  pos += 8; }
                if (xsum >= 16)   { xsum >>= 4;  pos += 4; }
                if (xsum >= 4)    { xsum >>= 2;  pos += 2; }
                if (xsum >= 2)    {           pos += 1; }

                long long fraction = (pos >= 4) ? (sum_exp - (1LL << pos)) >> (pos - 4) : (sum_exp - (1LL << pos)) << (4 - pos);
                log_sum = (pos << 4) + fraction - 64;
            }
            my_loss += (log_sum - target_logit_shared[warp_id]);
        }

        uint32_t seed_head_x = (step_seed + pair_idx) + SEED_OFF_HEAD_X;
        partial_head = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            partial_head += (long long)x[i] * noise_from_hash(seed_head_x + HIDDEN_DIM, i);
        }
        xB_head = WarpReduce(temp_storage[warp_id]).Sum(partial_head);
        xB_head = warpBroadcast(xB_head, 0);

        local_exp = 0;
        for (int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
            long long acc = 0;
            for (int k_idx = 0; k_idx < HIDDEN_DIM; k_idx++) {
                acc += (long long)x[k_idx] * model->head_x[k_idx * VOCAB_SIZE + i];
            }
            if (ns != 0) acc += ((xB_head * (long long)noise_from_hash(seed_head_x, i)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            int8_t logit_i = clip(acc >> RWKV_MATMUL_SHIFT);
            int idx = (int32_t)logit_i + 128;
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            local_exp += d_EXP2_TABLE[idx];
            if (i == input_token) target_logit_shared[warp_id] = (int32_t)logit_i + 128;
        }
        sum_exp = WarpReduce(temp_storage[warp_id]).Sum(local_exp);
        sum_exp = warpBroadcast(sum_exp, 0);
        __syncwarp();

        if (lane_id == 0) {
            long long log_sum = 0;
            long long xsum = sum_exp;
            if (xsum > 0) {
                int pos = 0;
                while (xsum >= 65536) { xsum >>= 16; pos += 16; }
                if (xsum >= 256)  { xsum >>= 8;  pos += 8; }
                if (xsum >= 16)   { xsum >>= 4;  pos += 4; }
                if (xsum >= 4)    { xsum >>= 2;  pos += 2; }
                if (xsum >= 2)    {           pos += 1; }

                long long fraction = (pos >= 4) ? (sum_exp - (1LL << pos)) >> (pos - 4) : (sum_exp - (1LL << pos)) << (4 - pos);
                log_sum = (pos << 4) + fraction - 64;
            }
            long long loss_x = (log_sum - target_logit_shared[warp_id]);
            my_loss += (loss_x * LOSS_X_NUM) / LOSS_X_DEN;
        }
    }

    if (lane_id == 0) {
        accum_loss[p_idx] = (int32_t)my_loss;
    }

    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE;
        for (int l = 0; l < N_LAYERS; l++) {
            state_x[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = x_prev_local[l][sub];
            state_ffn[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = x_prev_ffn_local[l][sub];
        }
    }
}

__global__ void compute_fitness_kernel(
    const int32_t *__restrict__ accum_loss, 
    int32_t *__restrict__ fitnesses, 
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int32_t loss_pos = accum_loss[2*idx];
    int32_t loss_neg = accum_loss[2*idx+1];
    
    int32_t fit = 0;
    if (loss_pos < loss_neg) fit = 1;
    else if (loss_neg < loss_pos) fit = -1;
    
    fitnesses[idx] = fit;
}

__global__ void update_matrix_kernel(
    int8_t * __restrict__ W,
    int rows,
    int cols,
    int offset_A,
    int offset_B,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed,
    int threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    
    int r = idx % rows; 
    int c = idx / rows; 
    
    long long vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        if (fit == 0) continue;

        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + offset_A, r);
        int8_t b = noise_from_hash(s + offset_B, c);
        
        vote += (long long)fit * (int)a * (int)b;
    }
    
    int8_t w_curr = W[c * rows + r]; 

    if(vote > threshold && w_curr < MAX_VAL) {
        w_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -threshold && w_curr > MIN_VAL) {
        w_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    W[c * rows + r] = w_curr; 
}

__global__ void update_vector_kernel(
    int8_t * __restrict__ V,
    int len,
    int seed_off_A,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed,
    int threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    
    long long vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        if (fit == 0) continue;
        
        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + seed_off_A, idx);
        
        vote += (long long)fit * (int)a;
    }
    
    int8_t v_curr = V[idx];

    if(vote > threshold && v_curr < MAX_VAL) {
        v_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -threshold && v_curr > MIN_VAL) {
        v_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    V[idx] = v_curr; 
}

struct abs_functor {
    __host__ __device__ int operator()(const int& x) const { return abs(x); }
};

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);
    srand(time(NULL));

    char datetime[64];
    format_local_datetime(datetime, sizeof(datetime));
    printf("Starting EGGROLL HIP/CUDA Training | Binary: %s | BUILD=%s | Datetime: %s\n",
           (argc > 0 && argv && argv[0]) ? argv[0] : "(unknown)",
           build_name(),
           datetime[0] ? datetime : "(unknown)");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    
    init_tables();
    CHECK_CUDA(cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_SIGMOID_TABLE, h_SIGMOID_TABLE, 256*sizeof(int8_t)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_TANH_TABLE, h_TANH_TABLE, 256*sizeof(int8_t)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_DECAY_TABLE, h_DECAY_TABLE, 128*sizeof(int8_t)));
    
    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("Error: input.txt not found!\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET); 
    ds.data=(uint8_t*)malloc(ds.length); 
    if(!fread(ds.data,1,ds.length,f)) { printf("Error reading input.txt\n"); exit(1); }
    fclose(f); 

    // --- Config Dump ---
    {
        long long params_embedding = (long long)VOCAB_SIZE * HIDDEN_DIM;
        long long params_ln0 = 2LL * HIDDEN_DIM;
        long long params_ln1 = 2LL * N_LAYERS * HIDDEN_DIM;
        long long params_ln2 = 2LL * N_LAYERS * HIDDEN_DIM;
        long long params_ln_x = 2LL * N_LAYERS * HIDDEN_DIM;
        long long params_ln_out = 2LL * HIDDEN_DIM;
        long long params_mix_vec = (long long)N_LAYERS * (6 + 3 + 3 + 1) * HIDDEN_DIM;
        long long params_mix_mat = (long long)N_LAYERS * 8 * RWKV_MIX_DIM * HIDDEN_DIM;
        long long params_time_mat = (long long)N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM;
        long long params_ffn_mat = (long long)N_LAYERS * 2 * HIDDEN_DIM * HIDDEN_DIM;
        long long params_heads = (long long)2 * HIDDEN_DIM * VOCAB_SIZE;
        long long params_couple = (long long)VOCAB_SIZE * COUPLE_RANK + (long long)COUPLE_RANK * HIDDEN_DIM;

        long long total_params = params_embedding + params_ln0 + params_ln1 + params_ln2 + params_ln_x +
                                 params_ln_out + params_mix_vec + params_mix_mat + params_time_mat +
                                 params_ffn_mat + params_heads + params_couple;

        printf("\n================ CONFIGURATION DUMP ================\n");
        printf("  Device: %s\n", prop.name);
        printf("  SM Cores:        %d\n", SM_CORES);
        printf("  Population Size: %d\n", POPULATION_SIZE);
        printf("  Hidden Dim:      %d\n", HIDDEN_DIM);
        printf("  Mix Dim:         %d\n", RWKV_MIX_DIM);
        printf("  Couple Rank:     %d\n", COUPLE_RANK);
        printf("  Seq Len:         %d\n", SEQ_LEN);
        printf("  Total Params:    %.2f M\n", total_params/1000000.0);
        printf("====================================================\n\n");
    }

    EggModel *h_model = (EggModel*)malloc(sizeof(EggModel));
    init_model(h_model); 
    
    EggModel *d_model;
    CHECK_CUDA(cudaMalloc(&d_model, sizeof(EggModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(EggModel), cudaMemcpyHostToDevice));
    
    uint8_t *d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));
    
    int32_t *d_accum_loss;
    CHECK_CUDA(cudaMalloc(&d_accum_loss, POPULATION_SIZE * sizeof(int32_t)));
    
    int32_t *d_fitnesses;
    CHECK_CUDA(cudaMalloc(&d_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    int8_t *d_state_x = NULL;
    int8_t *d_state_ffn = NULL;
    int32_t *d_state_wkv = NULL;
    size_t state_size = (size_t)POPULATION_SIZE * N_LAYERS * HIDDEN_DIM * sizeof(int8_t);
    CHECK_CUDA(cudaMalloc(&d_state_x, state_size));
    CHECK_CUDA(cudaMalloc(&d_state_ffn, state_size));
    CHECK_CUDA(cudaMemset(d_state_x, 0, state_size));
    CHECK_CUDA(cudaMemset(d_state_ffn, 0, state_size));
    size_t wkv_state_size = (size_t)POPULATION_SIZE * N_LAYERS * RWKV_N_HEAD * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE * sizeof(int32_t);
    CHECK_CUDA(cudaMalloc(&d_state_wkv, wkv_state_size));
    CHECK_CUDA(cudaMemset(d_state_wkv, 0, wkv_state_size));
    
    printf("Starting EGGROLL RWKV v7 CUDA Training (Batch=%d)...\n", BATCH);
    long max_steps = (ds.length - 1) / SEQ_LEN;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    unsigned long total_tokens = 0;

    for(long step=0; step<max_steps && keep_running; step++) {
        struct timespec t0, t1, t2, t3;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        int threads_per_block = BLOCK_THREADS;
        int warps_per_block = BLOCK_THREADS / WARP_SIZE;
        int blocks = POPULATION_SIZE / warps_per_block;
        size_t shared_mem_size = warps_per_block * SHARED_STRIDE * sizeof(int8_t);
        
        train_sequence_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_dataset, ds.length, start_idx, d_model,
            d_state_x, d_state_ffn, d_state_wkv,
            d_accum_loss, seed
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        // --- Host Control Removed: GPU logic via Thrust & Custom Kernels ---
        
        // 1. Compute Fitness on GPU
        int half_pop = POPULATION_SIZE / 2;
        compute_fitness_kernel<<< (half_pop + 255) / 256, 256 >>>(
            d_accum_loss, d_fitnesses, half_pop
        );
        
        // 2. Reduce Loss (Thrust)
        thrust::device_ptr<int32_t> t_accum_loss(d_accum_loss);
        long long current_accum_total = thrust::reduce(
            thrust::device, 
            t_accum_loss, 
            t_accum_loss + POPULATION_SIZE, 
            (long long)0, 
            thrust::plus<long long>()
        );
        
        double current_avg_loss = (double)current_accum_total / POPULATION_SIZE;
        double current_loss_per_token = current_avg_loss / (SEQ_LEN * (1 << FIXED_POINT));
        int current_threshold = get_update_threshold(current_loss_per_token);

        clock_gettime(CLOCK_MONOTONIC, &t2);
        
        if (!keep_running) break;

        // Debug prints
        if (step % 10 == 0) {
            thrust::device_ptr<int32_t> t_fitnesses(d_fitnesses);
            int fit_sum = thrust::transform_reduce(
                thrust::device,
                t_fitnesses, 
                t_fitnesses + half_pop,
                abs_functor(),
                0,
                thrust::plus<int>()
            );
            
            // Just peek at the first item's debug info by copying minimal data
            int32_t h_debug_loss[2];
            int32_t h_debug_fit[1];
            CHECK_CUDA(cudaMemcpy(h_debug_loss, d_accum_loss, 2*sizeof(int32_t), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_debug_fit, d_fitnesses, sizeof(int32_t), cudaMemcpyDeviceToHost));

            char dbg_datetime[64];
            format_local_datetime(dbg_datetime, sizeof(dbg_datetime));
            printf("[Debug] Step %ld Pair 0: Pos=%d Neg=%d Fit=%d | Threshold=%d | Total NonZero Fits: %d | Datetime: %s\n",
                step, h_debug_loss[0], h_debug_loss[1], h_debug_fit[0], current_threshold, fit_sum,
                dbg_datetime[0] ? dbg_datetime : "(unknown)");
        }
        
        int32_t zeros[2] = {0, 0};
        CHECK_CUDA(cudaMemcpyToSymbol(d_debug_updates, zeros, 2*sizeof(int32_t)));

        long off_emb = offsetof(EggModel, embedding);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_emb, HIDDEN_DIM, VOCAB_SIZE,
            HIDDEN_DIM, 0,
            SEED_OFF_EMB, d_fitnesses, seed, current_threshold
        );

        for (int l = 0; l < N_LAYERS; l++) {
            int seed_base = l * 1000;

            long off_xr = offsetof(EggModel, x_r) + l * HIDDEN_DIM;
            long off_xw = offsetof(EggModel, x_w) + l * HIDDEN_DIM;
            long off_xk = offsetof(EggModel, x_k) + l * HIDDEN_DIM;
            long off_xv = offsetof(EggModel, x_v) + l * HIDDEN_DIM;
            long off_xa = offsetof(EggModel, x_a) + l * HIDDEN_DIM;
            long off_xg = offsetof(EggModel, x_g) + l * HIDDEN_DIM;

            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xr, HIDDEN_DIM, SEED_OFF_XR, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xw, HIDDEN_DIM, SEED_OFF_XW, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xk, HIDDEN_DIM, SEED_OFF_XK, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xv, HIDDEN_DIM, SEED_OFF_XV, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xa, HIDDEN_DIM, SEED_OFF_XA, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_xg, HIDDEN_DIM, SEED_OFF_XG, seed_base, d_fitnesses, seed, current_threshold);

            long off_w0 = offsetof(EggModel, w0) + l * HIDDEN_DIM;
            long off_a0 = offsetof(EggModel, a0) + l * HIDDEN_DIM;
            long off_v0 = offsetof(EggModel, v0) + l * HIDDEN_DIM;
            long off_kk = offsetof(EggModel, k_k) + l * HIDDEN_DIM;
            long off_ka = offsetof(EggModel, k_a) + l * HIDDEN_DIM;
            long off_rk = offsetof(EggModel, r_k) + l * HIDDEN_DIM;
            long off_ffn_xk = offsetof(EggModel, ffn_xk) + l * HIDDEN_DIM;

            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_w0, HIDDEN_DIM, SEED_OFF_W0, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_a0, HIDDEN_DIM, SEED_OFF_A0, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_v0, HIDDEN_DIM, SEED_OFF_V0, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_kk, HIDDEN_DIM, SEED_OFF_KK, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_ka, HIDDEN_DIM, SEED_OFF_KA, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_rk, HIDDEN_DIM, SEED_OFF_RK, seed_base, d_fitnesses, seed, current_threshold);
            update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_ffn_xk, HIDDEN_DIM, SEED_OFF_FFN_XK, seed_base, d_fitnesses, seed, current_threshold);

            long off_w1 = offsetof(EggModel, w1) + l * (RWKV_MIX_DIM * HIDDEN_DIM);
            long off_w2 = offsetof(EggModel, w2) + l * (HIDDEN_DIM * RWKV_MIX_DIM);
            long off_a1 = offsetof(EggModel, a1) + l * (RWKV_MIX_DIM * HIDDEN_DIM);
            long off_a2 = offsetof(EggModel, a2) + l * (HIDDEN_DIM * RWKV_MIX_DIM);
            long off_v1 = offsetof(EggModel, v1) + l * (RWKV_MIX_DIM * HIDDEN_DIM);
            long off_v2 = offsetof(EggModel, v2) + l * (HIDDEN_DIM * RWKV_MIX_DIM);
            long off_g1 = offsetof(EggModel, g1) + l * (RWKV_MIX_DIM * HIDDEN_DIM);
            long off_g2 = offsetof(EggModel, g2) + l * (HIDDEN_DIM * RWKV_MIX_DIM);

            update_matrix_kernel<<< (RWKV_MIX_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_w1, RWKV_MIX_DIM, HIDDEN_DIM,
                SEED_OFF_W1, SEED_OFF_W1 + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*RWKV_MIX_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_w2, HIDDEN_DIM, RWKV_MIX_DIM,
                SEED_OFF_W2, SEED_OFF_W2 + RWKV_MIX_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );

            update_matrix_kernel<<< (RWKV_MIX_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_a1, RWKV_MIX_DIM, HIDDEN_DIM,
                SEED_OFF_A1, SEED_OFF_A1 + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*RWKV_MIX_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_a2, HIDDEN_DIM, RWKV_MIX_DIM,
                SEED_OFF_A2, SEED_OFF_A2 + RWKV_MIX_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );

            update_matrix_kernel<<< (RWKV_MIX_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_v1, RWKV_MIX_DIM, HIDDEN_DIM,
                SEED_OFF_V1, SEED_OFF_V1 + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*RWKV_MIX_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_v2, HIDDEN_DIM, RWKV_MIX_DIM,
                SEED_OFF_V2, SEED_OFF_V2 + RWKV_MIX_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );

            update_matrix_kernel<<< (RWKV_MIX_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_g1, RWKV_MIX_DIM, HIDDEN_DIM,
                SEED_OFF_G1, SEED_OFF_G1 + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*RWKV_MIX_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_g2, HIDDEN_DIM, RWKV_MIX_DIM,
                SEED_OFF_G2, SEED_OFF_G2 + RWKV_MIX_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );

            long off_r = offsetof(EggModel, receptance) + l * (HIDDEN_DIM * HIDDEN_DIM);
            long off_k = offsetof(EggModel, key) + l * (HIDDEN_DIM * HIDDEN_DIM);
            long off_v = offsetof(EggModel, value) + l * (HIDDEN_DIM * HIDDEN_DIM);
            long off_o = offsetof(EggModel, output) + l * (HIDDEN_DIM * HIDDEN_DIM);
            long off_fk = offsetof(EggModel, ffn_key) + l * (HIDDEN_DIM * HIDDEN_DIM);
            long off_fv = offsetof(EggModel, ffn_value) + l * (HIDDEN_DIM * HIDDEN_DIM);

            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_r, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_R, SEED_OFF_R + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_k, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_K, SEED_OFF_K + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_v, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_V, SEED_OFF_V + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_o, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_O, SEED_OFF_O + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_fk, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_FFN_K, SEED_OFF_FFN_K + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
            update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                (int8_t*)d_model + off_fv, HIDDEN_DIM, HIDDEN_DIM,
                SEED_OFF_FFN_V, SEED_OFF_FFN_V + HIDDEN_DIM,
                seed_base, d_fitnesses, seed, current_threshold
            );
        }

        long off_head_x = offsetof(EggModel, head_x);
        long off_head_y = offsetof(EggModel, head_y);
        long off_couple_a = offsetof(EggModel, couple_a);
        long off_couple_b = offsetof(EggModel, couple_b);

        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head_x, VOCAB_SIZE, HIDDEN_DIM,
            0, HIDDEN_DIM,
            SEED_OFF_HEAD_X, d_fitnesses, seed, current_threshold
        );
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head_y, VOCAB_SIZE, HIDDEN_DIM,
            0, HIDDEN_DIM,
            SEED_OFF_HEAD_Y, d_fitnesses, seed, current_threshold
        );

        update_matrix_kernel<<< (VOCAB_SIZE*COUPLE_RANK + 511)/512, 512 >>>(
            (int8_t*)d_model + off_couple_a, VOCAB_SIZE, COUPLE_RANK,
            0, COUPLE_RANK,
            SEED_OFF_COUPLE_A, d_fitnesses, seed, current_threshold
        );
        update_matrix_kernel<<< (COUPLE_RANK*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_couple_b, COUPLE_RANK, HIDDEN_DIM,
            0, HIDDEN_DIM,
            SEED_OFF_COUPLE_B, d_fitnesses, seed, current_threshold
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t3);
        
        if (!keep_running) break;

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_tokens += POPULATION_SIZE * SEQ_LEN;
        
        if (step % 1 == 0) { 
            int32_t h_updates[2];
            CHECK_CUDA(cudaMemcpyFromSymbol(h_updates, d_debug_updates, 2*sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
            
            static uint8_t *d_output = NULL;
            static uint8_t *h_output = NULL;
            int seed_len = 30; 
            int gen_len = 50;
            if(!d_output) {
                CHECK_CUDA(cudaMalloc(&d_output, gen_len));
                h_output = (uint8_t*)malloc(gen_len);
            }
            
            generate_sequence_kernel<<<1, 256, 0>>>(
                d_model, seed + 12345, d_dataset + start_idx, seed_len, gen_len, d_output
            );
            CHECK_CUDA(cudaDeviceSynchronize()); 
            CHECK_CUDA(cudaMemcpy(h_output, d_output, gen_len, cudaMemcpyDeviceToHost));
            
            printf("\033[32m");
            for(int i=0; i<seed_len; i++) {
                char c = ds.data[start_idx+i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[36m");
            for(int i=0; i<gen_len; i++) {
                char c = h_output[i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[0m\n");

            double fwd_ms = get_time_diff_ms(t0, t1);
            double host_ms = get_time_diff_ms(t1, t2);
            double upd_ms = get_time_diff_ms(t2, t3);
            double step_ms = get_time_diff_ms(t0, t3);
            double steps_per_sec = (step_ms > 0) ? (1000.0 / step_ms) : 0.0;
            
            double total_t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
            double mtok_per_s = (total_t > 0) ? ((total_tokens / total_t) / 1000000.0) : 0.0;
            
            printf("Step %ld | Loss: %.4f | Up+: %d Up-: %d | Fwd: %.1fms Host: %.1fms Upd: %.1fms | %.2f Steps/s | MTok/s: %.6f\n",
               step, current_loss_per_token, h_updates[0], h_updates[1], 
               fwd_ms, host_ms, upd_ms, steps_per_sec, mtok_per_s);
        }
    }

    if (!keep_running) {
        printf("\nTraining interrupted by User. Exiting gracefully...\n");
    }

    printf("Cleaning up resources...\n");
    free(ds.data);
    free(h_model);
    
    CHECK_CUDA(cudaFree(d_model));
    CHECK_CUDA(cudaFree(d_dataset));
    CHECK_CUDA(cudaFree(d_accum_loss));
    CHECK_CUDA(cudaFree(d_fitnesses));
    CHECK_CUDA(cudaFree(d_state_x));
    CHECK_CUDA(cudaFree(d_state_ffn));
    CHECK_CUDA(cudaFree(d_state_wkv));
    
    return 0;
}
