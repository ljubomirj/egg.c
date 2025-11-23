#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// --- Hyperparameters from Paper/Code ---
#define POPULATION_SIZE 128     // Number of workers (must be even for antithetical)
#define SIGMA_SHIFT 4           // Noise scaling (bitwise shift) [cite: 1010]
#define FIXED_POINT 4           // 4 bits for fractional part
#define UPDATE_THRESHOLD 500    // Threshold to trigger an integer flip
#define MAX_VAL 127
#define MIN_VAL -127
#define HIDDEN_DIM 256
#define INPUT_DIM 256

// --- Lookup Tables for Loss Calculation ---
int32_t LOG2_TABLE[1024]; 
int32_t EXP2_TABLE[256];

void init_lookup_tables() {
    // Initialize approximated log/exp tables to avoid floats during training loop
    // [cite: 998]
    for(int i=0; i<256; i++) {
        // exp2(i / 16) * 16
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
    }
    for(int i=0; i<1024; i++) {
        // log2(i / 16) * 16
        if(i==0) LOG2_TABLE[i] = 0; // handle 0
        else LOG2_TABLE[i] = (int32_t)(log2((double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
    }
}

// --- Pseudo-Random Number Generator ---
// Simple Xorshift for speed. In real implementation, use a counter-based RNG 
// (like Philox) to match JAX's stateless behavior. [cite: 69]
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Generate a noise vector element (Gaussian approx via CLT or simple uniform)
// Scaled to int8 range
int8_t gen_noise_val(uint32_t *rng_state) {
    // Generate value roughly between -16 and 16 (scaled normal)
    uint32_t r = xorshift32(rng_state) & 0xFF;
    return (int8_t)((r % 32) - 16);
}

// --- Core EGGROLL Operations ---

// 1. Variational Forward Pass (The "Trick")
// Computes: x(W + sigma*AB^T)^T = xW^T + (xB)A^T
// [cite: 72, 225]
void matmul_perturbed(
    const int8_t *input, 
    const int8_t *weight, 
    const int8_t *A,      // Perturbation vector A (m x 1)
    const int8_t *B,      // Perturbation vector B (n x 1)
    int8_t *output, 
    int m, int n, 
    int noise_sign        // +1 or -1 for antithetical sampling
) {
    // 1. Compute xB (Dot product of input and B)
    int32_t xB = 0;
    for (int j = 0; j < n; j++) {
        xB += (int32_t)input[j] * (int32_t)B[j];
    }

    // 2. Standard Matrix Mul + Low Rank Correction
    int shift = 8; // Scaling factor (16 * sqrt(n))
    
    for (int i = 0; i < m; i++) {
        int32_t base_acc = 0;
        
        // Standard xW^T
        for (int j = 0; j < n; j++) {
            base_acc += (int32_t)input[j] * (int32_t)weight[i * n + j];
        }

        // Add Perturbation: (xB * A[i])
        // Apply sigma_shift (learning rate/noise scale control)
        int32_t perturbation = (xB * (int32_t)A[i]) * noise_sign;
        perturbation = perturbation >> SIGMA_SHIFT; 

        int32_t total = (base_acc + perturbation) >> shift;

        // Clip to int8
        if (total > MAX_VAL) output[i] = MAX_VAL;
        else if (total < MIN_VAL) output[i] = MIN_VAL;
        else output[i] = (int8_t)total;
    }
}

// 2. Cross Entropy Loss (Integer Only)
// [cite: 994-997]
int32_t compute_loss(int8_t *logits, int target_idx, int dim) {
    // Shift logits to positive range for table lookup [0, 255]
    int32_t sum_exp = 0;
    for (int i = 0; i < dim; i++) {
        int idx = (int32_t)logits[i] + 128;
        if (idx < 0) idx = 0;
        if (idx > 255) idx = 255;
        sum_exp += EXP2_TABLE[idx];
    }
    
    // logsumexp approximation
    int32_t log_sum_exp = 0;
    if (sum_exp < 1024) log_sum_exp = LOG2_TABLE[sum_exp];
    else log_sum_exp = LOG2_TABLE[1023]; // clamp

    int32_t target_logit = (int32_t)logits[target_idx] + 128;
    
    // Negative Log Likelihood
    return log_sum_exp - target_logit;
}

// 3. Training Step for a Single Matrix
void train_matrix_layer(
    int8_t *W, 
    int m, int n, 
    const int8_t *inputs,     // Batch of inputs
    const int *targets,       // Batch of target indices
    int batch_size,
    uint32_t master_seed
) {
    int32_t *update_accumulator = (int32_t *)calloc(m * n, sizeof(int32_t));
    int8_t *A = (int8_t *)malloc(m * sizeof(int8_t));
    int8_t *B = (int8_t *)malloc(n * sizeof(int8_t));
    int8_t *outputs = (int8_t *)malloc(m * sizeof(int8_t));

    // --- POPULATION LOOP (Parallelizable) ---
    // Iterate in steps of 2 for Antithetical Sampling (+Noise, -Noise)
    // 
    for (int p = 0; p < POPULATION_SIZE; p += 2) {
        
        uint32_t worker_seed = master_seed + p;
        
        // Generate A and B for this worker pair
        // In reality, A/B are generated via seeded PRNG on the fly
        uint32_t rng = worker_seed;
        for(int i=0; i<m; i++) A[i] = gen_noise_val(&rng);
        for(int i=0; i<n; i++) B[i] = gen_noise_val(&rng);

        // --- Evaluate Positive Perturbation (+1) ---
        int32_t loss_pos = 0;
        for(int b=0; b<batch_size; b++) {
            matmul_perturbed(&inputs[b*n], W, A, B, outputs, m, n, 1);
            loss_pos += compute_loss(outputs, targets[b], m);
        }

        // --- Evaluate Negative Perturbation (-1) ---
        int32_t loss_neg = 0;
        for(int b=0; b<batch_size; b++) {
            matmul_perturbed(&inputs[b*n], W, A, B, outputs, m, n, -1);
            loss_neg += compute_loss(outputs, targets[b], m);
        }

        // --- Fitness Shaping ---
        // Simple sign-based fitness: Did Pos beat Neg?
        // 
        int fitness = 0;
        if (loss_pos < loss_neg) fitness = 1;      // Pos was better
        else if (loss_neg < loss_pos) fitness = -1; // Neg was better
        
        // --- Accumulate Gradient Estimate ---
        // Instead of storing full matrices, we reconstruct the update on the fly
        // Update += Fitness * (A * B^T)
        // [cite: 1021]
        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                int32_t noise_val = (int32_t)A[i] * (int32_t)B[j];
                update_accumulator[i*n + j] += fitness * noise_val;
            }
        }
    }

    // --- UPDATE STEP (Discrete Voting) ---
    // 
    int changes = 0;
    for(int i=0; i < m*n; i++) {
        int32_t vote = update_accumulator[i];
        
        // If the aggregated vote exceeds threshold, move integer weight
        if (abs(vote) > UPDATE_THRESHOLD) {
            if (vote > 0) {
                if (W[i] < MAX_VAL) W[i]++;
            } else {
                if (W[i] > MIN_VAL) W[i]--;
            }
            changes++;
        }
    }
    
    printf("Layer Update: %d/%d parameters changed.\n", changes, m*n);

    free(update_accumulator);
    free(A);
    free(B);
    free(outputs);
}

int main() {
    init_lookup_tables();

    // 1. Initialize Dummy Weight Matrix (1 Layer)
    // 256 inputs -> 256 outputs
    int8_t *W = (int8_t *)malloc(HIDDEN_DIM * INPUT_DIM * sizeof(int8_t));
    for(int i=0; i<HIDDEN_DIM*INPUT_DIM; i++) W[i] = 0; // Init to 0

    // 2. Create Dummy Batch Data
    int batch_size = 4;
    int8_t *inputs = (int8_t *)malloc(batch_size * INPUT_DIM * sizeof(int8_t));
    int *targets = (int *)malloc(batch_size * sizeof(int));
    
    // Fill dummy data
    for(int i=0; i<batch_size * INPUT_DIM; i++) inputs[i] = (i % 30) - 15;
    for(int i=0; i<batch_size; i++) targets[i] = 5; // Arbitrary target class

    // 3. Run Training Loop
    printf("Starting EGGROLL Training Loop...\n");
    for (int epoch = 0; epoch < 5; epoch++) {
        printf("Epoch %d: ", epoch);
        
        // Train the layer
        // In a full model, you would call this for every weight matrix (Wf, Wh, MLP, etc.)
        train_matrix_layer(W, HIDDEN_DIM, INPUT_DIM, inputs, targets, batch_size, epoch * 12345);
    }

    free(W);
    free(inputs);
    free(targets);
    return 0;
}
