#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Constants matching the paper/code
#define HIDDEN_DIM 256
#define VOCAB_SIZE 256
#define N_LAYERS 6
#define FIXED_POINT_SHIFT 4
#define MAX_VAL 127
#define MIN_VAL -127
#define LOG_MAX 7 

// ------------------------------------------------------------------
// Core Math Operations
// ------------------------------------------------------------------

// Saturated Addition: Adds integers and clips to [-127, 127]
int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// Saturated Addition for 3 inputs (helper for GRU)
int8_t clipped_add3(int32_t a, int32_t b, int32_t c) {
    int32_t res = a + b + c;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// Scaled Matrix-Vector Multiplication
// Concept: output = (vec @ matrix) / (16 * sqrt(n))
// In fixed point logic, this usually becomes a bit shift.
// Paper: "dividing by 16*sqrt(n)... typically equivalent to right-shift" [cite: 957, 958]
void matmul_scaled(const int8_t *vec, const int8_t *mat, int8_t *out, int in_dim, int out_dim) {
    // Scaling factor approximation. For n=256, sqrt(n)=16. 16*16=256. 
    // Division by 256 is >> 8.
    int shift = 8; 

    for (int i = 0; i < out_dim; i++) {
        int32_t acc = 0;
        for (int j = 0; j < in_dim; j++) {
            acc += (int32_t)vec[j] * (int32_t)mat[i * in_dim + j];
        }
        
        // Apply scaling (right shift)
        int32_t scaled = acc >> shift;
        
        // Clip to int8 range
        if (scaled > MAX_VAL) out[i] = MAX_VAL;
        else if (scaled < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)scaled;
    }
}

// ------------------------------------------------------------------
// Layers
// ------------------------------------------------------------------

typedef struct {
    int8_t weight[HIDDEN_DIM];
} EGG_LN;

// Custom L1 LayerNorm [cite: 286, 965]
void layer_norm(const EGG_LN *ln, int8_t *x) {
    int32_t abs_sum = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        abs_sum += (x[i] >= 0) ? x[i] : -x[i];
    }
    
    // Mean Absolute Value (avoiding division by zero)
    if (abs_sum == 0) abs_sum = 1;
    int32_t mean_abs = abs_sum / HIDDEN_DIM;
    if (mean_abs == 0) mean_abs = 1;

    for (int i = 0; i < HIDDEN_DIM; i++) {
        // x * weight / mean_abs
        int32_t val = ((int32_t)x[i] * (int32_t)ln->weight[i]) / mean_abs;
        
        if (val > MAX_VAL) x[i] = MAX_VAL;
        else if (val < MIN_VAL) x[i] = MIN_VAL;
        else x[i] = (int8_t)val;
    }
}

typedef struct {
    int8_t Wf[HIDDEN_DIM * HIDDEN_DIM];
    int8_t Uf[HIDDEN_DIM * HIDDEN_DIM];
    int8_t bf[HIDDEN_DIM];
    int8_t Wh[HIDDEN_DIM * HIDDEN_DIM];
    int8_t Uh[HIDDEN_DIM * HIDDEN_DIM];
    int8_t bh[HIDDEN_DIM];
} EGG_GRU;

// Evolved Generative GRU Step [cite: 981, 983-986]
void gru_step(const EGG_GRU *gru, const int8_t *x, int8_t *state, int8_t *out_h) {
    int8_t f_lin[HIDDEN_DIM], f_rec[HIDDEN_DIM];
    int8_t h_lin[HIDDEN_DIM], h_rec[HIDDEN_DIM];
    int8_t ft[HIDDEN_DIM];
    int8_t gated_past[HIDDEN_DIM];
    int8_t ht_hat[HIDDEN_DIM];

    // 1. Calculate Update Gate 'ft'
    matmul_scaled(x, gru->Wf, f_lin, HIDDEN_DIM, HIDDEN_DIM);
    matmul_scaled(state, gru->Uf, f_rec, HIDDEN_DIM, HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM; i++) {
        // Nonlinearity is just clipped add (Sigmoid is Identity)
        ft[i] = clipped_add3(f_lin[i], f_rec[i], gru->bf[i]);
    }

    // 2. Calculate Gated Past State
    // Paper Logic: ((ft + 127) * state) >> 8  [cite: 984]
    for (int i = 0; i < HIDDEN_DIM; i++) {
        int32_t gate_val = (int32_t)ft[i] + 127; // Shift to [0, 254]
        int32_t res = (gate_val * (int32_t)state[i]) >> 8;
        
        // Manual clip (though logically should fit if state is bounded)
        if (res > MAX_VAL) gated_past[i] = MAX_VAL;
        else if (res < MIN_VAL) gated_past[i] = MIN_VAL;
        else gated_past[i] = (int8_t)res;
    }

    // 3. Calculate Candidate State 'ht_hat'
    matmul_scaled(x, gru->Wh, h_lin, HIDDEN_DIM, HIDDEN_DIM);
    matmul_scaled(gated_past, gru->Uh, h_rec, HIDDEN_DIM, HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM; i++) {
        // Nonlinearity is just clipped add (Tanh is Identity)
        ht_hat[i] = clipped_add3(h_lin[i], h_rec[i], gru->bh[i]);
    }

    // 4. Final State Update (Linear Interpolation)
    // Paper Logic: state + ((ft + 127) * (ht_hat - state)) >> 8 [cite: 986]
    for (int i = 0; i < HIDDEN_DIM; i++) {
        int32_t gate_val = (int32_t)ft[i] + 127;
        int32_t diff = (int32_t)ht_hat[i] - (int32_t)state[i];
        int32_t update = (gate_val * diff) >> 8;
        
        int32_t final_val = (int32_t)state[i] + update;
        
        // Clip final state
        if (final_val > MAX_VAL) out_h[i] = MAX_VAL;
        else if (final_val < MIN_VAL) out_h[i] = MIN_VAL;
        else out_h[i] = (int8_t)final_val;
    }
}

typedef struct {
    int8_t w1[HIDDEN_DIM * (HIDDEN_DIM * 4)];
    int8_t w2[(HIDDEN_DIM * 4) * HIDDEN_DIM]; 
} EGG_MLP;

// MLP Block [cite: 977]
void mlp_forward(const EGG_MLP *mlp, int8_t *x) {
    int8_t hidden[HIDDEN_DIM * 4];
    int8_t out[HIDDEN_DIM];

    // Linear 1 (Expansion)
    // No activation needed as matmul_scaled includes clipping/saturation
    matmul_scaled(x, mlp->w1, hidden, HIDDEN_DIM, HIDDEN_DIM * 4);
    
    // Linear 2 (Projection)
    matmul_scaled(hidden, mlp->w2, out, HIDDEN_DIM * 4, HIDDEN_DIM);

    // Copy back to x
    memcpy(x, out, HIDDEN_DIM * sizeof(int8_t));
}

// ------------------------------------------------------------------
// Full Model Struct
// ------------------------------------------------------------------

typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    
    struct {
        EGG_LN ln1;
        EGG_GRU gru;
        EGG_LN ln2;
        EGG_MLP mlp;
    } layers[N_LAYERS];

    EGG_LN ln_out;
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
} EggModel;

// ------------------------------------------------------------------
// Main Forward Pass
// ------------------------------------------------------------------

void egg_forward(const EggModel *model, const uint8_t *tokens, int seq_len, 
                 int8_t *states, int8_t *logits_out) {
    
    // Working buffer for the current token vector
    int8_t x[HIDDEN_DIM];
    int8_t residual[HIDDEN_DIM];

    // Loop over time (tokens)
    for (int t = 0; t < seq_len; t++) {
        uint8_t token_id = tokens[t];

        // 1. Embedding Lookup [cite: 963]
        memcpy(x, &model->embedding[token_id * HIDDEN_DIM], HIDDEN_DIM * sizeof(int8_t));

        // 2. Loop over Layers
        for (int l = 0; l < N_LAYERS; l++) {
            // -- GRU Block --
            memcpy(residual, x, HIDDEN_DIM); // Save residual
            
            layer_norm(&model->layers[l].ln1, x);
            
            // Pointer to this layer's state for this batch/sequence
            // (Assuming single batch for this simplified C code)
            int8_t *layer_state = &states[l * HIDDEN_DIM];
            gru_step(&model->layers[l].gru, x, layer_state, layer_state);
            
            // GRU output is the new state
            memcpy(x, layer_state, HIDDEN_DIM);
            
            // Add Residual
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            // -- MLP Block --
            memcpy(residual, x, HIDDEN_DIM); // Save residual
            
            layer_norm(&model->layers[l].ln2, x);
            mlp_forward(&model->layers[l].mlp, x);
            
            // Add Residual
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // 3. Final Head
        layer_norm(&model->ln_out, x);
        
        // To save memory in this example, only calculating logits for the LAST token
        if (t == seq_len - 1) {
            matmul_scaled(x, model->head, logits_out, HIDDEN_DIM, VOCAB_SIZE);
        }
    }
}

// ------------------------------------------------------------------
// Example Usage
// ------------------------------------------------------------------

int main() {
    // 1. Allocate Model (In real use, this would be mapped from a file)
    EggModel *model = (EggModel *)calloc(1, sizeof(EggModel));
    if (!model) { fprintf(stderr, "Memory allocation failed\n"); return 1; }

    // Initialize dummy weights for demonstration (e.g. Identity/Ones)
    // In practice, you would load 'minipile_train.npy' or similar here
    for(int i=0; i<HIDDEN_DIM; i++) {
        model->layers[0].ln1.weight[i] = 16; // Standard init from paper
    }

    // 2. Prepare Inputs
    uint8_t input_tokens[] = { 65, 66, 67 }; // "ABC"
    int seq_len = 3;

    // States buffer: [Layers * Hidden]
    int8_t *states = (int8_t *)calloc(N_LAYERS * HIDDEN_DIM, sizeof(int8_t));
    
    // Output buffer
    int8_t logits[VOCAB_SIZE];

    // 3. Run Inference
    printf("Running EGG Forward Pass (Pure C, Int8)...\n");
    egg_forward(model, input_tokens, seq_len, states, logits);

    printf("Logits for last token:\n");
    for(int i=0; i<10; i++) {
        printf("%d ", logits[i]);
    }
    printf("\n");

    free(states);
    free(model);
    return 0;
}
