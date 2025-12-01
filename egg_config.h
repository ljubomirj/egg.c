#ifndef EGG_CONFIG_H
#define EGG_CONFIG_H

// --- Configuration ---
#define VOCAB_SIZE 256        // Byte-level tokenization
#define HIDDEN_DIM 512        // Model width
#define N_LAYERS 4            // Number of layers
//#define SEQ_LEN 4096          // Sequence length for BPTT (truncated)
#define SEQ_LEN 256          // Sequence length for BPTT (truncated)
//#define POPULATION_SIZE 128   // Number of perturbations per step
#define POPULATION_SIZE 32   // Number of perturbations per step
#define BATCH_SIZE 8          // Parallel streams
#define FIXED_POINT 4         // 4 bits for fractional part
#define SIGMA_SHIFT 4         // Noise scale (bitwise shift)
#define UPDATE_THRESHOLD 160  // Votes needed to flip a weight
#define MAX_VAL 127
#define MIN_VAL -127

// Derived Constants for Optimization
#define MAX_MATRIX_DIM (HIDDEN_DIM * 4) // Largest dimension (MLP expansion)
#define MAX_POP_PAIRS (POPULATION_SIZE / 2)

#endif // EGG_CONFIG_H


