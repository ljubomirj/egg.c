#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include "egg_gpu_metal.h"
#include "egg_gpu_metal_optimized.h"
#endif

#include "egg_config.h"

// Optimized Metal functions are declared in egg_gpu_metal_optimized.h

// --- Data Structures ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM];
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

typedef struct {
    int8_t h[N_LAYERS][HIDDEN_DIM];
} RecurrentState;

// --- RNG and Noise Generation ---
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise(uint32_t *rng) {
    uint32_t r = xorshift32(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

// --- Optimized Population Evaluation ---
typedef struct {
    const EggModel *model;
    const Dataset *ds;
    long start_idx;
    int *pair_fitnesses;
    uint32_t step_seed;
    int *population_losses;
} OptimizedEvalContext;

// Evaluate entire population in one GPU call
static bool evaluate_population_optimized(
    const EggModel *model,
    const Dataset *ds,
    long start_idx,
    int *pair_fitnesses,
    uint32_t step_seed,
    int *population_losses
) {
#if defined(__APPLE__)
    // Use optimized Metal implementation
    static bool metal_initialized = false;
    static bool metal_attempted = false;

    if (!metal_attempted) {
        metal_attempted = true;
        // Check if we should use optimized Metal
        const char *use_optimized = getenv("EGG_USE_OPTIMIZED_METAL");
        if (use_optimized && use_optimized[0] == '1') {
            metal_initialized = egg_gpu_metal_optimized_init();
            if (metal_initialized) {
                printf("[EGG OPT] Using optimized Metal implementation\n");
            }
        }
    }

    if (metal_initialized) {
        // Compute streams for each population member
        long stride = ds->length / (POPULATION_SIZE / 2);

        // For now, we'll evaluate pairs sequentially with GPU optimization
        // In a full implementation, we'd batch all population members
        for (int p = 0; p < POPULATION_SIZE / 2; p++) {
            long stream_idx = (start_idx + (p * stride)) % (ds->length - SEQ_LEN);
            const uint8_t *tokens = &ds->data[stream_idx];

            int losses[2];
            if (!egg_gpu_metal_optimized_forward_pass(model, tokens, SEQ_LEN, step_seed + p, losses)) {
                return false;
            }

            if (losses[0] < losses[1]) {
                pair_fitnesses[p] = 1;
            } else if (losses[1] < losses[0]) {
                pair_fitnesses[p] = -1;
            } else {
                pair_fitnesses[p] = 0;
            }

            if (population_losses) {
                population_losses[p * 2] = losses[0];
                population_losses[p * 2 + 1] = losses[1];
            }
        }
        return true;
    }
#endif

    // Fallback to CPU
    return false;
}

// --- Matrix Update (CPU fallback for now) ---
static void update_matrix(
    int8_t *W, int rows, int cols,
    uint32_t layer_seed,
    const int *pair_fitnesses,
    int pairs
) {
    // Allocate transposed noise matrices
    int8_t (*A_T)[pairs] = (int8_t (*)[pairs])malloc(rows * pairs * sizeof(int8_t));
    int8_t (*B_T)[pairs] = (int8_t (*)[pairs])malloc(cols * pairs * sizeof(int8_t));

    // Generate noise and compute votes
    for (int p = 0; p < pairs; p++) {
        int fitness = pair_fitnesses[p];

        if (fitness == 0) {
            // Zero out columns for this pair
            for (int r = 0; r < rows; r++) A_T[r][p] = 0;
            for (int c = 0; c < cols; c++) B_T[c][p] = 0;
        } else {
            uint32_t rng = layer_seed + p * 1000;
            for (int r = 0; r < rows; r++) {
                A_T[r][p] = gen_noise(&rng);
            }
            for (int c = 0; c < cols; c++) {
                B_T[c][p] = gen_noise(&rng);
            }
            if (fitness < 0) {
                for (int r = 0; r < rows; r++) A_T[r][p] = -A_T[r][p];
            }
        }
    }

    // Apply updates
    for (int r = 0; r < rows; r++) {
        int8_t *w_row = &W[r * cols];
        for (int c = 0; c < cols; c++) {
            int32_t vote = 0;
            for (int p = 0; p < pairs; p++) {
                vote += (int32_t)A_T[r][p] * (int32_t)B_T[c][p];
            }

            if (vote > UPDATE_THRESHOLD && w_row[c] < MAX_VAL) w_row[c]++;
            else if (vote < -UPDATE_THRESHOLD && w_row[c] > MIN_VAL) w_row[c]--;
        }
    }

    free(A_T);
    free(B_T);
}

// --- Model Initialization ---
static void init_model(EggModel *model) {
    uint32_t rng = 42;

    // Initialize with random noise
    for (int i = 0; i < VOCAB_SIZE * HIDDEN_DIM; i++) {
        model->embedding[i] = gen_noise(&rng);
    }

    for (int l = 0; l < N_LAYERS; l++) {
        for (int g = 0; g < 4; g++) {
            for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) {
                model->gru_weights[l][g][i] = gen_noise(&rng);
            }
        }
        for (int i = 0; i < HIDDEN_DIM; i++) {
            model->gru_biases[l][0][i] = 0;
            model->gru_biases[l][1][i] = 0;
        }

        // MLP weights
        for (int i = 0; i < HIDDEN_DIM * (HIDDEN_DIM * 4); i++) {
            model->mlp_weights[l][0][i] = gen_noise(&rng);
        }
        for (int i = 0; i < (HIDDEN_DIM * 4) * HIDDEN_DIM; i++) {
            model->mlp_weights[l][1][i] = gen_noise(&rng);
        }

        // Layer norm weights
        for (int i = 0; i < HIDDEN_DIM; i++) {
            model->ln_weights[l][0][i] = 16;
            model->ln_weights[l][1][i] = 16;
        }
    }

    // Head
    for (int i = 0; i < HIDDEN_DIM * VOCAB_SIZE; i++) {
        model->head[i] = gen_noise(&rng);
    }

    // Output layer norm
    for (int i = 0; i < HIDDEN_DIM; i++) {
        model->ln_out[i] = 16;
    }
}

// --- Load Dataset ---
static Dataset load_dataset(const char *filename) {
    Dataset ds = {0};

    // Try compressed version first
    char zst_filename[1024];
    snprintf(zst_filename, sizeof(zst_filename), "%s.zst", filename);

    FILE *f = fopen(zst_filename, "rb");
    if (!f) {
        // Try uncompressed
        f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Error: Could not open %s or %s\n", filename, zst_filename);
            exit(1);
        }
    }

    if (strstr(filename, ".zst") != NULL || f != fopen(zst_filename, "rb")) {
        // Load compressed file
        fclose(f);

        // Use zstd to decompress
        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "zstd -dc %s", zst_filename);
        f = popen(cmd, "rb");
        if (!f) {
            fprintf(stderr, "Error: Could not decompress %s\n", zst_filename);
            exit(1);
        }

        // Get file size
        fseek(f, 0, SEEK_END);
        ds.length = ftell(f);
        fseek(f, 0, SEEK_SET);

        ds.data = (uint8_t*)malloc(ds.length);
        if (!ds.data) {
            fprintf(stderr, "Error: Could not allocate memory for dataset\n");
            exit(1);
        }

        if (!fread(ds.data, 1, ds.length, f)) {
            fprintf(stderr, "Error: Could not read dataset\n");
            exit(1);
        }

        pclose(f);
    } else {
        // Load uncompressed file
        fseek(f, 0, SEEK_END);
        ds.length = ftell(f);
        fseek(f, 0, SEEK_SET);

        ds.data = (uint8_t*)malloc(ds.length);
        if (!ds.data) {
            fprintf(stderr, "Error: Could not allocate memory for dataset\n");
            exit(1);
        }

        if (!fread(ds.data, 1, ds.length, f)) {
            fprintf(stderr, "Error: Could not read dataset\n");
            exit(1);
        }

        fclose(f);
    }

    printf("Loaded dataset: %ld bytes\n", ds.length);
    return ds;
}

// --- Signal Handling ---
volatile sig_atomic_t keep_running = 1;
void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    keep_running = 0;
}

// --- Main Training Loop ---
int main(int argc, char *argv[]) {
    signal(SIGINT, handle_sigint);
    srand(time(NULL));

    printf("\n=== EGGROLL Optimized GPU Training ===\n");
    printf("Configuration:\n");
    printf("  Population Size: %d\n", POPULATION_SIZE);
    printf("  Hidden Dim: %d\n", HIDDEN_DIM);
    printf("  Layers: %d\n", N_LAYERS);
    printf("  Seq Len: %d\n", SEQ_LEN);
    printf("  Update Threshold: %d\n\n", UPDATE_THRESHOLD);

    // Load dataset
    Dataset ds = load_dataset("input.txt");

    // Initialize model
    EggModel model;
    init_model(&model);
    printf("Model initialized\n");

    // Training state
    int *pair_fitnesses = (int*)malloc((POPULATION_SIZE / 2) * sizeof(int));
    int *population_losses = (int*)malloc(POPULATION_SIZE * sizeof(int));

    if (!pair_fitnesses || !population_losses) {
        fprintf(stderr, "Error: Could not allocate fitness arrays\n");
        exit(1);
    }

    // Training loop
    const int max_steps = (ds.length / SEQ_LEN) - 10;
    long total_tokens = 0;
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int step = 0; step < max_steps && keep_running; step++) {
        long start_idx = step * SEQ_LEN;
        uint32_t step_seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);

        // Evaluate population
        bool gpu_success = evaluate_population_optimized(
            &model, &ds, start_idx, pair_fitnesses,
            step_seed, population_losses
        );

        if (!gpu_success) {
            fprintf(stderr, "GPU evaluation failed at step %d\n", step);
            break;
        }

        // Compute average loss
        int total_loss = 0;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            total_loss += population_losses[i];
        }
        double avg_loss = (double)total_loss / (POPULATION_SIZE * SEQ_LEN * (1 << FIXED_POINT));

        // Update weights
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t layer_seed = step_seed + (l * 100);

            // GRU weights
            for (int g = 0; g < 4; g++) {
                update_matrix(model.gru_weights[l][g], HIDDEN_DIM, HIDDEN_DIM,
                             layer_seed + g + 1, pair_fitnesses, POPULATION_SIZE / 2);
            }

            // GRU biases
            update_matrix(model.gru_biases[l][0], 1, HIDDEN_DIM,
                         layer_seed + 7, pair_fitnesses, POPULATION_SIZE / 2);
            update_matrix(model.gru_biases[l][1], 1, HIDDEN_DIM,
                         layer_seed + 8, pair_fitnesses, POPULATION_SIZE / 2);

            // MLP weights
            update_matrix(model.mlp_weights[l][0], HIDDEN_DIM * 4, HIDDEN_DIM,
                         layer_seed + 5, pair_fitnesses, POPULATION_SIZE / 2);
            update_matrix(model.mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM * 4,
                         layer_seed + 6, pair_fitnesses, POPULATION_SIZE / 2);

            // Layer norm weights
            update_matrix(model.ln_weights[l][0], 1, HIDDEN_DIM,
                         layer_seed + 9, pair_fitnesses, POPULATION_SIZE / 2);
            update_matrix(model.ln_weights[l][1], 1, HIDDEN_DIM,
                         layer_seed + 10, pair_fitnesses, POPULATION_SIZE / 2);
        }

        // Head
        update_matrix(model.head, VOCAB_SIZE, HIDDEN_DIM,
                     step_seed + 999, pair_fitnesses, POPULATION_SIZE / 2);

        // Output layer norm
        update_matrix(model.ln_out, 1, HIDDEN_DIM,
                     step_seed + 11, pair_fitnesses, POPULATION_SIZE / 2);

        // Print progress
        if (step % 10 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                            (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            total_tokens += POPULATION_SIZE * SEQ_LEN;

            printf("Step %d/%d | Loss: %.4f | Tok/s: %.2f\n",
                   step, max_steps, avg_loss, total_tokens / elapsed);
        }
    }

    // Cleanup
    printf("\nTraining completed. Cleaning up...\n");

#if defined(__APPLE__)
    egg_gpu_metal_optimized_shutdown();
#endif

    free(ds.data);
    free(pair_fitnesses);
    free(population_losses);

    return 0;
}