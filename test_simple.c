#include <stdio.h>
#include <stdlib.h>
#include "egg_config.h"

int main() {
    printf("Testing basic configuration:\n");
    printf("POPULATION_SIZE: %d\n", POPULATION_SIZE);
    printf("HIDDEN_DIM: %d\n", HIDDEN_DIM);
    printf("N_LAYERS: %d\n", N_LAYERS);
    printf("SEQ_LEN: %d\n", SEQ_LEN);
    printf("Test passed!\n");
    return 0;
}