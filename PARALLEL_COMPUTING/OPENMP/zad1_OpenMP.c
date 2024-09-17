#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 6048576

int main() {
    float *B = malloc(N * sizeof(float));
    float *C = malloc(N * sizeof(float));
    float *A = malloc(N * sizeof(float));

    if (!B || !C || !A) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Tablica nasionek dla wektorów B i C
    unsigned int *seeds_B = malloc(N * sizeof(unsigned int));
    unsigned int *seeds_C = malloc(N * sizeof(unsigned int));

    if (!seeds_B || !seeds_C) {
        fprintf(stderr, "Memory allocation for seeds failed\n");
        free(B);
        free(C);
        free(A);
        return 1;
    }

    double start_time, end_time;

    // Start measuring time
    start_time = omp_get_wtime();

    // Initialize seeds for B and C with different values
    #pragma omp parallel default(none) shared(A,B,C,seeds_B,seeds_C)
    {
        int thread_id = omp_get_num_threads() + omp_get_thread_num();
        unsigned int local_seed = 1234 + thread_id; // Unique seed per thread
        // printf("Thread %d has local seed %u\n", thread_id, local_seed);


        // Assign unique seeds for B and C within each thread
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            seeds_B[i] = local_seed + i;
            seeds_C[i] = local_seed + i + N;
        }

        // Fill B and C with random numbers in [0, 1) using unique seeds
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            B[i] = (float)rand_r(&seeds_B[i]) / RAND_MAX;
            C[i] = (float)rand_r(&seeds_C[i]) / RAND_MAX;
        }
    

    // Compute A = B + C
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            A[i] = B[i] + C[i];
        }
    }
    // End measuring time
    end_time = omp_get_wtime();

    // Print the required values
    printf("A[0] = %f, B[0] = %f, C[0] = %f\n", A[0], B[0], C[0]);
    printf("A[%d] = %f, B[%d] = %f, C[%d] = %f\n", N-1, A[N-1], N-1, B[N-1], N-1, C[N-1]);

    // Print elapsed time
    printf("Elapsed time: %f seconds\n", end_time - start_time);

    // Clean up
    free(B);
    free(C);
    free(A);
    free(seeds_B);
    free(seeds_C);

    return 0;
}
