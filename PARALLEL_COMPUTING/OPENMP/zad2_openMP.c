#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 200000000

int main() {
    double *A = malloc(N * sizeof(double));

    if (!A) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Tablica seedow dla wektora A
    unsigned int *seeds_A = malloc(N * sizeof(unsigned int));
    if (!seeds_A) {
        fprintf(stderr, "Memory allocation for seeds failed\n");
        free(A);
        return 1;
    }

    double start_time, end_time;

    // Zmienna do przechowania sumy kwadratów
    double sum = 0.0;

    // Start measuring time
    start_time = omp_get_wtime();

    // Równolegle inicjalizowanie seedow i wypelnianie tablicy losowymi wartosciami
    #pragma omp parallel default(none) shared(A, seeds_A, sum)
    {
        int thread_id = omp_get_num_threads() + omp_get_thread_num();
        unsigned int local_seed = 1234 + thread_id;  // Unikalne seed dla kazdego watku

        // Przypisanie unikalnych seedow do wektora A w kazdym watku
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            seeds_A[i] = local_seed + i;
        }

        // Wypelnianie A losowymi liczbami w przedziale [0, 1) przy uzyciu unikalnych seedow
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            A[i] = (double)rand_r(&seeds_A[i]) / RAND_MAX;
        }

        // Równolegle obliczenie sumy kwadratów
        #pragma omp for schedule(guided) reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += A[i] * A[i];
        }
    }
    // Obliczenie normy drugiej
    double norma_dwa = sqrt(sum);

    // End measuring time
    end_time = omp_get_wtime();

    // Wypisanie wyniku
    printf("Norma druga wektora wynosi: %f\n", norma_dwa);

    // Wypisanie czasu dzialania programu
    printf("Elapsed time: %f seconds\n", end_time - start_time);

    // Zwolnienie pamieci
    free(A);
    free(seeds_A);

    return 0;
}

