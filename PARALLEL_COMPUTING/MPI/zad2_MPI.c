#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 200000000  // Dlugosc wektora

int main(int argc, char** argv) {
    int rank, size;
    double *local_A = NULL;
    double local_sum = 0.0, global_sum = 0.0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Obliczanie lokalnej dlugosci tablicy
    int local_n = N / size;

    // Przydzielanie pamieci dla lokalnej tablicy
    local_A = malloc(local_n * sizeof(double));
    if (!local_A) {
        fprintf(stderr, "Memory allocation for local array failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Rozpocznij pomiar czasu
    start_time = MPI_Wtime();
    
    // Inicjalizacja seeda dla kazdego procesu
    unsigned int seed = 1234 + rank * 1000;  // Unikalny seed na podstawie ranku
    srand(seed);  // Inicjalizacja generatora liczb losowych

    // Wypelnianie lokalnej tablicy losowymi wartosciami
    for (int i = 0; i < local_n; i++) {
        local_A[i] = (double)rand() / RAND_MAX;
    }


    // Oblicz lokalna sume kwadratów
    for (int i = 0; i < local_n; i++) {
        local_sum += local_A[i] * local_A[i];
    }

    // Redukcja lokalnych sum kwadratów do sumy globalnej
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Obliczanie i wypisywanie normy drugiej tylko na procesie 0
    if (rank == 0) {
        double norma_dwa = sqrt(global_sum);
        end_time = MPI_Wtime();

        // Wypisanie wyniku
        printf("Norma druga wektora wynosi: %f\n", norma_dwa);
        printf("Elapsed time: %f seconds\n", end_time - start_time);
    }

    // Zwolnienie pamieci
    free(local_A);

    MPI_Finalize();
    return 0;
}
