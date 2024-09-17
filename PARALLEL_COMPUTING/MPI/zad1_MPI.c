#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Funkcja do generowania losowej liczby zmiennoprzecinkowej z przedzialu [0,1]
double random_double() {
    return (double)rand() / RAND_MAX;
}

int main(int argc, char** argv) {
    int rank, size;
    const int n = 3048576;
    double *A_local, *B_local, *C_local;
    int local_n;
    double start_time, end_time, total_time;

    // Inicjalizacja MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Obliczenie ilosci elementów na proces
    local_n = n / size;

    // Alokacja pamieci dla lokalnych czesci wektorów
    A_local = (double*)malloc(local_n * sizeof(double));
    B_local = (double*)malloc(local_n * sizeof(double));
    C_local = (double*)malloc(local_n * sizeof(double));

    // Inicjalizacja generatora losowego
    srand(time(NULL) + rank);

     // Start pomiaru czasu
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Wypelnienie lokalnych wektorów B i C losowymi wartosciami
    for (int i = 0; i < local_n; i++) {
        B_local[i] = random_double();
        C_local[i] = random_double();
        A_local[i] = B_local[i] + C_local[i];
    }

    // Proces 0 wypisuje A[0], B[0], C[0]
    if (rank == 0) {
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
        printf("Czas wykonania operacji: %f sekund\n", total_time);
        printf("Proces %d: A[0] = %f, B[0] = %f, C[0] = %f\n", rank, A_local[0], B_local[0], C_local[0]);
        
        
    }

    // Proces o ranku size-1 wypisuje A[n-1], B[n-1], C[n-1]
    if (rank == size - 1) {
        printf("Proces %d: A[n-1] = %f, B[n-1] = %f, C[n-1] = %f\n", rank, A_local[local_n - 1], B_local[local_n - 1], C_local[local_n - 1]);
    }

    // Zwolnienie pamieci lokalnych wektorów
    free(A_local);
    free(B_local);
    free(C_local);

    // Zakonczenie MPI
    MPI_Finalize();

    return 0;
}
