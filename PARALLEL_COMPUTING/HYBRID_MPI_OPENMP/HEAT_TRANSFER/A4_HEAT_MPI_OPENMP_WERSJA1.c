#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define N 1024  // liczba wierszy/kolumn
#define MAX_ITER 1000000 // maksymalna liczba iteracji
#define EPS 0.00006 // kryterium bledu dla warunku stopu

#define VAL_UPPPER 70.0 // górna wartosc brzegowa
#define VAL_LOWER 1.0 // dolna wartosc brzegowa
#define VAL_LEFT 20.0 // lewa wartosc brzegowa
#define VAL_RIGHT 50.0 // prawa wartosc brzegowa

void fillSides(double arr[][N], int local_n, int rank, int size);
void exchangeBoundaries(double local_arr[][N], int local_n, int rank, int size, MPI_Comm comm);
double calculateJacobi(double local_old[][N], double local_new[][N], int local_n);
void save2D(double arr[][N], int m, int n, const char *filename);

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    // Obliczanie liczby wierszy dla kazdego procesu
    int base_rows = N / size;  // Podstawowa liczba wierszy
    int extra_row = (rank == 0 || rank == size - 1) ? 1 : 2;  // Rank 0 i ostatni maja 1 marginesowy, reszta po 2.
    int local_n = base_rows + extra_row;

    // Sprawdzenie, czy local_n nie przekracza rozmiaru tablicy
    if (local_n > N + 2) {
        printf("Error: local_n exceeds the array bounds for process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Tworzenie lokalnych tablic dla kazdego procesu
    double local_arr_0[local_n][N];  // Tablica lokalna nr 0
    double local_arr_1[local_n][N];  // Tablica lokalna nr 1

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD); // Synchronizacja procesów przed rozpoczeciem pomiaru czasu
    start_time = MPI_Wtime();

    // Inicjalizacja tablic na 0
    for (int i = 0; i < local_n; ++i) {
        for (int j = 0; j < N; ++j) {
            local_arr_0[i][j] = 0.0;
            local_arr_1[i][j] = 0.0;
        }
    }

    // Wypelnienie tablic warunkami brzegowymi
    fillSides(local_arr_0, local_n, rank, size);
    fillSides(local_arr_1, local_n, rank, size);

    double global_error = 1.0;
    for (int iter = 0; iter < MAX_ITER && global_error > EPS; ++iter) {
        double local_error = 0.0;

        switch (iter % 2) {
            case 0:
                // Wymiana danych z sasiednimi procesami (dla local_arr_0)
                exchangeBoundaries(local_arr_0, local_n, rank, size, MPI_COMM_WORLD);
                // Obliczenia Jacobiego (z local_arr_0 do local_arr_1) przed wymiana granic
                local_error = calculateJacobi(local_arr_0, local_arr_1, local_n);
                break;
            case 1:
                // Wymiana danych z sasiednimi procesami (dla local_arr_1)
                exchangeBoundaries(local_arr_1, local_n, rank, size, MPI_COMM_WORLD);
                // Obliczenia Jacobiego (z local_arr_1 do local_arr_0) przed wymiana granic
                local_error = calculateJacobi(local_arr_1, local_arr_0, local_n);
                break;
        }

        // Redukcja globalna dla sprawdzenia konwergencji
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_error = sqrt(global_error / ((N-2) * (N-2)));

        // Wyswietlanie co 10000 iteracji
        if (rank == 0 && iter % 10000 == 0) {
            printf("Iteration %d, global error: %lf\n", iter, global_error);
        }

        // Sprawdzenie warunku stopu
        if (global_error < EPS) {
            if (rank == 0) {
                printf("Converged after %d iterations, final error: %lf\n", iter, global_error);
            }
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronizacja procesów po zakonczeniu iteracji
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total execution time: %lf seconds\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}

// Ustawianie wartosci brzegowych dla kazdego procesu
void fillSides(double arr[][N], int local_n, int rank, int size) {
    // Wypelnianie bocznych wartosci brzegowych
    for (int i = 0; i < local_n; ++i) {
        arr[i][0] = VAL_LEFT;
        arr[i][N - 1] = VAL_RIGHT;
    }

    // Proces 0 ustawia górna krawedz
    if (rank == 0) {
        for (int j = 0; j < N; ++j) {
            arr[0][j] = VAL_UPPPER;
        }
    }

    // Ostatni proces ustawia dolna krawedz
    if (rank == size - 1) {
        for (int j = 0; j < N; ++j) {
            arr[local_n - 1][j] = VAL_LOWER;
        }
    }
}

// Wymiana granicznych wierszy miedzy sasiednimi procesami z nieblokujaca komunikacja
void exchangeBoundaries(double local_arr[][N], int local_n, int rank, int size, MPI_Comm comm) {
    MPI_Request requests[4];
    int req_count = 0;

    // Asynchroniczna komunikacja z poprzednim procesem
    if (rank > 0) {
        MPI_Irecv(local_arr[0], N, MPI_DOUBLE, rank - 1, 0, comm, &requests[req_count++]);
        MPI_Isend(local_arr[1], N, MPI_DOUBLE, rank - 1, 0, comm, &requests[req_count++]);
    }

    // Asynchroniczna komunikacja z nastepnym procesem
    if (rank < size - 1) {
        MPI_Irecv(local_arr[local_n - 1], N, MPI_DOUBLE, rank + 1, 0, comm, &requests[req_count++]);
        MPI_Isend(local_arr[local_n - 2], N, MPI_DOUBLE, rank + 1, 0, comm, &requests[req_count++]);
    }

    // Oczekiwanie na zakonczenie wszystkich operacji
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

// Obliczanie jednej iteracji metody Jacobiego
double calculateJacobi(double local_old[][N], double local_new[][N], int local_n) {
    double local_error = 0.0;

    #pragma omp parallel default(none) shared(local_new, local_old, temp)
    {
        #pragma omp for collapse(2) reduction(+:local_error)
        for (int i = 1; i < local_n - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                local_new[i][j] = 0.25 * (local_old[i-1][j] + local_old[i+1][j] + local_old[i][j-1] + local_old[i][j+1]);
                local_error += (local_new[i][j] - local_old[i][j]) * (local_new[i][j] - local_old[i][j]);
            }
        }

        return local_error;
    }
}
