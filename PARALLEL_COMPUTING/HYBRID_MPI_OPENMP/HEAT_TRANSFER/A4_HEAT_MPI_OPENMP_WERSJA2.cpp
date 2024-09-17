#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <omp.h>

#define N 1024  // liczba wierszy/kolumn
#define MAX_ITER 1000000 // maksymalna liczba iteracji
#define EPS 0.00006 // kryterium bledu dla warunku stopu

#define VAL_UPPPER 70.0 // górna wartosc brzegowa
#define VAL_LOWER 1.0 // dolna wartosc brzegowa
#define VAL_LEFT 20.0 // lewa wartosc brzegowa
#define VAL_RIGHT 50.0 // prawa wartosc brzegowa

void fillSides(double arr[][N], int local_n, int rank, int size);

int main(int argc, char *argv[])
{
    int rank, size, provided;

    // Inicjalizacja MPI z obsługą wątków
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    
    // Sprawdzanie, czy MPI obsługuje wymagany poziom wątków
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI does not provide MPI_THREAD_SERIALIZED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Obliczanie liczby wierszy dla kazdego procesu
    int base_rows = N / size;  // Podstawowa liczba wierszy
    int extra_row = (rank == 0 || rank == size - 1) ? 1 : 2;  // Rank 0 i ostatni maja 1 marginesowy, reszta po 2.
    int local_n = base_rows + extra_row;
    double local_error = 1.0;
    omp_set_num_threads(2);

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

    #pragma omp parallel default(none) shared(local_arr_0, local_arr_1, local_n, local_error, rank, size)
    {
        #pragma omp for collapse(2) schedule(guided)
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
            #pragma omp barrier
            #pragma omp single
            {
                local_error = 0.0;   
            }

            switch (iter % 2) {
                case 0:
                    // Wymiana danych z sasiednimi procesami (dla local_arr_0)
                    #pragma omp single
                    {
                        MPI_Request requests[4];
                        int req_count = 0;

                        // Asynchroniczna komunikacja z poprzednim procesem
                        if (rank > 0) {
                            MPI_Irecv(local_arr_0[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                            MPI_Isend(local_arr_0[1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                        }

                        // Asynchroniczna komunikacja z nastepnym procesem
                        if (rank < size - 1) {
                            MPI_Irecv(local_arr_0[local_n - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                            MPI_Isend(local_arr_0[local_n - 2], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                        }

                        // Oczekiwanie na zakonczenie wszystkich operacji
                        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
                    }

                    #pragma omp barrier

                    // Obliczenia Jacobiego (z local_arr_0 do local_arr_1)
                    #pragma omp for collapse(2) reduction(+:local_error)
                    for (int i = 1; i < local_n - 1; ++i) {
                        for (int j = 1; j < N - 1; ++j) {
                            local_arr_1[i][j] = 0.25 * (local_arr_0[i-1][j] + local_arr_0[i+1][j] + local_arr_0[i][j-1] + local_arr_0[i][j+1]);
                            local_error += (local_arr_1[i][j] - local_arr_0[i][j]) * (local_arr_1[i][j] - local_arr_0[i][j]);
                        }
                    }
                    break;
                case 1:
                    // Wymiana danych z sasiednimi procesami (dla local_arr_1)
                    #pragma omp single
                    {
                        MPI_Request requests[4];
                        int req_count = 0;

                        // Asynchroniczna komunikacja z poprzednim procesem
                        if (rank > 0) {
                            MPI_Irecv(local_arr_1[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                            MPI_Isend(local_arr_1[1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                        }

                        // Asynchroniczna komunikacja z nastepnym procesem
                        if (rank < size - 1) {
                            MPI_Irecv(local_arr_1[local_n - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                            MPI_Isend(local_arr_1[local_n - 2], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
                        }

                        // Oczekiwanie na zakonczenie wszystkich operacji
                        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
                    }
                    #pragma omp barrier

                    // Obliczenia Jacobiego (z local_arr_1 do local_arr_0)
                    #pragma omp for collapse(2) reduction(+:local_error)
                    for (int i = 1; i < local_n - 1; ++i) {
                        for (int j = 1; j < N - 1; ++j) {
                            local_arr_0[i][j] = 0.25 * (local_arr_1[i-1][j] + local_arr_1[i+1][j] + local_arr_1[i][j-1] + local_arr_1[i][j+1]);
                            local_error += (local_arr_0[i][j] - local_arr_1[i][j]) * (local_arr_0[i][j] - local_arr_1[i][j]);
                        }
                    }
                    break;
            }

            // Redukcja globalna dla sprawdzenia konwergencji
            #pragma omp single
            {
                MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                global_error = sqrt(global_error / ((N-2) * (N-2)));
            }
            
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
