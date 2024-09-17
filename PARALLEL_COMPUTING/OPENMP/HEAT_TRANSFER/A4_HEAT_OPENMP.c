#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 1024  // liczba wierszy i kolumn
#define MAX_ITER 1000000 // maksymalna liczba iteracji
#define EPS 0.00006 // kryterium bledu do zatrzymania

#define VAL_UPPPER 70.0 // wartosci na górnej krawedzi
#define VAL_LOWER 1.0 // wartosci na dolnej krawedzi
#define VAL_LEFT 20.0 // wartosci na lewej krawedzi
#define VAL_RIGHT 50.0 // wartosci na prawej krawedzi

// Definicje statycznych tablic
double arr[N][N];
double old_arr[N][N];

void fillSides(double arr[N][N]);
void fillRow(int n, double arr[N][N], int k);
void copy_matrix(double arr[N][N], double old_arr[N][N]);

int main()
{
	int nthreads = omp_get_num_threads();
    	int id = omp_get_thread_num();
	double start, end;
	double temp = 1;
	start = omp_get_wtime();

	// Inicjalizacja tablic
	#pragma omp parallel default(none) shared(arr, old_arr, temp)
	{
		#pragma omp for schedule(guided)
		for (int i = 0; i < N; ++i) {
			fillRow(N, arr, i);
			fillRow(N, old_arr, i);
		}

		fillSides(arr);
		fillSides(old_arr);

		////////////////////////////////////////////////////////

		for (int t = 0; t < MAX_ITER && temp > EPS; ++t) {

			#pragma omp barrier
			#pragma omp single
			{
				temp = 0;
			}

			switch (t % 2 == 0) {
				case 0:
					#pragma omp for collapse(2) reduction(+:temp)
					for (int i = 1; i < N - 1; ++i) {
						for (int j = 1; j < N - 1; ++j) {
							arr[i][j] = (old_arr[i + 1][j] + old_arr[i - 1][j] + old_arr[i][j + 1] + old_arr[i][j - 1]) / 4;
							temp += (arr[i][j] - old_arr[i][j]) * (arr[i][j] - old_arr[i][j]);
						}
					}
					break;

				case 1:
					#pragma omp for collapse(2) reduction(+:temp)
					for (int i = 1; i < N - 1; ++i) {
						for (int j = 1; j < N - 1; ++j) {
							old_arr[i][j] = (arr[i + 1][j] + arr[i - 1][j] + arr[i][j + 1] + arr[i][j - 1]) / 4;
							temp += (arr[i][j] - old_arr[i][j]) * (arr[i][j] - old_arr[i][j]);
						}
					}
					break;
			}

			#pragma omp single
			{
				temp = sqrt(temp / ((N - 2) * (N - 2)));
				if (t % 10000 == 0) {
					printf("\n current eps = %lf in %d iteration ", temp, t);
				}
				if (temp < EPS) {
                    			printf("\n Number of iterations = %d\n final eps = %lf ", t,temp);
		        	}
			}
		}
	}

	end = omp_get_wtime();

	printf("Work took %lf seconds\n", end - start);

	return 0;
}

void fillRow(int n, double arr[N][N], int k) {
	for (int i = 0; i < n; ++i) {
		arr[k][i] = 0;
	}
}

void fillSides(double arr[N][N]) {
	int k = N - 1;

	for (int i = 1; i < k; ++i) {
		arr[i][0] = VAL_LEFT;
	}
	for (int i = 1; i < k; ++i) {
		arr[i][k] = VAL_RIGHT;
	}
	k = N - 1;
	for (int j = 1; j < k; ++j) {
		arr[0][j] = VAL_UPPPER;
	}
	for (int j = 1; j < k; ++j) {
		arr[k][j] = VAL_LOWER;
	}
}
