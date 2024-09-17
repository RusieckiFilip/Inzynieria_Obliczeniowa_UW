#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 512  // number of rows/columns
#define MAX_ITER 1000000 // max number of iterations    
#define EPS 0.00006 // error val for stop criterion

#define VAL_UPPPER 70.0 // upper side values
#define VAL_LOWER 1.0 // lower side values
#define VAL_LEFT 20.0 // left side values
#define VAL_RIGHT 50.0 // right side values

// Declare statically sized 2D arrays
double arr[N][N];
double old_arr[N][N];

void fillSides(double arr[N][N], int m, int n);
void fillRow(int n, double arr[N][N], int k);

int main()
{
    for (int i = 0; i < N; ++i) {
        fillRow(N, arr, i);
        fillRow(N, old_arr, i);
    } 

    fillSides(arr, N, N);
    fillSides(old_arr, N, N);

    double start, end;
    start = omp_get_wtime();

    for (int t = 0; t < MAX_ITER; ++t) {
        double temp = 0;
        
        switch (t % 2 == 0) {
        case 0:
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    arr[i][j] = (old_arr[i + 1][j] + old_arr[i - 1][j] + old_arr[i][j + 1] + old_arr[i][j - 1]) / 4;
                    temp += (arr[i][j] - old_arr[i][j]) * (arr[i][j] - old_arr[i][j]);
                }
            }
                break;

        case 1:
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    old_arr[i][j] = (arr[i + 1][j] + arr[i - 1][j] + arr[i][j + 1] + arr[i][j - 1]) / 4;
                    temp += (arr[i][j] - old_arr[i][j]) * (arr[i][j] - old_arr[i][j]);
                }
            }
                break;
        }
        
        temp = sqrt(temp / ((N - 2) * (N - 2)));
        if (t % 10000 == 0) {
            printf("\n current eps = %lf in %d iteration ", temp, t);
        }
        if (temp < EPS) {
            printf("\n Number of iterations = %d\n final eps = %lf ", t,temp);
            break;
        }
        // stopping criterion
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

void fillSides(double arr[N][N], int m, int n) {
    int k = n - 1;
    for (int i = 1; i < k; ++i) {
        arr[i][0] = VAL_LEFT;
    }
    for (int i = 1; i < k; ++i) {
        arr[i][k] = VAL_RIGHT;
    }
    k = m - 1;
    for (int j = 1; j < k; ++j) {
        arr[0][j] = VAL_UPPPER;
    }
    for (int j = 1; j < k; ++j) {
        arr[k][j] = VAL_LOWER;
    }
}
