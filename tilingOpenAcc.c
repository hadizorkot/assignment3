#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#define TILE_SIZE 16

void matrixMultiplication(int* Input1, int* Input2, int* Output, int Height1, int Width1, int Width2) {
    #pragma acc data copyin(Input1[0:Height1*Width1], Input2[0:Width1*Width2]) copyout(Output[0:Height1*Width2])
    {
        #pragma acc kernels
        for (int x = 0; x < Height1; ++x) {
            for (int y = 0; y < Width2; ++y) {
                int answer = 0;
                for (int p = 0; p < (Width1 + TILE_SIZE - 1) / TILE_SIZE; ++p) {
                    #pragma acc loop tile(TILE_SIZE,TILE_SIZE)
                    for (int i = 0; i < TILE_SIZE; ++i) {
                        for (int j = 0; j < TILE_SIZE; ++j) {
                            int k = p * TILE_SIZE + j;
                            if (k < Width1) {
                                answer += Input1[x * Width1 + k] * Input2[k * Width2 + y];
                            }
                        }
                    }
                }
                Output[x * Width2 + y] = answer;
            }
        }
    }
}

int main() {
    int Height1, Width1, Width2;
    printf("Enter the dimensions of Input1 (Height1): ");
    scanf("%d", &Height1);
    printf("Enter the dimensions of Input2 (Width1 x Width2): ");
    scanf("%d %d", &Width1, &Width2);

    int *Input1, *Input2, *Output;
    size_t sizeA = Height1 * Width1 * sizeof(int);
    size_t sizeB = Width1 * Width2 * sizeof(int);
    size_t sizeC = Height1 * Width2 * sizeof(int);

    Input1 = (int *)malloc(sizeA);
    Input2 = (int *)malloc(sizeB);
    Output = (int *)malloc(sizeC);

    srand(time(NULL));
    for (int i = 0; i < Height1 * Width1; ++i) {
        Input1[i] = rand() % 10;
    }
    for (int i = 0; i < Width1 * Width2; ++i) {
        Input2[i] = rand() % 10;
    }

    double start_time = acc_get_time();
    matrixMultiplication(Input1, Input2, Output, Height1, Width1, Width2);
    double end_time = acc_get_time();

    printf("%f seconds\n", end_time - start_time);

    free(Input1);
    free(Input2);
    free(Output);

    return 0;
}