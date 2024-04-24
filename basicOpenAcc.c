:#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplication(int* Input1, int* Input2, int* Output, int Height1, int Width1, int Width2) {
    #pragma acc parallel loop collapse(2) copyin(Input1[0:Height1*Width1], Input2[0:Width1*Width2]) copyout(Output[0:Height1*Width2])
    for (int x = 0; x < Height1; x++) {
        for (int y = 0; y < Width2; y++) {
            int Pvalue = 0;
            for (int k = 0; k < Width1; k++) {
                Pvalue += Input1[x * Width1 + k] * Input2[k * Width2 + y];
            }
            Output[x * Width2 + y] = Pvalue;
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

    Input1 = (int*)malloc(sizeA);
    Input2 = (int*)malloc(sizeB);
    Output = (int*)malloc(sizeC);

    srand(time(NULL));
    for (int i = 0; i < Height1 * Width1; ++i) {
        Input1[i] = rand() % 10;
    }
    for (int i = 0; i < Width1 * Width2; ++i) {
        Input2[i] = rand() % 10;
    }

    // Using OpenACC to manage memory and computation
    #pragma acc data copyin(Input1[0:Height1*Width1], Input2[0:Width1*Width2]) copyout(Output[0:Height1*Width2])
    {
        double start_time = omp_get_wtime();
        matrixMultiplication(Input1, Input2, Output, Height1, Width1, Width2);
        double time_taken = omp_get_wtime() - start_time;
        printf("Time taken: %f seconds\n", time_taken);
    }

    free(Input1);
    free(Input2);
    free(Output);
   
    return 0;
}