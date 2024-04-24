#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 16

__global__ void matrixMultiplication(int* Input1, int* Input2, int* Output, int Height1, int Width1, int Width2) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < Height1 && y < Width2) {
        int Pvalue = 0;
        
        for (int k = 0; k < Width1; ++k) {
            Pvalue += Input1[x * Width1 + k] * Input2[k * Width2 + y];
        }
        
        Output[x * Width2 + y] = Pvalue;
    }
}

int main() {

    int  Height1 , Width1, Width2;
    
    printf("Enter the dimensions of Input1 (Height1): ");
    scanf("%d", &Height1);
    printf ("Enter the dimensions of Input2 (Width1 x Width2): ");
    
    scanf("%d %d", &Width1, &Width2);

  
  
    int* Input1, * Input2, * Output;
  
    int* dev_Input1, * dev_Input2, * dev_Output;
  
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
    
    cudaMalloc((void**)&dev_Input1, sizeA);
    cudaMalloc((void**)&dev_Input2, sizeB);
    
    cudaMalloc((void**)&dev_Output, sizeC);
    cudaMemcpy(dev_Input1, Input1, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Input2, Input2, sizeB, cudaMemcpyHostToDevice);
       
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((Width2 + blockDim.x - 1) / blockDim.x, (Height1AA + blockDim.y - 1) / blockDim.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplication<<<gridDim, blockDim>>>(dev_Input1, dev_Input2, dev_Output, Height1, Width1, Width2);

  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(Output, dev_Output, sizeC, cudaMemcpyDeviceToHost);

    printf("%f milliseconds", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(dev_Input1);
    cudaFree(dev_Input2);
    cudaFree(dev_Output);
    
    free(Input1);
    free(Input2);
    free(Output);
    
    return 0;
}