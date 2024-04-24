#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define BLOCK_SIZE 16

#define TILE_SIZE 16


__global__ void matrixMultiplication(int* Input1, int* Input2, int* Output, int Height1, int Width1, int Width2) {

  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  
    __shared__ int TILE1[TILE_SIZE][TILE_SIZE];
    __shared__ int TILE2[TILE_SIZE][TILE_SIZE];

  
    int pRow, pCol;
    int answer = 0;
  
    for (int p = 0; p < (Width1 + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        
        pRow = threadIdx.y;
        pCol = threadIdx.x;
        
        if (p * TILE_SIZE + pCol < Width1 && x < Height1){
    		TILE1[pRow][pCol] = Input1[x * Width1 + p * TILE_SIZE + pCol];
    	}
    	
        else{
            TILE1[pRow][pCol] = 0;
	}
	
        if (p * TILE_SIZE + pRow < Width1 && y < Width2){
            TILE2[pRow][pCol] = Input2[(p * TILE_SIZE + pRow) * Width2 + y];
        }
        
        else{
            TILE2[pRow][pCol] = 0;
	}
	
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            answer += TILE1[pRow][k] * TILE2[k][pCol];
        }

        __syncthreads();
        
    }
    
    if (x < Height1 && y < Width2) {
        Output[x * Width2 + y] = answer;
    }
}

int main() {
    
    int Height1, Width1, Width2;
    printf("Enter the dimensions of Input1 (Height1): ");
    scanf  ("%d", &Height1);
    printf ("Enter the dimensions of Input2 (Width1 x Width2): ");
    scanf("%d %d", &Width1, &Width2);

    int* Input1, * Input2, * Output;
    int* dev_Input1, * dev_Input2, * dev_Output;

    size_t sizeA = Height1 * Width1 * sizeof(int);
    size_t sizeB = Width1 * Width2 *sizeof(int);
    size_t sizeC = Height1 * Width2 *sizeof(int);  

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
    dim3 gridDim((Width2 + blockDim.x - 1) / blockDim.x, (Height1 + blockDim.y - 1) / blockDim.y);

  
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