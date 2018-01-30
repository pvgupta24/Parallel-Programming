#include<stdio.h>
#include<iostream>
#include<cuda.h>
using namespace std;

//Catch Cuda errors
void catchCudaError(cudaError_t error){
    if(error!=cudaSuccess) {
        printf("\n====== Cuda Error Code %i ======\n %s\n",error,cudaGetErrorString(error)); 
        exit(-1); 
    }
}
//=====================================================================

#define DIM 32

#define ROW1 40
#define COMMON_WIDTH 30
#define COL2 4000

//Kernel function
__global__ void multiply(int a[][COMMON_WIDTH], int b[][COL2], int c[][COL2]){
    //Skip till required block + the required thread index in the block
    uint x = blockDim.x * blockIdx.x + threadIdx.x;
    uint y = blockDim.y * blockIdx.y + threadIdx.y;

    int cell = 0;
    if(x < ROW1 && y < COL2){
        for(uint i = 0; i < COMMON_WIDTH; ++i)
            cell += a[x][i]*b[i][y];
        c[x][y] = cell;//c has ROW1 x COL2 dim.
    }
}

int main(){

    int a[ROW1][COMMON_WIDTH], b[COMMON_WIDTH][COL2], c[ROW1][COL2]; //Host 2-d arrays
    int (*d_a)[COMMON_WIDTH], (*d_b)[COL2], (*d_c)[COL2]; //Device 2-d arrays

    clock_t start, end;
    cudaEvent_t d_start, d_end;
    catchCudaError(cudaEventCreate(&d_start));
    catchCudaError(cudaEventCreate(&d_end));


    size_t sizeA = ROW1*COMMON_WIDTH*sizeof(int);
    size_t sizeB = COMMON_WIDTH*COL2*sizeof(int);
    size_t sizeC = ROW1*COL2* sizeof(int);
    //Allocate device memory(double ptr as assigning value to a pointer as defined in CUDA API)
    catchCudaError(cudaMalloc((void **)&d_a, sizeA));
    catchCudaError(cudaMalloc((void **)&d_b, sizeB));
    catchCudaError(cudaMalloc((void **)&d_c, sizeC));

    //Initial values of a,b random
    for(uint i=0; i < ROW1; ++i)
        for(uint j=0; j < COMMON_WIDTH; ++j)
            a[i][j] = i+j;

    for(uint i=0; i < COMMON_WIDTH; ++i)
        for(uint j=0; j < COL2; ++j)
            b[i][j] = i-j;    
    

    //Copy to Device
    catchCudaError(cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice));
    catchCudaError(cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice));

    catchCudaError(cudaEventRecord(d_start));

    dim3 dimGrid(DIM, DIM);
    dim3 dimBlock(ceil(1.0*ROW1/DIM), ceil(1.0*COL2/DIM)) ;
    //Max 1024 threads in each block(max 65,535)
    multiply <<< dimGrid, dimBlock >>>(d_a, d_b, d_c);
    catchCudaError(cudaEventRecord(d_end));    
    
    //Copy to Host
    catchCudaError(cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost));

    //Wait for all threads to finish
    //catchCudaError(cudaDeviceSynchronize());

    //Waits till event is recorded
    catchCudaError(cudaEventSynchronize(d_end));
    int cell;
    start = clock();
    for(uint i=0; i<ROW1; ++i)
        for(uint j=0; j<COL2; ++j){
            cell = 0;        
            for(uint k=0; k<COMMON_WIDTH; ++k)
                cell += a[i][k]*b[k][j];

            if(cell != c[i][j]){
                printf("Incorrect Matrix Multiplication");
                exit(-3);
            }
        }
    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;
    float d_time_taken;
    cudaEventElapsedTime(&d_time_taken, d_start, d_end);

    printf("Correct matrix multiplication\n");
    printf("Host time = %f ms\nDevice Time = %f ms\n", time_taken, d_time_taken);    
    //Free device memory
    catchCudaError(cudaFree(d_a));
    catchCudaError(cudaFree(d_b));
    catchCudaError(cudaFree(d_c));

}   

/*
Output
Correct matrix multiplication
Host time = 11.943000 ms
Device Time = 0.252992 ms
*/
