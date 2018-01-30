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
#define ROW 40
#define COL 4000

//Kernel function
__global__ void add(int a[][COL], int b[][COL], int c[][COL]){
    //Skip till required block + the required thread index in the block
    uint x = blockDim.x * blockIdx.x + threadIdx.x;
    uint y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < ROW && y < COL)
        c[x][y] = a[x][y] + b[x][y];
}

int main(){

    int  a[ROW][COL], b[ROW][COL], c[ROW][COL]; //Host 2-d arrays
    int (*d_a)[COL], (*d_b)[COL], (*d_c)[COL]; //Device 2-d arrays

    clock_t start, end;
    cudaEvent_t d_start, d_end;
    catchCudaError(cudaEventCreate(&d_start));
    catchCudaError(cudaEventCreate(&d_end));

    size_t size = ROW*COL* sizeof(int);
    //Allocate device memory(double ptr as assigning value to a pointer as defined in CUDA API)
    catchCudaError(cudaMalloc((void **)&d_a, size));
    catchCudaError(cudaMalloc((void **)&d_b, size));
    catchCudaError(cudaMalloc((void **)&d_c, size));

    //Initial values of a,b random
    for(uint i=0; i < ROW; ++i){
        for(uint j=0; j < COL; ++j){
            a[i][j] = i+j;
            b[i][j] = i-j;
        }
    }

    //Copy to Device
    catchCudaError(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    catchCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    catchCudaError(cudaEventRecord(d_start));

    dim3 dimGrid(DIM, DIM);
    dim3 dimBlock(ceil(1.0*ROW/DIM), ceil(1.0*COL/DIM)) ;
    //Max 1024 threads in each block(max 65,535)
    add <<< dimGrid, dimBlock >>>(d_a, d_b, d_c);
    catchCudaError(cudaEventRecord(d_end));    
    
    //Copy to Host
    catchCudaError(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    //Wait for all threads to finish
    //catchCudaError(cudaDeviceSynchronize(d_end));

    //Waits till event is recorded
    catchCudaError(cudaEventSynchronize(d_end));

    start = clock();
    for(uint i=0; i<ROW; ++i)
        for(uint j=0; j<COL; ++j)
            if(a[i][j] + b[i][j] != c[i][j]){
                printf("Incorrect matrix addition (%d,%d)\n", i, j);
                exit(-3);
            }
    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;
    float d_time_taken;
    cudaEventElapsedTime(&d_time_taken, d_start, d_end);

    printf("Correct matrix addition\n");
    printf("Host time = %f ms\nDevice Time = %f ms\n", time_taken, d_time_taken);    
    //Free device memory
    catchCudaError(cudaFree(d_a));
    catchCudaError(cudaFree(d_b));
    catchCudaError(cudaFree(d_c));

}   

/*
Output
Correct matrix addition
Host time = 0.422000 ms
Device Time = 0.143072 ms
*/
