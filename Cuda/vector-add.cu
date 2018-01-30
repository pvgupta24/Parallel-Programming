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

#define N 400000
#define MAX_THREAD 1024

//Kernel function
__global__ void add(int *a, int *b, int *c){
    //Skip till required block + the required thread index in the block
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id<N)
        c[id] = a[id] + b[id];
}

int main(){

    int  a[N], b[N], c[N]; //Host arrays
    int *d_a, *d_b, *d_c; //Device arrays
    clock_t start, end;
    cudaEvent_t d_start, d_end;
    catchCudaError(cudaEventCreate(&d_start));
    catchCudaError(cudaEventCreate(&d_end));

    //Allocate device memory(double ptr as assigning value to a pointer as defined in CUDA API)
    catchCudaError(cudaMalloc((void **)&d_a, N * sizeof(int)));
    catchCudaError(cudaMalloc((void **)&d_b, N * sizeof(int)));
    catchCudaError(cudaMalloc((void **)&d_c, N * sizeof(int)));

    //Initial values of a,b
    for(uint i=0; i<N; ++i){
        a[i] = i;
        b[i] = 2*i;
    }

    //Copy to Device
    catchCudaError(cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    catchCudaError(cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

    catchCudaError(cudaEventRecord(d_start));
    //Max 1024 threads in each block(max 65,535)
    add <<< ceil(1.0*N/MAX_THREAD), MAX_THREAD >>>(d_a, d_b, d_c);
    catchCudaError(cudaEventRecord(d_end));    
    
    //Copy to Host
    catchCudaError(cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));

    //Wait for all threads to finish
    //catchCudaError(cudaDeviceSynchronize(d_end));

    //Waits till event is recorded
    catchCudaError(cudaEventSynchronize(d_end));

    start = clock();
    for(uint i=0; i<N; ++i)
        if(a[i]+b[i] != c[i]){
            printf("Incorrect vector addition\n");
            exit(-3);
        }
    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;
    float d_time_taken;
    cudaEventElapsedTime(&d_time_taken, d_start, d_end);

    printf("Correct Vector addition\n");
    printf("Host time = %f ms\nDevice Time = %f ms\n", time_taken, d_time_taken);    
    //Free device memory
    catchCudaError(cudaFree(d_a));
    catchCudaError(cudaFree(d_b));
    catchCudaError(cudaFree(d_c));

}   

/*
Output
Correct Vector addition
Host time = 1.143000 ms
Device Time = 0.384800 ms
*/
