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

#define TILE 16
#define N 256
//Kernel function
__global__ void multiply(float *a, float *b, float *c, 
                         int r_a,int c_a,int r_b,int c_b,int r_c,int c_c){

    __shared__ float s_a[TILE][TILE];
    __shared__ float s_b[TILE][TILE];
    //Skip till required block + the required thread index in the block
    uint row = blockDim.y * blockIdx.y + threadIdx.y;
    uint col = blockDim.x * blockIdx.x + threadIdx.x;
    //Transpose
    float cell = 0;
    s_a[threadIdx.y][threadIdx.x] = 0;
    s_b[threadIdx.y][threadIdx.x] = 0;

    for (uint k = 0; k < ((c_a + TILE -1)/ TILE); k++)
    {
        if (row < r_a && (threadIdx.x + (k*TILE)) < c_a)
            s_a[threadIdx.y][threadIdx.x] = a[(row*c_a) + threadIdx.x + (k*TILE)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0;
                   
        if (col < c_b && (threadIdx.y + k*TILE) < r_b)
            s_b[threadIdx.y][threadIdx.x] = b[(threadIdx.y + k*TILE)*c_b + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0;
                   
        __syncthreads();
        for (uint j = 0; j < TILE; ++j)
            cell += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        __syncthreads();        
    }
    if (row < r_c && col < c_c)
        c[row*c_c + col] += cell;
           
    //C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

int main(){

    //Host Arrays
    float *a, *b, *c;
    //Device Arrays
    float *d_a, *d_b, *d_c;

    //Set dimensions
    int r_a = N; 
    int c_a = N;    
    int r_b = N; 
    int c_b = N;
    int r_c = r_a; 
    int c_c = c_b;

    if(c_a != r_b){
        cout<< "Matrix dimensions wrong for multiplication"<<endl;
        exit(1);
    }

    clock_t start, end;
    cudaEvent_t d_start, d_end;
    catchCudaError(cudaEventCreate(&d_start));
    catchCudaError(cudaEventCreate(&d_end));


    size_t sizeA = r_a * c_a * sizeof(float);
    size_t sizeB = r_b * c_b * sizeof(float);
    size_t sizeC = r_c * c_c * sizeof(float);
    //Allocate host memory
    a = (float *)malloc(sizeA);
    b = (float *)malloc(sizeB);
    c = (float *)malloc(sizeC);

    //Allocate device memory(double ptr as assigning value to a pointer as defined in CUDA API)
    catchCudaError(cudaMalloc((void **)&d_a, sizeA));
    catchCudaError(cudaMalloc((void **)&d_b, sizeB));
    catchCudaError(cudaMalloc((void **)&d_c, sizeC));

    //Initial values of a,b random
    for(uint i=0; i < r_a; ++i){
        for(uint j=0; j < c_a; ++j){
            a[i * c_a + j] = i+j;
        }
    }
    
    for(uint i=0; i < r_b; ++i){
        for(uint j=0; j < c_b; ++j){
            b[i * c_b + j] = i-j;    
        }
    }
    

    //Copy to Device
    catchCudaError(cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice));
    catchCudaError(cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice));

    catchCudaError(cudaEventRecord(d_start));

    dim3 dimGrid(1+r_a/TILE, 1+c_b/TILE, 1);
    dim3 dimBlock(TILE, TILE, 1) ;
    //Max 1024 threads in each block(max 65,535)
    multiply <<< dimGrid, dimBlock >>>(d_a, d_b, d_c, r_a, c_a, r_b, c_b, r_c, c_c);
    catchCudaError(cudaEventRecord(d_end));    
    
    //Copy to Host
    catchCudaError(cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost));

    //Wait for all threads to finish
    //catchCudaError(cudaDeviceSynchronize());

    //Waits till event is recorded
    catchCudaError(cudaEventSynchronize(d_end));
    float cell;
    start = clock();
    for(uint i=0; i < r_c; ++i){
        for(uint j=0; j < c_c; ++j){
            cell = 0;        
            for(uint k=0; k < c_a; ++k)
                cell += a[i * c_a + k]*b[k * c_b + j];
        }
    }
    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;
    float d_time_taken;
    cudaEventElapsedTime(&d_time_taken, d_start, d_end);

    printf("Host time = %f ms\nDevice Time = %f ms\n", time_taken, d_time_taken);    
    //Free Host memory
    free(a);
    free(b);
    free(c);
    //Free device memory
    catchCudaError(cudaFree(d_a));
    catchCudaError(cudaFree(d_b));
    catchCudaError(cudaFree(d_c));

}   

/*
Output
Correct matrix multiplication
Host time = 76.949997 ms
Device Time = 0.398816 ms
*/
