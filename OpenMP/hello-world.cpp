/**
* Program to print hello world with multiple threads
*/

#include <iostream>
// OpenMP Compiler Directive
#include <omp.h>

using namespace std;

int main()
{
// Public variables available to all threads in process heap

// Give me the default number of threads specified by the ENV VAR OMP_NUM_THREADS
#pragma omp parallel
    {
        // Private variable in thread's stack
        int tid = omp_get_thread_num();
        printf("Hello %d !!", tid);
        printf("World %d !!\n", tid);
    }
    return 0;
}
/*
export OMP_NUM_THREADS=4
echo $OMP_NUM_THREADS
g++ -fopenmp hello-world.cpp
./a.out
*/