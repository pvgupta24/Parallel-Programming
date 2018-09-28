/**
* Program to find sum at odd and even places in an array
*/

#include <iostream>
// OpenMP Compiler Directive
#include <omp.h>

static int array_size = 1024;

using namespace std;

int main()
{
    int *arr = new int[array_size];
    for (int i = 0; i < array_size; ++i)
        arr[i] = i;

    // Public variables available to all threads in process heap
    int alternate_sum[] = {0, 0};

// Makes a team of threads by forking other threads from the parent thread
// (which becomes 0th thread in the team)
#pragma omp parallel num_threads(2)
    {
        // Private variable in thread's stack
        int tid = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        int sum = 0;
        for (int i = tid; i < array_size; i += thread_count)
            sum += arr[i];
        alternate_sum[tid] = sum;
    }
    cout << "Sum at even places = " << alternate_sum[0] << endl;
    cout << "Sum at odd places = " << alternate_sum[1] << endl;
    return 0;
}
/*
g++ -fopenmp alternate-sum-in-array.cpp
./a.out
*/