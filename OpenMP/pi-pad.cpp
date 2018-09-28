/**
* Program to calculate the value of pi by padding to prevent false sharing
*/

#include <iostream>
// OpenMP Compiler Directive
#include <omp.h>

#define THREAD_COUNT 4
#define CACHE_PAD 8

// Higher step_count => More accurate pi as complete blocks are added to
// approximate area under the curve
static long step_count = 1e8;

using namespace std;

int main()
{

    auto partial_sums = new double[THREAD_COUNT][CACHE_PAD];
    double pi = 0.0;
    double step = 1.0 / step_count;

    omp_set_num_threads(THREAD_COUNT);

// Round Robin Distribution of rectangles between threads
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        // OpenMP may not be able to provide the requested number of threads
        if (tid == 0)
            cout << "Launched " << thread_count << " Threads " << endl;

        for (int i = tid; i < step_count; i += thread_count)
        {
            double x = (i + 0.5) * step;

            // Performance issues will be there because of invalidating the data
            // due to cache conflicts between multiple partial_sums[tid]
            // Leads to False Sharing => Poor scaling
            // Dirty fix : Padd(offset) by 8 to ensure no conflicts
            // on cache lines
            partial_sums[tid][0] += 4.0 / (1 + x * x);
        }
    }
    // Accumulating the partial sums to get total area under the curve
    for (int i = 0; i < THREAD_COUNT; ++i)
        pi += partial_sums[i][0] * step;

    cout << "Pi = " << pi << " " << endl;
    return 0;
}
/*
g++ -fopenmp pi.cpp
./a.out
*/