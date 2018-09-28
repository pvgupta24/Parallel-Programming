/**
* Program to calculate the value of pi
*/

#include <iostream>
// OpenMP Compiler Directive
#include <omp.h>

#define THREAD_COUNT 4

// Higher step_count => More accurate pi as complete blocks are added to
// approximate area under the curve
static long step_count = 1e8;

using namespace std;

int main()
{

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
            cout << "Launched " << thread_count << "Threads " << endl;
        double sum = 0;
        for (int i = tid; i < step_count; i += thread_count)
        {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1 + x * x);
        }

        // Mutual exclusion of variable pi
#pragma omp critical
        pi += sum * step;
    }

    cout << "Pi = " << pi << " " << endl;
    return 0;
}
/*
g++ -fopenmp pi.cpp
./a.out
*/