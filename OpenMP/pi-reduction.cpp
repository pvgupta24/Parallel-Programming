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
    double pi = 0.0;
    double step = 1.0 / step_count;

// The shared variable is replaced by intermediate variables over the threads
// and uses the specified operator for aggregating.
#pragma omp parallel for reduction(+ : pi)
    for (int i = 0; i < step_count; i++)
    {
        double x = (i + 0.5) * step;
        pi += 4.0 / (1 + x * x);
    }

    pi *= step;
    cout << "Pi = " << pi << endl;
}
/*
g++ -fopenmp pi.cpp
./a.out
*/