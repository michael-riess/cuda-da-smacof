#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef CUDA_SMACOF_DISTANCE
#define CUDA_SMACOF_DISTANCE

void computeEuclideanDistances(float* Y, float* D, int m, int s, size_t size_Y, size_t size_D, int blocks, int threads);

void computeEuclideanDistancesSerial(float* Y, float* D, int m, int s);

#endif // ** UDA_SMACOF_DISTANCE **