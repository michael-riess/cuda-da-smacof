#ifndef CUDA_SMACOF_TEMPERATURE
#define CUDA_SMACOF_TEMPERATURE

#include <stdlib.h>
#include <cuda_runtime.h>

void computeNewDissimilarity(float* Delta, float* Delta_prime, size_t size_D, float temp, int m, int s, int blocks, int threads);

float computeTemperature(float* Delta, size_t size_D, int m, int s);

#endif // ** UDA_SMACOF_TEMPURATURE **