#ifndef CUDA_SMACOF_STRESS
#define CUDA_SMACOF_STRESS

#include <stdlib.h>
#include <cuda_runtime.h>

double computeStress(float* Delta, float* D, size_t size_D, double weight, int m, int blocks, int threads);

double computeNormalizedStress(float* Delta, float* D, size_t size_D, int m, int blocks, int threads);

double computeNormalizedStressSerial(float* Delta, float* D, int m);

#endif // ** CUDA_SMACOF_STRESS **