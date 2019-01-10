#ifndef CUDA_SMACOF_GUTTMAN
#define CUDA_SMACOF_GUTTMAN

#include <cuda_runtime.h>
#include "cublas_v2.h"

void computeGuttmanTransform(cublasHandle_t handle, float* Y, float* D, float* Delta, int m, int s, size_t size_Y, size_t size_D, int blocks, int threads);

#endif // ** UDA_SMACOF_GUTTMAN **