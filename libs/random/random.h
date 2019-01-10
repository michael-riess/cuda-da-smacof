#ifndef CUDA_SMACOF_RANDOM
#define CUDA_SMACOF_RANDOM

#include <stdlib.h>
#include <cuda_runtime.h>

void matrixRandomPopulate(float* matrix, int m, int s, int blocks, int threads);

#endif // ** CUDA_SMACOF_RANDOM **