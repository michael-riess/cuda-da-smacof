#ifndef CUDA_SMACOF_RANDOM
#define CUDA_SMACOF_RANDOM

#include <stdlib.h>
#include <cuda_runtime.h>

void matrixRandomPopulate(float* matrix, int m, int s, int blocks, int threads);

void matrixRandomPopulateSerial(float* matrix, int m, int s);

#endif // ** CUDA_SMACOF_RANDOM **