#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


// cuda macro for ensuring cuda errors are logged
#define __cuda__(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA-Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/* KERNEL: compute dissimilarity matrix based on annealing temperature
*  addend = temperature * sqrt(2 * s)
*/
__global__ void newDissimilarity(float* Delta_prime, int dataRows, float addend) {
    for (unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        ix < (dataRows * (dataRows + 1) / 2);
        ix += blockDim.x * gridDim.x
    ){
        // generate 2D indeces from 1D index, ix, in flattened matrix.
        int i = ix / (dataRows + 1);
        int j = ix % (dataRows + 1);
        // if generated indeces lie outside of lower triangle, generate new ones inside it
        if (j > i) {
            i = dataRows - i - 1;
            j = dataRows - j;
        }

        int idx = j * dataRows + i;
        if (Delta_prime[idx] > addend) {
            Delta_prime[idx] = (Delta_prime[idx] - addend); 
        } else {
            Delta_prime[idx] = 0.0;
        }
        Delta_prime[i * dataRows + j] = Delta_prime[idx];
    }  
}


/* Compute new dissimilarity matrix based on new annealing temperature
*  Employed by DA-SMACOF only
*/
void computeNewDissimilarity(float* Delta, float* Delta_prime, size_t size_D, float temp, int m, int s, int blocks, int threads) {
    float* cuda_Delta;
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );
    newDissimilarity<<<blocks, threads>>>(cuda_Delta, m, (float)(temp*sqrt((double)(2.0 * s))));
    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaMemcpy(Delta_prime, cuda_Delta, size_D, cudaMemcpyDeviceToHost) );
    __cuda__( cudaFree(cuda_Delta) );
}


/* Compute initial annealing temperature
*/
float computeTemperature(float* Delta, size_t size_D, int m, int s) {
    float* cuda_Delta;
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );

    // use thrust reduction to find maximum value in dissimilarity matrix
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(cuda_Delta);
    __cuda__( cudaPeekAtLastError() );
    float max = *(thrust::max_element(d_ptr, d_ptr + (m*m)));

    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaFree(cuda_Delta) );

    // return temperature s.t. at least one new dissimiliary value is > 0
    return (float)((double)(max * 0.90f))/(sqrt((double)(2.0 * s)));
}