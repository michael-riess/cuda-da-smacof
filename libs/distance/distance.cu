#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


// cuda macro for ensuring cuda errors are logged
#define __cuda__(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA-Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/* KERNEL: generate matrix, D, of euclidean distances between points in matrix "data"
*/
__global__ void euclideanDistance(float *data, float *D, int dataRows, int dataCols){ 
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

        float resta;
        float suma = 0.0f;

        for (int k = 0 ; k < dataCols; k++) {
            resta = data[i * dataCols + k] - data[j * dataCols + k];
            suma += resta * resta;
        }
                    
        suma = sqrt(suma);

        // ensure matrix is symetric
        D[i * dataRows + j] = suma;
        D[j * dataRows + i] = suma;
    }
}


/* Compute matrix of euclidean distances between points in matrix Y
*/
void computeEuclideanDistances(float* Y, float* D, int m, int s, size_t size_Y, size_t size_D, int blocks, int threads) {
    float* cuda_Y;
    float* cuda_D;

    __cuda__( cudaMalloc(&cuda_Y, size_Y) );
    __cuda__( cudaMalloc(&cuda_D, size_D) );

    __cuda__( cudaMemcpy(cuda_Y, Y, size_Y, cudaMemcpyHostToDevice) );
    __cuda__( cudaMemcpy(cuda_D, D, size_D, cudaMemcpyHostToDevice) );

    euclideanDistance<<<blocks, threads>>>(cuda_Y, cuda_D, m, s);
    __cuda__( cudaPeekAtLastError() );

    // copy updated distance matrix back to D
    __cuda__( cudaMemcpy(D, cuda_D, size_D, cudaMemcpyDeviceToHost) );
    
    // free gpu memory
    __cuda__( cudaFree(cuda_Y) );
    __cuda__( cudaFree(cuda_D) );
}