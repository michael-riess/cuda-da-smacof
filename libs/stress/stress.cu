#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <stdbool.h>
#include <cuda_runtime.h>
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


/* KERNEL: Populate the addends array with the set of stress addends (values when summed will equal the stress value)
*/
__global__ void generateStressAddends(float* Delta, float* D, double* addends, int dataRows) {
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

        // generate and insert stress addend into array for later summation
        double n = 0.0f;
        if (i != j) {
            n = (double)Delta[(i * dataRows) + j] - (double)D[(i * dataRows) + j]; //use doubles to preserve precision   
        }
        addends[ix] = (n*n);
    }
}

/* Compute stress with the aid of the gpu
*/
double computeStress(float* Delta, float* D, size_t size_D, int m, int blocks, int threads){
    size_t lowerTriangleSize = ((m * (m + 1)) / 2);

    // create array of stress addends
    double* cuda_stressAddends;
    float* cuda_Delta;
    float* cuda_D;
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );
    __cuda__( cudaMalloc(&cuda_D, size_D) );
    __cuda__( cudaMalloc(&cuda_stressAddends, (lowerTriangleSize * sizeof(double))) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );
    __cuda__( cudaMemcpy(cuda_D, D, size_D, cudaMemcpyHostToDevice) );
    generateStressAddends<<<blocks, threads>>>(cuda_Delta, cuda_D, cuda_stressAddends, m);
    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaDeviceSynchronize() );
    __cuda__( cudaFree(cuda_Delta) );
    __cuda__( cudaFree(cuda_D) );

    //sum reduction on all stress addends
    thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(cuda_stressAddends);
    __cuda__( cudaPeekAtLastError() );
    double stress = thrust::reduce(d_ptr, (d_ptr + lowerTriangleSize));
    __cuda__( cudaDeviceSynchronize() );
    __cuda__( cudaFree(cuda_stressAddends) );

    return stress;
}

/* Computes normalized stress without the aid of the gpu.
*/
double computeNormalizedStressSerial(float* Delta, float* D, int m) {
    double stress = 0.0f;
    double weight = 0.0f;
    for(int i = 0; i < m; i++) {
        for(int j = i+1; j < m; j++) {
            weight += (Delta[(i*m)+j] * Delta[(i*m)+j]);
        }
    }
    weight = 1.0/weight;
    for(int i = 0; i < m; i++) {
        for(int j = i+1; j < m; j++) {
            double n = (Delta[(i*m)+j] - D[(i*m)+j]);
            stress += (n*n);
        }
    }
    stress *= weight;
    return stress;
}