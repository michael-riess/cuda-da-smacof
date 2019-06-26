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
__global__ void generateStressAddends(float* Delta, float* D, double* addends, double weight, int dataRows) {
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
        addends[ix] = (weight*(n*n));
    }
}


/* Compute stress with the aid of the gpu
*/
double computeStress(float* Delta, float* D, size_t size_D, double weight, int m, int blocks, int threads){
    size_t lowerTriangleSize = ((m * (m + 1)) / 2);

    float* cuda_Delta;
    float* cuda_D;

    // create array of stress addends
    double* cuda_stressAddends;
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );
    __cuda__( cudaMalloc(&cuda_D, size_D) );
    __cuda__( cudaMalloc(&cuda_stressAddends, (lowerTriangleSize * sizeof(double))) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );
    __cuda__( cudaMemcpy(cuda_D, D, size_D, cudaMemcpyHostToDevice) );
    generateStressAddends<<<blocks, threads>>>(cuda_Delta, cuda_D, cuda_stressAddends, weight, m);
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


/* KERNEL: Populate the addends array with the set of normalized stress weight denominator values
*/
__global__ void generateNormalizedStressDenominatorAddends(float* Delta, double* addends, int dataRows) {
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

        // generate and insert stress weight denominator addend into array for later summation
        if (i != j) {
            addends[ix] = (double)Delta[(i * dataRows) + j] * (double)Delta[(i * dataRows) + j]; //use doubles to preserve precision   
        } else {
            addends[ix] = 0.0f;
        }
    }
}


/* Computes normalized stress with the aid of the gpu
*/
double computeNormalizedStress(float* Delta, float* D, size_t size_D, int m, int blocks, int threads) {
    size_t lowerTriangleSize = ((m * (m + 1)) / 2);

    float* cuda_Delta;
    float* cuda_D;

    // create array of normalized stress denominator addends
    double* cuda_denominatorAddends;
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );
    __cuda__( cudaMalloc(&cuda_denominatorAddends, (lowerTriangleSize * sizeof(double))) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );
    generateNormalizedStressDenominatorAddends<<<blocks, threads>>>(cuda_Delta, cuda_denominatorAddends, m);
    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaDeviceSynchronize() );

    //sum reduction on all normalized stress weight denominator addends
    thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(cuda_denominatorAddends);
    __cuda__( cudaPeekAtLastError() );
    double weight = 1.0f / thrust::reduce(d_ptr, (d_ptr + lowerTriangleSize));
    __cuda__( cudaDeviceSynchronize() );
    __cuda__( cudaFree(cuda_denominatorAddends) );

    // create array of normalized stress addends
    double* cuda_stressAddends;
    __cuda__( cudaMalloc(&cuda_D, size_D) );
    __cuda__( cudaMalloc(&cuda_stressAddends, (lowerTriangleSize * sizeof(double))) );
    __cuda__( cudaMemcpy(cuda_D, D, size_D, cudaMemcpyHostToDevice) );
    generateStressAddends<<<blocks, threads>>>(cuda_Delta, cuda_D, cuda_stressAddends, weight, m);
    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaDeviceSynchronize() );
    __cuda__( cudaFree(cuda_Delta) );
    __cuda__( cudaFree(cuda_D) );

    //sum reduction on all normalized stress addends
    d_ptr = thrust::device_pointer_cast(cuda_stressAddends);
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
    weight = 1.0f/weight;
    for(int i = 0; i < m; i++) {
        for(int j = i+1; j < m; j++) {
            double n = (Delta[(i*m)+j] - D[(i*m)+j]);
            stress += (n*n);
        }
    }
    stress *= weight;
    return stress;
}