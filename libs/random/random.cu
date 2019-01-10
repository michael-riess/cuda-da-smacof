#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>


// cuda macro for ensuring cuda errors are logged
#define __cuda__(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA-Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


/* KERNEL: Set up curand environment for populating matrix with pseudorandom values
*/
__global__ void cuda_rand_init(curandState *state, unsigned int size, int seed) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ){
        curand_init(seed, idx, 0, &state[idx]);
     }
}


/* KERNEL: Populate matrix with pseudorandom values
*/
__global__ void cuda_rand(curandState *state, float *matrix, unsigned int size) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ){
        matrix[idx] = curand_uniform(&state[idx]);
    }
}


/* Populate initial solution matrix with pseudorandom values between 0 and 10
*/
void matrixRandomPopulate(float* matrix, int m, int s, int blocks, int threads) {
    float* cuda_matrix;
    curandState* cuda_state;
    __cuda__( cudaMalloc(&cuda_matrix, m*s*sizeof(float)) );
    __cuda__( cudaMalloc(&cuda_state, m*s*sizeof(curandState)) );

    // initialize curand state with pseudorandom value for different initial pseudorandom solutions across executions
    srand(time(NULL));
    cuda_rand_init<<<blocks, threads>>>(cuda_state, m*s, (float)rand()/((float)RAND_MAX/10.0f));

    // populate initial solution matrix with pseudorandom values
    cuda_rand<<<blocks, threads>>>(cuda_state, cuda_matrix, m*s);
    __cuda__( cudaMemcpy(matrix, cuda_matrix, m*s*sizeof(float), cudaMemcpyDeviceToHost) );
    __cuda__( cudaFree(cuda_matrix) );
    __cuda__( cudaFree(cuda_state) );
}