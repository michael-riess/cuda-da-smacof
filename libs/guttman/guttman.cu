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


/* KERNEL: Calculate non-diagonal elements of the Guttman Transform Matrix
*/
__global__ void guttmanPart1(float *D, float *projD, float *GT_B, int dataRows){
    
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

        //non-diagonal elements
        if (i != j){
            unsigned int idx = i * dataRows + j;

            if (projD[idx] != 0.0 ){
                GT_B[idx] = -(D[idx]) / projD[idx];
                GT_B[j * dataRows + i] = GT_B[idx];

            }else{
                GT_B[idx] = 0.0f;
                GT_B[j * dataRows + i] = GT_B[idx];
            }
        }
    }
    
}


/* KERNEL: Calculate diagonal elements of the Guttman Transform Matrix
*/
__global__ void guttmanPart2(float *GT_B, int dataRows){

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < dataRows;
        idx += blockDim.x * gridDim.x
    ) {

        // diagonal elements
        int i = idx * dataRows + idx;
		GT_B[i] = 0.0f;
        for (int k = 0; k < dataRows; k++){
            if(idx != k) {                          
                GT_B[i] -= GT_B[idx * dataRows + k];
            }
        }
    }
}


/* KERNEL: Generate Moore-Penrose Inverse matrix
*/
__global__ void moorePenroseInverse(float* V, int dataRows) {
    for (unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        ix < (dataRows * (dataRows + 1) / 2);
        ix += blockDim.x * gridDim.x
    ) {
        float N = (float)dataRows;

        // generate 2D indeces from 1D index, ix, in flattened matrix.
        int i = ix / (dataRows + 1);
        int j = ix % (dataRows + 1);
        // if generated indeces lie outside of lower triangle, generate new ones inside it
        if (j > i) {
            i = dataRows - i - 1;
            j = dataRows - j;
        }

        if (j != i) {
            V[j * dataRows + i] = -1.0f/(N*N);
            V[i * dataRows + j] = V[j * dataRows + i];
        } else {
            V[j * dataRows + i] = (N-1.0f)/(N*N);
        }
    }
}


/* Compute the Guttman Transform
*  Y` <= V` * B * Y
*/
void computeGuttmanTransform(cublasHandle_t handle, float* Y, float* D, float* Delta, int m, int s, size_t size_Y, size_t size_D, int blocks, int threads) {
    float* cuda_D;
    float* cuda_Delta;
    float* cuda_B;
    float* cuda_B_prime;
    float* cuda_V;
    float* cuda_Y;
    float* cuda_Y_prime;

    // necessary scalar factors for cublasSgemm function
    float alpha = 1.0f;
    float beta = 0.0f;

    __cuda__( cudaMalloc(&cuda_D, size_D) );
    __cuda__( cudaMalloc(&cuda_B, size_D) );
    __cuda__( cudaMalloc(&cuda_Delta, size_D) );

    __cuda__( cudaMemcpy(cuda_D, D, size_D, cudaMemcpyHostToDevice) );
    __cuda__( cudaMemcpy(cuda_Delta, Delta, size_D, cudaMemcpyHostToDevice) );

    // generate the guttman transform matrix, B
    guttmanPart1<<<blocks, threads>>>(cuda_Delta, cuda_D, cuda_B, m);
    __cuda__( cudaPeekAtLastError() );
    __cuda__( cudaFree(cuda_D) );
    __cuda__( cudaFree(cuda_Delta) );
    guttmanPart2<<<blocks, threads>>>(cuda_B, m);
    __cuda__( cudaPeekAtLastError() );

    __cuda__( cudaMalloc(&cuda_V, size_D) );
    __cuda__( cudaMalloc(&cuda_B_prime, size_D) );

    // generate the moore-penrose inverse matrix, V`
    moorePenroseInverse<<<blocks, threads>>>(cuda_V, m);
    __cuda__( cudaPeekAtLastError() );


    /**** cuda_Y_prime <= cuda_V * cuda_B * cuda_Y ****
    * --------------------------------------------------
    * **** NOTE: **** 
    * Cublas expects column-major-order matrices, therefore row-major-ordered matrices (standard C) are viewed as
    * transposes. Additinally, we know that (A*B)^T == B^T * A^T. Thus we simply switch the order of matrices A
    * and B in the cublassgemm call and recieve the proper C value.
    * **** Cublas documention : ****
    * http://developer.download.nvidia.com/compute/cuda/2_0/docs/CUBLAS_Library_2.0.pdf
    * http://rpm.pbone.net/index.php3/stat/45/idpl/12463013/numer/3/nazwa/cublasSgemm
    * **** Explenation: ****
    * https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
    */

    // cuda_B_prime <= cuda_V * cuda_B   //   B` <= V` * B
    // multiply cuda_V and cuda_B e.g. the moore-penrose inverse matrix and the guttman transform matrix.
    cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, cuda_B, m, cuda_V, m, &beta, cuda_B_prime, m);

    __cuda__( cudaFree(cuda_B) );
    __cuda__( cudaFree(cuda_V) );

    __cuda__( cudaMalloc(&cuda_Y, size_Y) );
    __cuda__( cudaMalloc(&cuda_Y_prime, size_Y) );

    __cuda__( cudaMemcpy(cuda_Y, Y, size_Y, cudaMemcpyHostToDevice) );

    // cuda_Y_prime <= cuda_B_prime * cuda_Y    //   Y` <= B` * Y
    // multply cuda_B_prime and cuda_Y e.g. the result of the previouse matrix multiplication and the latest solution matrix
    cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, s, m, m, &alpha, cuda_Y, s, cuda_B_prime, m, &beta, cuda_Y_prime, s);

    // save new cuda_Y_prime matrix to Y
    __cuda__( cudaMemcpy(Y, cuda_Y_prime, size_Y, cudaMemcpyDeviceToHost) );

    __cuda__( cudaFree(cuda_B_prime) );
    __cuda__( cudaFree(cuda_Y) );
    __cuda__( cudaFree(cuda_Y_prime) );
}
