#define _POSIX_C_SOURCE 200809L
#define BILLION 1000000000L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>

#include "distance.h"
#include "guttman.h"
#include "matrix-read.h"
#include "temperature.h"
#include "stress.h"
#include "analysis.h"
#include "random.h"


// Host code
int main(int argc, char** argv) {

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    int blocks;     //number of blocks
    int threads;    //number of threads per block
    
    int m;          // number of items / objects; aka 'N'
    int n;          // dimensions of high-dimensional space;
    int s;          // dimension of low-dimensional space; aka 'L'
    double epsilon; // threshhold for the stress variance; aka 'ε'
    int k_max;      // maximum number of iterations; aka 'MAX'
    float temp_min;
    float alpha;
    int iterations;
    bool track_median = false;

    float* matrix;

    // validate arguments
    if(argc > 11) {
        fprintf(stderr, "\nToo Many Arguments\n");
        return 1;
    } else if(argc < 10) {
        fprintf(stderr, "\nToo Few Arguments\n");
        return 1;
    } else if (argc == 11) {
        track_median = (strncmp(argv[10], "median", 5) == 0) ? true : false;
    }

    blocks = atoi(argv[2]);
    threads = atoi(argv[3]);
    s = atoi(argv[4]);
    epsilon = strtof(argv[5], NULL);
    k_max = atoi(argv[6]);
    temp_min = strtof(argv[7], NULL);
    alpha = strtof(argv[8], NULL);
    iterations = atoi(argv[9]);

    // read in matrix from file
    readMatrix(argv[1], &matrix, &m, &n);


    //fprintf(stderr, "\nM: %i, N: %i\nBlocks: %i, Threads: %i", m, n, blocks, threads);

    size_t size_D = m*m*sizeof(float);     // total size in memeory of dissimilarity & distance arrays
    size_t size_Y = m*s*sizeof(float);     // total size in memory of low-dimensional array;

    float* Delta = (float*)malloc(size_D);      // pointer to flattened MxM dissimilarity matrix; aka 'Δ' aka 'D'
    float* Delta_prime = (float*)malloc(size_D);// pointer to temperature based dissimilarity matrix; aka '⧊' aka 'delta hat'
    float* Y = (float*)malloc(size_Y);          // MxS set of finding points in the low-dimensional space
    float* D = (float*)malloc(size_D);          // MxM matrix of euclidean distance in target-dimensional space; aka 'projD'

    float* Y_med;
    struct stress* Stresses;
    if (track_median) {
        Stresses = (struct stress*)malloc(iterations*sizeof(struct stress));
        Y_med = (float*)malloc(size_Y*iterations);
        for (int i = 0; i < (m*s*iterations); i++) {
            Y_med[i] = 0.0;
        }
    }

    // Generate Dissimilarity matrix Delta from matrix
    computeEuclideanDistances(matrix, Delta, m, n, m*n*sizeof(float), size_D, blocks, threads);

    double total_stress = 0.0;
    double max_stress = 0.0;
    double min_stress = DBL_MAX;
    unsigned long total_time = 0;
    struct timespec* timer;

    for(int iter = 0; iter < iterations; iter++) {
        
        timer = startTimer();

        // create initial random solution Y^[0]
        matrixRandomPopulate(Y, m, s, blocks, threads);

        // compute first distance matrix from random Y^[0]
        computeEuclideanDistances(Y, D, m, s, size_Y, size_D, blocks, threads);

        
        float temp = computeTemperature(Delta, size_D, m, s);

        int k;          // current interation
        double error;   // error value to determine if close enough approximation in lower dimensional space

        double prev_stress;
        double stress;

        bool moreTemps = true;

        while(moreTemps) {
            k = 0;          // current interation
            error = 1.0f;   // error value to determine if close enough approximation in lower dimensional space
        
            prev_stress = 0.0f;
            stress = 0.0f;


            if (temp > 0) {
                computeNewDissimilarity(Delta, Delta_prime, size_D, temp, m, s, blocks, threads);
            } else {
                // compute first distance matrix from random Y^[0]
                computeEuclideanDistances(Y, D, m, s, size_Y, size_D, blocks, threads);
            }

            // SMACOF
            while(k < k_max && error > epsilon) {

                if (temp > 0) {
                    // perform guttman transform
                    computeGuttmanTransform(handle, Y, D, Delta_prime, m, s, size_Y, size_D, blocks, threads);
                    computeEuclideanDistances(Y, D, m, s, size_Y, size_D, blocks, threads);

                    //calculate STRESS
                    stress = computeStress(Delta_prime, D, size_D, m, blocks, threads);
                
                } else {
                    // perform guttman transform
                    computeGuttmanTransform(handle, Y, D, Delta, m, s, size_Y, size_D, blocks, threads);
                    computeEuclideanDistances(Y, D, m, s, size_Y, size_D, blocks, threads);

                    //calculate STRESS
                    stress = computeStress(Delta, D, size_D, m, blocks, threads);
                }

                // update error and prev_stress values
                error = fabs(stress - prev_stress);
                prev_stress = stress;
                stress = 0.0f;

                k += 1;
            }

            // quit after running with temp of 0
            if(temp == 0.0) {
                moreTemps = false;

            // run once more with temp of 0 once reached temp_min
            } else if(temp < temp_min || (temp * alpha) < temp_min) {
                temp = 0.0;
            
            // reduce temp by alpha and run smacof again
            } else {
                temp *= alpha;
            }
        }

        // end time
        total_time += stopTimer(timer);

        // compute normalized stress for comparing mapping quality
        stress = computeNormalizedStressSerial(Delta, D, m);

        // sum stress values for computing average stress
        total_stress += stress;

        // maintain maximum stress
        if(stress > max_stress) {
            max_stress = stress;
        }

        //maintain minimum stress
        if(stress < min_stress) {
            min_stress = stress;
        }

        // if tracking median results, 
        if (track_median) {
            for (int i = 0; i < (m*s); i++) {
                Y_med[(m*s*iter)+i] = Y[i];
            }
            Stresses[iter].value = stress;
            Stresses[iter].index = iter;
        }
    }

    // print results after 'iterations' number of test
    printf("\nAVG_TIME: %lf\nAVG_STRESS: %0.8lf\nMAX_STRESS: %0.8lf\nMIN_STRESS: %0.8lf\n", (double)(((long double)total_time/(long double)iterations)/(long double)BILLION), (total_stress/((double)iterations)), max_stress, min_stress);


    // if median is being tracked, print median results including median solution
    if (track_median) {
        struct stress* med = median(Stresses, iterations);
        printf("MEDIAN_STRESS: %0.8lf\nMEDIAN_SOLUTION: [\n", med->value);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < s; j++) {
                printf("%f", Y_med[(med->index*m*s)+(i*s)+j]);
                if (j != s-1) {
                    printf(" ");
                }
            }
            printf("\n");
        }
        printf("]\n");
        free(Y_med);
        free(Stresses);
    }

    free(matrix);
    free(Delta);
    free(Y);
    free(D);
}