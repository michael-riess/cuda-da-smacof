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
#include <vector>
#include <iostream>
#include <string>
#include <limits.h>

#include "distance.h"
#include "guttman.h"
#include "matrix-read.h"
#include "temperature.h"
#include "stress.h"
#include "analysis.h"
#include "random.h"


// Host code
int main(int argc, char** argv) {
    
    int m;          // number of items / objects; aka 'N'
    int n;          // dimensions of high-dimensional space;
    int s;          // dimension of low-dimensional space; aka 'L'
    double epsilon; // threshhold for the stress variance; aka 'ε'
    int k_max;      // maximum number of iterations; aka 'MAX'
    int iterations; // number of test runs for gathering average performance

    bool track_median;          // flag for tracking statistics from median solution
    bool track_median_solution; // flag for tracking median solution
    bool track_median_stresses; // flag for tracking stresses from median solution

    float* matrix;

    // validate arguments
    if(argc > 7) {
        fprintf(stderr, "\nToo Many Arguments\n");
        return 1;
    } else if(argc < 6) {
        fprintf(stderr, "\nToo Few Arguments\n");
        return 1;
    }
    if (argc > 6) {
        track_median = (strncmp(argv[6], "median", 6) == 0) ? true : false;
        track_median_solution = (strncmp(argv[6], "median_solution", 15) == 0) ? true : false;
        track_median_stresses = (strncmp(argv[6], "median_stresses", 15) == 0) ? true : false;
    }

    // parse arguments
    s = atoi(argv[2]);
    epsilon = strtof(argv[3], NULL);
    k_max = atoi(argv[4]);
    iterations = atoi(argv[5]);

    // read in matrix from file
    readMatrix(argv[1], &matrix, &m, &n);

    fprintf(stderr, "\nM: %i, N: %i\n", m, n);

    size_t size_D = m*m*sizeof(float);     // total size in memeory of dissimilarity & distance arrays
    size_t size_Y = m*s*sizeof(float);     // total size in memory of low-dimensional array;

    float* Delta = (float*)malloc(size_D);      // pointer to flattened MxM dissimilarity matrix; aka 'Δ' aka 'D'
    float* Delta_prime = (float*)malloc(size_D);// pointer to temperature based dissimilarity matrix; aka '⧊' aka 'delta hat'
    float* Y = (float*)malloc(size_Y);          // MxS set of finding points in the low-dimensional space
    float* D = (float*)malloc(size_D);          // MxM matrix of euclidean distance in target-dimensional space; aka 'projD'

    float* Y_med;
    struct stress* normalized_stresses;
    std::vector<struct stress> stresses[iterations];
    if (track_median) {
        normalized_stresses = (struct stress*)malloc(iterations*sizeof(struct stress));

        if (track_median_solution) {
            Y_med = (float*)malloc(size_Y*iterations);
            for (int i = 0; i < (m*s*iterations); i++) {
                Y_med[i] = 0.0;
            }
        }
    }

    // compute initial dissimiliary matrix
    computeEuclideanDistancesSerial(matrix, Delta, m, n);

    double total_stress = 0.0;
    double max_stress = 0.0;
    double min_stress = DBL_MAX;
    unsigned long total_time = 0;
    unsigned long max_time = 0;
    unsigned long min_time = ULONG_MAX;
    struct timespec* timer;


    for(int iter = 0; iter < iterations; iter++) {
        
        timer = startTimer();

        // create initial random solution Y^[0]
        matrixRandomPopulateSerial(Y, m, s);

        // compute first distance matrix from random Y^[0]
        computeEuclideanDistancesSerial(Y, D, m, s);
        
        int k = 0;          // current interation
        double error = 1.0f;// error value to determine if close enough approximation in lower dimensional space

        double prev_stress = 0.0f;
        double stress = 0.0f;

        while(k < k_max && error > epsilon) {

            // perform guttman transform
            computeGuttmanTransformSerial(&Y, D, Delta, m, s, size_Y, size_D);
            computeEuclideanDistancesSerial(Y, D, m, s);

            // calculate STRESS
            stress = computeNormalizedStressSerial(Delta, D, m);

            // update error and prev_stress values
            error = fabs(stress - prev_stress);
            prev_stress = stress;

            if (track_median_stresses) {
                stresses[iter].push_back((struct stress){stress, 0, stresses[iter].size()});
            }

            stress = 0.0f;

            k += 1;
        }

        // end time
        long int current_time = stopTimer(timer);
        total_time += current_time;


        if(current_time > max_time) {
            max_time = current_time;
        }

        if(current_time < min_time) {
            min_time = current_time;
        }

        // compute normalized stress for comparing mapping quality
        stress = computeNormalizedStressSerial(Delta, D, m);

        // sum stress values for computing average stress
        total_stress += stress;

        // maintain maximum stress
        if(stress > max_stress) {
            max_stress = stress;
        }

        // maintain minimum stress
        if(stress < min_stress) {
            min_stress = stress;
        }

        // if tracking median results, 
        if (track_median) {
            if(track_median_solution) {
                for (int i = 0; i < (m*s); i++) {
                    Y_med[(m*s*iter)+i] = Y[i];
                }
            }
            normalized_stresses[iter].value = stress;
            normalized_stresses[iter].index = iter;
            normalized_stresses[iter].time = current_time;
        }
    }


    // print time results
    printf("\nAVG_TIME: %0.8lf\n+MAX_TIME: %0.8lf\n-MIN_TIME: %0.8lf\n",
        (double)(((long double)total_time/(long double)iterations)/(long double)BILLION),   // average time
        (double)((long double)max_time/(long double)BILLION),                               // max time
        (double)((long double)min_time/(long double)BILLION)                                // min time
    );

    // print stress results
    printf("\nAVG_STRESS: %0.8lf\n+MAX_STRESS: %0.8lf\n-MIN_STRESS: %0.8lf\n",
        (total_stress/((double)iterations)),    // average stress     
        max_stress,                             // max stress
        min_stress                              // min stress
    );




    // if median is being tracked, print median results including median solution
    if (track_median) {
        struct stress* med = median(normalized_stresses, iterations);
        printf("MEDIAN_STRESS: %0.8lf\nMEDIAN_TIME: %0.8lf\n", med->value, (double)(((long double)med->time)/((long double)BILLION)));

        // if being tracked, print median solution
        if (track_median_solution) {
            printf("MEDIAN_SOLUTION: [\n");
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
        }


        if (track_median_stresses) {
            printf("MEDIAN_STRESSES: [\n");
            for (int i = 0; i < stresses[med->index].size(); i++) {
                printf("%i, %0.8lf\n", stresses[med->index][i].index, stresses[med->index][i].value);
            }
            printf("]\n");
        }

        free(normalized_stresses);
    }

    free(matrix);
    free(Delta);
    free(Y);
    free(D);
}