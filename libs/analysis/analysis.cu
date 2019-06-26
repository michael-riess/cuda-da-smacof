#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "analysis.h"


/* Function for comparing stress values.
*  Employed by qsort in "median" function.
*  Returns 0 if values are the same, 1 if the first is greater than the second,
*   and -1 if the firt value is smaller than the second.
*/
int compare(const void* a, const void* b) {
    struct stress _a = *((struct stress*)a);
    struct stress _b = *((struct stress*)b);
    if (_a.value == _b.value) {
        return 0;
    } else if (_a.value < _b.value) {
        return -1;
    } else {
        return 1;
    }
}


/* Function for determining the median stress value from an array of stresses.
*  A struct containing both the median stress and its original index is returned
*   in order to facilitate the retrieval of other associated data.
*/
struct stress* median(struct stress* array, unsigned int size) {
    qsort(array, size, sizeof(struct stress), compare);
    return &(array[size/2]);
}


/* Starts timer for evalutating performance
*  Total time calculated includes that taken by cpu bound execution, and that taken by gpu where cpu is idle.
*  Every call to "startTimer" should be followed by a corresponding call to "stopTimer"
*   which handles the freeing of memory allocated in "startTimer".
*/
struct timespec* startTimer() {
    struct timespec* start = (struct timespec*)malloc(sizeof(struct timespec));
    clock_gettime(CLOCK_MONOTONIC, start);
    return start;
}


/* Stops timer for evaluating performance and returns total time from start of timer till its end.
*  Total time calculated includes that taken by cpu bound execution, and that taken by gpu where cpu is idle.
*  Every call to "startTimer" should be followed by a corresponding call to "stopTimer"
*   which handles the freeing of memory allocated in "startTimer".
*/
long int stopTimer(struct timespec* start) {
    struct timespec end;
    long int time = 0;

    // ensure timer start struct exists before attempting to access it
    if (start) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = ((BILLION * (end.tv_sec - start->tv_sec)) + (end.tv_nsec - start->tv_nsec));
        free(start);
    }

    return time;
}


/* Returns total time from start of timer till this call, for evalutiong performace.
*  Total time calculated includes that taken by cpu bound execution, and that taken by gpu where cpu is idle.
*  Unlike "stopTimer", "stopwatchTimer" does not free memory allocated by "startTimer".
*/
long int stopwatchTimer(struct timespec* start) {
    struct timespec end;
    long int time = 0;

    // ensure timer start struct exists before attempting to access it
    if (start) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = ((BILLION * (end.tv_sec - start->tv_sec)) + (end.tv_nsec - start->tv_nsec));
    }

    return time;
}