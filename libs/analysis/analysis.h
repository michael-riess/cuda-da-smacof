#include <time.h>

#ifndef CUDA_SMACOF_ANALYSIS
#define CUDA_SMACOF_ANALYSIS

#define BILLION 1000000000L

typedef struct stress {
    double value;
    long int time;
    int index;
} stress;

int compare(const void* a, const void* b);

struct stress* median(struct stress* array, unsigned int size);

struct timespec* startTimer();

long int stopTimer(struct timespec* start);

long int stopwatchTimer(struct timespec* start);

#endif // ** CUDA_SMACOF_ANALYSIS **