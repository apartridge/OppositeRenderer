#pragma once

#include <time.h>

inline double currentTime()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return now.tv_sec + now.tv_nsec * 1.0e-9;
}
