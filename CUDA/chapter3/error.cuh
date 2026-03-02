#pragma once
#include <stdio.h>

#define CHECK(call)                                                       \
do                                                                        \
{\
    const cudaError_t error_code = call;\
    if (error_code != cudaSuccess) {\
        printf("CUDA Error\n");\
        printf("    FILE:    %s\n", __FILE__);\
        printf("    LINE:    %s\n", __LINE__);\
        printf("    ERROR CODE:    %d\n", error_code);\
        printf("    ERROR MESSAGE: %d\n", cudaGetErrorString(error_code))\
        exit(1);
    }\
} while(0)