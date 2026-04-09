#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 捕获 CUDA 错误的宏
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", err);                            \
            printf("    Error text: %s\n", cudaGetErrorString(err));       \
            exit(1);                                                        \
        }                                                                   \
    } while (0)