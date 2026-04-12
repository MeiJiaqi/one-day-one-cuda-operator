#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cfloat>

// ==========================================================
// 归约底层原语 (Reduction Primitives)
// ==========================================================

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared_max[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val); 
    if (lane == 0) shared_max[wid] = val; 
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32.0f) ? shared_max[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared_sum[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32.0f) ? shared_sum[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

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
