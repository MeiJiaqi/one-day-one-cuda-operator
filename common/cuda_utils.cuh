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

// ==========================================================
// Welford 并行算法底层原语 (LayerNorm 专属)
// ==========================================================

// 定义 Welford 状态结构体
struct WelfordState {
    float mean;
    float m2;    // 平方差之和 (sum of squared differences)
    float count; // 参与计算的元素个数
};

// 核心数学：合并两个 Welford 状态
__inline__ __device__ WelfordState combine_welford(WelfordState a, WelfordState b) {
    WelfordState res;
    res.count = a.count + b.count;
    if (res.count == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }
    float delta = b.mean - a.mean;
    // 更新均值
    res.mean = a.mean + delta * b.count / res.count;
    // 更新平方差之和
    res.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / res.count;
    return res;
}

// Warp 级 Welford 归约
__inline__ __device__ WelfordState warpReduceWelford(WelfordState state) {
    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordState other;
        other.mean = __shfl_down_sync(0xffffffff, state.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        state = combine_welford(state, other);
    }
    return state;
}

// Block 级 Welford 归约
__inline__ __device__ WelfordState blockReduceWelford(WelfordState state) {
    __shared__ float s_mean[32];
    __shared__ float s_m2[32];
    __shared__ float s_count[32];

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. Warp 内部先归约
    state = warpReduceWelford(state);

    // 2. Warp 代表将结果写入 Shared Memory
    if (lane == 0) {
        s_mean[wid] = state.mean;
        s_m2[wid] = state.m2;
        s_count[wid] = state.count;
    }
    __syncthreads();

    // 3. 第一个 Warp 读取 Shared Memory 并做最终归约
    if (wid == 0) {
        WelfordState shared_state;
        shared_state.mean = (threadIdx.x < blockDim.x / 32.0f) ? s_mean[lane] : 0.0f;
        shared_state.m2 = (threadIdx.x < blockDim.x / 32.0f) ? s_m2[lane] : 0.0f;
        shared_state.count = (threadIdx.x < blockDim.x / 32.0f) ? s_count[lane] : 0.0f;
        state = warpReduceWelford(shared_state);
    }
    return state;
}
