#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"
#include <iostream>

// ==========================================================
// V1: 单 Block 级的 Blelloch Scan (互斥前缀和 Exclusive Scan)
// 限制: 输入数组长度必须是 2 的幂次，且不能超过 Block 线程数的 2 倍 (最高 2048)
// ==========================================================
__global__ void prefix_sum_v1_block_kernel(const float* input, float* output, int n) {
    // 申请共享内存，用于树状归约
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    
    // 1. 将数据从 Global Memory 搬运到 Shared Memory
    // 每个线程负责搬运 2 个元素 (因为 n 个元素的二叉树底端有 n 个叶子，需要 n/2 个线程来操作)
    int offset = 1;
    temp[2 * tid]     = input[2 * tid];
    temp[2 * tid + 1] = input[2 * tid + 1];
    
    // ==========================================
    // Phase 1: Up-Sweep (归约建树)
    // ==========================================
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // ==========================================
    // Phase 2: Down-Sweep (分发推导)
    // ==========================================
    // 根节点清零 (Exclusive Scan 的核心要求)
    if (tid == 0) {
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // 3. 将计算好的 Exclusive Scan 结果写回 Global Memory
    // (如果想要 Inclusive Scan，只需要加上 input 对应位置的值即可)
    output[2 * tid]     = temp[2 * tid] + input[2 * tid];
    output[2 * tid + 1] = temp[2 * tid + 1] + input[2 * tid + 1];
}


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// 转换宏：将逻辑索引映射到带 Padding 的物理地址
#define SHMEM_IDX(n) ((n) + CONFLICT_FREE_OFFSET(n))

__global__ void prefix_sum_v2_conflict_free_kernel(const float* input, float* output, int n) {
    // Shared Memory 大小需要比 n 稍微大一点，以容纳 Padding
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int a = tid;
    int b = tid + (n / 2);

    // 计算带 Padding 的地址
    int ai = SHMEM_IDX(a);
    int bi = SHMEM_IDX(b);

    // 1. 搬运数据
    temp[ai] = input[a];
    temp[bi] = input[b];

    // Up-Sweep (归约阶段)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int i = SHMEM_IDX(offset * (2 * tid + 1) - 1);
            int j = SHMEM_IDX(offset * (2 * tid + 2) - 1);
            temp[j] += temp[i];
        }
        offset *= 2;
    }

    // Down-Sweep (下推阶段)
    if (tid == 0) {
        temp[SHMEM_IDX(n - 1)] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int i = SHMEM_IDX(offset * (2 * tid + 1) - 1);
            int j = SHMEM_IDX(offset * (2 * tid + 2) - 1);
            
            float t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }
    __syncthreads();

    // 写回 (Inclusive Scan = Exclusive Scan + 原输入)
    output[a] = temp[ai] + input[a];
    output[b] = temp[bi] + input[b];
}


// ==========================================================
// V3-1 (修复版): 局部扫描并提取 Block Sums
// ==========================================================
__global__ void prefix_sum_v3_local_kernel(const float* input, float* output, float* block_sums, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    
    int block_offset = blockIdx.x * (blockDim.x * 2);
    int a = tid;
    int b = tid + blockDim.x;

    int global_a = block_offset + a;
    int global_b = block_offset + b;

    int ai = SHMEM_IDX(a);
    int bi = SHMEM_IDX(b);

    // 搬运数据
    temp[ai] = (global_a < n) ? input[global_a] : 0.0f;
    temp[bi] = (global_b < n) ? input[global_b] : 0.0f;

    // --- Phase 1: Up-Sweep ---
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads(); // 完美的屏障，保证 temp 已经全部写完
        if (tid < d) {
            int i = SHMEM_IDX(offset * (2 * tid + 1) - 1);
            int j = SHMEM_IDX(offset * (2 * tid + 2) - 1);
            temp[j] += temp[i];
        }
        offset *= 2;
    }

    // ✨ 修复点：Up-Sweep 结束后，根节点天然就是整个块的总和，直接写入即可！
    if (tid == 0 && block_sums != nullptr) {
        block_sums[blockIdx.x] = temp[SHMEM_IDX(blockDim.x * 2 - 1)];
    }

    // --- Phase 2: Down-Sweep ---
    if (tid == 0) temp[SHMEM_IDX(blockDim.x * 2 - 1)] = 0;

    for (int d = 1; d < blockDim.x * 2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int i = SHMEM_IDX(offset * (2 * tid + 1) - 1);
            int j = SHMEM_IDX(offset * (2 * tid + 2) - 1);
            float t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }
    __syncthreads();

    // 写回 Global Memory (Exclusive Scan + 原数据 = Inclusive Scan)
    if (global_a < n) output[global_a] = temp[ai] + input[global_a];
    if (global_b < n) output[global_b] = temp[bi] + input[global_b];
}
// ==========================================================
// V3-2: 广播叠加 Base 偏移量
// ==========================================================
__global__ void add_block_sum_kernel(float* output, const float* block_sums, int n) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * (blockDim.x * 2);
    
    // 特别注意：第 0 个 Block 不需要加偏移量，因为它前面没有元素
    if (blockIdx.x == 0) return;

    int global_a = block_offset + tid;
    int global_b = block_offset + tid + blockDim.x;

    float base_val = block_sums[blockIdx.x - 1]; // 注意是上一个块的 Inclusive Sum

    if (global_a < n) output[global_a] += base_val;
    if (global_b < n) output[global_b] += base_val;
}
// ==========================================================
// C++ 接口
// ==========================================================
// void prefix_sum_forward(torch::Tensor input, torch::Tensor output) {
//     int n = input.numel();
    
//     // V1 限制：只支持 n <= 2048 且为 2 的幂次
//     // 因为一个 Block 最多 1024 个线程，每个线程处理 2 个元素
//     int threads = n / 2; 
    
//     // 动态分配 Shared Memory: n 个 float
//     int shared_mem_size = n * sizeof(float);

//     prefix_sum_v1_block_kernel<<<1, threads, shared_mem_size>>>(
//         input.data_ptr<float>(), 
//         output.data_ptr<float>(), 
//         n
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

// void prefix_sum_forward(torch::Tensor input, torch::Tensor output) {
//     int n = input.numel();
//     int threads = n / 2; 

//     // 考虑到 Padding，多申请一些空间：n + (n / 32)
//     int shared_mem_elements = n + (n / NUM_BANKS);
//     int shared_mem_size = shared_mem_elements * sizeof(float);

//     prefix_sum_v2_conflict_free_kernel<<<1, threads, shared_mem_size>>>(
//         input.data_ptr<float>(), 
//         output.data_ptr<float>(), 
//         n
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

// ==========================================================
// C++ 接口 (支持百万级数据的递归调度)
// ==========================================================
// 定义常量：一个 Block 最多处理 2048 个元素 (1024 线程 * 2)
#define ELEMENTS_PER_BLOCK 2048 
#define THREADS_PER_BLOCK 1024

// 这是一个内部的递归函数
void prefix_sum_recursive(float* d_in, float* d_out, int n) {
    int blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    int shared_mem_elements = ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK / NUM_BANKS);
    int shared_mem_size = shared_mem_elements * sizeof(float);

    if (blocks == 1) {
        // 基本情况：只需要一个 Block，直接扫完收工
        prefix_sum_v3_local_kernel<<<1, THREADS_PER_BLOCK, shared_mem_size>>>(d_in, d_out, nullptr, n);
    } else {
        // 需要跨 Block：申请 block_sums 数组
        float* d_block_sums;
        cudaMalloc(&d_block_sums, blocks * sizeof(float));

        // 1. 局部扫描，并提取 block_sums
        prefix_sum_v3_local_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_in, d_out, d_block_sums, n);

        // 2. 递归调用：对 block_sums 自身进行前缀和！
        // 这一步是点睛之笔，如果 blocks 数还是很大，它会继续递归分块
        float* d_block_sums_scanned;
        cudaMalloc(&d_block_sums_scanned, blocks * sizeof(float));
        prefix_sum_recursive(d_block_sums, d_block_sums_scanned, blocks);

        // 3. 广播叠加：把扫描后的 block_sums 补偿到 output 上
        add_block_sum_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_block_sums_scanned, n);

        cudaFree(d_block_sums);
        cudaFree(d_block_sums_scanned);
    }
}

void prefix_sum_forward(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    prefix_sum_recursive(input.data_ptr<float>(), output.data_ptr<float>(), n);
    CUDA_CHECK(cudaGetLastError());
}