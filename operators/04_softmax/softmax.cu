#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"


// ==========================================================
// 1. 核心 Kernel: 融合型 Safe Softmax
// ==========================================================

__global__ void softmax_v1_fused_kernel(const float* input, float* output, int num_rows, int seq_len) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int offset = row * seq_len;
    int tid = threadIdx.x;

    // ✨ 新增：申请共享内存，用于 Thread 0 向全员广播
    __shared__ float s_row_max;
    __shared__ float s_row_sum;

    // ==========================================================
    // Pass 1: 找最大值 max(x)
    // ==========================================================
    float thread_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[offset + i]);
    }
    float block_max = blockReduceMax(thread_max);
    
    // ✨ 广播机制：0 号线程把结果写到小黑板上，大家同步等它写完，然后各自抄写
    if (tid == 0) s_row_max = block_max;
    __syncthreads(); 
    float row_max = s_row_max; 

    // ==========================================================
    // Pass 2: 计算 e^(x - max) 并求和 sum
    // ==========================================================
    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        thread_sum += expf(input[offset + i] - row_max);
    }
    float block_sum = blockReduceSum(thread_sum);
    
    // ✨ 广播机制：同步 sum
    if (tid == 0) s_row_sum = block_sum;
    __syncthreads();
    float row_sum = s_row_sum;

    // ==========================================================
    // Pass 3: 归一化并写回 Global Memory
    // ==========================================================
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[offset + i] = expf(input[offset + i] - row_max) / row_sum;
    }
}


// 向量化读取宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

__global__ void softmax_v2_online_kernel(const float* input, float* output, int num_rows, int seq_len) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int offset = row * seq_len;
    int tid = threadIdx.x;

    // 每一个线程维护自己的局部 max 和 sum
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // ==========================================================
    // Pass 1: 一次遍历，同时搞定 Max 和 Sum (Online 核心)
    // ==========================================================
    // 使用 float4 向量化读取，一次读 4 个，大幅提升带宽
    int vec_size = seq_len / 4;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 val4 = FETCH_FLOAT4(input[offset + i * 4]);
        float vals[4] = {val4.x, val4.y, val4.z, val4.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = vals[j];
            if (x > local_max) {
                // 关键修正逻辑
                local_sum = local_sum * expf(local_max - x) + 1.0f;
                local_max = x;
            } else {
                local_sum += expf(x - local_max);
            }
        }
    }
    // 处理剩余的不满 float4 的部分 (如果有)
    for (int i = tid + vec_size * 4; i < seq_len; i += blockDim.x) {
        float x = input[offset + i];
        if (x > local_max) {
            local_sum = local_sum * expf(local_max - x) + 1.0f;
            local_max = x;
        } else {
            local_sum += expf(x - local_max);
        }
    }

    // ==========================================================
    // Block 级归约：合并所有线程的 local_max 和 local_sum
    // ==========================================================
    
    // 1. 先求全局最大值，并立刻广播给全员！
    float block_max = blockReduceMax(local_max);
    __shared__ float s_max;
    if (tid == 0) s_max = block_max;
    __syncthreads();
    float row_max = s_max; // 现在所有线程都有真正的 row_max 了

    // 2. 所有线程使用极其准确的 row_max 来修正自己的 local_sum
    local_sum = local_sum * expf(local_max - row_max);

    // 3. 再求全局和，并立刻广播给全员！
    float block_sum = blockReduceSum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = block_sum;
    __syncthreads();
    float row_sum = s_sum; // 现在所有线程都有真正的 row_sum 了

    // ==========================================================
    // Pass 2: 写回结果 (这是第二次访存，也是最后一次)
    // ==========================================================
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 val4 = FETCH_FLOAT4(input[offset + i * 4]);
        float4 out4;
        out4.x = expf(val4.x - row_max) / row_sum;
        out4.y = expf(val4.y - row_max) / row_sum;
        out4.z = expf(val4.z - row_max) / row_sum;
        out4.w = expf(val4.w - row_max) / row_sum;
        // 向量化写回
        reinterpret_cast<float4*>(&output[offset + i * 4])[0] = out4;
    }
    // 处理剩余部分
    for (int i = tid + vec_size * 4; i < seq_len; i += blockDim.x) {
        output[offset + i] = expf(input[offset + i] - row_max) / row_sum;
    }
}
// ==========================================================
// 3. C++ 接口暴露
// ==========================================================
void softmax_forward(torch::Tensor input, torch::Tensor output) {
    // 将多维 Tensor 压缩为 2D 矩阵: [Batch*Heads*SeqLen, SeqLen]
    // 最后一维是 seq_len，前面所有的维度拍平当成 num_rows
    int seq_len = input.size(-1);
    int num_rows = input.numel() / seq_len;

    // 一个 Block 负责一行，最多给 1024 个线程
    int threads = std::min(seq_len, 1024);
    // 如果 threads 不是 32 的整数倍，向上取整到 32 倍数 (为了 Warp Shuffle 安全)
    threads = ((threads + 31) / 32) * 32; 

    int blocks = num_rows; // 多少行就起多少个 Block

    // softmax_v1_fused_kernel<<<blocks, threads>>>(
    //     input.data_ptr<float>(), 
    //     output.data_ptr<float>(), 
    //     num_rows, 
    //     seq_len
    // );
    softmax_v2_online_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_rows, 
        seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}