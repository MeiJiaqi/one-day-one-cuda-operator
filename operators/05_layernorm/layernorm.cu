#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"
#include <cmath>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// ==========================================================
// 融合 LayerNorm Kernel
// ==========================================================
__global__ void layernorm_v1_fused_kernel(
    const float* input, 
    const float* gamma, 
    const float* beta, 
    float* output, 
    int num_rows, 
    int hidden_size, 
    float epsilon) {
    
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int offset = row * hidden_size;
    int tid = threadIdx.x;

    // 初始化每个线程局部的 Welford 状态
    WelfordState thread_state = {0.0f, 0.0f, 0.0f};

    // ==========================================================
    // Pass 1: 计算均值和平方差和 (使用 float4 向量化读取)
    // ==========================================================
    int vec_size = hidden_size / 4;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        float4 val4 = FETCH_FLOAT4(input[offset + i * 4]);
        float vals[4] = {val4.x, val4.y, val4.z, val4.w};
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = vals[j];
            // Welford 单点更新逻辑
            thread_state.count += 1.0f;
            float delta = x - thread_state.mean;
            thread_state.mean += delta / thread_state.count;
            thread_state.m2 += delta * (x - thread_state.mean);
        }
    }
    
    // 处理不足一个 float4 的尾部数据
    for (int i = tid + vec_size * 4; i < hidden_size; i += blockDim.x) {
        float x = input[offset + i];
        thread_state.count += 1.0f;
        float delta = x - thread_state.mean;
        thread_state.mean += delta / thread_state.count;
        thread_state.m2 += delta * (x - thread_state.mean);
    }

    // Block 级归约，合并所有线程的状态
    WelfordState block_state = blockReduceWelford(thread_state);

    // ==========================================================
    // 广播机制：让所有线程拿到最终的均值和方差
    // ==========================================================
    __shared__ float s_mean;
    __shared__ float s_var;
    if (tid == 0) {
        s_mean = block_state.mean;
        // 方差 = M2 / N
        s_var = block_state.m2 / block_state.count; 
    }
    __syncthreads();

    float mean = s_mean;
    // 预先计算出 1 / sqrt(var + eps)，乘法比除法快得多！
    float rsqrt_var = rsqrtf(s_var + epsilon); 

    // ==========================================================
    // Pass 2: 归一化并进行 Scale & Shift (使用 float4 向量化)
    // ==========================================================
    for (int i = tid; i < vec_size; i += blockDim.x) {
        int idx = i * 4;
        float4 x4 = FETCH_FLOAT4(input[offset + idx]);
        float4 g4 = FETCH_FLOAT4(gamma[idx]); // 读取 Scale
        float4 b4 = FETCH_FLOAT4(beta[idx]);  // 读取 Shift
        
        float4 out4;
        out4.x = (x4.x - mean) * rsqrt_var * g4.x + b4.x;
        out4.y = (x4.y - mean) * rsqrt_var * g4.y + b4.y;
        out4.z = (x4.z - mean) * rsqrt_var * g4.z + b4.z;
        out4.w = (x4.w - mean) * rsqrt_var * g4.w + b4.w;
        
        reinterpret_cast<float4*>(&output[offset + idx])[0] = out4;
    }
    
    for (int i = tid + vec_size * 4; i < hidden_size; i += blockDim.x) {
        float x = input[offset + i];
        float g = gamma[i];
        float b = beta[i];
        output[offset + i] = (x - mean) * rsqrt_var * g + b;
    }
}

// // ==========================================================
// // C++ 接口暴露
// // ==========================================================
// void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps) {
//     int hidden_size = input.size(-1);
//     int num_rows = input.numel() / hidden_size;

//     // 一个 Block 负责一行，最多 1024 线程
//     int threads = std::min(hidden_size, 1024);
//     threads = ((threads + 31) / 32) * 32; 
//     int blocks = num_rows;

//     layernorm_v1_fused_kernel<<<blocks, threads>>>(
//         input.data_ptr<float>(), 
//         gamma.data_ptr<float>(), 
//         beta.data_ptr<float>(), 
//         output.data_ptr<float>(), 
//         num_rows, 
//         hidden_size, 
//         eps
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

// ==========================================================
// 融合 LayerNorm Kernel (V2: Warp-Level 极致版)
// 专为中小型 Hidden Size (如 1024, 2048) 设计
// ==========================================================
__global__ void layernorm_v2_warp_kernel(
    const float* input, 
    const float* gamma, 
    const float* beta, 
    float* output, 
    int num_rows, 
    int hidden_size, 
    float epsilon) {
    
    // 1. 计算当前线程属于全局的第几个 Warp
    // 一个 Warp 负责处理矩阵的一行
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= num_rows) return;

    // 当前 Warp 处理的行指针偏移量
    int offset = warp_id * hidden_size;
    
    // 当前线程在 Warp 内的编号 (0 ~ 31)
    int lane_id = threadIdx.x % 32;

    WelfordState thread_state = {0.0f, 0.0f, 0.0f};

    // ==========================================================
    // Pass 1: 计算均值和平方差和 (float4 向量化)
    // ==========================================================
    int vec_size = hidden_size / 4;
    
    // 注意循环步长：现在是 32 (一个 Warp 的大小)，而不是 blockDim.x
    for (int i = lane_id; i < vec_size; i += 32) {
        float4 val4 = FETCH_FLOAT4(input[offset + i * 4]);
        float vals[4] = {val4.x, val4.y, val4.z, val4.w};
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = vals[j];
            thread_state.count += 1.0f;
            float delta = x - thread_state.mean;
            thread_state.mean += delta / thread_state.count;
            thread_state.m2 += delta * (x - thread_state.mean);
        }
    }
    
    // 尾部数据处理 (不足 float4 的部分)
    for (int i = lane_id + vec_size * 4; i < hidden_size; i += 32) {
        float x = input[offset + i];
        thread_state.count += 1.0f;
        float delta = x - thread_state.mean;
        thread_state.mean += delta / thread_state.count;
        thread_state.m2 += delta * (x - thread_state.mean);
    }

    // ==========================================================
    // Warp 级归约与广播 (彻底抛弃 Shared Memory 和 __syncthreads)
    // ==========================================================
    // 1. Warp 内 32 个线程合并 Welford 状态
    WelfordState warp_state = warpReduceWelford(thread_state);

    // 2. 此时只有 lane 0 拥有完美的最终状态，我们需要把它广播给其余 31 个兄弟
    // __shfl_sync 的第三个参数 0 表示从 lane 0 提取数据
    float row_mean = __shfl_sync(0xffffffff, warp_state.mean, 0);
    float row_m2   = __shfl_sync(0xffffffff, warp_state.m2, 0);
    float row_count = __shfl_sync(0xffffffff, warp_state.count, 0);

    float row_var = row_m2 / row_count;
    float rsqrt_var = rsqrtf(row_var + epsilon); 

    // ==========================================================
    // Pass 2: 归一化并进行 Scale & Shift
    // ==========================================================
    for (int i = lane_id; i < vec_size; i += 32) {
        int idx = i * 4;
        float4 x4 = FETCH_FLOAT4(input[offset + idx]);
        float4 g4 = FETCH_FLOAT4(gamma[idx]);
        float4 b4 = FETCH_FLOAT4(beta[idx]);
        
        float4 out4;
        out4.x = (x4.x - row_mean) * rsqrt_var * g4.x + b4.x;
        out4.y = (x4.y - row_mean) * rsqrt_var * g4.y + b4.y;
        out4.z = (x4.z - row_mean) * rsqrt_var * g4.z + b4.z;
        out4.w = (x4.w - row_mean) * rsqrt_var * g4.w + b4.w;
        
        reinterpret_cast<float4*>(&output[offset + idx])[0] = out4;
    }
    
    // 尾部归一化处理
    for (int i = lane_id + vec_size * 4; i < hidden_size; i += 32) {
        float x = input[offset + i];
        float g = gamma[i];
        float b = beta[i];
        output[offset + i] = (x - row_mean) * rsqrt_var * g + b;
    }
}

// void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps) {
//     int hidden_size = input.size(-1);
//     int num_rows = input.numel() / hidden_size;

//     // V2 策略：Warp-Level
//     // 假设一个 Block 我们分配 128 个线程 (包含 4 个 Warp)
//     int threads_per_block = 128; 
//     int warps_per_block = threads_per_block / 32;

//     // 计算总共需要多少个 Block (向上取整)
//     int blocks = (num_rows + warps_per_block - 1) / warps_per_block;

//     layernorm_v2_warp_kernel<<<blocks, threads_per_block>>>(
//         input.data_ptr<float>(), 
//         gamma.data_ptr<float>(), 
//         beta.data_ptr<float>(), 
//         output.data_ptr<float>(), 
//         num_rows, 
//         hidden_size, 
//         eps
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

// ==========================================================
// C++ 接口暴露与启发式调度 (Heuristic Dispatcher)
// ==========================================================
void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps) {
    int hidden_size = input.size(-1);
    int num_rows = input.numel() / hidden_size;

    // 工业级核心逻辑：根据特征维度大小，动态选择最优 Kernel
    if (hidden_size <= 2048) {
        // 中小尺寸：使用 Warp-Level，避免线程发呆
        int threads_per_block = 128; // 4 个 Warp
        int warps_per_block = threads_per_block / 32;
        int blocks = (num_rows + warps_per_block - 1) / warps_per_block;
        
        layernorm_v2_warp_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), 
            output.data_ptr<float>(), num_rows, hidden_size, eps
        );
    } else {
        // 超大尺寸：使用 Block-Level，用海量线程填满显存总线
        int threads = std::min(hidden_size / 4, 1024); // 结合 float4 估算
        threads = ((threads + 31) / 32) * 32; 
        int blocks = num_rows;

        layernorm_v1_fused_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), 
            output.data_ptr<float>(), num_rows, hidden_size, eps
        );
    }
    CUDA_CHECK(cudaGetLastError());
}
