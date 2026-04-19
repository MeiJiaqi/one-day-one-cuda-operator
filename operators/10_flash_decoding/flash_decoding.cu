#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"
#include <math.h>
#include <float.h>

// ==========================================================
// V1: Flash Decoding (Online Softmax 融合算子)
// ==========================================================
__global__ void flash_decoding_v1_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, float scale
) {
    // blockIdx.x 代表当前处理的 (Batch * Head) 的一维索引
    int bx = blockIdx.x; 
    // threadIdx.x 代表特征维度 (0 到 d-1)
    int tid = threadIdx.x;

    // 寻址到当前 Head 的起点
    const float* q_head = Q + bx * d;
    const float* k_head = K + bx * N * d;
    const float* v_head = V + bx * N * d;
    float* o_head = O + bx * d;

    // 1. 读取当前特征维度的 Query 并缩放 (驻留在寄存器)
    float q_val = (tid < d) ? q_head[tid] : 0.0f;
    q_val *= scale;

    // 2. 初始化 Online Softmax 变量
    float m_i = -FLT_MAX; // 历史最大值
    float l_i = 0.0f;     // 历史指数和
    float o_acc = 0.0f;   // 当前维度的输出累加器

    // 申请共享内存用于计算点积 (Q * K^T)
    extern __shared__ float smem[];

    // 3. 时间循环：顺着序列长度 (KV Cache) 往后扫
    for (int j = 0; j < N; ++j) {
        // --- 步骤 3.1: 计算当前 Token 的 Q * K^T ---
        float k_val = (tid < d) ? k_head[j * d + tid] : 0.0f;
        smem[tid] = q_val * k_val;
        __syncthreads();

        // 并行树状归约求和 (求出一个标量 Attention Score)
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }
        float score = smem[0]; // 当前 Token 的最终分数

        // --- 步骤 3.2: 终极魔法 Online Softmax ---
        float m_i_new = max(m_i, score);
        // 算出由于 Max 变大而导致的“历史贬值率”
        float exp_diff = expf(m_i - m_i_new);
        // 算出当前 Token 的“实际含金量”
        float exp_val = expf(score - m_i_new);

        // 更新全局分母
        l_i = l_i * exp_diff + exp_val;

        // --- 步骤 3.3: 融合 V 的乘加 ---
        float v_val = (tid < d) ? v_head[j * d + tid] : 0.0f;
        // 把历史累加值按贬值率缩水，再加上当前 Token 带来的新价值
        o_acc = o_acc * exp_diff + exp_val * v_val;

        // 将新的 Max 设为历史 Max
        m_i = m_i_new;
    }

    // 4. 最终归一化并写回显存 (仅 1 次内存写)
    if (tid < d) {
        o_head[tid] = o_acc / l_i;
    }
}

// ==========================================================
// V2 Phase 1: 分块计算 (Chunk)
// ==========================================================
__global__ void flash_decoding_v2_chunk_kernel(
    const float* Q, const float* K, const float* V, 
    float* mid_O, float* mid_m, float* mid_l,
    int N, int d, int chunk_size, float scale
) {
    int bx = blockIdx.y;          // Batch * Head 索引
    int chunk_idx = blockIdx.x;   // 当前处理的 Chunk 索引
    int tid = threadIdx.x;        // 特征维度索引

    // 确定当前 Chunk 的起始和结束位置
    int start_seq = chunk_idx * chunk_size;
    int end_seq = min(start_seq + chunk_size, N);

    const float* q_head = Q + bx * d;
    const float* k_head = K + bx * N * d;
    const float* v_head = V + bx * N * d;

    float q_val = (tid < d) ? q_head[tid] : 0.0f;
    q_val *= scale;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o_acc = 0.0f;

    extern __shared__ float smem[];

    // 只遍历当前 Chunk 内的序列
    for (int j = start_seq; j < end_seq; ++j) {
        float k_val = (tid < d) ? k_head[j * d + tid] : 0.0f;
        smem[tid] = q_val * k_val;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        float score = smem[0]; 

        float m_i_new = max(m_i, score);
        float exp_diff = expf(m_i - m_i_new);
        float exp_val = expf(score - m_i_new);

        l_i = l_i * exp_diff + exp_val;

        float v_val = (tid < d) ? v_head[j * d + tid] : 0.0f;
        o_acc = o_acc * exp_diff + exp_val * v_val;

        m_i = m_i_new;
    }

    // 将局部结果写入全局中间变量
    if (tid < d) {
        // mid_O 形状 [Batch*Head, num_chunks, d]
        mid_O[(bx * gridDim.x + chunk_idx) * d + tid] = o_acc;
    }
    if (tid == 0) {
        // mid_m, mid_l 形状 [Batch*Head, num_chunks]
        mid_m[bx * gridDim.x + chunk_idx] = m_i;
        mid_l[bx * gridDim.x + chunk_idx] = l_i;
    }
}

// ==========================================================
// V2 Phase 2: 全局规约合并 (Reduce)
// ==========================================================
__global__ void flash_decoding_v2_reduce_kernel(
    const float* mid_O, const float* mid_m, const float* mid_l,
    float* O, int num_chunks, int d
) {
    int bx = blockIdx.x;   // Batch * Head 索引
    int tid = threadIdx.x;

    float global_m = -FLT_MAX;
    float global_l = 0.0f;
    float global_o = 0.0f;

    // 遍历所有 Chunk 的局部结果，应用 Online Softmax 合并公式
    for (int i = 0; i < num_chunks; ++i) {
        float m_local = mid_m[bx * num_chunks + i];
        float l_local = mid_l[bx * num_chunks + i];
        float o_local = mid_O[(bx * num_chunks + i) * d + tid];

        float global_m_new = max(global_m, m_local);
        float exp_diff = expf(global_m - global_m_new);
        float exp_local = expf(m_local - global_m_new);

        global_l = global_l * exp_diff + l_local * exp_local;
        global_o = global_o * exp_diff + o_local * exp_local;
        
        global_m = global_m_new;
    }

    if (tid < d) {
        O[bx * d + tid] = global_o / global_l;
    }
}

// ==========================================================
// C++ 接口
// ==========================================================
// void flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
//     int B = Q.size(0);
//     int H = Q.size(1);
//     int d = Q.size(2);
//     int N = K.size(2);

//     float scale = 1.0f / sqrt(static_cast<float>(d));

//     // Grid: B * H 个 Block
//     // Block: d 个线程 (完美映射特征维度)
//     int blocks = B * H;
//     int threads = d;
    
//     // 分配共享内存供归约使用
//     int shared_mem_size = d * sizeof(float);

//     flash_decoding_v1_kernel<<<blocks, threads, shared_mem_size>>>(
//         Q.data_ptr<float>(),
//         K.data_ptr<float>(),
//         V.data_ptr<float>(),
//         O.data_ptr<float>(),
//         N, d, scale
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

void flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
    int B = Q.size(0);
    int H = Q.size(1);
    int d = Q.size(2);
    int N = K.size(2);
    float scale = 1.0f / sqrt(static_cast<float>(d));

    // 切片大小：每个 Block 处理 256 个 Token
    int chunk_size = 256; 
    int num_chunks = (N + chunk_size - 1) / chunk_size;

    // 申请中间变量内存
    auto options = torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32);
    torch::Tensor mid_O = torch::empty({B * H, num_chunks, d}, options);
    torch::Tensor mid_m = torch::empty({B * H, num_chunks}, options);
    torch::Tensor mid_l = torch::empty({B * H, num_chunks}, options);

    // Phase 1: 启动 B * H * num_chunks 个 Block！彻底打满 GPU！
    dim3 grid_chunk(num_chunks, B * H);
    int threads = d;
    int shared_mem_size = d * sizeof(float);

    flash_decoding_v2_chunk_kernel<<<grid_chunk, threads, shared_mem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        mid_O.data_ptr<float>(), mid_m.data_ptr<float>(), mid_l.data_ptr<float>(),
        N, d, chunk_size, scale
    );

    // Phase 2: 启动 B * H 个 Block 进行合并
    flash_decoding_v2_reduce_kernel<<<B * H, threads>>>(
        mid_O.data_ptr<float>(), mid_m.data_ptr<float>(), mid_l.data_ptr<float>(),
        O.data_ptr<float>(), num_chunks, d
    );
    CUDA_CHECK(cudaGetLastError());
}