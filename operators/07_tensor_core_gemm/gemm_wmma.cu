#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"
#include <cuda_fp16.h>
#include <mma.h>


// 必须使用 nvcuda 命名空间
using namespace nvcuda;

// 定义 WMMA 的魔法数字：一次计算的 Block 大小 (M, N, K)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;



// ==========================================================
// V1 纯 Tensor Core GEMM (Global Memory 直读版)
// ==========================================================
__global__ void wmma_gemm_kernel(const half* a, const half* b, float* c, int M, int N, int K) {
    // 1. Warp 级别定位
    // 一个 Warp (32 threads) 负责计算 C 矩阵中的一个 16x16 的 Block
    // blockDim.x 和 blockDim.y 我们在外面会配置成能整除 32 的数字
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y); 
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int m_idx = warpM * WMMA_M;
    int n_idx = warpN * WMMA_N;

    // 边界检查
    if (m_idx >= M || n_idx >= N) return;

    // 2. 声明 Tensor Core 专用的 Fragment (寄存器堆)
    // 注意：A和B的类型是 half (FP16)，Accumulator 是 float (FP32)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 3. 将累加器清零
    wmma::fill_fragment(c_frag, 0.0f);

    // 4. 在 K 维度上循环，每次吃下 16 个深度
    for (int k_idx = 0; k_idx < K; k_idx += WMMA_K) {
        // Warp 协同从 Global Memory 加载数据到 Fragment
        wmma::load_matrix_sync(a_frag, a + m_idx * K + k_idx, K);
        wmma::load_matrix_sync(b_frag, b + k_idx * N + n_idx, N);

        // 🚀 核心爆发：调用 Tensor Core 进行 16x16x16 矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 5. Warp 协同将结果写回 Global Memory
    wmma::store_matrix_sync(c + m_idx * N + n_idx, c_frag, N, wmma::mem_row_major);
}

// ==========================================================
// V2 Tensor Core GEMM (Shared Memory Tiling 进阶版)
// ==========================================================

// 定义 Block 级别计算的矩阵块大小
#define M_TILE 64
#define N_TILE 64
#define K_TILE 32

__global__ void wmma_gemm_shared_kernel(const half* a, const half* b, float* c, int M, int N, int K) {
    // 1. 申请 Shared Memory 作为高速缓存中转站
    // 注意：为了防止之前学过的 Bank Conflict，可以考虑加上 Padding，这里为了直观先写标准大小
    __shared__ half s_a[M_TILE][K_TILE];
    __shared__ half s_b[K_TILE][N_TILE];

    // Warp 的局部索引
    int warp_id = threadIdx.x / 32;     // 0 ~ 3 (因为我们分配了 128 个线程)
    // int lane_id = threadIdx.x % 32;     // 0 ~ 31

    // 确定当前 Warp 负责计算 C 矩阵中的哪一块 (32x32 的区域)
    // 4 个 Warp 刚好分摊 64x64 的 M_TILE x N_TILE 区域
    int warp_row = (warp_id / 2) * 32;  // 0 或 32
    int warp_col = (warp_id % 2) * 32;  // 0 或 32

    // 确定当前 Block 负责的全局坐标起始点
    int block_m = blockIdx.y * M_TILE;
    int block_n = blockIdx.x * N_TILE;

    // 2. 声明 Fragment：注意，每个 Warp 要计算 32x32 的面积，所以需要 2x2 = 4 个 16x16 的累加器
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // A 和 B 的读取 Fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

    // 3. 核心 K 维度分块循环
    for (int k_step = 0; k_step < K; k_step += K_TILE) {
        
        // --- 步骤 A：整个 Block 协同将数据从 Global 搬到 Shared ---
        // 128 个线程要搬运 64x32=2048 个 A 的元素，每个线程搬运 16 个
        #pragma unroll
        for (int i = 0; i < (M_TILE * K_TILE) / 128; i++) {
            int idx = i * 128 + threadIdx.x;
            int r = idx / K_TILE;
            int c_idx = idx % K_TILE;
            s_a[r][c_idx] = a[(block_m + r) * K + (k_step + c_idx)];
        }
        
        // 128 个线程搬运 32x64=2048 个 B 的元素
        #pragma unroll
        for (int i = 0; i < (K_TILE * N_TILE) / 128; i++) {
            int idx = i * 128 + threadIdx.x;
            int r = idx / N_TILE;
            int c_idx = idx % N_TILE;
            s_b[r][c_idx] = b[(k_step + r) * N + (block_n + c_idx)];
        }
        
        // 等待所有人搬运完毕
        __syncthreads();

        // --- 步骤 B：Warp 从 Shared Memory 吸取数据给 Tensor Core 计算 ---
        // K_TILE 是 32，WMMA_K 是 16，所以内部还要循环 2 次
        for (int k_inner = 0; k_inner < K_TILE; k_inner += WMMA_K) {
            
            // 加载 A 的两块 16x16 (因为 Warp 要负责 32 行)
            wmma::load_matrix_sync(a_frag[0], &s_a[warp_row][k_inner], K_TILE);
            wmma::load_matrix_sync(a_frag[1], &s_a[warp_row + 16][k_inner], K_TILE);
            
            // 加载 B 的两块 16x16 (因为 Warp 要负责 32 列)
            wmma::load_matrix_sync(b_frag[0], &s_b[k_inner][warp_col], N_TILE);
            wmma::load_matrix_sync(b_frag[1], &s_b[k_inner][warp_col + 16], N_TILE);

            // 爆发：执行 4 次 WMMA 乘加！
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        // 等待所有 Warp 算完，防止下一轮加载覆盖 Shared Memory
        __syncthreads();
    }

    // 4. 将结果写回 Global Memory
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int global_r = block_m + warp_row + i * 16;
            int global_c = block_n + warp_col + j * 16;
            if (global_r < M && global_c < N) {
                wmma::store_matrix_sync(c + global_r * N + global_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// ==========================================================
// C++ 接口
// ==========================================================
// void wmma_gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
//     int M = a.size(0);
//     int K = a.size(1);
//     int N = b.size(1);

//     // Grid 和 Block 划分策略
//     // 让一个 Block 包含 4 个 Warp (128 线程)，计算 1 个 16x64 的区域
//     dim3 threads(128, 1); 
//     dim3 blocks((N + 64 - 1) / 64, (M + 16 - 1) / 16);

//     // 注意：这里的指针强制转换，把 PyTorch 的 at::Half 转换成了 CUDA 的 half
//     wmma_gemm_kernel<<<blocks, threads>>>(
//         reinterpret_cast<const half*>(a.data_ptr<at::Half>()), 
//         reinterpret_cast<const half*>(b.data_ptr<at::Half>()), 
//         c.data_ptr<float>(), 
//         M, N, K
//     );
//     CUDA_CHECK(cudaGetLastError());
// }


// 我们的 Block-level 分块大小
#define M_TILE 64
#define N_TILE 64
#define K_TILE 32

// 魔法数字：打乱 FP16 的 Bank Conflict
#define PAD 8 

// ==========================================================
// V3: 究极形态 (Double Buffering 软件流水线 + Padding 零冲突)
// ==========================================================
__global__ void wmma_gemm_v3_double_buffer_kernel(const half* a, const half* b, float* c, int M, int N, int K) {
    // ✨ 核心升级：第一维为 2，代表 Ping-Pong 两个缓冲区
    __shared__ half s_a[2][M_TILE][K_TILE + PAD];
    __shared__ half s_b[2][K_TILE][N_TILE + PAD];

    // Warp 的局部与全局定位
    int warp_id = threadIdx.x / 32;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    int block_m = blockIdx.y * M_TILE;
    int block_n = blockIdx.x * N_TILE;

    // 声明并清零 4 个 16x16 的累加器 Fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

    // ==========================================================
    // Preamble (序幕): 将第 0 个 K_TILE 搬入 Buffer 0
    // ==========================================================
    #pragma unroll
    for (int i = 0; i < (M_TILE * K_TILE) / 128; i++) {
        int idx = i * 128 + threadIdx.x;
        int r = idx / K_TILE;
        int c_idx = idx % K_TILE;
        s_a[0][r][c_idx] = a[(block_m + r) * K + c_idx]; 
    }
    #pragma unroll
    for (int i = 0; i < (K_TILE * N_TILE) / 128; i++) {
        int idx = i * 128 + threadIdx.x;
        int r = idx / N_TILE;
        int c_idx = idx % N_TILE;
        s_b[0][r][c_idx] = b[r * N + (block_n + c_idx)];
    }
    __syncthreads();

    // 读写指针初始化
    int write_stage = 1; // 接下来要搬砖写入 Buffer 1
    int read_stage = 0;  // Tensor Core 从 Buffer 0 读取

    // ==========================================================
    // Main Loop (核心流水线)
    // ==========================================================
    for (int k_step = 0; k_step < K; k_step += K_TILE) {

        // 1. 搬砖工 (Global -> Shared)：异步搬运下一个 Tile 到 write_stage
        if (k_step + K_TILE < K) {
            #pragma unroll
            for (int i = 0; i < (M_TILE * K_TILE) / 128; i++) {
                int idx = i * 128 + threadIdx.x;
                int r = idx / K_TILE;
                int c_idx = idx % K_TILE;
                s_a[write_stage][r][c_idx] = a[(block_m + r) * K + (k_step + K_TILE + c_idx)];
            }
            #pragma unroll
            for (int i = 0; i < (K_TILE * N_TILE) / 128; i++) {
                int idx = i * 128 + threadIdx.x;
                int r = idx / N_TILE;
                int c_idx = idx % N_TILE;
                s_b[write_stage][r][c_idx] = b[(k_step + K_TILE + r) * N + (block_n + c_idx)];
            }
        }

        // 2. 吞金兽 (Shared -> Tensor Core)：处理 read_stage 中的当前 Tile
        for (int k_inner = 0; k_inner < K_TILE; k_inner += WMMA_K) {
            // ✨ 重点：跨度参数必须加上 PAD，保证硬件能正确寻址被错开的元素！
            wmma::load_matrix_sync(a_frag[0], &s_a[read_stage][warp_row][k_inner], K_TILE + PAD);
            wmma::load_matrix_sync(a_frag[1], &s_a[read_stage][warp_row + 16][k_inner], K_TILE + PAD);
            
            wmma::load_matrix_sync(b_frag[0], &s_b[read_stage][k_inner][warp_col], N_TILE + PAD);
            wmma::load_matrix_sync(b_frag[1], &s_b[read_stage][k_inner][warp_col + 16], N_TILE + PAD);

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        // 同步屏障：确保搬砖工搬完了，计算兽也算完了，再进入下一轮
        __syncthreads();

        // ✨ 身份互换：利用异或操作 (XOR) 无缝切换 0 和 1
        write_stage ^= 1;
        read_stage ^= 1;
    }

    // ==========================================================
    // Epilogue (收尾): 将寄存器堆里的最终结果写回 Global Memory
    // ==========================================================
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int global_r = block_m + warp_row + i * 16;
            int global_c = block_n + warp_col + j * 16;
            if (global_r < M && global_c < N) {
                wmma::store_matrix_sync(c + global_r * N + global_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// ==========================================================
// V4: 绝杀形态 (Double Buffering + Padding + float4 向量化搬砖)
// ==========================================================
__global__ void wmma_gemm_v4_vectorized_kernel(const half* a, const half* b, float* c, int M, int N, int K) {
    __shared__ half s_a[2][M_TILE][K_TILE + PAD];
    __shared__ half s_b[2][K_TILE][N_TILE + PAD];

    int warp_id = threadIdx.x / 32;
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 32;

    int block_m = blockIdx.y * M_TILE;
    int block_n = blockIdx.x * N_TILE;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

    // ==========================================================
    // 序幕 (Preamble): 使用 float4 向量化加载 Buffer 0
    // ==========================================================
    // A 矩阵: M_TILE(64) * K_TILE(32) = 2048 个 half. 
    // 用 float4(8个half) 搬运，只需要 256 次 load. 128 个线程，每人刚好搬 2 次！
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = i * 128 + threadIdx.x;    // 向量级索引
        int r = idx / (K_TILE / 8);         // 行号
        int c_vec = idx % (K_TILE / 8);     // 向量列号 (0~3)

        float4 val = reinterpret_cast<const float4*>(&a[(block_m + r) * K + c_vec * 8])[0];
        reinterpret_cast<float4*>(&s_a[0][r][c_vec * 8])[0] = val;
    }

    // B 矩阵: K_TILE(32) * N_TILE(64) = 2048 个 half. 同理，每人搬 2 次！
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = i * 128 + threadIdx.x;
        int r = idx / (N_TILE / 8);
        int c_vec = idx % (N_TILE / 8);

        float4 val = reinterpret_cast<const float4*>(&b[r * N + (block_n + c_vec * 8)])[0];
        reinterpret_cast<float4*>(&s_b[0][r][c_vec * 8])[0] = val;
    }
    __syncthreads();

    int write_stage = 1; 
    int read_stage = 0;  

    // ==========================================================
    // Main Loop
    // ==========================================================
    for (int k_step = 0; k_step < K; k_step += K_TILE) {

        // 1. 搬砖工：极其残暴的 float4 异步搬运
        if (k_step + K_TILE < K) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = i * 128 + threadIdx.x;
                int r = idx / (K_TILE / 8);
                int c_vec = idx % (K_TILE / 8);
                float4 val = reinterpret_cast<const float4*>(&a[(block_m + r) * K + (k_step + K_TILE + c_vec * 8)])[0];
                reinterpret_cast<float4*>(&s_a[write_stage][r][c_vec * 8])[0] = val;
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = i * 128 + threadIdx.x;
                int r = idx / (N_TILE / 8);
                int c_vec = idx % (N_TILE / 8);
                float4 val = reinterpret_cast<const float4*>(&b[(k_step + K_TILE + r) * N + (block_n + c_vec * 8)])[0];
                reinterpret_cast<float4*>(&s_b[write_stage][r][c_vec * 8])[0] = val;
            }
        }

        // 2. 吞金兽：Tensor Core 狂飙
        for (int k_inner = 0; k_inner < K_TILE; k_inner += WMMA_K) {
            wmma::load_matrix_sync(a_frag[0], &s_a[read_stage][warp_row][k_inner], K_TILE + PAD);
            wmma::load_matrix_sync(a_frag[1], &s_a[read_stage][warp_row + 16][k_inner], K_TILE + PAD);
            
            wmma::load_matrix_sync(b_frag[0], &s_b[read_stage][k_inner][warp_col], N_TILE + PAD);
            wmma::load_matrix_sync(b_frag[1], &s_b[read_stage][k_inner][warp_col + 16], N_TILE + PAD);

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();

        write_stage ^= 1;
        read_stage ^= 1;
    }

    // ==========================================================
    // Epilogue (写入部分如果也能向量化就更完美，但为了避免边界问题，暂用单 float 写入)
    // ==========================================================
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int global_r = block_m + warp_row + i * 16;
            int global_c = block_n + warp_col + j * 16;
            if (global_r < M && global_c < N) {
                wmma::store_matrix_sync(c + global_r * N + global_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}
void wmma_gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // V2 策略：每个 Block 128 个线程 (4个 Warp)
    // 负责计算 64 (M_TILE) x 64 (N_TILE) 的输出区域
    dim3 threads(128); 
    dim3 blocks((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    wmma_gemm_v4_vectorized_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()), 
        c.data_ptr<float>(), 
        M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}