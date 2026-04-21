#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"



__global__ void gemm_naive_kernel(const float* A, const float* B,float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M &&col < N)
    {
        float sum = 0.0f;
        for(int k=0;k<K;++k)
        {
            sum += A[row * K + k] * B[k * N+col];
        }
        C[row * N + col] = sum;
    }
}
// 2. 暴露给 C++ 的接口
void gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // 严谨的输入检查
    TORCH_CHECK(a.device().is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor B must be on CUDA");
    TORCH_CHECK(c.device().is_cuda(), "Tensor C must be on CUDA");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Tensors must be 2D");
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // 定义 2D Block 大小 (16x16 = 256 threads)
    dim3 blockDim(16, 16);
    // 计算 2D Grid 大小
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // 启动 Kernel
    gemm_naive_kernel<<<gridDim, blockDim>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        M, N, K
    );
    
    CUDA_CHECK(cudaGetLastError());
}


// ---------------- Day 2 进阶: Shared Memory Tiling ----------------
// 定义分块大小，32x32 是一个非常经典的数字，因为一个 Warp 是 32 个线程
#define TILE_SIZE 32 

__global__ void gemm_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 申请两块共享内存，存当前 Tile 的 A 和 B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 全局行号和列号 (用于写回 C)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 局部行号和列号 (用于读写 Shared Memory)
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float value = 0.0f;

    // 沿着 K 维度，一步步滑动 Tile
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // 1. 协同加载：当前线程负责把一个元素从 Global 搬到 Shared
        int tiled_k_A = t * TILE_SIZE + tx; // A 矩阵在当前 Tile 的列索引
        int tiled_k_B = t * TILE_SIZE + ty; // B 矩阵在当前 Tile 的行索引

        // 边界保护：如果超出实际矩阵大小，补 0
        if (row < M && tiled_k_A < K) {
            As[ty][tx] = A[row * K + tiled_k_A];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (tiled_k_B < K && col < N) {
            Bs[ty][tx] = B[tiled_k_B * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 2. 同步：必须等 Block 内所有线程都把数据搬完，才能开始算！
        __syncthreads();

        // 3. 在高速的 Shared Memory 中计算当前 Tile 的点积并累加
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += As[ty][i] * Bs[i][tx];
        }

        // 4. 同步：必须等所有线程都算完当前 Tile，才能进入下一个循环覆盖 Shared Memory！
        __syncthreads();
    }

    // 将最终累加结果写回 Global Memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}
#define TILE_SIZE 32

__global__ void gemm_shared_mem_kernel(const float* A,const float* B,float* C,int M,int N,int K)
{
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    double value=0.0f;

    int numsTile=(K+TILE_SIZE-1)/TILE_SIZE;

    for(int t=0;t<numsTile;t++){

        int tile_k_a = t*TILE_SIZE+tx;
        int tile_k_b = t*TILE_SIZE+ty;
        
        if(row<M&&tile_k_a<K){
            s_a[ty][tx]=A[row*K+tile_k_a];
        }else{
            s_a[ty][tx]=0.0f;
        }

        if(col<N&&tile_k_b<K){
            s_b[ty][tx]=B[tile_k_b*N+col];
        }else{
            s_b[ty][tx]=0.0f;
        }

        __syncthreads();
        #pragma unroll
        for(int i=0;i<TILE_SIZE;++i){
            value+=s_a[ty][i]*s_b[i][tx];
        }

        __syncthreads();
    }

    if(row<M&&col<N){
        C[row*N+col]=value;
    }

    
}

// // 供 C++ 调用的 v2 接口
// void gemm_v2_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
//     int M = a.size(0);
//     int K = a.size(1);
//     int N = b.size(1);

//     dim3 blockDim(TILE_SIZE, TILE_SIZE);
//     dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
//     gemm_shared_kernel<<<gridDim, blockDim>>>(
//         a.data_ptr<float>(), 
//         b.data_ptr<float>(), 
//         c.data_ptr<float>(), 
//         M, N, K
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

void gemm_v2_forward(torch::Tensor a,torch::Tensor b,torch::Tensor c){
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(0);
    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);

    gemm_shared_mem_kernel<<<gridDim,blockDim>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M,N,K
    );
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- Day 2 终极进阶: 1D Register Tiling ----------------
#define TILE_SIZE 32
#define THREAD_WORK_X 4 // 每个线程负责计算 X 维度的 4 个元素

__global__ void gemm_v3_register_tiling_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 当前线程在 Grid 中的全局坐标 (注意 X 维度乘了 THREAD_WORK_X)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * THREAD_WORK_X;

    int ty = threadIdx.y;
    int tx = threadIdx.x; // tx 的范围是 0~7 (因为 TILE_SIZE=32, /4=8)

    // 申请寄存器数组，用于存储当前线程负责的 4 个 C 元素的结果
    // 寄存器存在于每个线程的私有空间，速度极快
    float c_reg[THREAD_WORK_X] = {0.0f};

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // 1. 协同加载到 Shared Memory
        // 因为现在一个 Block 只有 32 * 8 = 256 个线程，要搬运 32 * 32 = 1024 个元素
        // 所以每个线程需要循环搬运 4 次！
        #pragma unroll
        for (int i = 0; i < THREAD_WORK_X; ++i) {
            int load_idx = ty * TILE_SIZE + tx * THREAD_WORK_X + i; // 1D 展平索引
            int load_y = load_idx / TILE_SIZE;
            int load_x = load_idx % TILE_SIZE;

            // 搬运 A
            int global_a_row = blockIdx.y * TILE_SIZE + load_y;
            int global_a_col = t * TILE_SIZE + load_x;
            if (global_a_row < M && global_a_col < K) {
                As[load_y][load_x] = A[global_a_row * K + global_a_col];
            } else {
                As[load_y][load_x] = 0.0f;
            }

            // 搬运 B
            int global_b_row = t * TILE_SIZE + load_y;
            int global_b_col = blockIdx.x * TILE_SIZE + load_x;
            if (global_b_row < K && global_b_col < N) {
                Bs[load_y][load_x] = B[global_b_row * N + global_b_col];
            } else {
                Bs[load_y][load_x] = 0.0f;
            }
        }
        __syncthreads();

        // 2. 寄存器分块核心计算！
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            // 将 A 的 1 个元素读入寄存器 (复用 4 次！)
            float a_reg = As[ty][k];
            
            #pragma unroll
            for (int i = 0; i < THREAD_WORK_X; ++i) {
                // 此时 Bs 是从 Shared Memory 读，a_reg 和 c_reg 都在寄存器里
                c_reg[i] += a_reg * Bs[k][tx * THREAD_WORK_X + i];
            }
        }
        __syncthreads();
    }

    // 3. 将寄存器中的 4 个结果写回 Global Memory
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < THREAD_WORK_X; ++i) {
            int current_col = col + i;
            if (current_col < N) {
                C[row * N + current_col] = c_reg[i];
            }
        }
    }
}


void gemm_v3_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // Block 维度：Y 是 32，X 是 32/4 = 8。总计 256 个线程
    dim3 blockDim(TILE_SIZE / THREAD_WORK_X, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_v3_register_tiling_kernel<<<gridDim, blockDim>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}


// __global__ void gemm_v3_register_tiling_my_kernel(const float* A,const float* B,float* C,int M,int N,int K)
// {
//     __shared__ float s_a[TILE_SIZE][TILE_SIZE];
//     __shared__ float s_b[TILE_SIZE][TILE_SIZE];

//     int row = blockIdx.y*TILE_SIZE+threadIdx.y;
//     int col = blockIdx.x*TILE_SIZE+threadIdx.x*THREAD_WORK_X;

//     int ty=threadIdx.y;
//     int tx=threadIdx.x;

//     float c_reg[THREAD_WORK_X] ={0.0f};

//     int numTiles = (K+TILE_SIZE-1)/TILE_SIZE;

//     for(int t=0;t<numTiles;++t)
//     {
//         #pragma unroll
//         for(int i=0;i<THREAD_WORK_X;++i)
//         {
//             int tid=ty*(TILE_SIZE/THREAD_WORK_X)+tx;
//             int total_idx = tid+i*256;
//             int row_in_tile = total_idx/TILE_SIZE;
//             int col_in_tile = total_idx%TILE_SIZE;

//              // 搬运 A
//              int global_a_row = blockIdx.y * TILE_SIZE + row_in_tile;
//              int global_a_col = t * TILE_SIZE + col_in_tile;
//              if(global_a_row<M&&global_a_col<K){
//                 s_a[row_in_tile][col_in_tile]=A[global_a_row*K+global_a_col];
//              }else{
//                 s_a[row_in_tile][col_in_tile]=0.0f;
//              }

//              // 搬运 B
//              int global_b_col = blockIdx.x * TILE_SIZE + col_in_tile;
//              int global_b_row = t * TILE_SIZE + row_in_tile;

//              if(global_b_row<K&&global_b_col<N)
//                 s_b[row_in_tile][col_in_tile]=B[global_b_row*N+global_b_col];
//              else
//                 s_b[row_in_tile][col_in_tile]=0.0f;      
//         }
//         __syncthreads();
//         // 2. 寄存器分块核心计算！
//         #pragma unroll
//         for(int k=0;k<TILE_SIZE;++k)
//         {
//             float a_reg =s_a[ty][k];
//             #pragma unroll
//             for(int i=0;i<THREAD_WORK_X;++i){
//                 c_reg[i]+=a_reg*s_b[k][tx*THREAD_WORK_X+i];
//             }
//         }
//         __syncthreads();
//     }
//     // 3. 将寄存器中的 4 个结果写回 Global Memory
//     if(row<M){
//         #pragma unroll
//         for(int i=0;i<THREAD_WORK_X;++i){
//             int current_col=col+i;
//             if(current_col<N){
//                 C[row*N+current_col]=c_reg[i];
//             }
//         }
//     }
// }

#define TILE_SIZE 32
#define THREAD_WORK_X 4

__global__ void gemm_v3_register_tiling_my_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y; // 0~31
    int tx = threadIdx.x; // 0~7
    
    // 当前线程负责计算的 C 矩阵起始坐标
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx * THREAD_WORK_X;

    float c_reg[THREAD_WORK_X] = {0.0f};

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // --- 1. 完美的合并访问搬运 ---
        #pragma unroll
        for (int i = 0; i < THREAD_WORK_X; ++i) {
            int tid = ty * (TILE_SIZE / THREAD_WORK_X) + tx; // 0~255
            int total_idx = tid + i * 256;                   // 每次循环步进 256
            int row_in_tile = total_idx / TILE_SIZE;
            int col_in_tile = total_idx % TILE_SIZE;

            // 搬运 A (注意：row_in_tile 对应全局的 row 方向，col_in_tile 对应 K 方向)
            int global_a_row = blockIdx.y * TILE_SIZE + row_in_tile;
            int global_a_col = t * TILE_SIZE + col_in_tile;
            if (global_a_row < M && global_a_col < K)
                s_a[row_in_tile][col_in_tile] = A[global_a_row * K + global_a_col];
            else
                s_a[row_in_tile][col_in_tile] = 0.0f;

            // 搬运 B 矩阵
            int global_b_row = t * TILE_SIZE + row_in_tile;    // row_in_tile 对应 K 维度
            int global_b_col = blockIdx.x * TILE_SIZE + col_in_tile; // col_in_tile 对应 N 维度

            if (global_b_row < K && global_b_col < N) {
                s_b[row_in_tile][col_in_tile] = B[global_b_row * N + global_b_col];
            } else {
                s_b[row_in_tile][col_in_tile] = 0.0f; // 必须填0，否则边缘计算会炸
            }
        }
        
        // 必须同步：确保 Shared Memory 全部填满
        __syncthreads();

        // --- 2. 寄存器分块计算 ---
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_reg = s_a[ty][k]; // 广播 A 元素到寄存器
            #pragma unroll
            for (int i = 0; i < THREAD_WORK_X; ++i) {
                c_reg[i] += a_reg * s_b[k][tx * THREAD_WORK_X + i];
            }
        }
        
        // 必须同步：确保计算完成，才能进入下一轮 Tile 覆盖 Shared Memory
        __syncthreads();
    }

    // --- 3. 写回 Global Memory ---
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < THREAD_WORK_X; ++i) {
            int current_col = col + i;
            if (current_col < N) {
                C[row * N + current_col] = c_reg[i];
            }
        }
    }
}
void gemm_v3_my_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {

    int M=a.size(0);
    int K=a.size(1);
    int N=b.size(1);

    dim3 blockDim(TILE_SIZE / THREAD_WORK_X, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_v3_register_tiling_my_kernel<<<gridDim,blockDim>>>(
        a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),M,N,K
    );
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- Day 2 绝对死磕: 2D Register Tiling ----------------
// 定义工业级经典分块参数
#define BM 128  // Block 在 M 维度的跨度
#define BN 128  // Blo}ck 在 N 维度的跨度
#define BK 8    // Block 在 K 维度的跨度 (K维度不用太大，避免 Shared Memory 撑爆)
#define TM 8    // 每个线程在 M 维度计算 8 个元素
#define TN 8    // 每个线程在 N 维度计算 8 个元素

__global__ void gemm_v4_2d_register_tiling_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 申请 Shared Memory (一维数组，方便避免复杂的行列映射计算)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Block ID 和 Thread ID
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0 ~ 15
    int ty = threadIdx.y; // 0 ~ 15

    // 申请寄存器
    float c_reg[TM][TN] = {0.0f}; // 存储 8x8 = 64 个累加结果
    float a_reg[TM] = {0.0f};     // 缓存 8 个 A 元素
    float b_reg[TN] = {0.0f};     // 缓存 8 个 B 元素

    // 将 2D Thread ID 展平为 1D，用于协同搬运 Global -> Shared
    // 一个 Block 有 16 * 16 = 256 个线程
    int tid = ty * blockDim.x + tx; 

    // 计算当前线程搬运 A 和 B 时的偏移量
    // As 大小为 128 * 8 = 1024。256个线程，每个线程一次搬运 4 个元素
    int a_load_row = tid / BK; // 0 ~ 31
    int a_load_col = tid % BK; // 0 ~ 7
    int a_load_step = 256 / BK; // 32 (每次往下跨 32 行)

    // Bs 大小为 8 * 128 = 1024。
    int b_load_row = tid / BN; // 0 ~ 1
    int b_load_col = tid % BN; // 0 ~ 127
    int b_load_step = 256 / BN; // 2 (每次往下跨 2 行)

    // 外层循环：在 K 维度上滑动
    for (int t = 0; t < (K + BK - 1) / BK; ++t) {
        // 1. 协同搬运 A 到 Shared Memory (循环 4 次搬完 128 行)
        #pragma unroll
        for (int i = 0; i < BM; i += a_load_step) {
            int global_row = by * BM + a_load_row + i;
            int global_col = t * BK + a_load_col;
            if (global_row < M && global_col < K) {
                As[(a_load_row + i) * BK + a_load_col] = A[global_row * K + global_col];
            } else {
                As[(a_load_row + i) * BK + a_load_col] = 0.0f;
            }
        }

        // 2. 协同搬运 B 到 Shared Memory (循环 4 次搬完 8 行)
        #pragma unroll
        for (int i = 0; i < BK; i += b_load_step) {
            int global_row = t * BK + b_load_row + i;
            int global_col = bx * BN + b_load_col;
            if (global_row < K && global_col < N) {
                Bs[(b_load_row + i) * BN + b_load_col] = B[global_row * N + global_col];
            } else {
                Bs[(b_load_row + i) * BN + b_load_col] = 0.0f;
            }
        }

        __syncthreads();

        // 3. 核心计算：在当前 BK 块内，计算 8x8 的点积
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 将 A 的 8 个元素读入寄存器
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[(ty * TM + i) * BK + k];
            }
            
            // 将 B 的 8 个元素读入寄存器
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[k * BN + (tx * TN + j)];
            }

            // 执行 8x8 = 64 次 FFMA 运算 (全在寄存器内发生！)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. 将计算好的 8x8 寄存器结果写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = by * BM + ty * TM + i;
            int global_col = bx * BN + tx * TN + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = c_reg[i][j];
            }
        }
    }
}

// 供 C++ 调用的 v4 接口
void gemm_v4_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // Block 维度设为 16x16 = 256 个线程
    dim3 blockDim(16, 16);
    // Grid 维度基于 128x128 的大块划分
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_v4_2d_register_tiling_kernel<<<gridDim, blockDim>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- Day 2 巅峰形态: 向量化访存 + Bank Conflict 消除 ----------------

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 向量化读取宏：将地址强转为 float4 指针，一次性读写 16 Bytes
// #define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
__global__ void gemm_v5_vectorized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 魔法优化：BK 和 BN 后面加上 Padding (+1 / +8)，完美错开 Bank Conflict！
    __shared__ float As[BM * (BK + 1)]; 
    __shared__ float Bs[BK * (BN + 8)]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0~15
    int ty = threadIdx.y; // 0~15

    int tid = ty * blockDim.x + tx; // 0~255

    // 申请寄存器
    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM] = {0.0f};
    float b_reg[TN] = {0.0f};

    // ---------------- 向量化搬运的索引计算 ----------------
    // 一个线程一次搬运 float4 (4个元素)。256个线程一次能搬 1024 个元素。
    // As 需要 128*8 = 1024 个元素，刚好一次搬完！
    // Bs 需要 8*128 = 1024 个元素，也刚好一次搬完！

    // 搬运 A 时的索引 (A 是按行读取，每行 K 个元素，我们按 float4 读取，跨度缩小 4 倍)
    int load_a_row = tid / (BK / 4);     // tid / 2 
    int load_a_col = (tid % (BK / 4)) * 4; // (tid % 2) * 4

    // 搬运 B 时的索引 (B 也是按行读取，每行 N 个元素)
    int load_b_row = tid / (BN / 4);     // tid / 32
    int load_b_col = (tid % (BN / 4)) * 4; // (tid % 32) * 4

    for (int t = 0; t < (K + BK - 1) / BK; ++t) {
        // 1. 极速向量化搬运 A -> Shared Memory
        int global_a_row = by * BM + load_a_row;
        int global_a_col = t * BK + load_a_col;
        if (global_a_row < M && global_a_col < K) {
            // 一次性从 Global 读 4 个 float
            float4 tmp = FETCH_FLOAT4(A[global_a_row * K + global_a_col]);
            // 分别存入 Shared Memory (注意 As 的 stride 现在是 BK + 1)
            As[load_a_row * (BK + 1) + load_a_col + 0] = tmp.x;
            As[load_a_row * (BK + 1) + load_a_col + 1] = tmp.y;
            As[load_a_row * (BK + 1) + load_a_col + 2] = tmp.z;
            As[load_a_row * (BK + 1) + load_a_col + 3] = tmp.w;
        } else {
            As[load_a_row * (BK + 1) + load_a_col + 0] = 0.0f;
            As[load_a_row * (BK + 1) + load_a_col + 1] = 0.0f;
            As[load_a_row * (BK + 1) + load_a_col + 2] = 0.0f;
            As[load_a_row * (BK + 1) + load_a_col + 3] = 0.0f;
        }

        // 2. 极速向量化搬运 B -> Shared Memory
        int global_b_row = t * BK + load_b_row;
        int global_b_col = bx * BN + load_b_col;
        if (global_b_row < K && global_b_col < N) {
            // 一次性读 4 个 float
            float4 tmp = FETCH_FLOAT4(B[global_b_row * N + global_b_col]);
            // 注意 Bs 的 stride 是 BN + 8
            Bs[load_b_row * (BN + 8) + load_b_col + 0] = tmp.x;
            Bs[load_b_row * (BN + 8) + load_b_col + 1] = tmp.y;
            Bs[load_b_row * (BN + 8) + load_b_col + 2] = tmp.z;
            Bs[load_b_row * (BN + 8) + load_b_col + 3] = tmp.w;
        } else {
            Bs[load_b_row * (BN + 8) + load_b_col + 0] = 0.0f;
            Bs[load_b_row * (BN + 8) + load_b_col + 1] = 0.0f;
            Bs[load_b_row * (BN + 8) + load_b_col + 2] = 0.0f;
            Bs[load_b_row * (BN + 8) + load_b_col + 3] = 0.0f;
        }

        __syncthreads();

        // 3. 核心计算 (带有 Padding 补偿的读取)
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                // 注意这里加上了 BK + 1
                a_reg[i] = As[(ty * TM + i) * (BK + 1) + k];
            }
            
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                // 注意这里加上了 BN + 8
                b_reg[j] = Bs[k * (BN + 8) + (tx * TN + j)];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. 将结果写回 (此处为了极致性能，也应该用 float4 写回，
    // 但为了避免越界导致 core dump，我们在此依然保留基础的逐个写回机制，
    // 因为这部分的开销仅占总时间的极小一部分)
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_row = by * BM + ty * TM + i;
            int global_col = bx * BN + tx * TN + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = c_reg[i][j];
            }
        }
    }
}

// 供 C++ 调用的 v5 接口
void gemm_v5_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_v5_vectorized_kernel<<<gridDim, blockDim>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}