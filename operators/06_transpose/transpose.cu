#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"

// Tile 大小定义为 32x32
#define TILE_DIM 32
#define BLOCK_ROWS 8 // 每个 Block 实际分配的线程是 32x8=256 个，每个线程处理 4 个元素

// ==========================================================
// V1 共享内存基础版 (含有严重的 Bank Conflict)
// ==========================================================
__global__ void transpose_v1_shmem_kernel(const float* input, float* output, int num_rows, int num_cols) {
    // 申请 32x32 的 Shared Memory 中转站
    // 🚨 警告：这就是导致 32-way Bank Conflict 的罪魁祸首形状！
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // 计算当前线程负责读入的全局坐标 (x 是列，y 是行)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取 Global Memory -> 写入 Shared Memory (按行)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_cols && (y + j) < num_rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * num_cols + x];
        }
    }

    __syncthreads(); // 等待整个 Tile 都加载到 Shared Memory

    // 计算转置后的写出全局坐标
    // 注意：网格的 blockIdx.x 和 blockIdx.y 也互换了！
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 2. 从 Shared Memory 按列读取 -> 合并写入 Global Memory (按行)
    // 🚨 就在这里：当 threadIdx.x 保持不变，改变 [threadIdx.x][threadIdx.y + j] 提取列数据时，引发冲突！
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_rows && (y + j) < num_cols) {
            // output 现在的形状是 [num_cols, num_rows]
            output[(y + j) * num_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ==========================================================
// V2 共享内存 Padding 魔法版 (零冲突 Conflict-Free)
// ==========================================================
__global__ void transpose_v2_padded_kernel(const float* input, float* output, int num_rows, int num_cols) {
    // ✨ 核心魔法就在这里：列维度加上 1
    // 从 [32][32] 变成了 [32][33]
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 1. 合并读取 Global Memory -> 写入 Shared Memory (按行)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_cols && (y + j) < num_rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * num_cols + x];
        }
    }

    __syncthreads(); 

    // 翻转坐标，准备写回
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 2. 从 Shared Memory 按列读取 -> 合并写入 Global Memory (按行)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_rows && (y + j) < num_cols) {
            // ✨ 由于每一行多加了一个元素的空间，
            // 当 threadIdx.y 改变时，同一列的元素在物理地址上刚好错开了 33 个 float，
            // 完美规避了 32-way Bank Conflict！
            output[(y + j) * num_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ==========================================================
// C++ 接口
// ==========================================================
void transpose_forward(torch::Tensor input, torch::Tensor output) {
    int num_rows = input.size(0);
    int num_cols = input.size(1);

    // 线程块配置：x维32，y维8。一共256个线程处理一个 32x32 的块
    dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
    dim3 blocks((num_cols + TILE_DIM - 1) / TILE_DIM, 
                (num_rows + TILE_DIM - 1) / TILE_DIM, 1);

    transpose_v2_padded_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_rows, 
        num_cols
    );
    CUDA_CHECK(cudaGetLastError());
}