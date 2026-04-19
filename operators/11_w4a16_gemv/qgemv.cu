#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"
#include <cuda_fp16.h>

// ==========================================================
// W4A16 GEMV: 极限内存压缩的动态反量化算子
// X: [K] (FP16) - 激活值
// W_packed: [N, K/2] (UINT8) - 权重，每行连续存放 K 个 INT4，2个挤在1个Byte里
// Scales: [N] (FP16) - 反量化缩放因子 (Per-Channel)
// Y: [N] (FP32) - 输出累加
// ==========================================================
__global__ void w4a16_gemv_v1_kernel(
    const half* X, 
    const uint8_t* W_packed, 
    const half* scales, 
    float* Y, 
    int N, 
    int K
) {
    // 1D Grid: 每个线程负责输出 Y 的一个元素 (对应 W 的第 n 行)
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    float scale = __half2float(scales[n]);

    // 每一行在物理内存上只有 K/2 个 Byte
    int row_offset = n * (K / 2);

    // 遍历这行的所有 Byte，每个 Byte 吐出 2 个权重进行计算
    for (int k_step = 0; k_step < K / 2; ++k_step) {
        // 从极其拥挤的 Global Memory 读出 1 个 Byte (包含 2 个 INT4 权重)
        uint8_t packed = W_packed[row_offset + k_step];

        // ----------------------------------------------------
        // 解包 1: 解析低 4 位 (对应第 2*k_step 个权重)
        // ----------------------------------------------------
        int8_t nibble0 = packed & 0x0F;
        // 有符号整数映射 (0~15 映射到 -8~7)
        if (nibble0 >= 8) nibble0 -= 16;
        float w0 = (float)nibble0 * scale;
        sum += __half2float(X[k_step * 2]) * w0;

        // ----------------------------------------------------
        // 解包 2: 解析高 4 位 (对应第 2*k_step + 1 个权重)
        // ----------------------------------------------------
        int8_t nibble1 = (packed >> 4) & 0x0F;
        if (nibble1 >= 8) nibble1 -= 16;
        float w1 = (float)nibble1 * scale;
        sum += __half2float(X[k_step * 2 + 1]) * w1;
    }

    Y[n] = sum;
}
// ==========================================================
// V2: 向量化 W4A16 GEMV (float4 加载 + 优化解包)
// ==========================================================
__global__ void w4a16_gemv_v2_vectorized_kernel(
    const half* X, 
    const uint32_t* W_packed, // 改用 uint32_t 一次读 8 个权重 (4 Bytes)
    const half* scales, 
    float* Y, 
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.0f;
    float scale = __half2float(scales[n]);

    // 指向当前行的起始物理位置
    // 每行 K 个权重，对应 K/2 字节，对应 K/8 个 uint32_t
    const uint32_t* row_ptr = W_packed + n * (K / 8);

    for (int k_idx = 0; k_idx < K / 8; ++k_idx) {
        // 核心突破：一次性从显存读入 8 个权重 (32-bit)
        uint32_t packed_vals = row_ptr[k_idx];
        
        // 获取 X 向量对应的 8 个输入值
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // 通过位移和掩码快速提取 4-bit 权重
            // 无需 if 分支，使用位运算技巧转换有符号数
            uint32_t u_w = (packed_vals >> (i * 4)) & 0x0F;
            int32_t i_w = (u_w >= 8) ? (int32_t)u_w - 16 : (int32_t)u_w;
            
            float x_val = __half2float(X[k_idx * 8 + i]);
            sum += x_val * (float)i_w * scale;
        }
    }
    Y[n] = sum;
}

// ==========================================================
// C++ 接口
// ==========================================================
// void w4a16_gemv_forward(torch::Tensor X, torch::Tensor W_packed, torch::Tensor scales, torch::Tensor Y) {
//     int K = X.size(0);
//     int N = scales.size(0);

//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;

//     w4a16_gemv_v1_kernel<<<blocks, threads>>>(
//         reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
//         W_packed.data_ptr<uint8_t>(),
//         reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
//         Y.data_ptr<float>(),
//         N, K
//     );
//     CUDA_CHECK(cudaGetLastError());
// }

// ==========================================================
// C++ 接口 (适配 V2 向量化版本)
// ==========================================================
void w4a16_gemv_forward(torch::Tensor X, torch::Tensor W_packed, torch::Tensor scales, torch::Tensor Y) {
    int K = X.size(0);
    int N = scales.size(0);

    // Python 端传进来的是 uint8_t 类型的 Tensor
    // 我们在这里直接强转为 uint32_t 指针，让 GPU 每次按 4 字节 (8 个 INT4) 吞吐！
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(W_packed.data_ptr<uint8_t>());

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    w4a16_gemv_v2_vectorized_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        w_ptr,
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        Y.data_ptr<float>(),
        N, K
    );
    CUDA_CHECK(cudaGetLastError());
}