#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"

// ==========================================================
// V1: 朴素版 Conv2d (NCHW 布局, 步长 Stride=1, 填充 Padding=0)
// 这是一个被称为“七重循环”噩梦的最基础实现
// ==========================================================
__global__ void conv2d_v1_naive_kernel(
    const float* X,       // [N, C, H, W]
    const float* Weight,  // [K, C, R, S]
    float* Y,             // [N, K, H_out, W_out]
    int N, int C, int H, int W,
    int K_out, int R, int S, 
    int H_out, int W_out
) {
    // 1. 获取当前线程的全局 1D 索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 输出张量的总元素个数
    int total_elements = N * K_out * H_out * W_out;
    if (idx >= total_elements) return;

    // 2. 将 1D 索引解码为 4D 坐标 (n, k, h_out, w_out)
    // 采用 NCHW 内存布局的逆向取余法
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k     = (idx / (W_out * H_out)) % K_out;
    int n     = idx / (W_out * H_out * K_out);

    // 3. 开始执行卷积核的乘加聚合运算 (Inner loops)
    float sum = 0.0f;

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                // 计算输入图像 X 的 1D 物理索引
                int h_in = h_out + r;
                int w_in = w_out + s;
                int x_idx = ((n * C + c) * H + h_in) * W + w_in;

                // 计算权重 Weight 的 1D 物理索引
                int w_idx = ((k * C + c) * R + r) * S + s;

                sum += X[x_idx] * Weight[w_idx];
            }
        }
    }

    // 4. 写回输出
    Y[idx] = sum;
}

// ==========================================================
// C++ 接口
// ==========================================================
void conv2d_forward(torch::Tensor X, torch::Tensor Weight, torch::Tensor Y) {
    // 提取维度信息
    int N = X.size(0);
    int C = X.size(1);
    int H = X.size(2);
    int W = X.size(3);

    int K_out = Weight.size(0);
    int R = Weight.size(2);
    int S = Weight.size(3);

    // 这里先硬编码 Stride=1, Padding=0
    int H_out = H - R + 1;
    int W_out = W - S + 1;

    int total_elements = N * K_out * H_out * W_out;
    
    // Grid 和 Block 划分
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv2d_v1_naive_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        Weight.data_ptr<float>(),
        Y.data_ptr<float>(),
        N, C, H, W,
        K_out, R, S,
        H_out, W_out
    );
    CUDA_CHECK(cudaGetLastError());
}

// ==========================================================
// V2 辅助 Kernel: Im2Col 内存展开
// 将 [N, C, H, W] 展开为 [N * H_out * W_out, C * R * S]
// ==========================================================
__global__ void im2col_kernel(
    const float* data_im, 
    float* data_col,
    int N, int C, int H, int W,
    int R, int S, 
    int H_out, int W_out
) {
    // 总共需要展开的列数 (即目标矩阵的行数 N * H_out * W_out)
    int num_kernels = N * H_out * W_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_kernels) return;

    // 解析当前属于哪个 N, h_out, w_out
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int n     = index / (W_out * H_out);

    // 目标矩阵 data_col 的每一行长度为 C * R * S
    int col_row_offset = index * (C * R * S);
    
    // 把滑窗内的所有元素提取并拉平
    int col_idx = 0;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = h_out + r;
                int w_in = w_out + s;
                int im_idx = ((n * C + c) * H + h_in) * W + w_in;
                
                data_col[col_row_offset + col_idx] = data_im[im_idx];
                col_idx++;
            }
        }
    }
}

// ==========================================================
// C++ 接口 (混合 Im2Col 与 cuBLAS/PyTorch GEMM)
// ==========================================================
void conv2d_im2col_forward(torch::Tensor X, torch::Tensor Weight, torch::Tensor Y_col) {
    int N = X.size(0);
    int C = X.size(1);
    int H = X.size(2);
    int W = X.size(3);
    int K_out = Weight.size(0);
    int R = Weight.size(2);
    int S = Weight.size(3);
    int H_out = H - R + 1;
    int W_out = W - S + 1;

    int num_kernels = N * H_out * W_out;
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;

    im2col_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        Y_col.data_ptr<float>(),
        N, C, H, W,
        R, S, H_out, W_out
    );
    CUDA_CHECK(cudaGetLastError());
}