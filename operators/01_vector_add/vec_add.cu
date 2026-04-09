#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh" // 如果你有错误检查宏的话

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// 暴露给 ops.h 的接口
void vector_add_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        n
    );
}