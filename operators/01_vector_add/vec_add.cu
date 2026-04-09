#include <iostream>
#include <vector>
#include "../../common/cuda_utils.cuh"
#include "../../common/tester.cuh"

// 1. 算子核函数 (专注计算逻辑)
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// 2. CPU Reference (专注正确性)
void vecAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 设置数据规模 (10M 个 float)
    int n = 10 * 1024 * 1024; 
    size_t size = n * sizeof(float);

    // 分配 Host 内存并初始化
    std::vector<float> h_A(n, 1.2f);
    std::vector<float> h_B(n, 3.4f);
    std::vector<float> h_C(n, 0.0f);
    std::vector<float> h_C_ref(n, 0.0f);

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Host to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // 配置线程块
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // ------------------- 核心测试流程 -------------------
    
    // 步骤 A: 运行 CPU 版本作为 Baseline
    vecAddCPU(h_A.data(), h_B.data(), h_C_ref.data(), n);

    // 步骤 B: 使用 Lambda 封装 Kernel 调用，传入 Tester 进行 Benchmark
    auto run_kernel = [&]() {
        vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    };
    
    float avg_time_ms = CUDATester::benchmark(run_kernel, 10, 3);

    // 步骤 C: Device to Host 并验证正确性
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));
    CUDATester::check_correctness(h_C_ref.data(), h_C.data(), n);

    // --------------------------------------------------

    // 打印专业的性能指标
    // Vector Add 的访存量：读 A，读 B，写 C，总计 3 次
    double bytes_processed = 3.0 * size; 
    double bandwidth_GBs = (bytes_processed / 1e9) / (avg_time_ms / 1000.0);

    std::cout << "--- Performance Report ---" << std::endl;
    std::cout << "Array Size : " << n << " elements" << std::endl;
    std::cout << "Latency    : " << avg_time_ms << " ms" << std::endl;
    std::cout << "Bandwidth  : " << bandwidth_GBs << " GB/s" << std::endl;

    // 释放内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}