#pragma once
#include <iostream>
#include <functional>
#include <cmath>
#include "cuda_utils.cuh"

class CUDATester {
public:
    // 性能 Benchmark 工具 (支持预热和多次迭代求平均)
    static float benchmark(std::function<void()> kernel_launch, int num_iters = 10, int warmup_iters = 3) {
        // 1. 预热 (Warmup)：唤醒 GPU，避免首次启动的开销干扰计时
        for (int i = 0; i < warmup_iters; ++i) {
            kernel_launch();
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // 确保预热完成

        // 2. 创建 CUDA 事件用于高精度计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 3. 开始计时
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iters; ++i) {
            kernel_launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop)); // 阻塞等待所有操作完成

        // 4. 计算耗时
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // 5. 销毁事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return milliseconds / num_iters; // 返回单次执行的平均耗时
    }

    // 通用的结果比对工具
    template <typename T>
    static bool check_correctness(const T* cpu_ref, const T* gpu_res, size_t size, double tol = 1e-5) {
        for (size_t i = 0; i < size; ++i) {
            // 注意：对于浮点数，最好使用相对误差或带容差的绝对误差
            if (std::abs(cpu_ref[i] - gpu_res[i]) > tol) {
                std::cerr << "❌ [FAILED] Mismatch at index " << i 
                          << ": CPU=" << cpu_ref[i] 
                          << ", GPU=" << gpu_res[i] << std::endl;
                return false;
            }
        }
        std::cout << "✅ [PASSED] Result is CORRECT!" << std::endl;
        return true;
    }
};