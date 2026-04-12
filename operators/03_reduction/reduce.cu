#include "../../csrc/ops.h"
#include "../../common/cuda_utils.cuh"

// v1: 基于 Shared Memory 的树状归约 (经典实现)
__global__ void reduce_v1_shared_mem(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 加载数据到 Shared Memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 2. 树状归约过程 (In-place reduction in shared memory)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 将每个 Block 的局部和写入全局内存
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// v2: 基于 Warp Shuffle 的极致优化版

__global__ void reduce_v2_warp_shuffle(const float* input, float* output, int n) {
    __shared__ float warp_sums[32]; // 最多支持 1024 线程 (32 warps)

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (i < n) ? input[i] : 0.0f;

    // 1. 每个 Warp 内部先求和
    sum = warpReduceSum(sum);

    // 2. 每个 Warp 的首线程将结果存入共享内存
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    // 3. 最后由第一个 Warp 对共享内存中的各 Warp 和进行最后一次归约
    sum = (threadIdx.x < blockDim.x / 32.0) ? warp_sums[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    // 4. 写回
    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}

// 接口封装
void reduce_forward(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    // 暂时调用 v2 版本
    reduce_v2_warp_shuffle<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
}

// ---------------- Day 3 进阶: Thread Coarsening (Grid-Stride Loop) ----------------

__global__ void reduce_v3_grid_stride_kernel(const float* input, float* output, int n) {
    __shared__ float warp_sums[32];

    // 获取当前线程在整个 Grid 中的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 获取整个 Grid 的跨度 (总线程数)
    int stride = blockDim.x * gridDim.x;

    // 1. 线程粗化：每个线程不再只读 1 个元素，而是循环读取多个元素！
    // 假设有 100 万数据，但只启动了 1 万个线程，每个线程就会在这里循环 100 次
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }

    // 2. Warp 内部归约
    sum = warpReduceSum(sum);

    // 3. 跨 Warp 归约 (存入 Shared Memory)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    // 4. 最后一个 Warp 完成 Block 内的最终归约
    sum = (threadIdx.x < blockDim.x / 32.0) ? warp_sums[lane] : 0.0f;
    if (wid == 0) sum = warpReduceSum(sum);

    // 5. 写回
    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}

// 供 C++ 调用的 v3 接口
void reduce_v3_forward(torch::Tensor input, torch::Tensor output) {
    int n = input.numel();
    int threads = 1024;
    
    // 【核心改变】：我们不再启动无穷无尽的 Block。
    // 限制 Block 的最大数量，通常设为 SM 数量的整数倍。
    // 假设高端显卡有 100+ 个 SM，我们限制最多启动 1024 个 Block 足矣。
    int max_blocks = 1024; 
    int blocks = std::min((n + threads - 1) / threads, max_blocks);

    reduce_v3_grid_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    CUDA_CHECK(cudaGetLastError());
}