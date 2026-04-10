# 🚀 Day 02: General Matrix Multiply (GEMM)

## 📖 算子描述
通用矩阵乘法（GEMM, $C = A \times B$）是深度学习（尤其是 Transformer 和 CNN）中最核心的底层算子。与 Day 01 的 Vector Add 不同，GEMM 是典型的**计算密集型 (Compute Bound)** 任务。

本次挑战的目标：不依赖任何第三方库（如 cuBLAS），纯手工使用 C++ 和 CUDA (FP32) 压榨 GPU 的极限浮点运算算力。

## 💻 测试环境与硬件基准
* **GPU**: vGPU-48GB-425W (Ampere/Ada 架构顶级计算卡)
* **矩阵规模**: $M=4096, K=4096, N=4096$
* **Baseline (PyTorch)**: `~48.58 TFLOPS` (注：PyTorch 默认开启了 TF32 截断加速，触发了 Tensor Core)
* **FP32 理论物理峰值**: 约 40 TFLOPS (纯 CUDA Core)

---

## 📈 性能演进纪实 (The Optimization Journey)

我们在一天之内迭代了 5 个大版本，将吞吐量提升了 **43 倍**，最终达到了 PyTorch (TF32) 性能的 **73.5%**，几乎榨干了这块硅晶片的纯 FP32 物理极限。

| Version | Optimization Technique | Latency (ms) | Throughput (TFLOPS) | Speedup |
| :---: | :--- | :--- | :--- | :--- |
| **v1** | Naive Global Memory | 40.0+ | **0.82** | 1.0x |
| **v2** | Shared Memory Tiling (32x32) | 28.0+ | **3.73** | 4.5x |
| **v3** | 1D Register Tiling (1x4) | 12.31 | **15.02** | 18.3x |
| **v4** | 2D Register Tiling (8x8) | 4.09 | **23.64** | 28.8x |
| **v5** | Vectorized Float4 + Bank Conflict Padding | **3.84** | **35.74** | **43.5x** |

---

## 🔬 核心优化技术解析

### 1. Shared Memory Tiling (共享内存分块)
* **痛点**：v1 版本中，每次计算乘加都需要从极慢的 Global Memory 读取数据，导致计算单元（ALU）大量闲置。
* **解法**：按 `32x32` 对矩阵进行分块（Tiling）。由 Thread Block 协同将数据预载到 Shared Memory 中。全局访存量理论上下降到原来的 1/32。

### 2. 2D Register Tiling (寄存器分块)
* **痛点**：即使是 Shared Memory，读取（LDS指令）依然有延迟。在 v2 中，2条读取指令只能换来 1条乘加指令（FFMA），读写计算比极低。
* **解法**：让每个线程负责计算 `8x8` 的结果块。通过在**寄存器 (Register)** 中复用数据，将 16 次 Shared Memory 读取转化为 64 次 FFMA 运算。指令读写计算比从 `2:1` 暴降至 `0.25:1`，彻底解放计算单元。

### 3. Vectorized Memory Access (向量化访存)
* **解法**：将普通的 `float*` 指针通过 `reinterpret_cast<const float4*>` 强转。在从 Global Memory 向 Shared Memory 搬运数据时，一条指令同时读取 16 Bytes (4个float)，极大降低了访存指令的发射频率，拉满显存带宽。

### 4. Shared Memory Bank Conflict Resolution (存储体冲突消除)
* **痛点**：在 Shared Memory 中按列读取数据时，由于步长为 128（32个Bank的整数倍），导致了严重的内存踩踏事故（Bank Conflict），并行读取退化为串行。
* **解法**：应用 **Padding（填充）魔法**。在申请 Shared Memory 时将步长设为 `BK + 1` 和 `BN + 8`。如同错开的齿轮，完美打散了所有线程的访存地址，实现了 0 Conflict 的极速读取。

---

## 🎯 总结与反思

* **内存层级是 CUDA 的灵魂**：优化的本质，就是不断地把数据从慢速存储（Global）搬运到快速存储（Shared -> Register），并最大化“数据复用率”。
* **关于 PyTorch 的差距**：PyTorch 利用了 Ampere 架构的特性，在底层默默开启了 **TF32**，将原本属于 CUDA Core 的 FP32 运算转移到了算力恐怖的 **Tensor Core** 上。纯净的 FP32 跑到 35.7 TFLOPS 已是当前技术路线的极限。
* **Next Steps**: 未来若需突破百 TFLOPS 大关，需引入 `<mma.h>` 库，从 Thread 级编程跃迁至 Warp 级编程，直接调用硬件底层的 WMMA 接口。