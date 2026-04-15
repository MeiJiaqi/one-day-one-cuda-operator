# 🚀 Day 07: Tensor Core GEMM (WMMA)

## 📖 算子描述
本次挑战突破了传统的 CUDA Core (单精度/FP32) 编程范式，全面拥抱 Ampere/Ada 架构的最强算力单元——**Tensor Core (张量核心)**。
我们通过 NVIDIA 提供的 WMMA (Warp Matrix Multiply and Accumulate) API，实现了半精度输入 (FP16) 与单精度累加 (FP32) 的混合精度通用矩阵乘法 (GEMM)。

**核心计算维度**：
$$C_{M \times N} = A_{M \times K} \times B_{K \times N} + C_{M \times N}$$
(其中 A 和 B 为 FP16，C 为 FP32)

## 💻 测试环境与硬件基准
* **GPU**: vGPU-48GB-425W (Ampere/Ada 架构)
* **核心指标**: TFLOPS (每秒万亿次浮点运算)
* **Baseline**: PyTorch 底层调用的 `cuBLAS` (FP16 混合精度)
* **最终成绩**: **127.75 TFLOPS** (达成官方算力极限的 **89%**)

---

## 📈 性能演进纪实 (The TFLOPS Journey)

| Version | Optimization Technique | 4096x4096x4096 (TFLOPS) | 核心突破点 |
| :---: | :--- | :---: | :--- |
| **Baseline**| **PyTorch cuBLAS** | **143.40** | NVIDIA 官方汇编级天花板 |
| **v1** | Global Memory 直读 (Naive WMMA) | 36.33 | 极其严重的 I/O 瓶颈，Tensor Core 处于饥饿状态 |
| **v2** | Shared Memory Tiling (共享内存分块) | 60.14 | 用 SRAM 作为缓冲池，算力跃升 1.6 倍 |
| **v3** | Double Buffering + Padding | 87.34 | 软件流水线掩盖延迟，消灭 FP16 Bank Conflict |
| **v4** | **`float4` Vectorized Load (究极向量化)**| **127.75** | **128-bit 狂暴吞吐，打满物理带宽，摸到神之领域** |

---

## 🔬 核心优化技术解析

### 1. WMMA 编程模型 (Warp-Level Collaboration)
放弃单线程单元素的标量思维。将 32 个线程组成的 Warp 视为一个整体，协同完成 $16 \times 16 \times 16$ 矩阵块的加载、计算和写回。算力密度呈指数级上升。

### 2. 存储体冲突防御 (Bank Conflict Free for FP16)
在 FP16 下，两个 `half` 元素 (2 Bytes) 会挤在同一个 Bank 中，导致 16-way 甚至 32-way 的极其严重的读取冲突。
**对策**：在声明 Shared Memory 时引入 `PAD = 8` 的偏移魔法，彻底打乱内存映射，让 Tensor Core 的吸入速度毫无阻碍。

### 3. 软件流水线 (Double Buffering / Pipelining)
开辟 `Ping-Pong` 两个缓冲区。当 Tensor Core 在 `Buffer 0` 中疯狂进行矩阵乘加时，Warp 的线程正在异步地将下一个矩阵块从 Global Memory 搬运到 `Buffer 1` 中。完美重叠了“内存延迟”和“计算耗时”。

### 4. 128-bit 终极向量化访存 (Vectorized Load)
将极其低效的 16-bit (`half`) 搬运，通过 `reinterpret_cast<float4*>` 强行打包为 128-bit 的宽位指令。不仅将访存指令数砍掉 87.5%，更瞬间打满了显存控制器的物理吞吐极限。

---

## 🎯 总结与反思：最后的 10% 差距在哪里？
在不写 PTX/SASS 汇编的前提下，纯 C++ CUDA API 跑到 **127+ TFLOPS** 已是极高水准。距离 cuBLAS 的最后一点差距在于：
1. **Cache Thrashing (缓存颠簸)**：在 $8192 \times 8192$ 等超大维度下，未能实现 Thread Block Swizzling (线程块分发重排) 来最大化 L2 Cache 命中率。
2. **硬件级异步拷贝**：未使用 Ampere 专属的 `cp.async` 指令绕过寄存器直接进行 Global -> Shared 的搬运。
3. **更深的流水线**：cuBLAS 通常使用 3~4 级以上的 Multi-Stage 流水线来应对极致规模。

> **感悟**：算子开发不是敲代码，而是一场在硅片上与物理定律（时钟周期、带宽、缓存机制）展开的终极博弈。