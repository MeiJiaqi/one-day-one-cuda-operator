# 🚀 Day 04: Softmax (Online & Fused)

## 📖 算子描述
Softmax 是 Transformer 架构中的核心激活函数。本次挑战的目标是实现一个**数值安全（Safe）**且**性能极致（Fused）**的 Softmax 算子。

我们经历了从 **3-Pass**（三次访存）到 **2-Pass**（在线归一化）的飞跃，最终在 $4096 \times 4096$ 规模下跑出了硬件物理带宽的极限。

## 💻 测试环境与硬件基准
* **GPU**: vGPU-48GB-425W (Ampere/Ada 架构)
* **核心指标**: 显存带宽 (Effective Bandwidth, GB/s)
* **Baseline (PyTorch)**: `946.20 GB/s`
* **终极成绩 (My CUDA)**: **946.26 GB/s** (微秒级绝杀官方库)

---

## 📈 性能演进纪实 (The Optimization Journey)

| Version | Optimization Technique | Bandwidth (GB/s) | Note |
| :---: | :--- | :--- | :--- |
| **v1** | Fused 3-Pass (Max -> Sum -> Div) | 896.77 | 传统的三次全局内存遍历 |
| **v2** | **Online Softmax + float4** | **946.26** | **一次遍历搞定 Max/Sum，带宽起飞** |

---

## 🔬 核心优化技术解析

### 1. 数值安全 (Safe Softmax)
为了防止 $e^x$ 在 FP32 下溢出（Overflow），我们在计算前必须减去该行的最大值 $m$。
$$\text{Softmax}(x_i) = \frac{e^{x_i - m}}{\sum e^{x_j - m}}$$

### 2. Online Softmax (在线归一化算法)
这是本次挑战的**灵魂**。传统算法需要先找 Max 才能算 Sum，导致必须扫两遍数据。
**Online 算法**利用递推公式：当遇到新的局部最大值 $m_{new}$ 时，通过补偿因子 $e^{m_{old} - m_{new}}$ 动态修正之前的局部指数和。
* **收益**：将算子的访存压力从“3次读/2次写”降低到了理论极限的“1次读/1次写”。

### 3. Vectorized Load/Store (float4)
利用 `reinterpret_cast<float4*>`，让 GPU 指令流水线一次性吞掉 128-bit 数据。在大尺寸（4096）下，这保证了显存总线被完全填满，没有一丝空转。

### 4. 线程同步与广播 (Sync & Broadcast)
修正了 v2 初期出现的“自私线程”问题。通过 Shared Memory 建立了两道广播关卡：
* **Max 广播**：确保所有线程基于同一最大值计算补偿因子。
* **Sum 广播**：确保归一化阶段的全局和绝对一致。

---

## 🎯 总结与反思
* **算法胜过微调**：Online Softmax 的数学重构比任何代码微调带来的提升都大。
* **特化胜过通用**：我们的算子因为省去了 PyTorch 的维度检查和启发式算法开销，在特定维度（4096）下跑出了超越工业级库的性能。
* **精度意识**：在并行计算中，同步的时机（Barrier）直接决定了结果的死活（NaN/inf）。