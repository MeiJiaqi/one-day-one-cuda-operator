# 🚀 Day 05: Layer Normalization (LayerNorm)

## 📖 算子描述

LayerNorm 是 Transformer 模型（如 GPT, BERT）中最关键的正则化层。它不仅能加速模型收敛，还能极大地提升数值稳定性。
与 BatchNorm 不同，LayerNorm 是在**特征维度 (Hidden Size)** 上独立进行归一化的。

**核心数学公式**：
给定输入的一行数据 $x \in \mathbb{R}^H$，LayerNorm 的计算过程分为 4 步：

## **核心数学公式**：

给定输入的一行数据 $x \in \mathbb{R}^H$，LayerNorm 的计算过程分为 4 步：

1. 计算均值：

$$

\mu = \frac{1}{H} \sum_{i=1}^H x_i

$$

2. 计算方差：

$$

\sigma^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2

$$

3. 归一化：

$$

\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}

$$

4. 缩放与平移 (Scale & Shift)：

$$

y_i = \gamma_i \hat{x}_i + \beta_i

$$

💻 测试环境与硬件基准

- **GPU**: vGPU-48GB-425W
- **测试场景**: 大语言模型常见的 Hidden Size 维度 (如 1024, 4096, 8192)。
- **Baseline**: PyTorch `torch.nn.functional.layer_norm`
- **核心指标**: 显存带宽 (Effective Bandwidth, GB/s)

---

## 🔬 核心优化技术解析

### 1. Welford 算法 (一趟求均值与方差)

朴素的方差公式 $E[x^2] - (E[x])^2$ 会发生**精度截断 (Catastrophic Cancellation)**，导致算出负数的方差进而出现 `NaN`。
我们将引入 **Welford 算法**，通过增量递推的方式，在一次遍历中极其稳定地同时算出 $\mu$ 和 $\sigma^2$。

### 2. Kernel Fusion (终极融合)

传统的实现可能会产生大量临时 Tensor。我们将把“求均值 -> 求方差 -> 归一化 -> 乘 Gamma 加 Beta”这 4 个步骤，全部塞进**同一个 Kernel** 中。

- **终极目标**：做到极其奢侈的 **1次读入，1次写出**，绝不触碰多余的 Global Memory。

### 3. 向量化访存 (float4)

传承 Day 4 的绝招，使用 `float4` 一次性读取 16 Bytes 数据，拉满显存总线利用率。