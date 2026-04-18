# 🚀 Day 09: 2D Convolution (Conv2d)

## 📖 算子描述
二维卷积是卷积神经网络 (CNN) 的核心计算单元。它通过一个可学习的卷积核 (Kernel) 在输入图像上滑动，提取局部空间特征。

**物理维度定义**：
* 输入 (Input, X): `[N, C, H, W]`
* 权重 (Weight, K): `[K, C, R, S]`
* 输出 (Output, Y): `[N, K, H_out, W_out]`

其中：
* `N` = Batch Size
* `C` = 输入通道数 (In Channels)
* `K` = 输出通道数 (Out Channels)
* `H, W` = 输入图像的高和宽
* `R, S` = 卷积核的高和宽 (Kernel Size)

**计算公式 (Stride=1, Padding=0)**：
$$Y_{n, k, h_{out}, w_{out}} = \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} X_{n, c, h_{out}+r, w_{out}+s} \cdot Weight_{k, c, r, s}$$

## 💻 优化路线
* **V1**: 朴素滑窗法 (Naive Sliding Window)。一个线程计算输出张量中的一个元素，内部嵌套 3 重循环 (C, R, S)。这是极度 Memory Bound 且包含大量冗余访存的基线版本。
* **V2 (预留)**: Im2Col + GEMM 策略，将卷积彻底转化为矩阵乘法。
* **V3 (预留)**: 隐式 GEMM (Implicit GEMM) 与 Shared Memory 优化。