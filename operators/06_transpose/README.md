# 🚀 Day 06: Matrix Transpose & Bank Conflicts

## 📖 算子描述
矩阵转置是将形状为 $[M, N]$ 的矩阵 $A$ 转换为形状为 $[N, M]$ 的矩阵 $A^T$。
数学定义极为简单：$$A^T_{j, i} = A_{i, j}$$

虽然没有浮点运算，但它是极致的 **Memory Bound (访存密集型)** 算子。简单的 Global Memory 直接读写会导致极其严重的未合并访存（Uncoalesced Memory Access）。

## 💻 测试环境与硬件基准
* **GPU**: vGPU-48GB-425W
* **测试场景**: 大尺度正方形与长方形矩阵 (e.g., 4096x4096, 4096x8192)。
* **Baseline**: PyTorch `A.t().contiguous()` (注意：必须调用 contiguous 触发真实的数据搬运)。
* **核心指标**: 显存带宽 (Effective Bandwidth, GB/s)

---

## 🔬 核心优化技术解析

### 1. 共享内存作为“中转站” (Shared Memory Tiling)
为了保证 Global Memory 的读和写都能合并（Coalesced），我们将矩阵划分为 $32 \times 32$ 的小块（Tile）。
1. **读入**：多个线程从 Global Memory 连续读取一行，写入 Shared Memory 的一行。（全局连续，完美合并）
2. **转置**：在 Shared Memory 内部进行局部转置。
3. **写出**：多个线程从 Shared Memory 连续读取一列（实际上此时已变成转置后的行），写入 Global Memory 连续的一行。（全局连续，完美合并）

### 2. Bank Conflict 与 Padding 魔法
在上述第 3 步中，当 32 个线程试图从 Shared Memory 的一列读取数据时，由于 Shared Memory 分为 32 个 Bank，且一行的长度恰好是 32，导致这一列的所有元素全部落在同一个 Bank 上！
这会引发极度致命的 **32-way Bank Conflict**，硬件被迫将并发读取变成了 32 次串行读取。

**绝杀技 (Padding)**：只需在声明 Shared Memory 时将形状从 `[32][32]` 改为 `[32][33]`。每一行末尾加一个空位，巧妙地将下一行的 Bank 索引错开 1 位。这一行看似无用的代码，能让带宽瞬间翻倍！