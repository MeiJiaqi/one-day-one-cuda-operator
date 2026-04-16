# 🚀 Day 08: Prefix Sum / Scan (前缀和与并行扫描)

## 📖 算子描述
前缀和是打破并行计算中“强数据依赖”的终极试金石。本次挑战实现了极致的 **Work-Efficient Blelloch Scan**，支持跨 Block 级别的百万级大规模张量扫描。

## 💻 测试环境与硬件基准
* **GPU**: vGPU-48GB-425W
* **Baseline**: PyTorch `torch.cumsum(A, dim=0)`
* **正确性**: 在 1,000,000 规模下达成绝对零误差 (Max Error = 0.0000)

---

## 🔬 核心优化技术解析

### 1. 树状并行归约 (Up-Sweep & Down-Sweep)
抛弃 CPU $O(N)$ 的串行依赖，在 Shared Memory 中构建隐式二叉树。利用多线程在 $O(\log N)$ 的时间复杂度内完成局部扫描，极大地释放了 GPU 的并发潜力。

### 2. 存储体零冲突 (Bank Conflict-Free Padding)
Blelloch 算法中 2 的幂次跨度天然会引发极其严重的 32-way Bank Conflict。
**绝杀技**：引入 `SHMEM_IDX(n)` 宏，通过位移运算 `((n) >> 5)` 动态计算 Padding 偏移量，完美打乱内存对齐，实现零冲突的高速显存访问。

### 3. 全局递归调度 (Scan-Then-Propagate)
打破单 Block 2048 个元素的物理上限。
* **分治策略**：CPU 端动态调度，将百万级数组切分为 Block Tile。
* **层级抽象**：提取每个 Tile 的 Block Sum，对其进行原地的递归扫描，最后通过 `Propagate Kernel` 广播叠加 Base 偏移量。

## 🎯 总结与反思
* **瓶颈转移**：通过递归算法成功突破了长度上限，但多趟内核启动 (Multi-Pass) 和反复的 Global Memory 读写成为了新的瓶颈。
* **未来展望**：要达到工业级 CUB 库的性能（0.01ms），必须重构调度逻辑，引入原子锁机制实现 **单趟解耦回看 (Single-Pass Decoupled Look-back)**，彻底消灭内核启动开销。