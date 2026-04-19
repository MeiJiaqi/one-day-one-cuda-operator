# 🚀 Day 10: Flash Decoding (Online Softmax)

## 📖 算子描述
针对 Transformer 大语言模型生成阶段 (Generation/Decoding) 的终极带宽优化。
在 Decoding 阶段，每次只生成一个 Token，因此 Query 的长度为 1。我们要让这 1 个 Query 与长达数万的 KV Cache 进行注意力计算。

**物理维度定义**：
* $Q$ (Query): `[Batch, Heads, Head_Dim]`
* $K$ (Key): `[Batch, Heads, Seq_Len, Head_Dim]`
* $V$ (Value): `[Batch, Heads, Seq_Len, Head_Dim]`
* $O$ (Output): `[Batch, Heads, Head_Dim]`

## 💻 核心优化：Online Softmax 融合
通过单趟循环 (Single-Pass) 遍历序列长度 `Seq_Len`，在寄存器中实时维护 `Running Max` 和 `Running Sum`，彻底消除 $N \times N$ 注意力分数矩阵的 Global Memory 读写开销。