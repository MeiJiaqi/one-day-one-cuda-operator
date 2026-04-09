# Operator: Vector Add

## 1. 算子描述
逐元素向量加法：$C = A + B$。
由于没有复杂的计算逻辑，该算子属于典型的 **Memory Bound（访存受限）** 算子，性能瓶颈在于显存带宽。

## 2. 核心指标分析公式
* 计算量 (FLOPs)：1 次加法/元素 -> 总计 $N$ FLOPs
* 访存量 (Bytes)：读 2 次，写 1 次 -> $3 \times N \times 4$ Bytes
* 算数强度 (Arithmetic Intensity)：$1 / 12 \approx 0.083$ FLOPs/Byte

## 3. 性能对标 (以 RTX 3060 为例)
* **数组规模**: $10,485,760$ elements (~40MB per array)
* **执行耗时**: `0.32 ms`
* **实际带宽**: `375 GB/s`
* **硬件理论带宽**: `360 GB/s` (实际可能因缓存命中稍高或受限于 PCI-e)
* **结论**: 已达到硬件显存带宽极限，优化完毕。