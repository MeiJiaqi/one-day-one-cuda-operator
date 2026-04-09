
# 🚀 One-Day-One-CUDA-Operator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

深入底层，从零开始的 CUDA 算子开发实战。本项目旨在通过每日实现和优化一个 CUDA 算子，系统性地掌握 GPU 并行计算原理、内存架构优化以及底层性能调优。

不同于网上松散的 `.cu` 脚本片段，本项目构建了一个**现代化、可复用的工业级算子开发框架**。底层采用纯 CUDA/C++ 压榨极限性能，上层通过 PyTorch C++ Extension 暴露友好的 Python 接口，实现真正的**零拷贝 (Zero-Copy)** 交互与自动化评测。

## ✨ 核心特性 (Key Features)

* **🌉 极致架构解耦**：算子核心逻辑 (`operators/`) 与 Python 绑定层 (`csrc/`) 完全分离。写 CUDA 就是纯粹写 CUDA，无需被环境配置干扰。
* **⚡ 零拷贝 (Zero-Copy) 交互**：全面接入 PyTorch Tensor 生态，直接向底层传递 GPU 显存指针，拒绝高昂的 Host-to-Device 传输开销，评测最真实的算子性能。
* **🛠️ 自动化构建系统**：智能的 `setup.py` 自动递归扫描新增的 `.cu` 文件并完成编译。新增算子无需修改任何编译脚本，真正的“开箱即用”。
* **📊 硬件级基准测试**：提供内置的 Python 基准测试脚本，自动计算 Kernel 耗时、实际内存带宽 (GB/s) 与吞吐量 (GFLOPS)，并与 PyTorch 原生算子硬碰硬对标。

## 📂 目录架构 (Architecture)

```text
one-day-one-cuda-operator/
├── csrc/                           # C++ 与 Python 的桥梁层 (统一注册入口)
│   ├── ops.h                       # 集中管理所有算子的 C++ 接口声明
│   └── extension.cpp               # 唯一的 Pybind11 模块注册中心
├── common/                         # 公共基础设施层
│   └── cuda_utils.cuh              # 包含 CUDA_CHECK 等错误捕获宏
├── operators/                      # 核心算子开发区 (主战场)
│   ├── 01_vector_add/              # Day 1: 向量加法 (Memory Bound)
│   └── ...                         
├── my_cuda_ops/                    # Python API 封装层 
│   └── __init__.py                 # 提供带 Type Hint 的友好 Python 接口
├── scripts/                        # 性能评测脚本
│   └── benchmark.py                # 自动化正确性校验与性能测试
└── setup.py                        # 自动化 C++ 扩展编译脚本
```

## 🛠️ 快速开始 (Quick Start)

### 1. 环境依赖
请确保你的系统中已安装：
* NVIDIA 显卡驱动 & CUDA Toolkit (推荐 >= 11.8)
* Python >= 3.8
* PyTorch (需带有 CUDA 支持)

### 2. 编译与安装
克隆仓库后，在项目根目录执行以下命令，将算子库安装到当前 Python 环境中：
```bash
git clone [https://github.com/your-username/one-day-one-cuda-operator.git](https://github.com/your-username/one-day-one-cuda-operator.git)
cd one-day-one-cuda-operator

# 编译并以开发者模式安装
pip install -e .
```

### 3. 运行基准测试
```bash
python scripts/benchmark.py
```
*预期输出示例：*
```text
Testing Vector Add with 10485760 elements...
✅ [PASSED] CUDA Result matches PyTorch Reference!

Running Benchmark (100 iterations)...
------------------------------
Latency   : 0.3150 ms
Bandwidth : 399.45 GB/s
------------------------------
```

## 🧑‍💻 开发指南：如何新增一个算子？

借助本框架，新增算子只需简单的 **3 步**：

1. **写核函数**：在 `operators/` 新建文件夹（如 `02_gemm/gemm.cu`），编写 CUDA 核函数及 C++ 启动器，并在 `csrc/ops.h` 中声明。
2. **注册绑定**：在 `csrc/extension.cpp` 中通过 Pybind11 注册你的新算子（仅需加一行代码）。
3. **Python 包装**：在 `my_cuda_ops/__init__.py` 中添加带注释的 Python 包装函数。

最后，运行 `pip install -e .`，即可在 Python 中畅快调用！

## 📅 算子路线图 (Roadmap)

| Day | Operator | Category | Core Optimizations | Status |
| :---: | :--- | :--- | :--- | :---: |
| 01 | **Vector Add** | Element-wise | Memory Coalescing | ✅ |
| 02 | **GEMM** | MatMul | Shared Memory, Tiling, Bank Conflict Fix | 🚧 |
| 03 | **Reduce Sum** | Reduction | Tree Reduction, Warp Shuffle | 🗓️ |
| 04 | **Softmax** | Transformer | Online Softmax, Register Tiling | 🗓️ |
| 05 | **LayerNorm** | Transformer | Fused Kernels | 🗓️ |
| 06 | **Prefix Sum** | Scan | Blelloch Scan | 🗓️ |

## 🤝 贡献 (Contributing)
如果你发现了 Bug、有更好的优化思路，或者愿意贡献新的算子，非常欢迎提交 Pull Request 或发起 Issue 讨论！

## 📄 开源协议 (License)
本项目基于 [MIT License](LICENSE) 开源。

---
*Powered by C++ & CUDA. Keep Coding, Keep Optimizing.*
