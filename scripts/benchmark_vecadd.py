import torch
import my_cuda_ops
import time

def benchmark_function(func, *args, warmup=20, iters=100):
    """
    高精度 CUDA 函数耗时评测
    """
    # 1. Warmup: 唤醒 GPU，确保时钟频率达到最高
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # 2. 创建 CUDA 事件计时器
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 3. 开始计时
    start_event.record()
    for _ in range(iters):
        func(*args)
    end_event.record()
    
    # 4. 强制等待 GPU 队列中的任务全部执行完毕
    torch.cuda.synchronize()

    # 返回单次平均耗时 (毫秒)
    return start_event.elapsed_time(end_event) / iters

def run_performance_test():
    # 测试的数组规模：从 2^10 (极小) 到 2^28 (极大，约 1GB 数据)
    sizes = [2**10, 2**14, 2**18, 2**22, 2**24, 2**26, 2**28]
    
    print(f"{'Size (N)':<12} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch GB/s':<15} | {'My CUDA GB/s':<15}")
    print("-" * 80)

    for n in sizes:
        # 分配数据 (确保内存连续)
        A = torch.randn(n, device='cuda', dtype=torch.float32)
        B = torch.randn(n, device='cuda', dtype=torch.float32)
        C_out = torch.empty_like(A)

        # ---------------- 验证正确性 ----------------
        my_cuda_ops.vector_add(A, B, C_out) # 假设你修改了接口，直接把 C 传进去
        C_ref = A + B
        max_err = torch.max(torch.abs(C_out - C_ref)).item()
        assert max_err < 1e-5, f"计算错误！规模 {n} 下的最大误差为 {max_err}"

        # ---------------- 评测 PyTorch 原生算子 ----------------
        # 使用 Lambda 封装，防止 Python 垃圾回收带来的耗时干扰
        pytorch_time = benchmark_function(lambda: torch.add(A, B, out=C_out))

        # ---------------- 评测 你的 CUDA 算子 ----------------
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.vector_add(A, B, C_out))

        # ---------------- 计算内存带宽 ----------------
        # Vector Add 的访存量：读取 A，读取 B，写入 C_out。一共 3 次。
        # 字节数 = 3 * N * 4 (float32 占 4 字节)
        bytes_accessed = 3 * n * 4 
        
        # Bandwidth (GB/s) = (Bytes / 10^9) / (Time / 1000)
        pytorch_bw = (bytes_accessed / 1e9) / (pytorch_time / 1000)
        my_cuda_bw = (bytes_accessed / 1e9) / (my_cuda_time / 1000)

        # 打印对齐的表格数据
        print(f"{n:<12} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_bw:<15.2f} | {my_cuda_bw:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 Vector Add 算子性能对标 (Baseline: PyTorch)")
    run_performance_test()