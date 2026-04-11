import torch
import my_cuda_ops

def benchmark_function(func, *args, warmup=20, iters=100):
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters

def run_reduce_benchmark():
    # 测试的数据规模：从 100万 到 2.6亿 元素
    sizes = [2**20, 2**22, 2**24, 2**26, 2**28]
    
    print(f"{'Size (N)':<12} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch GB/s':<15} | {'My CUDA GB/s':<15}")
    print("-" * 80)

    for n in sizes:
        # 归约操作对于浮点数累加的顺序极其敏感
        # 为了防止随机数累加误差过大，我们改用 0~1 的均匀分布随机数
        A = torch.rand(n, device='cuda', dtype=torch.float32)

        # 1. 验证正确性
        C_custom = my_cuda_ops.reduce_sum_v3(A)
        C_ref = torch.sum(A)
        
        # 注意：由于并行累加顺序不同，FP32 会产生较大的截断误差。
        # 当数组有几千万大小时，PyTorch 和我们算出来的尾数不一样是完全正常的！
        # 我们使用相对误差来进行校验
        rel_err = torch.abs((C_custom - C_ref) / C_ref).item()
        assert rel_err < 1e-4, f"精度误差过大！相对误差: {rel_err}, Ref: {C_ref.item()}, Custom: {C_custom.item()}"

        # 2. 评测耗时
        pytorch_time = benchmark_function(lambda: torch.sum(A))
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.reduce_sum_v3(A))

        # 3. 计算带宽 (GB/s)
        # Reduction 的有效访存量约为：把 N 个 float 读一遍 (忽略极小量的写回)
        bytes_accessed = n * 4.0 
        
        pytorch_bw = (bytes_accessed / 1e9) / (pytorch_time / 1000.0)
        my_cuda_bw = (bytes_accessed / 1e9) / (my_cuda_time / 1000.0)

        print(f"{n:<12} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_bw:<15.2f} | {my_cuda_bw:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 Reduce Sum 算子性能对标 (Baseline: PyTorch)")
    run_reduce_benchmark()