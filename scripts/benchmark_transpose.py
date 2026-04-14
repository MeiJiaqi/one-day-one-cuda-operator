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

def run_transpose_benchmark():
    shapes = [
        (4096, 4096),
        (8192, 8192),
        (4096, 8192),
        (10000, 10000), # 非 32 整数倍的奇葩尺寸
    ]
    
    print(f"{'Shape':<20} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch GB/s':<15} | {'My CUDA GB/s':<15}")
    print("-" * 85)

    for shape in shapes:
        A = torch.randn(shape, device='cuda', dtype=torch.float32)

        # 1. 验证正确性
        C_custom = my_cuda_ops.transpose(A)
        C_ref = A.t().contiguous()
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        assert max_err < 1e-5, f"精度验证失败！最大误差: {max_err}"

        # 2. 评测耗时 (PyTorch 必须加 contiguous 迫使它产生真正的物理内存搬运)
        pytorch_time = benchmark_function(lambda: A.t().contiguous())
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.transpose(A))
        # my_cuda_time = 999.0 

        # 3. 计算带宽 (GB/s)
        # 转置的访存：读取 1 遍，写入 1 遍
        bytes_accessed = 2 * A.numel() * 4.0 
        
        pytorch_bw = (bytes_accessed / 1e9) / (pytorch_time / 1000.0)
        my_cuda_bw = (bytes_accessed / 1e9) / (my_cuda_time / 1000.0)

        print(f"{str(shape):<20} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_bw:<15.2f} | {my_cuda_bw:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 Transpose 算子性能对标 (Baseline: PyTorch)")
    run_transpose_benchmark()