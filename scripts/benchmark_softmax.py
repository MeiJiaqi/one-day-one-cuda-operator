import torch
import torch.nn.functional as F
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

def run_softmax_benchmark():
    # 模拟大模型中的 Attention 矩阵形状: [BatchSize, NumHeads, SeqLen, SeqLen]
    # 我们将固定前面的维度，测试不同 SeqLen 下的性能
    shapes = [
        (8, 12, 128, 128),
        (8, 12, 512, 512),
        (8, 12, 1024, 1024),
        (8, 12, 2048, 2048),
        (8, 12, 4096, 4096),
    ]
    
    print(f"{'Shape':<25} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch GB/s':<15} | {'My CUDA GB/s':<15}")
    print("-" * 90)

    for shape in shapes:
        A = torch.randn(shape, device='cuda', dtype=torch.float32)

        # 1. 验证正确性 (等我们写完 Kernel 再解开注释)
        C_custom = my_cuda_ops.softmax(A)
        C_ref = F.softmax(A, dim=-1)
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        assert max_err < 1e-4, f"精度验证失败！最大误差: {max_err}"

        # 2. 评测耗时
        pytorch_time = benchmark_function(lambda: F.softmax(A, dim=-1))
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.softmax(A))
        # my_cuda_time = 999.0 # 占位符

        # 3. 计算带宽 (GB/s)
        # Softmax 的理论最低访存：读入 1 次 A，写出 1 次 C
        # 字节数 = 2 * numel * 4
        bytes_accessed = 2 * A.numel() * 4.0 
        
        pytorch_bw = (bytes_accessed / 1e9) / (pytorch_time / 1000.0)
        my_cuda_bw = (bytes_accessed / 1e9) / (my_cuda_time / 1000.0)

        print(f"{str(shape):<25} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_bw:<15.2f} | {my_cuda_bw:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 Softmax 算子性能对标 (Baseline: PyTorch)")
    run_softmax_benchmark()