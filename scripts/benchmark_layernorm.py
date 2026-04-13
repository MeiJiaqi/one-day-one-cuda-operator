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

def run_layernorm_benchmark():
    # 模拟 [BatchSize * SeqLen, HiddenSize]
    # 我们固定前面的行数，重点测试不同 HiddenSize 的表现
    shapes = [
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
        (8192, 8192),
        (4096, 16384), # 极大 Hidden Size 测试
    ]
    
    eps = 1e-5
    
    print(f"{'Shape':<20} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch GB/s':<15} | {'My CUDA GB/s':<15}")
    print("-" * 85)

    for shape in shapes:
        hidden_size = shape[-1]
        
        # 初始化输入和权重
        A = torch.randn(shape, device='cuda', dtype=torch.float32)
        # Gamma 和 Beta 的维度是 [HiddenSize]
        gamma = torch.ones(hidden_size, device='cuda', dtype=torch.float32)
        beta = torch.zeros(hidden_size, device='cuda', dtype=torch.float32)

        # =========================================================
        # 1. 验证正确性 (等 CUDA Kernel 写完后解开注释)
        # =========================================================
        C_custom = my_cuda_ops.layernorm(A, gamma, beta, eps)
        C_ref = F.layer_norm(A, (hidden_size,), gamma, beta, eps)
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        assert max_err < 1e-4, f"精度验证失败！最大误差: {max_err}"

        # 2. 评测耗时
        pytorch_time = benchmark_function(lambda: F.layer_norm(A, (hidden_size,), gamma, beta, eps))
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.layernorm(A, gamma, beta, eps))
        # my_cuda_time = 999.0 # 占位符

        # 3. 计算带宽 (GB/s)
        # LayerNorm 理论最低访存：读 1 次 A, 写 1 次 C。
        # (忽略 gamma 和 beta 的读取，因为它们相对极小且被所有行复用并留在 L2 Cache 中)
        bytes_accessed = 2 * A.numel() * 4.0 
        
        pytorch_bw = (bytes_accessed / 1e9) / (pytorch_time / 1000.0)
        my_cuda_bw = (bytes_accessed / 1e9) / (my_cuda_time / 1000.0)

        print(f"{str(shape):<20} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_bw:<15.2f} | {my_cuda_bw:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 LayerNorm 算子性能对标 (Baseline: PyTorch)")
    run_layernorm_benchmark()