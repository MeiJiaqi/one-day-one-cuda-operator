import torch
import my_cuda_ops

def benchmark_function(func, *args, warmup=10, iters=50):
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

def run_gemm_benchmark():
    # 测试不同的正方形矩阵规模
    sizes = [256, 512, 1024, 2048, 4096]
    
    print(f"{'Matrix Size':<15} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'PyTorch TFLOPS':<15} | {'My CUDA TFLOPS':<15}")
    print("-" * 85)

    for N in sizes:
        M = K = N
        A = torch.randn((M, K), device='cuda', dtype=torch.float32)
        B = torch.randn((K, N), device='cuda', dtype=torch.float32)

        # 1. 验证正确性
        C_custom = my_cuda_ops.matmul_v5(A, B)
        C_ref = torch.matmul(A, B)
        
        # 矩阵乘法会有浮点累加误差，容差(atol)设为 1e-3
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        assert max_err < 1e-3, f"精度验证失败！最大误差: {max_err}"

        # 2. 评测耗时
        pytorch_time = benchmark_function(lambda: torch.matmul(A, B))
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.matmul_v5(A, B))

        # 3. 计算 TFLOPS (Tera Floating Point Operations Per Second)
        # GEMM 计算量为: 2 * M * N * K 次浮点运算
        flops = 2.0 * M * N * K
        
        # TFLOPS = (FLOPs / 1e12) / (Time_ms / 1000)
        pytorch_tflops = (flops / 1e12) / (pytorch_time / 1000)
        my_cuda_tflops = (flops / 1e12) / (my_cuda_time / 1000)

        print(f"{N:<15} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {pytorch_tflops:<15.2f} | {my_cuda_tflops:<15.2f}")

if __name__ == "__main__":
    print("🚀 开始 GEMM 算子性能对标 (Baseline: PyTorch)")
    run_gemm_benchmark()