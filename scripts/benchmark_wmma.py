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

def run_wmma_benchmark():
    # 维度最好是 16 的倍数
    shapes = [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    print(f"{'Shape (M, K, N)':<25} | {'PyTorch (ms)':<15} | {'WMMA (ms)':<15} | {'PyTorch TFLOPS':<15} | {'WMMA TFLOPS':<15}")
    print("-" * 95)

    for M, K, N in shapes:
        # 输入必须是 FP16
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # 1. 验证精度 (PyTorch 用 autocast 来模拟混合精度)
        # ⚠️ 注意：FP16 计算必定带来一定的精度损失，所以误差容忍度要调大到 1e-2 或更高
        C_custom = my_cuda_ops.wmma_gemm(A, B)
        C_ref = torch.matmul(A.float(), B.float()) # 以 FP32 为基准
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        assert max_err < 0.1, f"精度验证失败！最大误差: {max_err}"

        pytorch_time = benchmark_function(lambda: torch.matmul(A, B))
        wmma_time = benchmark_function(lambda: my_cuda_ops.wmma_gemm(A, B))
        # wmma_time = 999.0 

        # 计算 TFLOPS: GEMM 的浮点运算次数是 2 * M * N * K
        flops = 2.0 * M * N * K
        pytorch_tflops = (flops / (pytorch_time / 1000.0)) / 1e12
        wmma_tflops = (flops / (wmma_time / 1000.0)) / 1e12

        print(f"{str((M, K, N)):<25} | {pytorch_time:<15.4f} | {wmma_time:<15.4f} | {pytorch_tflops:<15.2f} | {wmma_tflops:<15.2f}")

if __name__ == "__main__":
    print("🚀 开启 Tensor Core TFLOPS 狂暴测试 (Baseline: PyTorch cuBLAS FP16)")
    run_wmma_benchmark()