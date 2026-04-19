import torch
import my_cuda_ops

def benchmark_function(func, *args, warmup=10, iters=100):
    for _ in range(warmup): func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def run_w4a16_benchmark():
    # 模拟 LLaMA-7B 的隐藏层维度
    shapes = [
        (4096, 4096),   # 常规线性层
        (8192, 4096),   # FFN 层扩展
        (16384, 8192)   # 超大尺寸对标
    ]
    
    print(f"{'Shape (N, K)':<15} | {'PyTorch FP16 (ms)':<20} | {'W4A16 INT4 (ms)':<20}")
    print("-" * 60)

    for N, K in shapes:
        # 1. 模拟激活值 (LLM 解码阶段 X 只是一个向量)
        X = torch.randn((K,), device='cuda', dtype=torch.float16)

        # 2. 模拟官方 FP16 权重 
        W_fp16 = torch.randn((N, K), device='cuda', dtype=torch.float16)
        
        # 3. 模拟 INT4 压缩权重 (物理列数直接砍半，类型变为 uint8)
        # 为测速仅填充随机乱码，不进行严格的 PTQ/QAT 量化对齐
        W_packed = torch.randint(0, 255, (N, K // 2), device='cuda', dtype=torch.uint8)
        Scales = torch.randn((N,), device='cuda', dtype=torch.float16)

        # 测速
        pytorch_time = benchmark_function(lambda: torch.matmul(W_fp16, X))
        w4a16_time = benchmark_function(lambda: my_cuda_ops.w4a16_gemv(X, W_packed, Scales))

        print(f"{str((N, K)):<15} | {pytorch_time:<20.4f} | {w4a16_time:<20.4f}")

if __name__ == "__main__":
    print("🚀 开始 W4A16 量化算子对标 (纯带宽碾压测试)")
    run_w4a16_benchmark()