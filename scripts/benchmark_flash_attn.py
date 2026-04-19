import torch
import torch.nn.functional as F
import my_cuda_ops
import math

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

def pytorch_attention_baseline(q, k, v):
    # Q 需要补一个 Seq_Len 维度以符合 matmul
    q_expanded = q.unsqueeze(2) # [B, H, 1, d]
    scale = 1.0 / math.sqrt(q.size(-1))
    
    # 这一步产生 [B, H, 1, N] 的中间矩阵，狂吃显存带宽
    scores = torch.matmul(q_expanded, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    
    out = torch.matmul(attn, v) # [B, H, 1, d]
    return out.squeeze(2)

def run_benchmark():
    # 模拟大模型参数 (Batch=1, Heads=32, Head_Dim=128)
    configs = [
        (1, 32, 4096, 128),  # 4K 上下文
        (1, 32, 16384, 128), # 16K 长上下文
        (1, 32, 32768, 128)  # 32K 极限上下文
    ]
    
    print(f"{'Seq Length (N)':<20} | {'PyTorch Baseline(ms)':<25} | {'My FlashDecoding(ms)':<25} | {'Max Error':<10}")
    print("-" * 85)

    for B, H, N, d in configs:
        Q = torch.randn((B, H, d), device='cuda', dtype=torch.float32)
        K = torch.randn((B, H, N, d), device='cuda', dtype=torch.float32)
        V = torch.randn((B, H, N, d), device='cuda', dtype=torch.float32)

        # 精度验证
        out_ref = pytorch_attention_baseline(Q, K, V)
        out_custom = my_cuda_ops.flash_decoding(Q, K, V)
        max_err = torch.max(torch.abs(out_ref - out_custom)).item()

        # 测速
        time_ref = benchmark_function(pytorch_attention_baseline, Q, K, V)
        time_custom = benchmark_function(my_cuda_ops.flash_decoding, Q, K, V)

        print(f"{N:<20} | {time_ref:<25.4f} | {time_custom:<25.4f} | {max_err:<10.4f}")

if __name__ == "__main__":
    print("🚀 开始 Day 10 终极试炼：Flash Decoding (Online Softmax)")
    run_benchmark()