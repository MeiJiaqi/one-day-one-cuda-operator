import torch
import torch.cuda.nvtx as nvtx
import my_cuda_ops

def run_profiling():
    n = 1000000
    A = torch.randint(1, 10, (n,), device='cuda', dtype=torch.float32)

    # 1. 强制预热 (这部分的 Timeline 我们不看，让 GPU 预热好)
    for _ in range(5): 
        my_cuda_ops.prefix_sum(A)
    torch.cuda.synchronize()

    # 2. 精确打标签，只抓取 1 次完整的调用！
    nvtx.range_push("🔍_Single_Prefix_Sum_V3")
    my_cuda_ops.prefix_sum(A)
    torch.cuda.synchronize() # 确保抓取完整
    nvtx.range_pop()

if __name__ == "__main__":
    run_profiling()