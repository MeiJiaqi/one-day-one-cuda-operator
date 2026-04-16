import torch
import my_cuda_ops

def run_prefix_sum_benchmark():
    # 测试数组长度必须是 2 的幂次，且最大 2048
    shapes = [
        2048, 
        2048 * 4,        # 8192
        2048 * 100 + 73, # 204873 (边界测试)
        1000000          # 100万级别数据
    ]
    
    print(f"{'Size (Elements)':<20} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'Max Error':<15}")
    print("-" * 75)

    for n in shapes:
        # 使用随机整数方便观察结果
        A = torch.randint(1, 10, (n,), device='cuda', dtype=torch.float32)

        # 1. 验证精度 (非常关键！)
        C_custom = my_cuda_ops.prefix_sum(A)
        # PyTorch 的 cumsum 是 Inclusive Scan
        C_ref = torch.cumsum(A, dim=0) 
        
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        if max_err > 1e-4:
            print(f"❌ 精度严重错误! Size: {n}, Error: {max_err}")
            # 打印前 10 个元素对比，方便你 Debug
            print("PyTorch:", C_ref[:10].tolist())
            print("My CUDA:", C_custom[:10].tolist())
            return
            
        # 预热和测速...
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # PyTorch 测速
        for _ in range(10): torch.cumsum(A, dim=0)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100): torch.cumsum(A, dim=0)
        end_event.record()
        torch.cuda.synchronize()
        pytorch_time = start_event.elapsed_time(end_event) / 100.0

        # 自定义 CUDA 测速
        for _ in range(10): my_cuda_ops.prefix_sum(A)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100): my_cuda_ops.prefix_sum(A)
        end_event.record()
        torch.cuda.synchronize()
        my_cuda_time = start_event.elapsed_time(end_event) / 100.0

        print(f"{n:<20} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {max_err:<15.4f}")

if __name__ == "__main__":
    print("🚀 开始 Prefix Sum 算子挑战")
    run_prefix_sum_benchmark()