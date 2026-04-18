import torch
import torch.nn.functional as F
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

def run_conv2d_benchmark():
    # 模拟经典网络层: (N, C, H, W), (K, R, S)
    # 采用 ResNet 中常见的 3x3 卷积核尺寸
    configs = [
        # (Batch, In_C, H, W), (Out_K, Kernel_H, Kernel_W)
        ((32, 64, 56, 56), (64, 3, 3)),   # ResNet 早期阶段
        ((32, 128, 28, 28), (128, 3, 3)), # ResNet 中期阶段
        ((32, 256, 14, 14), (256, 3, 3)), # ResNet 后期阶段
    ]
    
    print(f"{'Shape (N,C,H,W)->(K,R,S)':<35} | {'PyTorch (ms)':<15} | {'My CUDA (ms)':<15} | {'Max Error':<10}")
    print("-" * 80)

    for in_shape, weight_shape in configs:
        X = torch.randn(in_shape, device='cuda', dtype=torch.float32)
        # weight 形状: (Out_K, In_C, R, S)
        Weight = torch.randn((weight_shape[0], in_shape[1], weight_shape[1], weight_shape[2]), 
                             device='cuda', dtype=torch.float32)

        # 1. 验证精度 (极其关键，多维寻址极易出错)
        C_custom = my_cuda_ops.conv2d_im2col(X, Weight)
        C_ref = F.conv2d(X, Weight, stride=1, padding=0)
        
        max_err = torch.max(torch.abs(C_custom - C_ref)).item()
        if max_err > 1e-1:
            print(f"❌ 精度错误! Error: {max_err}")
            return
            
        # 2. 评测耗时 (PyTorch 底层使用 cuDNN 高度优化的隐式 GEMM)
        pytorch_time = benchmark_function(lambda: F.conv2d(X, Weight, stride=1, padding=0))
        my_cuda_time = benchmark_function(lambda: my_cuda_ops.conv2d_im2col(X, Weight))

        config_str = f"{str(in_shape)}->{str(weight_shape)}"
        print(f"{config_str:<35} | {pytorch_time:<15.4f} | {my_cuda_time:<15.4f} | {max_err:<10.4f}")

if __name__ == "__main__":
    print("🚀 开始 Conv2d 算子性能对标 (Baseline: PyTorch cuDNN)")
    run_conv2d_benchmark()