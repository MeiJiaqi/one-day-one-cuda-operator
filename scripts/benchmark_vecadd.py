import torch
import my_cuda_ops # 导入你的高级算子库

# 准备数据
A = torch.randn(1024, device='cuda', dtype=torch.float32)
B = torch.randn(1024, device='cuda', dtype=torch.float32)

# 调用算子 (不需要手动传空的 Tensor C 了，Python 封装层已经做好了)
C = my_cuda_ops.vector_add(A, B)

# 验证
print("最大误差:", torch.max(torch.abs(C - (A + B))).item())