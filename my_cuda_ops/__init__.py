import torch

# 导入编译好的底层 C++ 动态链接库
from . import _C 

# 提供一个带有 Type Hint 和 Docstring 的 Python 接口
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    高性能 CUDA 向量加法
    
    Args:
        a (torch.Tensor): 输入张量 A (必须在 GPU 上且为 float32)
        b (torch.Tensor): 输入张量 B (必须在 GPU 上且为 float32)
        
    Returns:
        torch.Tensor: a + b 的结果
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape == b.shape, "Input shapes must match"
    
    # 自动帮用户创建输出 Tensor，提升易用性
    c = torch.empty_like(a)
    
    # 调用底层 C++ 算子
    _C.vector_add(a, b, c)
    
    return c