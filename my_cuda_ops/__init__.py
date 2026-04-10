import torch

# 导入编译好的底层 C++ 动态链接库
from . import _C 

# 提供一个带有 Type Hint 和 Docstring 的 Python 接口
def vector_add(a: torch.Tensor, b: torch.Tensor,c:torch.Tensor) -> torch.Tensor:
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
    
    
    # 调用底层 C++ 算子
    _C.vector_add(a, b, c)
    
    return c

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    高性能 CUDA 矩阵乘法 (GEMM)
    C = A @ B
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    assert a.size(1) == b.size(0), f"Shape mismatch: {a.shape} and {b.shape}"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Inputs must be float32"
    
    a = a.contiguous()
    b = b.contiguous()
    
    # 根据矩阵乘法规则，输出矩阵的 shape 是 (M, N)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    _C.gemm(a, b, c)
    
    return c

def matmul_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _C.gemm_v2(a, b, c)
    return c

def matmul_v3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _C.gemm_v3(a, b, c)
    return c

def matmul_v4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _C.gemm_v4(a, b, c)
    return c

def matmul_v5(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _C.gemm_v5(a, b, c)
    return c