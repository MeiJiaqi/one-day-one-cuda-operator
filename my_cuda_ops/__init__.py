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

def reduce_sum(a: torch.Tensor) -> torch.Tensor:
    """
    高性能 CUDA 并行归约求和 (Warp Shuffle 版)
    
    Args:
        a (torch.Tensor): 输入张量 (将被展平为 1D)
    Returns:
        torch.Tensor: 包含单个求和结果的 0 维张量 (标量)
    """
    assert a.is_cuda, "Input must be on CUDA"
    assert a.dtype == torch.float32, "Input must be float32"
    a = a.contiguous()
    
    # 展平为 1D，求和操作不关心原有的形状
    current_input = a.view(-1)
    
    # 多级归约：如果元素个数大于 1，就继续缩小
    while current_input.numel() > 1:
        n = current_input.numel()
        threads = 1024
        # 计算当前轮次需要多少个 Block
        blocks = (n + threads - 1) // threads
        
        # 分配当前轮次的输出内存
        current_output = torch.empty(blocks, device=a.device, dtype=a.dtype)
        
        # 调用底层的 block 级归约 (_C.reduce_sum 是你在 extension.cpp 里注册的名字)
        _C.reduce_sum(current_input, current_output)
        
        # 将输出作为下一轮的输入
        current_input = current_output
        
    # 返回一个标量 Tensor (去除维度)
    return current_input.view([])

def reduce_sum_v3(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.dtype == torch.float32
    current_input = a.contiguous().view(-1)
    
    # 必须要有一个 max_blocks，和 C++ 里写的保持一致
    max_blocks = 1024 
    
    while current_input.numel() > 1:
        n = current_input.numel()
        threads = 1024
        
        # 【修复点】：限制分配的内存大小，最多只能是 max_blocks
        blocks = min((n + threads - 1) // threads, max_blocks)
        
        # 现在分配的内存刚好装满 C++ 算出来的结果，没有一丝多余的垃圾
        current_output = torch.empty(blocks, device=a.device, dtype=a.dtype)
        
        _C.reduce_sum_v3(current_input, current_output)
        
        current_input = current_output
        
    return current_input.view([])

def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    高性能 CUDA Softmax
    目前仅支持在最后一维 (Row-wise) 上进行 Softmax
    """
    assert a.is_cuda and a.dtype == torch.float32
    # 确保最后一维在内存中是连续的
    a = a.contiguous()
    
    # 预分配输出 Tensor
    out = torch.empty_like(a)
    _C.softmax(a, out)
    return out

def layernorm(a: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    高性能 CUDA LayerNorm (基于 Welford 算法的一趟扫描)
    默认在最后一个维度 (Hidden Size) 上进行归一化
    """
    assert a.is_cuda and gamma.is_cuda and beta.is_cuda, "All inputs must be on CUDA"
    assert a.dtype == torch.float32, "Input must be float32"
    
    # 确保张量在内存中是连续的
    a = a.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    # 预分配输出内存
    out = torch.empty_like(a)
    
    # 调用底层 C++ 算子
    _C.layernorm(a, gamma, beta, out, eps)
    
    return out