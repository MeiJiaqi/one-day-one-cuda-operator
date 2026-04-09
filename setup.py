import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 1. 自动收集所有的 C++ 和 CUDA 源文件
# 包含统一注册入口
sources = ['csrc/extension.cpp'] 
# 自动递归搜索所有的 .cu 文件
cu_sources = glob.glob('operators/**/*.cu', recursive=True) 
sources.extend(cu_sources)

# 2. 编译配置
setup(
    name='my_cuda_ops',
    version='0.1.0',
    packages=find_packages(), # 自动识别 my_cuda_ops 这个 Python 包
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops._C', # 编译出的底层库名称
            sources=sources,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)