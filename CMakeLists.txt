cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 启用 CXX 和 CUDA 语言支持
project(OneDayOneCUDAOperator LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 允许针对不同架构编译 (根据你的显卡架构修改，80代表A100, 86代表RTX30系, 89代表RTX40系)
# 如果你不确定，可以暂时注释掉下面这行，CMake 会尝试自动检测
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# 添加全局包含目录，这样所有子目录都能找到 common/cuda_utils.cuh
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# 添加算子目录
add_subdirectory(operators/01_vector_add)