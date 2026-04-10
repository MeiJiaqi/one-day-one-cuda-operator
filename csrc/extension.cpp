#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_forward, "Vector Add (CUDA)");
    m.def("gemm", &gemm_forward, "GEMM Naive (CUDA)"); 
    m.def("gemm_v2", &gemm_v2_forward, "GEMM V2 Shared Memory (CUDA)");
    m.def("gemm_v3", &gemm_v3_forward, "GEMM V3 Register Tiling (CUDA)");
    m.def("gemm_v4", &gemm_v4_forward, "GEMM V4 2D Register Tiling (CUDA)");
    m.def("gemm_v5", &gemm_v5_forward, "GEMM V5 Vectorized Memory Access And Bank Conflict (CUDA)");
}