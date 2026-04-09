#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_forward, "Vector Add (CUDA)");
    // m.def("gemm", &gemm_forward, "GEMM (CUDA)"); // 以后有新算子，直接在这里解开注释
}