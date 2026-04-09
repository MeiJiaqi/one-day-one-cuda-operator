#pragma once
#include <torch/extension.h>

// Day 1: Vector Add
void vector_add_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// Day 2: GEMM (预留)
// void gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c, float alpha, float beta);