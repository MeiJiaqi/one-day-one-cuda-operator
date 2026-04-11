#pragma once
#include <torch/extension.h>

// Day 1: Vector Add
void vector_add_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

/// Day 02
void gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v2_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v3_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v4_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v5_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void reduce_forward(torch::Tensor input, torch::Tensor output);

void reduce_v3_forward(torch::Tensor input, torch::Tensor output);