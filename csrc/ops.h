#pragma once
#include <torch/extension.h>

// Day 1: Vector Add
void vector_add_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

/// Day 02
void gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v2_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v3_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v3_my_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v4_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void gemm_v5_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void reduce_forward(torch::Tensor input, torch::Tensor output);

void reduce_v3_forward(torch::Tensor input, torch::Tensor output);

void softmax_forward(torch::Tensor input, torch::Tensor output);

void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps);

void transpose_forward(torch::Tensor input, torch::Tensor output);

void wmma_gemm_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void prefix_sum_forward(torch::Tensor input, torch::Tensor output);

void conv2d_forward(torch::Tensor X, torch::Tensor Weight, torch::Tensor Y);

void conv2d_im2col_forward(torch::Tensor X, torch::Tensor Weight, torch::Tensor Y);

void flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);

void w4a16_gemv_forward(torch::Tensor X, torch::Tensor W_packed, torch::Tensor scales, torch::Tensor Y);