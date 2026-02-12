/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

//============================================================================
// C-style kernel launch functions (implemented in rocm_kernels.hip)
// These are compiled at build time using hipcc for zero runtime overhead
//============================================================================

extern "C" {

//--- Add Bias kernels (Conv) ---

void launch_add_bias_nchw_f32(float *data, const float *bias, int64_t channels,
                              int64_t spatial_size, int64_t total_size,
                              hipStream_t stream);

//--- Softmax kernels ---

void launch_softmax_f32(const float *input, float *output, int64_t batch,
                        int64_t dim, hipStream_t stream);

void launch_softmax_2d_f32(const float *input, float *output, int64_t axis_size,
                           int64_t outer_size, hipStream_t stream);

void launch_softmax_3d_f32(const float *input, float *output, int64_t axis_size,
                           int64_t inner_size, int64_t outer_size,
                           hipStream_t stream);

//--- Multiply kernels ---

void launch_mul_f32(const float *input_a, const float *input_b, float *output,
                    int64_t size_out, int64_t size_a, int64_t size_b,
                    hipStream_t stream);

void launch_mul_scalar_f32(const float *input, float scalar, float *output,
                           int64_t size, hipStream_t stream);

void launch_mul_elementwise_f32(const float *a, const float *b, float *output,
                                int64_t total_size, int64_t b_size,
                                hipStream_t stream);

//--- Transpose kernels ---

void launch_transpose_0213_f32(const float *input, float *output, int64_t n,
                               int64_t a, int64_t b, int64_t c,
                               hipStream_t stream);

void launch_transpose_4d_f32(const float *input, float *output, int64_t d0,
                             int64_t d1, int64_t d2, int64_t d3, int64_t p0,
                             int64_t p1, int64_t p2, int64_t p3,
                             hipStream_t stream);

//--- Tile kernels ---

void launch_tile_4d_f32(const float *input, float *output, int64_t in_d0,
                        int64_t in_d1, int64_t in_d2, int64_t in_d3, int64_t r0,
                        int64_t r1, int64_t r2, int64_t r3, int64_t out_size,
                        hipStream_t stream);

void launch_tile_5d_f32(const float *input, float *output, int64_t in_d0,
                        int64_t in_d1, int64_t in_d2, int64_t in_d3,
                        int64_t in_d4, int64_t r0, int64_t r1, int64_t r2,
                        int64_t r3, int64_t r4, int64_t out_size,
                        hipStream_t stream);

void launch_tile_nd_f32(const float *input, float *output,
                        const int64_t *input_shape, int ndims,
                        const int64_t *repeats, int64_t input_elements,
                        int64_t output_elements, hipStream_t stream);

//--- Utility kernels ---

void launch_reshape_copy_f32(const float *input, float *output, int64_t size,
                             hipStream_t stream);

//--- FMHA (Flash Multi-Head Attention) kernels ---

/**
 * Forward FMHA with causal masking for GQA (Grouped Query Attention).
 * Handles both prefill (multi-token) and decode (single-token) cases.
 *
 * Causal mask (bottom-right): Q[i] attends to K[j] where
 *   j <= i + (seqlen_k - seqlen_q)
 *
 * @param stream      HIP stream
 * @param q_ptr       Q tensor [B, nhead_q, seqlen_q, hdim] (fp16)
 * @param k_ptr       K tensor [B, nhead_k, seqlen_k, hdim] (fp16)
 * @param v_ptr       V tensor [B, nhead_k, seqlen_k, hdim] (fp16)
 * @param o_ptr       O tensor [B, nhead_q, seqlen_q, hdim] (fp16, output)
 * @param batch       Batch size
 * @param nhead_q     Number of query heads
 * @param nhead_k     Number of KV heads (nhead_q must be divisible by nhead_k)
 * @param seqlen_q    Query sequence length
 * @param seqlen_k    Key sequence length (total including past)
 * @param hdim        Head dimension (e.g., 64, 96, 128)
 * @param scale       Attention scale factor (typically 1/sqrt(hdim))
 * @return            hipSuccess on success, error code on failure
 */
hipError_t launch_fmha_fwd_fp16_causal(hipStream_t stream, const void *q_ptr,
                                       const void *k_ptr, const void *v_ptr,
                                       void *o_ptr, int32_t batch,
                                       int32_t nhead_q, int32_t nhead_k,
                                       int32_t seqlen_q, int32_t seqlen_k,
                                       int32_t hdim, float scale);

} // extern "C"

//============================================================================
// C++ wrapper namespace for convenience
//============================================================================

namespace rocm_kernels {

//============================================================================
// Bias Addition (for Conv)
//============================================================================

/**
 * Add bias to Conv output (broadcast along spatial dimensions)
 * For NCHW format: output[n,c,h,w] += bias[c]
 *
 * @param data Input/Output tensor [N, C, H, W] (in-place operation)
 * @param bias Bias tensor [C] (on device)
 * @param batch Batch size (N)
 * @param channels Number of channels (C)
 * @param spatial_size H * W
 * @param stream HIP stream
 */
void add_bias_nchw(float *data, const float *bias, int64_t batch,
                   int64_t channels, int64_t spatial_size, hipStream_t stream);

//============================================================================
// Mul (Element-wise Multiplication)
//============================================================================

/**
 * Element-wise multiplication with full broadcasting support
 * output = a * b with broadcasting based on sizes
 *
 * @param a First input tensor (on device)
 * @param b Second input tensor (on device)
 * @param output Output tensor (on device)
 * @param size_out Total number of elements in output
 * @param size_a Size of tensor a
 * @param size_b Size of tensor b
 * @param stream HIP stream for async execution
 */
void mul(const float *a, const float *b, float *output, int64_t size_out,
         int64_t size_a, int64_t size_b, hipStream_t stream);

/**
 * Element-wise multiplication: output = a * b
 * Supports broadcasting when one tensor is smaller (e.g., scalar or 1D)
 *
 * @param a First input tensor (on device)
 * @param b Second input tensor (on device)
 * @param output Output tensor (on device)
 * @param total_size Total number of elements in output
 * @param b_size Size of tensor b (for broadcasting, b_size <= total_size)
 * @param stream HIP stream for async execution
 */
void mul_elementwise(const float *a, const float *b, float *output,
                     int64_t total_size, int64_t b_size, hipStream_t stream);

/**
 * Scalar multiplication: output = a * scalar
 *
 * @param a Input tensor (on device)
 * @param scalar Scalar value
 * @param output Output tensor (on device)
 * @param size Number of elements
 * @param stream HIP stream
 */
void mul_scalar(const float *a, float scalar, float *output, int64_t size,
                hipStream_t stream);

//============================================================================
// Softmax
//============================================================================

/**
 * Softmax along the last dimension (2D case)
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param batch Number of batches (product of all dims except last)
 * @param dim Size of the last dimension (softmax dimension)
 * @param stream HIP stream
 */
void softmax(const float *input, float *output, int64_t batch, int64_t dim,
             hipStream_t stream);

/**
 * Softmax for 3D tensors with non-trivial inner dimension
 *
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param axis_size Size of the axis to apply softmax
 * @param inner_size Size of dimensions after the softmax axis
 * @param outer_size Size of dimensions before the softmax axis
 * @param stream HIP stream
 */
void softmax_3d(const float *input, float *output, int64_t axis_size,
                int64_t inner_size, int64_t outer_size, hipStream_t stream);

//============================================================================
// Reshape
//============================================================================

/**
 * Reshape is a zero-copy operation when data is contiguous.
 * This kernel handles the copy case when needed.
 *
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param size Total number of elements
 * @param stream HIP stream
 */
void reshape_copy(const float *input, float *output, int64_t size,
                  hipStream_t stream);

//============================================================================
// Transpose
//============================================================================

/**
 * General N-dimensional transpose
 *
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param in_shape Input shape array (on device or host pinned)
 * @param out_shape Output shape array
 * @param perm Permutation array (e.g., [0,2,1,3] for NHWC->NCHW)
 * @param ndim Number of dimensions
 * @param total_size Total number of elements
 * @param stream HIP stream
 */
void transpose(const float *input, float *output, const int64_t *in_shape,
               const int64_t *out_shape, const int32_t *perm, int32_t ndim,
               int64_t total_size, hipStream_t stream);

/**
 * Optimized 4D transpose for common attention patterns
 * Permutation: [0, 2, 1, 3] - swaps dims 1 and 2
 *
 * @param input Input tensor [N, A, B, C]
 * @param output Output tensor [N, B, A, C]
 * @param n Batch size (dim 0)
 * @param a Size of dim 1
 * @param b Size of dim 2
 * @param c Size of dim 3
 * @param stream HIP stream
 */
void transpose_0213(const float *input, float *output, int64_t n, int64_t a,
                    int64_t b, int64_t c, hipStream_t stream);

//============================================================================
// Tile (Repeat/Broadcast)
//============================================================================

/**
 * Tile operation - repeats tensor along specified dimensions
 *
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param in_shape Input shape array
 * @param repeats Repeat counts for each dimension
 * @param ndim Number of dimensions
 * @param in_size Total input elements
 * @param out_size Total output elements
 * @param stream HIP stream
 */
void tile(const float *input, float *output, const int64_t *in_shape,
          const int64_t *repeats, int32_t ndim, int64_t in_size,
          int64_t out_size, hipStream_t stream);

} // namespace rocm_kernels
