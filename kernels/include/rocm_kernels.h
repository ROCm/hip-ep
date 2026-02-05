// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

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
void add_bias_nchw(float* data, const float* bias,
                   int64_t batch, int64_t channels, int64_t spatial_size,
                   hipStream_t stream);

//============================================================================
// Mul (Element-wise Multiplication)
//============================================================================

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
void mul_elementwise(const float* a, const float* b, float* output,
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
void mul_scalar(const float* a, float scalar, float* output,
                int64_t size, hipStream_t stream);

//============================================================================
// Softmax
//============================================================================

/**
 * Softmax along the last dimension
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 * 
 * @param input Input tensor (on device)
 * @param output Output tensor (on device)
 * @param batch Number of batches (product of all dims except last)
 * @param dim Size of the last dimension (softmax dimension)
 * @param stream HIP stream
 */
void softmax(const float* input, float* output,
             int64_t batch, int64_t dim, hipStream_t stream);

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
void reshape_copy(const float* input, float* output,
                  int64_t size, hipStream_t stream);

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
void transpose(const float* input, float* output,
               const int64_t* in_shape, const int64_t* out_shape,
               const int32_t* perm, int32_t ndim,
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
void transpose_0213(const float* input, float* output,
                    int64_t n, int64_t a, int64_t b, int64_t c,
                    hipStream_t stream);

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
void tile(const float* input, float* output,
          const int64_t* in_shape, const int64_t* repeats,
          int32_t ndim, int64_t in_size, int64_t out_size,
          hipStream_t stream);

} // namespace rocm_kernels
