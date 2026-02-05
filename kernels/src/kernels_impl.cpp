/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// C++ wrapper implementations that call the precompiled HIP kernels
// The kernels are compiled at build time using hipcc (via hip_utils.cmake)
// This eliminates the ~800ms HIPRTC runtime compilation overhead

#include "rocm_kernels.h"
#include <hip/hip_runtime.h>

namespace rocm_kernels {

//============================================================================
// Add Bias Implementation (for Conv)
//============================================================================

void add_bias_nchw(float *data, const float *bias, int64_t batch,
                   int64_t channels, int64_t spatial_size, hipStream_t stream) {
  int64_t total_size = batch * channels * spatial_size;
  launch_add_bias_nchw_f32(data, bias, channels, spatial_size, total_size,
                           stream);
}

//============================================================================
// Mul Implementation
//============================================================================

void mul(const float *a, const float *b, float *output, int64_t size_out,
         int64_t size_a, int64_t size_b, hipStream_t stream) {
  launch_mul_f32(a, b, output, size_out, size_a, size_b, stream);
}

void mul_elementwise(const float *a, const float *b, float *output,
                     int64_t total_size, int64_t b_size, hipStream_t stream) {
  launch_mul_elementwise_f32(a, b, output, total_size, b_size, stream);
}

void mul_scalar(const float *a, float scalar, float *output, int64_t size,
                hipStream_t stream) {
  launch_mul_scalar_f32(a, scalar, output, size, stream);
}

//============================================================================
// Softmax Implementation - calls precompiled kernel
//============================================================================

void softmax(const float *input, float *output, int64_t batch, int64_t dim,
             hipStream_t stream) {
  launch_softmax_f32(input, output, batch, dim, stream);
}

void softmax_3d(const float *input, float *output, int64_t axis_size,
                int64_t inner_size, int64_t outer_size, hipStream_t stream) {
  if (inner_size == 1) {
    // Use optimized 2D kernel when inner_size is 1
    launch_softmax_2d_f32(input, output, axis_size, outer_size, stream);
  } else {
    // Use 3D kernel for general case
    launch_softmax_3d_f32(input, output, axis_size, inner_size, outer_size,
                          stream);
  }
}

//============================================================================
// Reshape Implementation (just a memory copy)
//============================================================================

void reshape_copy(const float *input, float *output, int64_t size,
                  hipStream_t stream) {
  launch_reshape_copy_f32(input, output, size, stream);
}

//============================================================================
// Transpose Implementation - calls precompiled kernels
//============================================================================

void transpose(const float *input, float *output, const int64_t *in_shape,
               const int64_t *out_shape, const int32_t *perm, int32_t ndim,
               int64_t total_size, hipStream_t stream) {
  if (ndim == 4) {
    // Convert int32_t perm to int64_t for kernel (implicit promotion)
    launch_transpose_4d_f32(
        input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
        static_cast<int64_t>(perm[0]), static_cast<int64_t>(perm[1]),
        static_cast<int64_t>(perm[2]), static_cast<int64_t>(perm[3]), stream);
  } else {
    // Fallback for non-4D - just copy (transpose is identity in 1D/2D with
    // perm=[0,1,...])
    hipMemcpyAsync(output, input, total_size * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
  }
}

void transpose_0213(const float *input, float *output, int64_t n, int64_t a,
                    int64_t b, int64_t c, hipStream_t stream) {
  launch_transpose_0213_f32(input, output, n, a, b, c, stream);
}

//============================================================================
// Tile Implementation - calls precompiled kernels
//============================================================================

void tile(const float *input, float *output, const int64_t *in_shape,
          const int64_t *repeats, int32_t ndim, int64_t in_size,
          int64_t out_size, hipStream_t stream) {
  if (in_size == out_size) {
    // No tiling needed
    hipMemcpyAsync(output, input, in_size * sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
    return;
  }

  if (ndim == 4) {
    launch_tile_4d_f32(input, output, in_shape[0], in_shape[1], in_shape[2],
                       in_shape[3], repeats[0], repeats[1], repeats[2],
                       repeats[3], out_size, stream);
  } else if (ndim == 5) {
    launch_tile_5d_f32(input, output, in_shape[0], in_shape[1], in_shape[2],
                       in_shape[3], in_shape[4], repeats[0], repeats[1],
                       repeats[2], repeats[3], repeats[4], out_size, stream);
  } else {
    // Use general N-dimensional tile kernel
    launch_tile_nd_f32(input, output, in_shape, ndim, repeats, in_size,
                       out_size, stream);
  }
}

} // namespace rocm_kernels
