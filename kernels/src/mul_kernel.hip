// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_kernels.h"

namespace rocm_kernels {

//============================================================================
// Mul Kernels
//============================================================================

/**
 * Element-wise multiplication kernel with broadcasting support
 * When b_size < total_size, b is broadcast (repeated cyclically)
 */
__global__ void mul_elementwise_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ output,
                                       int64_t total_size,
                                       int64_t b_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        // Handle broadcasting: b index wraps around
        int64_t b_idx = (b_size == total_size) ? idx : (idx % b_size);
        output[idx] = a[idx] * b[b_idx];
    }
}

/**
 * Scalar multiplication kernel
 */
__global__ void mul_scalar_kernel(const float* __restrict__ a,
                                  float scalar,
                                  float* __restrict__ output,
                                  int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = a[idx] * scalar;
    }
}

//============================================================================
// Public API Implementation
//============================================================================

void mul_elementwise(const float* a, const float* b, float* output,
                     int64_t total_size, int64_t b_size, hipStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(mul_elementwise_kernel,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       a, b, output, total_size, b_size);
}

void mul_scalar(const float* a, float scalar, float* output,
                int64_t size, hipStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(mul_scalar_kernel,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       a, scalar, output, size);
}

} // namespace rocm_kernels
