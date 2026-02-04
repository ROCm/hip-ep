// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_kernels.h"

namespace rocm_kernels {

//============================================================================
// Reshape Kernels
//============================================================================

/**
 * Simple copy kernel for reshape when data needs to be copied
 * (Most reshapes are zero-copy in practice)
 */
__global__ void copy_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx];
    }
}

/**
 * Vectorized copy kernel using float4 for better memory bandwidth
 */
__global__ void copy_vectorized_kernel(const float4* __restrict__ input,
                                       float4* __restrict__ output,
                                       int64_t size_vec4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size_vec4) {
        output[idx] = input[idx];
    }
}

//============================================================================
// Public API Implementation
//============================================================================

void reshape_copy(const float* input, float* output,
                  int64_t size, hipStream_t stream) {
    // Try to use vectorized copy if aligned
    bool use_vectorized = (size % 4 == 0) &&
                          (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(output) % 16 == 0);
    
    if (use_vectorized) {
        int64_t size_vec4 = size / 4;
        const int block_size = 256;
        const int num_blocks = (size_vec4 + block_size - 1) / block_size;
        
        hipLaunchKernelGGL(copy_vectorized_kernel,
                           dim3(num_blocks), dim3(block_size), 0, stream,
                           reinterpret_cast<const float4*>(input),
                           reinterpret_cast<float4*>(output),
                           size_vec4);
    } else {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        
        hipLaunchKernelGGL(copy_kernel,
                           dim3(num_blocks), dim3(block_size), 0, stream,
                           input, output, size);
    }
}

} // namespace rocm_kernels
