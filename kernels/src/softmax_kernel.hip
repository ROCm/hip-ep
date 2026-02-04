// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_kernels.h"
#include <float.h>

namespace rocm_kernels {

//============================================================================
// Softmax Kernels
//============================================================================

/**
 * Softmax kernel - one block per row (batch element)
 * Uses shared memory for reduction
 * 
 * Algorithm:
 * 1. Find max value in row (for numerical stability)
 * 2. Compute exp(x - max) and sum
 * 3. Normalize by sum
 */
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t batch,
                               int64_t dim) {
    extern __shared__ float shared_data[];
    
    int64_t batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch) return;
    
    const float* row_input = input + batch_idx * dim;
    float* row_output = output + batch_idx * dim;
    
    // Step 1: Find max value in this row
    float local_max = -FLT_MAX;
    for (int64_t i = tid; i < dim; i += block_size) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    
    // Reduce to find global max
    shared_data[tid] = local_max;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared_data[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (int64_t i = tid; i < dim; i += block_size) {
        float exp_val = expf(row_input[i] - row_max);
        row_output[i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    
    // Reduce to find global sum
    shared_data[tid] = local_sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_data[0];
    __syncthreads();
    
    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int64_t i = tid; i < dim; i += block_size) {
        row_output[i] *= inv_sum;
    }
}

/**
 * Optimized softmax for small dimensions (fits in single warp)
 * Uses warp-level primitives for reduction
 */
__global__ void softmax_warp_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t batch,
                                    int64_t dim) {
    int64_t batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    
    if (batch_idx >= batch) return;
    
    const float* row_input = input + batch_idx * dim;
    float* row_output = output + batch_idx * dim;
    
    // Each thread handles one or more elements
    float local_max = -FLT_MAX;
    float local_val[8];  // Handle up to 256 elements per warp (32 * 8)
    int num_iters = (dim + 31) / 32;
    
    for (int i = 0; i < num_iters && lane + i * 32 < dim; ++i) {
        local_val[i] = row_input[lane + i * 32];
        local_max = fmaxf(local_max, local_val[i]);
    }
    
    // Warp reduce for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    float row_max = __shfl(local_max, 0);
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = 0; i < num_iters && lane + i * 32 < dim; ++i) {
        local_val[i] = expf(local_val[i] - row_max);
        local_sum += local_val[i];
    }
    
    // Warp reduce for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset);
    }
    float inv_sum = 1.0f / __shfl(local_sum, 0);
    
    // Normalize and write output
    for (int i = 0; i < num_iters && lane + i * 32 < dim; ++i) {
        row_output[lane + i * 32] = local_val[i] * inv_sum;
    }
}

//============================================================================
// Public API Implementation
//============================================================================

void softmax(const float* input, float* output,
             int64_t batch, int64_t dim, hipStream_t stream) {
    
    if (dim <= 256) {
        // Use warp-optimized kernel for small dimensions
        // Multiple rows per block
        int rows_per_block = 4;
        int num_blocks = (batch + rows_per_block - 1) / rows_per_block;
        
        dim3 block(32, rows_per_block);
        dim3 grid(num_blocks);
        
        hipLaunchKernelGGL(softmax_warp_kernel,
                           grid, block, 0, stream,
                           input, output, batch, dim);
    } else {
        // Use shared memory kernel for larger dimensions
        int block_size = 256;
        size_t shared_mem_size = block_size * sizeof(float);
        
        hipLaunchKernelGGL(softmax_kernel,
                           dim3(batch), dim3(block_size), shared_mem_size, stream,
                           input, output, batch, dim);
    }
}

} // namespace rocm_kernels
