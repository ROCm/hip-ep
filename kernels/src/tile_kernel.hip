// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_kernels.h"

namespace rocm_kernels {

//============================================================================
// Tile Kernels
//============================================================================

/**
 * General N-dimensional tile kernel
 * Each output element maps back to an input element via modulo
 */
__global__ void tile_general_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    const int64_t* __restrict__ in_shape,
                                    const int64_t* __restrict__ in_strides,
                                    const int64_t* __restrict__ out_strides,
                                    int32_t ndim,
                                    int64_t out_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_size) return;
    
    // Convert linear output index to multi-dimensional indices
    // Then map to input indices using modulo
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    
    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        
        // Map to input coordinate using modulo
        int64_t in_coord = coord % in_shape[d];
        in_idx += in_coord * in_strides[d];
    }
    
    output[out_idx] = input[in_idx];
}

/**
 * Optimized tile for repeating along a single dimension
 * Common pattern: [1, heads, seq, dim] -> [1, heads*repeat, seq, dim]
 */
__global__ void tile_single_dim_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t before_size,  // Product of dims before repeat dim
                                       int64_t repeat_dim_size,  // Size of repeat dim in input
                                       int64_t after_size,  // Product of dims after repeat dim
                                       int64_t repeat_count,
                                       int64_t out_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= out_size) return;
    
    // Decompose output index
    int64_t after_idx = out_idx % after_size;
    int64_t temp = out_idx / after_size;
    int64_t repeat_idx = temp % (repeat_dim_size * repeat_count);
    int64_t before_idx = temp / (repeat_dim_size * repeat_count);
    
    // Map repeat index back to input
    int64_t in_repeat_idx = repeat_idx % repeat_dim_size;
    
    // Compute input index
    int64_t in_idx = before_idx * (repeat_dim_size * after_size) + 
                     in_repeat_idx * after_size + 
                     after_idx;
    
    output[out_idx] = input[in_idx];
}

//============================================================================
// Public API Implementation
//============================================================================

void tile(const float* input, float* output,
          const int64_t* in_shape, const int64_t* repeats,
          int32_t ndim, int64_t in_size, int64_t out_size,
          hipStream_t stream) {
    
    // Check if this is a single-dimension tile (optimization)
    int tile_dim = -1;
    for (int d = 0; d < ndim; ++d) {
        if (repeats[d] > 1) {
            if (tile_dim >= 0) {
                tile_dim = -2;  // Multiple dimensions have repeats
                break;
            }
            tile_dim = d;
        }
    }
    
    if (tile_dim >= 0) {
        // Single dimension tile - use optimized kernel
        int64_t before_size = 1;
        for (int d = 0; d < tile_dim; ++d) {
            before_size *= in_shape[d];
        }
        
        int64_t after_size = 1;
        for (int d = tile_dim + 1; d < ndim; ++d) {
            after_size *= in_shape[d];
        }
        
        const int block_size = 256;
        const int num_blocks = (out_size + block_size - 1) / block_size;
        
        hipLaunchKernelGGL(tile_single_dim_kernel,
                           dim3(num_blocks), dim3(block_size), 0, stream,
                           input, output, before_size, in_shape[tile_dim],
                           after_size, repeats[tile_dim], out_size);
    } else {
        // General case - use N-dimensional kernel
        
        // Compute strides
        int64_t h_in_strides[8], h_out_strides[8];
        int64_t h_out_shape[8];
        
        h_in_strides[ndim - 1] = 1;
        h_out_strides[ndim - 1] = 1;
        h_out_shape[ndim - 1] = in_shape[ndim - 1] * repeats[ndim - 1];
        
        for (int d = ndim - 2; d >= 0; --d) {
            h_in_strides[d] = h_in_strides[d + 1] * in_shape[d + 1];
            h_out_shape[d] = in_shape[d] * repeats[d];
            h_out_strides[d] = h_out_strides[d + 1] * h_out_shape[d + 1];
        }
        
        // Allocate and copy to device
        int64_t* d_in_shape;
        int64_t* d_in_strides;
        int64_t* d_out_strides;
        
        hipMalloc(&d_in_shape, ndim * sizeof(int64_t));
        hipMalloc(&d_in_strides, ndim * sizeof(int64_t));
        hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
        
        hipMemcpyAsync(d_in_shape, in_shape, ndim * sizeof(int64_t),
                       hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_in_strides, h_in_strides, ndim * sizeof(int64_t),
                       hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_out_strides, h_out_strides, ndim * sizeof(int64_t),
                       hipMemcpyHostToDevice, stream);
        
        const int block_size = 256;
        const int num_blocks = (out_size + block_size - 1) / block_size;
        
        hipLaunchKernelGGL(tile_general_kernel,
                           dim3(num_blocks), dim3(block_size), 0, stream,
                           input, output, d_in_shape, d_in_strides, d_out_strides,
                           ndim, out_size);
        
        // Cleanup
        hipStreamSynchronize(stream);
        hipFree(d_in_shape);
        hipFree(d_in_strides);
        hipFree(d_out_strides);
    }
}

} // namespace rocm_kernels
