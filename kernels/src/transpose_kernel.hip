// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_kernels.h"

namespace rocm_kernels {

//============================================================================
// Transpose Kernels
//============================================================================

/**
 * General N-dimensional transpose kernel
 * Each thread handles one element, computing its output position
 */
__global__ void transpose_general_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const int64_t* __restrict__ in_shape,
                                         const int64_t* __restrict__ in_strides,
                                         const int64_t* __restrict__ out_strides,
                                         const int32_t* __restrict__ perm,
                                         int32_t ndim,
                                         int64_t total_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= total_size) return;
    
    // Convert linear output index to multi-dimensional indices
    int64_t coords[8];  // Max 8 dimensions
    int64_t remaining = out_idx;
    
    for (int d = 0; d < ndim; ++d) {
        coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }
    
    // Map output coords to input coords using inverse permutation
    // out[i0, i1, ...] = in[i_perm[0], i_perm[1], ...]
    int64_t in_idx = 0;
    for (int d = 0; d < ndim; ++d) {
        // coords[d] is the index in output dim d
        // perm[d] tells us which input dim it came from
        in_idx += coords[d] * in_strides[perm[d]];
    }
    
    output[out_idx] = input[in_idx];
}

/**
 * Optimized 4D transpose for permutation [0, 2, 1, 3]
 * This is the common pattern in attention: [N, heads, seq, dim] -> [N, seq, heads, dim]
 * Uses shared memory tiling for better memory coalescing
 */
__global__ void transpose_0213_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int64_t n, int64_t a, int64_t b, int64_t c) {
    // Input:  [n, a, b, c] with strides [a*b*c, b*c, c, 1]
    // Output: [n, b, a, c] with strides [a*b*c, a*c, c, 1]
    
    // Each block handles a tile of the (a, b) dimensions
    const int TILE_DIM = 32;
    
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int64_t n_idx = blockIdx.z;
    int64_t c_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int64_t ab_block_x = blockIdx.x;
    
    // Compute which (a, b) tile we're processing
    int64_t num_ab_tiles_x = (b + TILE_DIM - 1) / TILE_DIM;
    int64_t a_tile = ab_block_x / num_ab_tiles_x;
    int64_t b_tile = ab_block_x % num_ab_tiles_x;
    
    int64_t a_idx = a_tile * TILE_DIM + threadIdx.x;
    int64_t b_idx = b_tile * TILE_DIM + threadIdx.y;
    
    // Load input[n, a_idx, b_idx, c_idx] into shared memory
    // Transpose the (a, b) dimensions in shared memory
    if (a_idx < a && b_idx < b && c_idx < c && n_idx < n) {
        int64_t in_offset = n_idx * (a * b * c) + a_idx * (b * c) + b_idx * c + c_idx;
        tile[threadIdx.y][threadIdx.x] = input[in_offset];
    }
    __syncthreads();
    
    // Write to output with transposed (a, b) indices
    // output[n, b_idx, a_idx, c_idx]
    int64_t out_a_idx = a_tile * TILE_DIM + threadIdx.y;
    int64_t out_b_idx = b_tile * TILE_DIM + threadIdx.x;
    
    if (out_a_idx < a && out_b_idx < b && c_idx < c && n_idx < n) {
        int64_t out_offset = n_idx * (a * b * c) + out_b_idx * (a * c) + out_a_idx * c + c_idx;
        output[out_offset] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * Simple version of [0,2,1,3] transpose without tiling
 * Better for small tensors or when c is small
 */
__global__ void transpose_0213_simple_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int64_t n, int64_t a, int64_t b, int64_t c) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = n * a * b * c;
    
    if (idx >= total) return;
    
    // Decode output index [n_i, b_i, a_i, c_i]
    int64_t c_i = idx % c;
    int64_t temp = idx / c;
    int64_t a_i = temp % a;
    temp = temp / a;
    int64_t b_i = temp % b;
    int64_t n_i = temp / b;
    
    // Input index for [n_i, a_i, b_i, c_i]
    int64_t in_idx = n_i * (a * b * c) + a_i * (b * c) + b_i * c + c_i;
    
    output[idx] = input[in_idx];
}

//============================================================================
// Public API Implementation
//============================================================================

void transpose(const float* input, float* output,
               const int64_t* in_shape, const int64_t* out_shape,
               const int32_t* perm, int32_t ndim,
               int64_t total_size, hipStream_t stream) {
    
    // Compute strides (on host, copy to device)
    int64_t h_in_strides[8], h_out_strides[8];
    
    h_in_strides[ndim - 1] = 1;
    h_out_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        h_in_strides[d] = h_in_strides[d + 1] * in_shape[d + 1];
        h_out_strides[d] = h_out_strides[d + 1] * out_shape[d + 1];
    }
    
    // Allocate device memory for strides and perm
    int64_t* d_in_strides;
    int64_t* d_out_strides;
    int32_t* d_perm;
    
    hipMalloc(&d_in_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_perm, ndim * sizeof(int32_t));
    
    hipMemcpyAsync(d_in_strides, h_in_strides, ndim * sizeof(int64_t), 
                   hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_out_strides, h_out_strides, ndim * sizeof(int64_t),
                   hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_perm, perm, ndim * sizeof(int32_t),
                   hipMemcpyHostToDevice, stream);
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(transpose_general_kernel,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       input, output, in_shape, d_in_strides, d_out_strides,
                       d_perm, ndim, total_size);
    
    // Free device memory after kernel completes
    // Note: This is a simplification - in production, use a memory pool
    hipStreamSynchronize(stream);
    hipFree(d_in_strides);
    hipFree(d_out_strides);
    hipFree(d_perm);
}

void transpose_0213(const float* input, float* output,
                    int64_t n, int64_t a, int64_t b, int64_t c,
                    hipStream_t stream) {
    int64_t total = n * a * b * c;
    
    // Use simple kernel for small tensors
    if (total < 65536 || c < 32) {
        const int block_size = 256;
        const int num_blocks = (total + block_size - 1) / block_size;
        
        hipLaunchKernelGGL(transpose_0213_simple_kernel,
                           dim3(num_blocks), dim3(block_size), 0, stream,
                           input, output, n, a, b, c);
    } else {
        // Use tiled version for larger tensors
        const int TILE_DIM = 32;
        dim3 block(TILE_DIM, TILE_DIM);
        
        int64_t num_ab_tiles = ((a + TILE_DIM - 1) / TILE_DIM) * 
                              ((b + TILE_DIM - 1) / TILE_DIM);
        int64_t num_c_tiles = (c + TILE_DIM - 1) / TILE_DIM;
        
        dim3 grid(num_ab_tiles, num_c_tiles, n);
        
        hipLaunchKernelGGL(transpose_0213_kernel,
                           grid, block, 0, stream,
                           input, output, n, a, b, c);
    }
}

} // namespace rocm_kernels
