// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Tile HIP Kernels
// ================
// GPU-accelerated tile (repeat) operations for N-dimensional tensors.
//
// Kernels:
//   - tile_nd_kernel: General N-dimensional tile with shared memory optimization

#include <hip/hip_runtime.h>

namespace rocm_ep {

//============================================================================
// Tile Kernel
// Uses N-dimensional kernel for all tile operations
//============================================================================

// Tile kernel with proper N-dimensional indexing
// Uses shared memory for shapes and strides to avoid global memory latency
template <typename T, int MAX_DIMS = 8>
__global__ void tile_nd_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t* __restrict__ input_shape,
    const int64_t* __restrict__ output_shape,
    int ndims,
    int64_t output_elements) {
  
  // Shared memory for shapes and strides
  __shared__ int64_t s_input_shape[MAX_DIMS];
  __shared__ int64_t s_output_shape[MAX_DIMS];
  __shared__ int64_t s_input_strides[MAX_DIMS];
  __shared__ int64_t s_output_strides[MAX_DIMS];
  
  // Load shapes into shared memory
  if (threadIdx.x < ndims) {
    s_input_shape[threadIdx.x] = input_shape[threadIdx.x];
    s_output_shape[threadIdx.x] = output_shape[threadIdx.x];
  }
  __syncthreads();
  
  // Compute strides (one thread computes, others wait)
  if (threadIdx.x == 0) {
    s_input_strides[ndims - 1] = 1;
    s_output_strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; --i) {
      s_input_strides[i] = s_input_strides[i + 1] * s_input_shape[i + 1];
      s_output_strides[i] = s_output_strides[i + 1] * s_output_shape[i + 1];
    }
  }
  __syncthreads();
  
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= output_elements) return;
  
  // Convert output linear index to multi-dimensional indices
  // Apply modulo with input shape, then convert back to input linear index
  int64_t input_idx = 0;
  int64_t remaining = idx;
  
  for (int d = 0; d < ndims; ++d) {
    int64_t coord = remaining / s_output_strides[d];
    remaining = remaining % s_output_strides[d];
    
    // Apply modulo to get input coordinate
    int64_t input_coord = coord % s_input_shape[d];
    input_idx += input_coord * s_input_strides[d];
  }
  
  output[idx] = input[input_idx];
}

//============================================================================
// Explicit Template Instantiations
//============================================================================

template __global__ void tile_nd_kernel<float, 8>(const float*, float*, const int64_t*, const int64_t*, int, int64_t);

} // namespace rocm_ep

//============================================================================
// Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

void launch_tile_nd_f32(
    const float* input,
    float* output,
    const int64_t* input_shape,
    int ndims,
    const int64_t* repeats,
    int64_t input_elements,
    int64_t output_elements,
    hipStream_t stream) {
  
  // Compute output shape = input_shape * repeats
  int64_t output_shape[8];
  for (int i = 0; i < ndims && i < 8; ++i) {
    output_shape[i] = input_shape[i] * repeats[i];
  }
  
  // Copy shapes to device
  int64_t *d_input_shape, *d_output_shape;
  hipMalloc(&d_input_shape, ndims * sizeof(int64_t));
  hipMalloc(&d_output_shape, ndims * sizeof(int64_t));
  hipMemcpyAsync(d_input_shape, input_shape, ndims * sizeof(int64_t), hipMemcpyHostToDevice, stream);
  hipMemcpyAsync(d_output_shape, output_shape, ndims * sizeof(int64_t), hipMemcpyHostToDevice, stream);
  
  int threads = 256;
  int blocks = static_cast<int>((output_elements + threads - 1) / threads);
  
  rocm_ep::tile_nd_kernel<float><<<blocks, threads, 0, stream>>>(
      input, output, d_input_shape, d_output_shape, ndims, output_elements);
  
  // Must wait for kernel to complete before freeing device memory
  hipStreamSynchronize(stream);
  
  hipFree(d_input_shape);
  hipFree(d_output_shape);
}

// Compatibility wrappers - redirect to tile_nd_f32
void launch_tile_4d_f32(
    const float* input,
    float* output,
    int64_t in_d0, int64_t in_d1, int64_t in_d2, int64_t in_d3,
    int64_t r0, int64_t r1, int64_t r2, int64_t r3,
    int64_t out_size,
    hipStream_t stream) {
  
  int64_t input_shape[4] = {in_d0, in_d1, in_d2, in_d3};
  int64_t repeats[4] = {r0, r1, r2, r3};
  int64_t input_elements = in_d0 * in_d1 * in_d2 * in_d3;
  launch_tile_nd_f32(input, output, input_shape, 4, repeats, input_elements, out_size, stream);
}

void launch_tile_5d_f32(
    const float* input,
    float* output,
    int64_t in_d0, int64_t in_d1, int64_t in_d2, int64_t in_d3, int64_t in_d4,
    int64_t r0, int64_t r1, int64_t r2, int64_t r3, int64_t r4,
    int64_t out_size,
    hipStream_t stream) {
  
  int64_t input_shape[5] = {in_d0, in_d1, in_d2, in_d3, in_d4};
  int64_t repeats[5] = {r0, r1, r2, r3, r4};
  int64_t input_elements = in_d0 * in_d1 * in_d2 * in_d3 * in_d4;
  launch_tile_nd_f32(input, output, input_shape, 5, repeats, input_elements, out_size, stream);
}

} // extern "C"
