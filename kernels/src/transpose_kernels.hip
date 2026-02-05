// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Transpose HIP Kernels
// =====================
// GPU-accelerated tensor transpose operations.
//
// Kernels:
//   - transpose_0213_kernel: Optimized for [0,2,1,3] permutation (attention)
//   - transpose_4d_kernel: General 4D transpose with arbitrary permutation

#include <hip/hip_runtime.h>

namespace hip_ep {

//============================================================================
// Transpose Kernels
//============================================================================

// Optimized transpose for [0, 2, 1, 3] permutation (common in attention)
// Input: [N, A, B, C] -> Output: [N, B, A, C]
template<typename T>
__global__ void transpose_0213_kernel(
    const T* input,
    T* output,
    int64_t n, int64_t a, int64_t b, int64_t c) {

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

// General 4D transpose kernel
template<typename T>
__global__ void transpose_4d_kernel(
    const T* input,
    T* output,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3) {

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_size = d0 * d1 * d2 * d3;

  if (idx >= total_size) return;

  // Compute input 4D indices
  int64_t d23 = d2 * d3;
  int64_t d123 = d1 * d23;

  int64_t i0 = idx / d123;
  int64_t rem = idx % d123;
  int64_t i1 = rem / d23;
  rem = rem % d23;
  int64_t i2 = rem / d3;
  int64_t i3 = rem % d3;

  // Store indices in array
  int64_t in_idx_arr[4];
  in_idx_arr[0] = i0;
  in_idx_arr[1] = i1;
  in_idx_arr[2] = i2;
  in_idx_arr[3] = i3;

  // Apply permutation to get output indices
  int64_t o0 = in_idx_arr[p0];
  int64_t o1 = in_idx_arr[p1];
  int64_t o2 = in_idx_arr[p2];
  int64_t o3 = in_idx_arr[p3];

  // Compute output dimensions after permutation
  int64_t dims[4] = {d0, d1, d2, d3};
  int64_t od0 = dims[p0];
  int64_t od1 = dims[p1];
  int64_t od2 = dims[p2];
  int64_t od3 = dims[p3];

  // Compute output linear index
  int64_t od23 = od2 * od3;
  int64_t out_idx = o0 * (od1 * od23) + o1 * od23 + o2 * od3 + o3;

  output[out_idx] = input[idx];
}

//============================================================================
// Explicit Template Instantiations
//============================================================================

template __global__ void transpose_0213_kernel<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t);
template __global__ void transpose_4d_kernel<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace hip_ep

//============================================================================
// Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

void launch_transpose_0213_f32(
    const float* input,
    float* output,
    int64_t n, int64_t a, int64_t b, int64_t c,
    hipStream_t stream) {

  int64_t total = n * a * b * c;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  hip_ep::transpose_0213_kernel<float><<<blocks, threads, 0, stream>>>(
      input, output, n, a, b, c);
}

void launch_transpose_4d_f32(
    const float* input,
    float* output,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3,
    hipStream_t stream) {

  int64_t total = d0 * d1 * d2 * d3;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  hip_ep::transpose_4d_kernel<float><<<blocks, threads, 0, stream>>>(
      input, output, d0, d1, d2, d3, p0, p1, p2, p3);
}

} // extern "C"
