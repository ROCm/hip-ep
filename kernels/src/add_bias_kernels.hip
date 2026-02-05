// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Add Bias HIP Kernels
// ====================
// GPU-accelerated bias addition for convolution outputs (NCHW format).
//
// Kernels:
//   - add_bias_nchw_kernel: Broadcasts bias [C] across [N, C, H, W] output

#include <hip/hip_runtime.h>

namespace hip_ep {

//============================================================================
// Add Bias Kernel (NCHW format)
//============================================================================

// Add bias to Conv output: output[n,c,h,w] += bias[c]
// For NCHW format: idx = n * (C * H * W) + c * (H * W) + hw
template<typename T>
__global__ void add_bias_nchw_kernel(
    T* data,
    const T* __restrict__ bias,
    int64_t channels,
    int64_t spatial_size,
    int64_t total_size) {
  
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_size) return;
  
  // For NCHW: idx = n * (C * H * W) + c * (H * W) + hw
  // We need to extract c from idx
  int64_t chw = channels * spatial_size;
  int64_t c = (idx % chw) / spatial_size;
  
  data[idx] += bias[c];
}

//============================================================================
// Explicit Template Instantiations
//============================================================================

template __global__ void add_bias_nchw_kernel<float>(float*, const float*, int64_t, int64_t, int64_t);

} // namespace hip_ep

//============================================================================
// Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

void launch_add_bias_nchw_f32(
    float* data,
    const float* bias,
    int64_t channels,
    int64_t spatial_size,
    int64_t total_size,
    hipStream_t stream) {
  
  int threads = 256;
  int blocks = static_cast<int>((total_size + threads - 1) / threads);
  hip_ep::add_bias_nchw_kernel<float><<<blocks, threads, 0, stream>>>(
      data, bias, channels, spatial_size, total_size);
}

} // extern "C"
