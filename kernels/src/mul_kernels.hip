// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Element-wise Multiplication HIP Kernels
// =======================================
// GPU-accelerated element-wise multiplication with broadcasting support.
//
// Kernels:
//   - elementwise_mul_kernel: General multiplication with broadcasting
//   - mul_scalar_kernel: Optimized scalar multiplication

#include <hip/hip_runtime.h>

namespace hip_ep {

//============================================================================
// Element-wise Multiplication Kernels
//============================================================================

// Element-wise multiplication kernel with broadcasting support
template<typename T>
__global__ void elementwise_mul_kernel(
    const T* input_a,
    const T* input_b,
    T* output,
    int64_t size_out,
    int64_t size_a,
    int64_t size_b) {
  
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size_out) return;

  // Broadcasting logic
  if (size_a == size_out && size_b == 1) {
    // B is scalar
    output[idx] = static_cast<T>(static_cast<float>(input_a[idx]) * 
                                  static_cast<float>(input_b[0]));
  } else if (size_a == 1 && size_b == size_out) {
    // A is scalar
    output[idx] = static_cast<T>(static_cast<float>(input_a[0]) * 
                                  static_cast<float>(input_b[idx]));
  } else if (size_a == size_out && size_b == size_out) {
    // Same shape
    output[idx] = static_cast<T>(static_cast<float>(input_a[idx]) * 
                                  static_cast<float>(input_b[idx]));
  } else {
    // General broadcasting using modulo
    int64_t idx_a = (size_a == 1) ? 0 : (idx % size_a);
    int64_t idx_b = (size_b == 1) ? 0 : (idx % size_b);
    output[idx] = static_cast<T>(static_cast<float>(input_a[idx_a]) * 
                                  static_cast<float>(input_b[idx_b]));
  }
}

// Scalar multiplication kernel (optimized when scalar is known)
template<typename T>
__global__ void mul_scalar_kernel(
    const T* input,
    T scalar,
    T* output,
    int64_t size) {
  
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  output[idx] = input[idx] * scalar;
}

//============================================================================
// Explicit Template Instantiations
//============================================================================

template __global__ void elementwise_mul_kernel<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template __global__ void mul_scalar_kernel<float>(const float*, float, float*, int64_t);

} // namespace hip_ep

//============================================================================
// Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

void launch_mul_f32(
    const float* input_a,
    const float* input_b,
    float* output,
    int64_t size_out,
    int64_t size_a,
    int64_t size_b,
    hipStream_t stream) {
  
  int threads = 256;
  int blocks = static_cast<int>((size_out + threads - 1) / threads);
  hip_ep::elementwise_mul_kernel<float><<<blocks, threads, 0, stream>>>(
      input_a, input_b, output, size_out, size_a, size_b);
}

void launch_mul_scalar_f32(
    const float* input,
    float scalar,
    float* output,
    int64_t size,
    hipStream_t stream) {
  
  int threads = 256;
  int blocks = static_cast<int>((size + threads - 1) / threads);
  hip_ep::mul_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
      input, scalar, output, size);
}

void launch_mul_elementwise_f32(
    const float* a,
    const float* b,
    float* output,
    int64_t total_size,
    int64_t b_size,
    hipStream_t stream) {
  
  launch_mul_f32(a, b, output, total_size, total_size, b_size, stream);
}

} // extern "C"
