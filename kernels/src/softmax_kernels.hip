// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Softmax HIP Kernels
// ===================
// GPU-accelerated softmax operations with double-precision accumulation
// for improved numerical accuracy.
//
// Kernels:
//   - softmax_kernel_2d: Optimized for 2D tensors (batch, features)
//   - softmax_kernel_3d: For 3D tensors with arbitrary axis

#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>

namespace rocm_ep {

//============================================================================
// Softmax Kernels
// Improved accuracy: uses double-precision for accumulation
//============================================================================

// Softmax kernel for 2D case (outer_size, axis_size, inner_size=1)
// Each block handles one row (outer_size index)
template<typename T>
__global__ void softmax_kernel_2d(
    const T* input,
    T* output,
    int64_t axis_size,
    int64_t outer_size) {

  int64_t outer_idx = blockIdx.x;
  if (outer_idx >= outer_size) return;

  const T* row_input = input + outer_idx * axis_size;
  T* row_output = output + outer_idx * axis_size;

  // Step 1: Find max for numerical stability (parallel reduction)
  __shared__ float shared_max[256];
  float thread_max = -FLT_MAX;

  for (int i = threadIdx.x; i < axis_size; i += blockDim.x) {
    float val = static_cast<float>(row_input[i]);
    thread_max = fmaxf(thread_max, val);
  }
  shared_max[threadIdx.x] = thread_max;
  __syncthreads();

  // Reduce to find global max
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x],
                                       shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_val = shared_max[0];

  // Step 2: Compute sum of exp(x - max) using DOUBLE precision for accumulation
  __shared__ double shared_sum[256];
  double thread_sum = 0.0;

  for (int i = threadIdx.x; i < axis_size; i += blockDim.x) {
    double val = exp(static_cast<double>(row_input[i]) - static_cast<double>(max_val));
    thread_sum += val;
  }
  shared_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  // Reduce to find global sum (in double precision)
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  double sum_exp = shared_sum[0];

  // Step 3: Compute final normalized values directly
  double inv_sum = 1.0 / sum_exp;
  for (int i = threadIdx.x; i < axis_size; i += blockDim.x) {
    double val = exp(static_cast<double>(row_input[i]) - static_cast<double>(max_val));
    row_output[i] = static_cast<T>(val * inv_sum);
  }
}

// Softmax kernel for 3D case (outer_size, axis_size, inner_size)
// Each block handles one (outer, inner) pair
template<typename T>
__global__ void softmax_kernel_3d(
    const T* input,
    T* output,
    int64_t axis_size,
    int64_t inner_size,
    int64_t outer_size) {

  int64_t idx = blockIdx.x;
  int64_t outer_idx = idx / inner_size;
  int64_t inner_idx = idx % inner_size;

  if (outer_idx >= outer_size || inner_idx >= inner_size) return;

  // Step 1: Find max for numerical stability
  float max_val = -FLT_MAX;
  for (int64_t a = 0; a < axis_size; ++a) {
    int64_t offset = outer_idx * axis_size * inner_size + a * inner_size + inner_idx;
    float val = static_cast<float>(input[offset]);
    max_val = fmaxf(max_val, val);
  }

  // Step 2: Compute sum of exp using DOUBLE precision
  double sum_exp = 0.0;
  for (int64_t a = 0; a < axis_size; ++a) {
    int64_t offset = outer_idx * axis_size * inner_size + a * inner_size + inner_idx;
    double val = exp(static_cast<double>(input[offset]) - static_cast<double>(max_val));
    sum_exp += val;
  }

  // Step 3: Compute final normalized values directly
  double inv_sum = 1.0 / sum_exp;
  for (int64_t a = 0; a < axis_size; ++a) {
    int64_t offset = outer_idx * axis_size * inner_size + a * inner_size + inner_idx;
    double val = exp(static_cast<double>(input[offset]) - static_cast<double>(max_val));
    output[offset] = static_cast<T>(val * inv_sum);
  }
}

//============================================================================
// Explicit Template Instantiations
//============================================================================

template __global__ void softmax_kernel_2d<float>(const float*, float*, int64_t, int64_t);
template __global__ void softmax_kernel_3d<float>(const float*, float*, int64_t, int64_t, int64_t);

} // namespace rocm_ep

//============================================================================
// Launcher Functions (extern "C" for C++ interop)
//============================================================================

extern "C" {

void launch_softmax_2d_f32(
    const float* input,
    float* output,
    int64_t axis_size,
    int64_t outer_size,
    hipStream_t stream) {

  int threads = 256;
  if (axis_size < 256) {
    threads = 1;
    while (threads < axis_size) threads <<= 1;
    if (threads > 256) threads = 256;
  }
  rocm_ep::softmax_kernel_2d<float><<<outer_size, threads, 0, stream>>>(
      input, output, axis_size, outer_size);
}

void launch_softmax_3d_f32(
    const float* input,
    float* output,
    int64_t axis_size,
    int64_t inner_size,
    int64_t outer_size,
    hipStream_t stream) {

  int blocks = static_cast<int>(outer_size * inner_size);
  rocm_ep::softmax_kernel_3d<float><<<blocks, 1, 0, stream>>>(
      input, output, axis_size, inner_size, outer_size);
}

void launch_softmax_f32(
    const float* input,
    float* output,
    int64_t batch,
    int64_t dim,
    hipStream_t stream) {

  launch_softmax_2d_f32(input, output, dim, batch, stream);
}

} // extern "C"
