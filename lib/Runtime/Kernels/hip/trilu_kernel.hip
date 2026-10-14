/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Keep (row, col) when upper != 0 and col - row >= k, otherwise when
// col - row <= k. The last two dimensions are the matrix; leading
// dimensions are batches of contiguous row-major matrices. A one-element
// input is broadcast over the output.

template <typename T>
__device__ T trilu_zero() {
  return static_cast<T>(0);
}

template <>
__device__ __half trilu_zero<__half>() {
  return __float2half(0.f);
}

template <>
__device__ __hip_bfloat16 trilu_zero<__hip_bfloat16>() {
  return __float2bfloat16(0.f);
}

template <typename T>
__global__ void trilu_kernel(const T *__restrict__ input, T *__restrict__ output,
                             int64_t num_elements, int64_t rows, int64_t cols,
                             int64_t diagonal, int64_t upper,
                             bool broadcast) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  int64_t col = idx % cols;
  int64_t row = (idx / cols) % rows;
  int64_t delta = col - row;
  bool keep = upper ? (delta >= diagonal) : (delta <= diagonal);
  output[idx] = keep ? input[broadcast ? 0 : idx] : trilu_zero<T>();
}

template <typename T>
static int launch_trilu(hipStream_t stream, const void *input, void *output,
                        int64_t num_elements, int64_t rows, int64_t cols,
                        int64_t diagonal, int64_t upper, bool broadcast,
                        int grid_size, int block_size) {
  // Drop a stale error so the caller's hipGetLastError reports this launch.
  (void)hipGetLastError();
  hipLaunchKernelGGL(trilu_kernel<T>, dim3(grid_size), dim3(block_size), 0,
                     stream, static_cast<const T *>(input),
                     static_cast<T *>(output), num_elements, rows, cols,
                     diagonal, upper, broadcast);
  return 0;
}

extern "C" int hip_trilu(void *stream, const void *input, void *output,
                         int64_t input_elements, int64_t num_elements,
                         int64_t rows, int64_t cols, int64_t diagonal,
                         int64_t upper, int hip_dtype) {
  if (num_elements <= 0)
    return 0;
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "[custom_kernels] hip_trilu: invalid matrix %lldx%lld\n",
            (long long)rows, (long long)cols);
    return -1;
  }
  if (input_elements != num_elements && input_elements != 1) {
    fprintf(stderr,
            "[custom_kernels] hip_trilu: input has %lld elements, output %lld\n",
            (long long)input_elements, (long long)num_elements);
    return -1;
  }
  bool broadcast = input_elements == 1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    launch_trilu<float>(hip_stream, input, output, num_elements, rows, cols,
                        diagonal, upper, broadcast, grid_size, kBlockSize);
    break;
  case HIP_DTYPE_FLOAT16:
    launch_trilu<__half>(hip_stream, input, output, num_elements, rows, cols,
                         diagonal, upper, broadcast, grid_size, kBlockSize);
    break;
  case HIP_DTYPE_BFLOAT16:
    launch_trilu<__hip_bfloat16>(hip_stream, input, output, num_elements, rows,
                                 cols, diagonal, upper, broadcast, grid_size,
                                 kBlockSize);
    break;
  case HIP_DTYPE_FLOAT64:
    launch_trilu<double>(hip_stream, input, output, num_elements, rows, cols,
                         diagonal, upper, broadcast, grid_size, kBlockSize);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_trilu: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_trilu launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
