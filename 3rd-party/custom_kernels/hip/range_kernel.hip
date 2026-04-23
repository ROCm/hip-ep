/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T>
__global__ void range_kernel(const T *start, const T *delta, T *output,
                             int64_t num_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_elements)
    return;
  output[tid] = *start + static_cast<T>(tid) * (*delta);
}

extern "C" int hip_range(void *stream, const void *start, const void *limit,
                         const void *delta, void *output,
                         int64_t output_num_elements, int hip_dtype) {
  (void)limit;
  if (!start || !delta || !output) {
    fprintf(stderr, "[custom_kernels] hip_range: null pointer argument\n");
    return -1;
  }
  if (output_num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((output_num_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_range: num_elements=%lld, hip_dtype=%d, "
      "grid=%d, block=%d\n",
      (long long)output_num_elements, hip_dtype, grid_size, kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_INT16:
    hipLaunchKernelGGL(range_kernel<int16_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int16_t *>(start),
                       static_cast<const int16_t *>(delta),
                       static_cast<int16_t *>(output), output_num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(range_kernel<int32_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int32_t *>(start),
                       static_cast<const int32_t *>(delta),
                       static_cast<int32_t *>(output), output_num_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(range_kernel<int64_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int64_t *>(start),
                       static_cast<const int64_t *>(delta),
                       static_cast<int64_t *>(output), output_num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(range_kernel<float>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const float *>(start),
                       static_cast<const float *>(delta),
                       static_cast<float *>(output), output_num_elements);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(range_kernel<double>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const double *>(start),
                       static_cast<const double *>(delta),
                       static_cast<double *>(output), output_num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_range: unsupported hip_dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_range launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
