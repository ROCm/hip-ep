/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void cast_i64_to_i32_kernel(
    const int64_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

extern "C" int udna_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  if (input_dtype == UDNA_DTYPE_INT64 && output_dtype == UDNA_DTYPE_INT32) {
    fprintf(stderr,
            "[custom_kernels] udna_cast: INT64->INT32, "
            "num_elements=%lld, grid=%d, block=%d\n",
            (long long)num_elements, grid_size, kBlockSize);

    hipLaunchKernelGGL(cast_i64_to_i32_kernel,
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t*>(input),
                       static_cast<int32_t*>(output),
                       num_elements);
  } else {
    fprintf(stderr,
            "[custom_kernels] udna_cast: unsupported conversion "
            "input_dtype=%d -> output_dtype=%d\n",
            input_dtype, output_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] udna_cast launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
