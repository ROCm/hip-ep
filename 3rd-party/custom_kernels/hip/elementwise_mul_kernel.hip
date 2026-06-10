/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Elementwise INT64 multiplication kernel.
 *
 * Modeled after elementwise_sub_kernel.hip. Exists because MIOpen's
 * miopenOpTensor (used by wrap_miopenOpTensor for fp16/fp32/bf16 mul)
 * has no INT64 path; some models (Qwen3 mrope: total = batch * seq_len)
 * produce scalar int64 multiplies that would otherwise silently no-op.
 *
 * No broadcasting: caller must ensure lhs/rhs/output all have the same
 * element count. wrap_miopenOpTensor enforces this before dispatch.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void elementwise_mul_i64_kernel(
    const int64_t* __restrict__ lhs,
    const int64_t* __restrict__ rhs,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] * rhs[idx];
  }
}

extern "C" int hip_elementwise_mul(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype) {
  if (num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
    case HIP_DTYPE_INT64: {
      CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_elementwise_mul: dtype=INT64, "
              "num_elements=%lld, grid=%d, block=%d\n",
              (long long)num_elements, grid_size, kBlockSize);

      hipLaunchKernelGGL(elementwise_mul_i64_kernel,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(lhs),
                         static_cast<const int64_t*>(rhs),
                         static_cast<int64_t*>(output),
                         num_elements);
      break;
    }
    default:
      fprintf(stderr,
              "[custom_kernels] hip_elementwise_mul: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_mul launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
