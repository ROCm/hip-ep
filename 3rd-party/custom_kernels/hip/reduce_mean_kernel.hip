/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Mean reduction = sum-reduction followed by elementwise division of every
 * output element by the count of input elements that contributed to it.
 * We piggy-back on hip_reduce_sum for the sum pass to avoid duplicating the
 * trailing-axes tree-reduction code, then scale the output in place.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

template <typename T>
__global__ void scale_inplace_kernel(T *__restrict__ data,
                                     int64_t num_elements, float inv) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  data[idx] = static_cast<T>(static_cast<float>(data[idx]) * inv);
}

__global__ void scale_inplace_f16_kernel(__half *__restrict__ data,
                                         int64_t num_elements, float inv) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  float v = __half2float(data[idx]) * inv;
  data[idx] = __float2half(v);
}

template <typename T>
__global__ void scale_inplace_int_kernel(T *__restrict__ data,
                                         int64_t num_elements, int64_t divisor) {
  // ONNX ReduceMean over integer T performs floating-point mean rounded to
  // int via truncation (numpy mean -> ORT cast). We approximate with
  // integer division — matches CPU ORT behaviour for the integer
  // specializations registered in this op.
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  data[idx] = static_cast<T>(data[idx] / static_cast<T>(divisor));
}

} // namespace

extern "C" int hip_reduce_mean(void *stream, const void *data, void *output,
                               int64_t num_input_elements,
                               int64_t num_output_elements, int64_t inner_size,
                               int hip_dtype) {
  if (num_output_elements <= 0)
    return 0;
  if (num_input_elements % num_output_elements != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_reduce_mean: input(%lld) not divisible "
            "by output(%lld)\n",
            (long long)num_input_elements, (long long)num_output_elements);
    return -1;
  }

  // The sum pass honors `inner_size` (strided reduce for non-trailing axes);
  // the trailing scale pass below is over the contiguous output, unaffected.
  int rc = hip_reduce_sum(stream, data, output, num_input_elements,
                          num_output_elements, inner_size, hip_dtype);
  if (rc != 0)
    return rc;

  int64_t reduce_size = num_input_elements / num_output_elements;
  if (reduce_size <= 1)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_output_elements + kBlockSize - 1) / kBlockSize);

  float inv = 1.0f / static_cast<float>(reduce_size);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(scale_inplace_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<__half *>(output), num_output_elements,
                       inv);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(scale_inplace_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<float *>(output), num_output_elements, inv);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(scale_inplace_int_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int32_t *>(output), num_output_elements,
                       reduce_size);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(scale_inplace_int_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<int64_t *>(output), num_output_elements,
                       reduce_size);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_reduce_mean: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_reduce_mean scale launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
