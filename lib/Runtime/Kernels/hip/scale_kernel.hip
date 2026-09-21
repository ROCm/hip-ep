/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * In-place strided scale: data[i * stride] *= beta for i in [0, count).
 * Used by wrap_gemm writeBroadcastC to scale a D2D-copied C seed on device
 * (scalar / row / column / full) without a host round-trip.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T, typename Acc>
__global__ void scale_strided_kernel(T *__restrict__ data, int64_t count,
                                     int64_t stride, Acc beta) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  T *p = data + i * stride;
  *p = static_cast<T>(static_cast<Acc>(*p) * beta);
}

__global__ void scale_strided_kernel_f16(__half *__restrict__ data,
                                         int64_t count, int64_t stride,
                                         float beta) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  __half *p = data + i * stride;
  *p = __float2half(__half2float(*p) * beta);
}

__global__ void scale_strided_kernel_bf16(__hip_bfloat16 *__restrict__ data,
                                          int64_t count, int64_t stride,
                                          float beta) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  __hip_bfloat16 *p = data + i * stride;
  *p = __float2bfloat16(__bfloat162float(*p) * beta);
}

extern "C" HIP_KERNEL_API int hip_scale_strided(void *stream, void *data,
                                                int64_t count, int64_t stride,
                                                float beta, int hip_dtype) {
  if (count <= 0)
    return 0;
  if (!data || stride <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_scale_strided: invalid data/stride "
            "(data=%p stride=%lld)\n",
            data, (long long)stride);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((count + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_scale_strided: dtype=%d count=%lld stride=%lld "
      "beta=%f\n",
      hip_dtype, (long long)count, (long long)stride, beta);

  (void)hipGetLastError();

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(scale_strided_kernel_f16, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<__half *>(data), count, stride, beta);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scale_strided_kernel<float, float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<float *>(data), count, stride, beta);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scale_strided_kernel<double, double>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<double *>(data), count, stride,
                       static_cast<double>(beta));
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(scale_strided_kernel_bf16, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<__hip_bfloat16 *>(data), count, stride,
                       beta);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_scale_strided: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_scale_strided launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
