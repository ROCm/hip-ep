/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Clip: y[i] = min(max(x[i], lo), hi)
 *
 * `lo` and `hi` are 0-rank scalar device pointers of the same dtype as the
 * input. We read them once into thread-local registers per kernel
 * invocation. Per ONNX spec, when `lo > hi` the result is `hi` everywhere,
 * which falls out of the `min(max(...), hi)` ordering naturally.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

template <typename T>
__global__ void elementwise_clip_kernel(const T *__restrict__ input,
                                        const T *__restrict__ lo_ptr,
                                        const T *__restrict__ hi_ptr,
                                        T *__restrict__ output,
                                        int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  T lo = lo_ptr[0];
  T hi = hi_ptr[0];
  T v = input[idx];
  T r = v < lo ? lo : v;
  r = r > hi ? hi : r;
  output[idx] = r;
}

// FP16 specialization: promote to float to avoid relying on __half operators.
__global__ void elementwise_clip_f16_kernel(const __half *__restrict__ input,
                                            const __half *__restrict__ lo_ptr,
                                            const __half *__restrict__ hi_ptr,
                                            __half *__restrict__ output,
                                            int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;
  float lo = __half2float(lo_ptr[0]);
  float hi = __half2float(hi_ptr[0]);
  float v = __half2float(input[idx]);
  float r = v < lo ? lo : v;
  r = r > hi ? hi : r;
  output[idx] = __float2half(r);
}

} // namespace

extern "C" int hip_elementwise_clip(void *stream, const void *input,
                                    const void *lo, const void *hi,
                                    void *output, int64_t num_elements,
                                    int hip_dtype) {
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_elementwise_clip: f16, num=%lld\n",
        (long long)num_elements);
    hipLaunchKernelGGL(elementwise_clip_f16_kernel, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<const __half *>(lo),
                       static_cast<const __half *>(hi),
                       static_cast<__half *>(output), num_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(elementwise_clip_kernel<float>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<const float *>(lo),
                       static_cast<const float *>(hi),
                       static_cast<float *>(output), num_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(elementwise_clip_kernel<int32_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<const int32_t *>(lo),
                       static_cast<const int32_t *>(hi),
                       static_cast<int32_t *>(output), num_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(elementwise_clip_kernel<int64_t>, dim3(grid_size),
                       dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<const int64_t *>(lo),
                       static_cast<const int64_t *>(hi),
                       static_cast<int64_t *>(output), num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_clip: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_elementwise_clip launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
