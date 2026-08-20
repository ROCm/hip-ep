/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

template <typename T>
__device__ __forceinline__
    typename std::enable_if<std::is_integral<T>::value &&
                                (sizeof(T) < sizeof(int64_t)),
                            T>::type
    range_value_impl(T start_val, T delta_val, int64_t tid) {
  // Avoid intermediate overflow for narrow integer types in tid*delta.
  // For int64/float/double we preserve the original arithmetic behavior.
  const int64_t v = static_cast<int64_t>(start_val) +
                    static_cast<int64_t>(tid) * static_cast<int64_t>(delta_val);
  return static_cast<T>(v);
}

template <typename T>
__device__ __forceinline__
    typename std::enable_if<!std::is_integral<T>::value ||
                                (sizeof(T) >= sizeof(int64_t)),
                            T>::type
    range_value_impl(T start_val, T delta_val, int64_t tid) {
  return start_val + static_cast<T>(tid) * delta_val;
}

template <typename T>
__device__ __forceinline__ T range_value(T start_val, T delta_val,
                                         int64_t tid) {
  return range_value_impl<T>(start_val, delta_val, tid);
}

template <typename T>
__device__ __forceinline__ bool range_delta_is_zero(T d) {
  // ORT treats +0 and -0 as zero for float/double (exact equality).
  return d == T(0);
}

template <typename T>
__global__ void range_kernel(const T *start, const T *delta, T *output,
                             int64_t num_elements, int *device_error_flag) {
  __shared__ T s_delta;
  if (threadIdx.x == 0)
    s_delta = *delta;
  __syncthreads();
  if (range_delta_is_zero(s_delta)) {
    if (device_error_flag && threadIdx.x == 0 && blockIdx.x == 0)
      *device_error_flag = -1;
    return;
  }

  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_elements)
    return;
  output[tid] = range_value(*start, *delta, tid);
}

extern "C" int hip_range(void *stream, const void *start, const void *limit,
                         const void *delta, void *output,
                         int64_t output_num_elements, int64_t hip_dtype,
                         void *device_error_flag) {
  // `limit` is intentionally unused in the kernel launch path. The output
  // length is precomputed by ONNX->HIP lowering and passed as
  // output_num_elements; kernel generation only needs start/delta.
  (void)limit;
  if (!start || !delta || !output) {
    fprintf(stderr, "[custom_kernels] hip_range: null pointer argument\n");
    return -1;
  }
  if (output_num_elements <= 0) {
    // No kernel for empty output. If runtime delta were 0, shape folding may
    // have produced 0 length without launching range_kernel; use compile-time
    // rejection (see RangeConversion) for constant delta==0, or a future
    // side-channel error flag for full dynamic ORT parity without per-empty sync.
    return 0;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  const uint64_t grid_size_u64 =
      static_cast<uint64_t>((output_num_elements + kBlockSize - 1) / kBlockSize);
  if (grid_size_u64 > std::numeric_limits<uint32_t>::max()) {
    fprintf(stderr,
            "[custom_kernels] hip_range: output_num_elements too large for "
            "grid dimension (%lld)\n",
            (long long)output_num_elements);
    return -1;
  }
  const uint32_t grid_size = static_cast<uint32_t>(grid_size_u64);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_range: num_elements=%lld, hip_dtype=%lld, "
      "grid=%u, block=%d\n",
      (long long)output_num_elements, (long long)hip_dtype, grid_size,
      kBlockSize);

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_INT16:
    hipLaunchKernelGGL(range_kernel<int16_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int16_t *>(start),
                       static_cast<const int16_t *>(delta),
                       static_cast<int16_t *>(output), output_num_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(range_kernel<int32_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int32_t *>(start),
                       static_cast<const int32_t *>(delta),
                       static_cast<int32_t *>(output), output_num_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(range_kernel<int64_t>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const int64_t *>(start),
                       static_cast<const int64_t *>(delta),
                       static_cast<int64_t *>(output), output_num_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(range_kernel<float>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const float *>(start),
                       static_cast<const float *>(delta),
                       static_cast<float *>(output), output_num_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(range_kernel<double>, dim3(grid_size), dim3(kBlockSize),
                       0, hip_stream, static_cast<const double *>(start),
                       static_cast<const double *>(delta),
                       static_cast<double *>(output), output_num_elements,
                       static_cast<int *>(device_error_flag));
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_range: unsupported hip_dtype=%lld\n",
            (long long)hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_range launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
