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

namespace {

template <typename T> struct BiasGeluTraits;

template <> struct BiasGeluTraits<float> {
  using ComputeType = float;
  __device__ static inline ComputeType load(float x) { return x; }
  __device__ static inline float store(ComputeType x) { return x; }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct BiasGeluTraits<__half> {
  using ComputeType = float;
  __device__ static inline ComputeType load(__half x) { return __half2float(x); }
  __device__ static inline __half store(ComputeType x) { return __float2half(x); }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct BiasGeluTraits<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static inline ComputeType load(__hip_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static inline __hip_bfloat16 store(ComputeType x) {
    return __float2bfloat16(x);
  }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erff(x); }
};

template <> struct BiasGeluTraits<double> {
  using ComputeType = double;
  __device__ static inline ComputeType load(double x) { return x; }
  __device__ static inline double store(ComputeType x) { return x; }
  __device__ static inline ComputeType erf_fn(ComputeType x) { return erf(x); }
};

template <typename T>
__device__ __forceinline__ T compute_bias_gelu_erf(T data_val, T bias_val) {
  using Traits = BiasGeluTraits<T>;
  using ComputeType = typename Traits::ComputeType;
  constexpr ComputeType kHalf = ComputeType(0.5);
  constexpr ComputeType kSqrt2Inv = ComputeType(0.7071067811865476);
  ComputeType x = Traits::load(data_val) + Traits::load(bias_val);
  ComputeType result =
      x * kHalf * (ComputeType(1.0) + Traits::erf_fn(x * kSqrt2Inv));
  return Traits::store(result);
}

template <typename T>
__global__ void bias_gelu_kernel(const T *__restrict__ data,
                                 const T *__restrict__ bias, T *__restrict__ output,
                                 int64_t num_elements, int64_t bias_len) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    int64_t bias_idx = idx % bias_len;
    output[idx] = compute_bias_gelu_erf<T>(data[idx], bias[bias_idx]);
  }
}

template <typename T>
static int launch_bias_gelu(hipStream_t stream, const void *data, const void *bias,
                            void *output, int64_t num_elements, int64_t bias_len) {
  if (num_elements <= 0 || bias_len <= 0)
    return 0;
  if (num_elements % bias_len != 0)
    return -1;

  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

  (void)hipGetLastError();
  hipLaunchKernelGGL(bias_gelu_kernel<T>, dim3(grid_size), dim3(kBlockSize), 0,
                     stream, static_cast<const T *>(data),
                     static_cast<const T *>(bias), static_cast<T *>(output),
                     num_elements, bias_len);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_bias_gelu launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

} // namespace

extern "C" int hip_bias_gelu(void *stream, const void *data, const void *bias,
                             void *output, int64_t num_elements,
                             int64_t bias_len, int hip_dtype) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_bias_gelu: dtype=%d, num_elements=%lld, "
      "bias_len=%lld\n",
      hip_dtype, (long long)num_elements, (long long)bias_len);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return launch_bias_gelu<float>(hip_stream, data, bias, output, num_elements,
                                   bias_len);
  case HIP_DTYPE_FLOAT16:
    return launch_bias_gelu<__half>(hip_stream, data, bias, output, num_elements,
                                    bias_len);
  case HIP_DTYPE_BFLOAT16:
    return launch_bias_gelu<__hip_bfloat16>(hip_stream, data, bias, output,
                                            num_elements, bias_len);
  case HIP_DTYPE_FLOAT64:
    return launch_bias_gelu<double>(hip_stream, data, bias, output, num_elements,
                                    bias_len);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_bias_gelu: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
