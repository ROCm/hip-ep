/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(X) = Q(Sigmoid(DQ(X))) -- and, as the family grows, the same shape
// for Q(Tanh(DQ(X))), Q(Softplus(DQ(X))), Q(Gelu(DQ(X))), etc.
//
// x = (X - x_zero_point) * x_scale
// y = activation(x)          // kind selects the device function below
// Y = saturate(round(y * out_recip_scale) + y_zero_point)
//
// out_recip_scale = 1 / y_scale, folded once by the host wrapper so the
// kernel never performs a runtime division.
#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define HIP_QACTIVATION_SIGMOID 0

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only. Device-side bounds therefore
// come from the <cstdint> macros through explicitly annotated helpers.
template <typename T>
struct QActBounds;

template <>
struct QActBounds<int8_t> {
  __device__ static float lowest() { return static_cast<float>(INT8_MIN); }
  __device__ static float highest() { return static_cast<float>(INT8_MAX); }
};

template <>
struct QActBounds<uint8_t> {
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT8_MAX); }
};

template <>
struct QActBounds<int16_t> {
  __device__ static float lowest() { return static_cast<float>(INT16_MIN); }
  __device__ static float highest() { return static_cast<float>(INT16_MAX); }
};

template <>
struct QActBounds<uint16_t> {
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT16_MAX); }
};

template <typename T>
__device__ inline T q_act_saturate(float v) {
  return static_cast<T>(
      fminf(QActBounds<T>::highest(), fmaxf(QActBounds<T>::lowest(), v)));
}

// Matches the float sigmoid kernel body (elementwise_unary_kernel.hip)
// exactly, so the quantized and float paths stay numerically identical for
// the same dequantized input -- the only difference is the surrounding
// dequant/requant.
__device__ inline float qact_sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void qactivation_kernel(const T *__restrict__ input,
                                   T *__restrict__ output,
                                   int64_t num_elements, int kind,
                                   float x_scale, int32_t x_zero_point,
                                   float out_recip_scale,
                                   int32_t y_zero_point) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  float x =
      (static_cast<float>(input[idx]) - static_cast<float>(x_zero_point)) *
      x_scale;

  float y;
  switch (kind) {
  case HIP_QACTIVATION_SIGMOID:
    y = qact_sigmoid_f(x);
    break;
  default:
    return; // Unsupported kind: no silent fallback.
  }

  output[idx] = q_act_saturate<T>(rintf(y * out_recip_scale) +
                                  static_cast<float>(y_zero_point));
}

template <typename T>
static int launch_qactivation(hipStream_t stream, const void *input,
                              void *output, int64_t num_elements, int kind,
                              float x_scale, int32_t x_zero_point,
                              float out_recip_scale, int32_t y_zero_point) {
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
  (void)hipGetLastError();
  hipLaunchKernelGGL(qactivation_kernel<T>, dim3(grid_size), dim3(kBlockSize),
                     0, stream, static_cast<const T *>(input),
                     static_cast<T *>(output), num_elements, kind, x_scale,
                     x_zero_point, out_recip_scale, y_zero_point);
  return 0;
}

extern "C" int hip_qactivation(void *stream, const void *input, void *output,
                               int64_t kind, int64_t num_elements,
                               int hip_dtype, float x_scale,
                               int64_t x_zero_point, float out_recip_scale,
                               int64_t y_zero_point) {
  if (kind != HIP_QACTIVATION_SIGMOID) {
    fprintf(stderr,
            "[custom_kernels] hip_qactivation: unsupported kind=%lld\n",
            (long long)kind);
    return -1;
  }
  if (num_elements <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int32_t xzp = static_cast<int32_t>(x_zero_point);
  int32_t yzp = static_cast<int32_t>(y_zero_point);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qactivation: dtype=%d kind=%lld num=%lld "
      "x_scale=%g x_zp=%d out_recip_scale=%g y_zp=%d\n",
      hip_dtype, (long long)kind, (long long)num_elements, x_scale, xzp,
      out_recip_scale, yzp);

  int rc = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_INT8:
    rc = launch_qactivation<int8_t>(hip_stream, input, output, num_elements,
                                    (int)kind, x_scale, xzp, out_recip_scale,
                                    yzp);
    break;
  case HIP_DTYPE_UINT8:
    rc = launch_qactivation<uint8_t>(hip_stream, input, output, num_elements,
                                     (int)kind, x_scale, xzp, out_recip_scale,
                                     yzp);
    break;
  case HIP_DTYPE_INT16:
    rc = launch_qactivation<int16_t>(hip_stream, input, output, num_elements,
                                     (int)kind, x_scale, xzp, out_recip_scale,
                                     yzp);
    break;
  case HIP_DTYPE_UINT16:
    rc = launch_qactivation<uint16_t>(hip_stream, input, output, num_elements,
                                      (int)kind, x_scale, xzp, out_recip_scale,
                                      yzp);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_qactivation: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qactivation launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
