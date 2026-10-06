/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX QLinearConv, NCHW, one spatial window per thread.
//
//   dot = conv(x - x_zp, w - w_zp) + B
//   y = saturate(round(dot * x_scale * w_scale / y_scale) + y_zp)
//
// B is optional int32, already quantized at scale x_scale * w_scale and zero
// point 0, so it adds in the integer domain. Input and output scales and zero
// points are per-tensor. Weight scale and zero point are per-tensor
// (count == 1) or per output channel (count == c_out). The dot product is
// accumulated in int64 so the only rounding is the final requantization,
// which uses rintf (ties to even), matching QuantizeLinear.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

template <typename T> __device__ float qlinear_lowest();
template <typename T> __device__ float qlinear_highest();

template <> __device__ float qlinear_lowest<uint8_t>() { return 0.0f; }
template <> __device__ float qlinear_highest<uint8_t>() { return 255.0f; }
template <> __device__ float qlinear_lowest<int8_t>() { return -128.0f; }
template <> __device__ float qlinear_highest<int8_t>() { return 127.0f; }

template <typename XT, typename WT, typename YT>
__global__ void
qlinear_conv_2d(const XT *x, const float *x_scale, const XT *x_zp, const WT *w,
                const float *w_scale, const WT *w_zp, const float *y_scale,
                const YT *y_zp, const int32_t *bias, YT *y, int64_t n,
                int64_t c_in, int64_t c_out, int64_t h_in, int64_t w_in,
                int64_t h_out, int64_t w_out, int64_t k_h, int64_t k_w,
                int64_t stride_h, int64_t stride_w, int64_t pad_h,
                int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t group,
                int64_t w_scale_count, int64_t w_zp_count) {
  const int64_t total = n * c_out * h_out * w_out;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t cin_per_g = c_in / group;
  const int64_t cout_per_g = c_out / group;
  const float xs = x_scale[0];
  const float ys = y_scale[0];
  const int x_zp_v = static_cast<int>(x_zp[0]);
  const int y_zp_v = static_cast<int>(y_zp[0]);

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    int64_t tmp = idx;
    const int64_t ow = tmp % w_out;
    tmp /= w_out;
    const int64_t oh = tmp % h_out;
    tmp /= h_out;
    const int64_t oc = tmp % c_out;
    const int64_t ni = tmp / c_out;

    const int64_t g = oc / cout_per_g;
    const int64_t ic_base = g * cin_per_g;
    const int w_zp_v = static_cast<int>(w_zp[w_zp_count == 1 ? 0 : oc]);
    const float ws = w_scale[w_scale_count == 1 ? 0 : oc];

    int64_t dot = 0;
    for (int64_t ic = 0; ic < cin_per_g; ++ic) {
      for (int64_t kh = 0; kh < k_h; ++kh) {
        const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        if (static_cast<uint64_t>(ih) >= static_cast<uint64_t>(h_in))
          continue;
        for (int64_t kw = 0; kw < k_w; ++kw) {
          const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
          if (static_cast<uint64_t>(iw) >= static_cast<uint64_t>(w_in))
            continue;
          const int64_t x_index =
              ((ni * c_in + ic_base + ic) * h_in + ih) * w_in + iw;
          const int64_t w_index = ((oc * cin_per_g + ic) * k_h + kh) * k_w + kw;
          const int xv = static_cast<int>(x[x_index]) - x_zp_v;
          const int wv = static_cast<int>(w[w_index]) - w_zp_v;
          dot += static_cast<int64_t>(xv) * static_cast<int64_t>(wv);
        }
      }
    }
    if (bias)
      dot += bias[oc];

    float q = rintf(static_cast<float>(dot) * xs * ws / ys) +
              static_cast<float>(y_zp_v);
    q = fminf(qlinear_highest<YT>(), fmaxf(qlinear_lowest<YT>(), q));
    y[idx] = static_cast<YT>(static_cast<int>(q));
  }
}

template <typename XT, typename WT, typename YT>
int launch_qlinear_conv(hipStream_t stream, const void *input,
                        const void *input_scale, const void *input_zp,
                        const void *weights, const void *weight_scale,
                        const void *weight_zp, const void *output_scale,
                        const void *output_zp, const void *bias, void *output,
                        int64_t n, int64_t c_in, int64_t c_out, int64_t h_in,
                        int64_t w_in, int64_t h_out, int64_t w_out, int64_t k_h,
                        int64_t k_w, int64_t stride_h, int64_t stride_w,
                        int64_t pad_h, int64_t pad_w, int64_t dil_h,
                        int64_t dil_w, int64_t group, int64_t w_scale_count,
                        int64_t w_zp_count) {
  const int64_t total = n * c_out * h_out * w_out;
  if (total <= 0)
    return 0;
  constexpr int kThreads = 256;
  int64_t blocks = (total + kThreads - 1) / kThreads;
  if (blocks > 65535)
    blocks = 65535;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qlinear_conv_2d<XT, WT, YT>),
      dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, stream,
      static_cast<const XT *>(input), static_cast<const float *>(input_scale),
      static_cast<const XT *>(input_zp), static_cast<const WT *>(weights),
      static_cast<const float *>(weight_scale),
      static_cast<const WT *>(weight_zp),
      static_cast<const float *>(output_scale),
      static_cast<const YT *>(output_zp), static_cast<const int32_t *>(bias),
      static_cast<YT *>(output), n, c_in, c_out, h_in, w_in, h_out, w_out, k_h,
      k_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, group, w_scale_count,
      w_zp_count);
  return 0;
}

int hip_dtype_pair(int dtype, bool *is_signed) {
  if (dtype == HIP_DTYPE_UINT8) {
    *is_signed = false;
    return 0;
  }
  if (dtype == HIP_DTYPE_INT8) {
    *is_signed = true;
    return 0;
  }
  return -1;
}

} // namespace

int hip_qlinear_conv(void *stream, const void *input, const void *input_scale,
                     const void *input_zero_point, const void *weights,
                     const void *weight_scale, const void *weight_zero_point,
                     const void *output_scale, const void *output_zero_point,
                     const void *bias, void *output, int64_t batch,
                     int64_t in_channels, int64_t out_channels,
                     int64_t height_in, int64_t width_in, int64_t height_out,
                     int64_t width_out, int64_t kernel_h, int64_t kernel_w,
                     int64_t stride_h, int64_t stride_w, int64_t pad_h,
                     int64_t pad_w, int64_t dilation_h, int64_t dilation_w,
                     int64_t group, int input_dtype, int weight_dtype,
                     int output_dtype, int64_t weight_scale_count,
                     int64_t weight_zp_count) {
  bool input_signed = false;
  bool weight_signed = false;
  bool output_signed = false;
  if (hip_dtype_pair(input_dtype, &input_signed) ||
      hip_dtype_pair(weight_dtype, &weight_signed) ||
      hip_dtype_pair(output_dtype, &output_signed)) {
    fprintf(stderr,
            "[custom_kernels] hip_qlinear_conv: unsupported dtypes "
            "x=%d w=%d y=%d\n",
            input_dtype, weight_dtype, output_dtype);
    return -1;
  }

  (void)hipGetLastError();
  int rc = 0;
  if (!input_signed && weight_signed && !output_signed) {
    rc = launch_qlinear_conv<uint8_t, int8_t, uint8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (!input_signed && !weight_signed && !output_signed) {
    rc = launch_qlinear_conv<uint8_t, uint8_t, uint8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (input_signed && weight_signed && !output_signed) {
    rc = launch_qlinear_conv<int8_t, int8_t, uint8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (!input_signed && weight_signed && output_signed) {
    rc = launch_qlinear_conv<uint8_t, int8_t, int8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (input_signed && !weight_signed && !output_signed) {
    rc = launch_qlinear_conv<int8_t, uint8_t, uint8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (input_signed && weight_signed && output_signed) {
    rc = launch_qlinear_conv<int8_t, int8_t, int8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else if (!input_signed && !weight_signed && output_signed) {
    rc = launch_qlinear_conv<uint8_t, uint8_t, int8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  } else {
    rc = launch_qlinear_conv<int8_t, uint8_t, int8_t>(
        static_cast<hipStream_t>(stream), input, input_scale, input_zero_point,
        weights, weight_scale, weight_zero_point, output_scale,
        output_zero_point, bias, output, batch, in_channels, out_channels,
        height_in, width_in, height_out, width_out, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        weight_scale_count, weight_zp_count);
  }
  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qlinear_conv launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
