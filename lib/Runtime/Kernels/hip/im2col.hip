// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "hip_custom_kernels.h"

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void im2d2col_f16(const __half *__restrict__ input, int64_t C,
                             int64_t H, int64_t W, int64_t kH, int64_t kW,
                             int64_t pad_top, int64_t pad_left,
                             int64_t stride_h, int64_t stride_w,
                             __half *__restrict__ output, int64_t out_H,
                             int64_t out_W) {
  unsigned int c = blockIdx.x;
  unsigned int out_h = blockIdx.y * stride_h;
  unsigned int kh = threadIdx.x;
  unsigned int kw = threadIdx.y;

  if (c >= C || out_h >= out_H || kh >= kH || kw >= kW)
    return;

  auto in_h = out_h + kh;

  auto pad_bottom_coord = pad_top + H;
  auto pad_right_coord = pad_left + W;

  for (unsigned int out_w = 0; out_w < out_W; out_w += stride_w) {
    auto in_w = out_w + kw;
    auto outputIdx = c * (kH * kW * out_H * out_W) + kh * (kW * out_H * out_W) +
                     kw * (out_H * out_W) + out_h * (out_W) + out_w;

    if (in_w < pad_left || in_w >= pad_right_coord) {
      output[outputIdx] = 0;
      continue;
    }

    if (in_h < pad_top || in_h >= pad_bottom_coord) {
      output[outputIdx] = 0;
      continue;
    }

    auto inputIdx = c * (H * W) + (in_h - pad_top) * (W) + in_w - pad_left;
    output[outputIdx] = input[inputIdx];
  }
}

__global__ void im2d2col_f32(const float *__restrict__ input, int64_t C,
                             int64_t H, int64_t W, int64_t kH, int64_t kW,
                             int64_t pad_top, int64_t pad_left,
                             int64_t stride_h, int64_t stride_w,
                             float *__restrict__ output, int64_t out_H,
                             int64_t out_W) {
  unsigned int c = blockIdx.x;
  unsigned int out_h = blockIdx.y * stride_h;
  unsigned int kh = threadIdx.x;
  unsigned int kw = threadIdx.y;

  if (c >= C || out_h >= out_H || kh >= kH || kw >= kW)
    return;

  auto in_h = out_h + kh;

  auto pad_bottom_coord = pad_top + H;
  auto pad_right_coord = pad_left + W;

  for (unsigned int out_w = 0; out_w < out_W; out_w += stride_w) {
    auto in_w = out_w + kw;
    auto outputIdx = c * (kH * kW * out_H * out_W) + kh * (kW * out_H * out_W) +
                     kw * (out_H * out_W) + out_h * (out_W) + out_w;

    if (in_w < pad_left || in_w >= pad_right_coord) {
      output[outputIdx] = 0;
      continue;
    }

    if (in_h < pad_top || in_h >= pad_bottom_coord) {
      output[outputIdx] = 0;
      continue;
    }

    auto inputIdx = c * (H * W) + (in_h - pad_top) * (W) + in_w - pad_left;
    output[outputIdx] = input[inputIdx];
  }
}

extern "C" int hip_im2d2col(void *stream, const void *input, int64_t data_type,
                            int64_t C, int64_t H, int64_t W, int64_t kh,
                            int64_t kw, int64_t pad_top, int64_t pad_bottom,
                            int64_t pad_left, int64_t pad_right,
                            int64_t stride_h, int64_t stride_w, void *output,
                            int64_t out_h, int64_t out_w) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  if (data_type == HIP_DTYPE_FLOAT16) {
    im2d2col_f16<<<dim3(C, out_h), dim3(kh, kw), 0, hip_stream>>>(
        static_cast<const __half *>(input), C, H, W, kh, kw, pad_top, pad_left,
        stride_h, stride_w, static_cast<__half *>(output), out_h, out_w);
  } else {
    im2d2col_f32<<<dim3(C, out_h), dim3(kh, kw), 0, hip_stream>>>(
        static_cast<const float *>(input), C, H, W, kh, kw, pad_top, pad_left,
        stride_h, stride_w, static_cast<float *>(output), out_h, out_w);
  }
  return 0;
}
