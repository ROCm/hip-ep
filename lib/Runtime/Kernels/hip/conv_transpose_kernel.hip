/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// ConvTranspose -- forward transposed convolution (2D spatial)
// ---------------------------------------------------------------------------
//
// ONNX defines ConvTranspose by scattering: each input element is multiplied by
// the whole filter and accumulated into a strided, dilated footprint of the
// output,
//
//   y[n][m][ih*sH + kh*dilH - padTop][iw*sW + kw*dilW - padLeft]
//       += x[n][c][ih][iw] * w[c][m_local][kh][kw]
//
// Written that way it needs atomics, because neighbouring input elements land
// on overlapping footprints whenever the stride is smaller than the kernel.
// Inverting it removes them: solving the same equation for the input position
// gives, for a fixed output cell, exactly the taps that can reach it --
//
//   ih = (oh + padTop - kh*dilH) / sH,  valid when that division is exact and
//                                       the quotient is inside the input
//
// so each thread owns one output element, reads only, and the result does not
// depend on scheduling order.
//
// Two things fall out of the ONNX definition rather than being handled:
//
//   * Trailing pads are absent from the ABI, exactly as in `hip_conv`. Padding
//     on a transposed convolution crops the output, and cropping at the far end
//     only removes output positions -- which the caller has already accounted
//     for in `out_h` / `out_w` -- so it changes nothing this kernel reads.
//   * `output_padding` likewise only extends `out_h` / `out_w`. The cells it
//     adds are ones no input reaches, and the divisibility and range tests
//     leave them at just the bias, which is what ONNX asks for.
//
// The weight layout is ONNX ConvTranspose's `[C, M/group, kH, kW]` -- input
// channels first, unlike forward Conv's `[M, C/group, ...]` -- and is indexed
// directly, so nothing has to be transposed first.
//
// Accumulation is in float for every storage dtype, matching `hip_conv`.
//
// This is deliberately a plain one-thread-per-output kernel with no tiling:
// no model the EP supports contains a ConvTranspose, so the shape that matters
// is the one where correctness is obvious.

namespace {

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}

struct ConvTransposeParams {
  int64_t N, Cin, Cout;
  int64_t CinPerGroup, MPerGroup;
  int64_t inH, inW;
  int64_t outH, outW;
  int64_t kH, kW;
  int64_t sH, sW;
  int64_t padTop, padLeft;
  int64_t dilH, dilW;
};

template <typename T>
__global__ void conv_transpose_direct_kernel(const T *__restrict__ x,
                                             const T *__restrict__ w,
                                             const T *__restrict__ bias,
                                             T *__restrict__ y,
                                             ConvTransposeParams pr,
                                             int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  // tid indexes Y as [N][Cout][outH][outW], row-major.
  const int64_t ow = tid % pr.outW;
  int64_t rest = tid / pr.outW;
  const int64_t oh = rest % pr.outH;
  rest /= pr.outH;
  const int64_t m = rest % pr.Cout;
  const int64_t n = rest / pr.Cout;

  const int64_t g = m / pr.MPerGroup;
  const int64_t mLocal = m - g * pr.MPerGroup;

  const int64_t inPlane = pr.inH * pr.inW;
  const int64_t wChanStride = pr.MPerGroup * pr.kH * pr.kW;
  // Both bases are already advanced to this group's first input channel, so the
  // reduction below indexes channels relative to the group.
  const T *xBase = x + (n * pr.Cin + g * pr.CinPerGroup) * inPlane;
  const T *wBase =
      w + (g * pr.CinPerGroup) * wChanStride + mLocal * pr.kH * pr.kW;

  float acc = bias ? to_float<T>(bias[m]) : 0.0f;

  for (int64_t kh = 0; kh < pr.kH; ++kh) {
    // Exact division is the whole test: a tap only reaches this row if the
    // stride steps onto it. Truncation toward zero makes a negative numerator
    // land on a negative quotient, which the range check then drops.
    const int64_t th = oh + pr.padTop - kh * pr.dilH;
    if (th % pr.sH != 0)
      continue;
    const int64_t ih = th / pr.sH;
    if (ih < 0 || ih >= pr.inH)
      continue;

    for (int64_t kw = 0; kw < pr.kW; ++kw) {
      const int64_t tw = ow + pr.padLeft - kw * pr.dilW;
      if (tw % pr.sW != 0)
        continue;
      const int64_t iw = tw / pr.sW;
      if (iw < 0 || iw >= pr.inW)
        continue;

      const T *xp = xBase + ih * pr.inW + iw;
      const T *wp = wBase + kh * pr.kW + kw;
      for (int64_t c = 0; c < pr.CinPerGroup; ++c)
        acc += to_float<T>(xp[c * inPlane]) * to_float<T>(wp[c * wChanStride]);
    }
  }

  y[tid] = from_float<T>(acc);
}

template <typename T>
int launch(hipStream_t stream, const void *input, const void *weights,
           const void *bias, void *output, const ConvTransposeParams &pr,
           int64_t total) {
  constexpr int kBlock = 256;
  const int64_t grid = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(conv_transpose_direct_kernel<T>, dim3(grid), dim3(kBlock),
                     0, stream, static_cast<const T *>(input),
                     static_cast<const T *>(weights),
                     static_cast<const T *>(bias), static_cast<T *>(output), pr,
                     total);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_conv_transpose: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_conv_transpose(void *stream, const void *input,
                                  const void *weights, const void *bias,
                                  void *output, int hip_dtype, int64_t N,
                                  int64_t Cin, int64_t Cout, int64_t in_h,
                                  int64_t in_w, int64_t out_h, int64_t out_w,
                                  int64_t k_h, int64_t k_w, int64_t s_h,
                                  int64_t s_w, int64_t pad_top,
                                  int64_t pad_left, int64_t dil_h,
                                  int64_t dil_w, int64_t group) {
  if (!input || !weights || !output) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: null input/weights/output\n");
    return -1;
  }
  if (group < 1 || Cin % group != 0 || Cout % group != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: group=%lld does not divide "
            "Cin=%lld / Cout=%lld\n",
            (long long)group, (long long)Cin, (long long)Cout);
    return -1;
  }
  if (s_h < 1 || s_w < 1 || dil_h < 1 || dil_w < 1) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: stride=[%lld,%lld] "
            "dilation=[%lld,%lld] must all be at least 1\n",
            (long long)s_h, (long long)s_w, (long long)dil_h,
            (long long)dil_w);
    return -1;
  }

  const int64_t total = N * Cout * out_h * out_w;
  if (total <= 0)
    return 0;

  ConvTransposeParams pr{};
  pr.N = N;
  pr.Cin = Cin;
  pr.Cout = Cout;
  pr.CinPerGroup = Cin / group;
  pr.MPerGroup = Cout / group;
  pr.inH = in_h;
  pr.inW = in_w;
  pr.outH = out_h;
  pr.outW = out_w;
  pr.kH = k_h;
  pr.kW = k_w;
  pr.sH = s_h;
  pr.sW = s_w;
  pr.padTop = pad_top;
  pr.padLeft = pad_left;
  pr.dilH = dil_h;
  pr.dilW = dil_w;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_conv_transpose: dtype=%d N=%lld Cin=%lld "
      "Cout=%lld group=%lld in=[%lld,%lld] out=[%lld,%lld] k=[%lld,%lld] "
      "s=[%lld,%lld] pbegin=[%lld,%lld] dil=[%lld,%lld] bias=%s\n",
      hip_dtype, (long long)N, (long long)Cin, (long long)Cout,
      (long long)group, (long long)in_h, (long long)in_w, (long long)out_h,
      (long long)out_w, (long long)k_h, (long long)k_w, (long long)s_h,
      (long long)s_w, (long long)pad_top, (long long)pad_left,
      (long long)dil_h, (long long)dil_w, bias ? "yes" : "null");

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return launch<float>(hip_stream, input, weights, bias, output, pr, total);
  case HIP_DTYPE_FLOAT16:
    return launch<__half>(hip_stream, input, weights, bias, output, pr, total);
  case HIP_DTYPE_BFLOAT16:
    return launch<hip_bfloat16>(hip_stream, input, weights, bias, output, pr,
                                total);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
