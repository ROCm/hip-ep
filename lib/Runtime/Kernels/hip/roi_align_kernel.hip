/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX RoiAlign (4-D NCHW). One thread per output element
 * (roi, c, ph, pw). Sampling, edge handling and max-mode semantics follow
 * the ONNX Runtime CPU kernel:
 *   * a sample with y < -1, y > H, x < -1 or x > W contributes 0
 *   * otherwise coordinates are clamped to [0, size - 1]
 *   * avg divides by max(grid_h * grid_w, 1)
 *   * max takes, per sample, the largest of the four weighted taps
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace {

template <typename T> struct AccType { using type = float; };
template <> struct AccType<double> { using type = double; };

template <typename T> __device__ inline typename AccType<T>::type load(T v);
template <> __device__ inline float load<float>(float v) { return v; }
template <> __device__ inline float load<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ inline float load<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ inline double load<double>(double v) { return v; }

template <typename T> __device__ inline T store(typename AccType<T>::type v);
template <> __device__ inline float store<float>(float v) { return v; }
template <> __device__ inline __half store<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline __hip_bfloat16 store<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ inline double store<double>(double v) { return v; }

template <typename T>
__global__ void roi_align_kernel(const T *__restrict__ input,
                                 const T *__restrict__ rois,
                                 const int64_t *__restrict__ batch_indices,
                                 T *__restrict__ output, int64_t N, int64_t C,
                                 int64_t H, int64_t W, int64_t outH,
                                 int64_t outW, int sampling_ratio,
                                 float spatial_scale, int half_pixel, int mode,
                                 int64_t total) {
  using Acc = typename AccType<T>::type;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t pw = tid % outW;
  int64_t ph = (tid / outW) % outH;
  int64_t c = (tid / (outW * outH)) % C;
  int64_t r = tid / (outW * outH * C);

  int64_t b = batch_indices[r];
  if (b < 0 || b >= N || H <= 0 || W <= 0) {
    output[tid] = store<T>(Acc(0));
    return;
  }

  const T *roi = rois + r * 4;
  Acc scale = static_cast<Acc>(spatial_scale);
  Acc offset = half_pixel ? Acc(0.5) : Acc(0);
  Acc roi_start_w = load<T>(roi[0]) * scale - offset;
  Acc roi_start_h = load<T>(roi[1]) * scale - offset;
  Acc roi_end_w = load<T>(roi[2]) * scale - offset;
  Acc roi_end_h = load<T>(roi[3]) * scale - offset;
  Acc roi_w = roi_end_w - roi_start_w;
  Acc roi_h = roi_end_h - roi_start_h;
  if (!half_pixel) {
    roi_w = roi_w > Acc(1) ? roi_w : Acc(1);
    roi_h = roi_h > Acc(1) ? roi_h : Acc(1);
  }
  Acc bin_h = roi_h / static_cast<Acc>(outH);
  Acc bin_w = roi_w / static_cast<Acc>(outW);
  int64_t grid_h = sampling_ratio > 0
                       ? sampling_ratio
                       : static_cast<int64_t>(ceil(roi_h / outH));
  int64_t grid_w = sampling_ratio > 0
                       ? sampling_ratio
                       : static_cast<int64_t>(ceil(roi_w / outW));
  int64_t count = grid_h * grid_w;
  if (count < 1)
    count = 1;

  const T *img = input + (b * C + c) * H * W;
  Acc Hf = static_cast<Acc>(H);
  Acc Wf = static_cast<Acc>(W);
  Acc result = Acc(0);
  bool have_max = false;

  for (int64_t iy = 0; iy < grid_h; ++iy) {
    Acc y = roi_start_h + static_cast<Acc>(ph) * bin_h +
            (static_cast<Acc>(iy) + Acc(0.5)) * bin_h /
                static_cast<Acc>(grid_h);
    for (int64_t ix = 0; ix < grid_w; ++ix) {
      Acc x = roi_start_w + static_cast<Acc>(pw) * bin_w +
              (static_cast<Acc>(ix) + Acc(0.5)) * bin_w /
                  static_cast<Acc>(grid_w);

      Acc v1 = 0, v2 = 0, v3 = 0, v4 = 0;
      if (!(y < Acc(-1) || y > Hf || x < Acc(-1) || x > Wf)) {
        Acc yy = y <= Acc(0) ? Acc(0) : y;
        Acc xx = x <= Acc(0) ? Acc(0) : x;
        int64_t y_low = static_cast<int64_t>(yy);
        int64_t x_low = static_cast<int64_t>(xx);
        int64_t y_high, x_high;
        if (y_low >= H - 1) {
          y_high = y_low = H - 1;
          yy = static_cast<Acc>(y_low);
        } else {
          y_high = y_low + 1;
        }
        if (x_low >= W - 1) {
          x_high = x_low = W - 1;
          xx = static_cast<Acc>(x_low);
        } else {
          x_high = x_low + 1;
        }
        Acc ly = yy - static_cast<Acc>(y_low);
        Acc lx = xx - static_cast<Acc>(x_low);
        Acc hy = Acc(1) - ly;
        Acc hx = Acc(1) - lx;
        v1 = hy * hx * load<T>(img[y_low * W + x_low]);
        v2 = hy * lx * load<T>(img[y_low * W + x_high]);
        v3 = ly * hx * load<T>(img[y_high * W + x_low]);
        v4 = ly * lx * load<T>(img[y_high * W + x_high]);
      }

      if (mode == 0) {
        result += v1 + v2 + v3 + v4;
      } else {
        Acc v = v1;
        v = v2 > v ? v2 : v;
        v = v3 > v ? v3 : v;
        v = v4 > v ? v4 : v;
        if (!have_max || v > result) {
          result = v;
          have_max = true;
        }
      }
    }
  }
  if (mode == 0)
    result /= static_cast<Acc>(count);
  output[tid] = store<T>(result);
}

template <typename T>
void launch(hipStream_t stream, dim3 grid, dim3 block, const void *input,
            const void *rois, const void *batch_indices, void *output,
            int64_t n, int64_t c, int64_t in_h, int64_t in_w, int64_t out_h,
            int64_t out_w, int sampling_ratio, float spatial_scale,
            int half_pixel, int mode, int64_t total) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(roi_align_kernel<T>), grid, block, 0,
                     stream, static_cast<const T *>(input),
                     static_cast<const T *>(rois),
                     static_cast<const int64_t *>(batch_indices),
                     static_cast<T *>(output), n, c, in_h, in_w, out_h, out_w,
                     sampling_ratio, spatial_scale, half_pixel, mode, total);
}

} // namespace

extern "C" int hip_roi_align(void *stream, const void *input, const void *rois,
                             const void *batch_indices, void *output, int64_t n,
                             int64_t c, int64_t in_h, int64_t in_w,
                             int64_t num_rois, int64_t out_h, int64_t out_w,
                             int sampling_ratio, float spatial_scale,
                             int half_pixel, int mode, int hip_dtype) {
  if (num_rois <= 0 || c <= 0 || out_h <= 0 || out_w <= 0)
    return 0;
  if (mode != 0 && mode != 1) {
    fprintf(stderr, "[custom_kernels] hip_roi_align: bad mode=%d\n", mode);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int64_t total = num_rois * c * out_h * out_w;
  dim3 block(256);
  dim3 grid_dim(static_cast<unsigned int>((total + 255) / 256));

  (void)hipGetLastError();
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_roi_align: dtype=%d N=%lld C=%lld in=%lldx%lld "
      "rois=%lld out=%lldx%lld sr=%d scale=%f half_pixel=%d mode=%d\n",
      hip_dtype, (long long)n, (long long)c, (long long)in_h, (long long)in_w,
      (long long)num_rois, (long long)out_h, (long long)out_w, sampling_ratio,
      (double)spatial_scale, half_pixel, mode);

  if (hip_dtype == HIP_DTYPE_FLOAT32) {
    launch<float>(hip_stream, grid_dim, block, input, rois, batch_indices,
                  output, n, c, in_h, in_w, out_h, out_w, sampling_ratio,
                  spatial_scale, half_pixel, mode, total);
  } else if (hip_dtype == HIP_DTYPE_FLOAT16) {
    launch<__half>(hip_stream, grid_dim, block, input, rois, batch_indices,
                   output, n, c, in_h, in_w, out_h, out_w, sampling_ratio,
                   spatial_scale, half_pixel, mode, total);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    launch<__hip_bfloat16>(hip_stream, grid_dim, block, input, rois,
                           batch_indices, output, n, c, in_h, in_w, out_h,
                           out_w, sampling_ratio, spatial_scale, half_pixel,
                           mode, total);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    launch<double>(hip_stream, grid_dim, block, input, rois, batch_indices,
                   output, n, c, in_h, in_w, out_h, out_w, sampling_ratio,
                   spatial_scale, half_pixel, mode, total);
  } else {
    fprintf(stderr, "[custom_kernels] hip_roi_align: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_roi_align launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
