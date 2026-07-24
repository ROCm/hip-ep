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
#include <cfloat>
#include <cmath>

// =============================================================================
// Pool — MaxPool / AveragePool / LpPool (1D / 2D / 3D)
// =============================================================================
//
// One thread per output element.  Output is laid out row-major over
// (N, C, O_1, O_2, O_3) where unused trailing axes are size 1.  For each
// thread we walk its `kernel_shape` window (with stride / pad / dilation)
// over the input plane(s).  Positions that fall in the pad region (outside
// the input bounds) are skipped — they contribute neither to value nor to
// indices, matching ONNX semantics.  `mode` selects the reduction:
//   MAX     : pick the max value; when `indices` is non-null also write the
//             row-major flat index in the unpadded input.
//   AVERAGE : sum the window, divide by the divisor (count_include_pad picks
//             the in-bounds element count vs the full kernel volume).
//   LP      : sum |v|^p over the window, then take the p-th root.
// Accumulation is in float regardless of dtype to keep precision on fp16 /
// bf16 inputs.

namespace {

// Reduction modes (mirror HIPDNN_EP_POOL_* in hipdnn_ep_runtime.h).
constexpr int kPoolAverage = 0;
constexpr int kPoolMax = 1;
constexpr int kPoolLp = 2;

template <typename T> __device__ inline float to_float(T v);
template <> __device__ inline float to_float<float>(float v) { return v; }
template <> __device__ inline float to_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ inline float to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}
template <> __device__ inline float to_float<double>(double v) {
  return static_cast<float>(v);
}

template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline float from_float<float>(float v) { return v; }
template <> __device__ inline __half from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}
template <> __device__ inline double from_float<double>(float v) {
  return static_cast<double>(v);
}

template <typename T>
__global__ void pool_kernel(const T* __restrict__ input,
                            T* __restrict__ output,
                            int64_t* __restrict__ indices,
                            int mode,
                            int count_include_pad,
                            int p,
                            int spatial_rank,
                            int64_t N, int64_t C,
                            int64_t in0, int64_t in1, int64_t in2,
                            int64_t out0, int64_t out1, int64_t out2,
                            int64_t k0, int64_t k1, int64_t k2,
                            int64_t s0, int64_t s1, int64_t s2,
                            int64_t p0, int64_t p1, int64_t p2,
                            int64_t d0, int64_t d1, int64_t d2,
                            int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  // Decode (n, c, o0, o1, o2) from tid using the row-major output layout.
  int64_t plane = out0 * out1 * out2;
  int64_t channel_stride = plane;          // per (n, c)
  int64_t batch_stride = C * channel_stride;

  int64_t n = tid / batch_stride;
  int64_t rem = tid - n * batch_stride;
  int64_t c = rem / channel_stride;
  rem -= c * channel_stride;
  int64_t o0 = rem / (out1 * out2);
  int64_t r1 = rem - o0 * (out1 * out2);
  int64_t o1 = r1 / out2;
  int64_t o2 = r1 - o1 * out2;

  int64_t base_in = (n * C + c) * (in0 * in1 * in2);
  int64_t in_stride0 = in1 * in2;
  int64_t in_stride1 = in2;

  // Window start (in input coords, may be negative when pad_begin > 0).
  int64_t start0 = o0 * s0 - p0;
  int64_t start1 = o1 * s1 - p1;
  int64_t start2 = o2 * s2 - p2;

  // Accumulators shared by the three modes (only the relevant one is used).
  float best = -FLT_MAX;     // MAX
  int64_t best_idx = -1;     // MAX
  float sum = 0.0f;          // AVERAGE / LP
  int64_t valid_count = 0;   // AVERAGE (count_include_pad == 0)
  float pf = static_cast<float>(p);

  // Loop bounds collapse to single iteration when spatial_rank < 3 because
  // the unused-axis kernel size is 1, stride 1, dilation 1, pad 0 — leaving
  // exactly index 0.
  for (int64_t i0 = 0; i0 < k0; ++i0) {
    int64_t x0 = start0 + i0 * d0;
    if (x0 < 0 || x0 >= in0) continue;
    for (int64_t i1 = 0; i1 < k1; ++i1) {
      int64_t x1 = start1 + i1 * d1;
      if (x1 < 0 || x1 >= in1) continue;
      for (int64_t i2 = 0; i2 < k2; ++i2) {
        int64_t x2 = start2 + i2 * d2;
        if (x2 < 0 || x2 >= in2) continue;

        int64_t in_off = x0 * in_stride0 + x1 * in_stride1 + x2;
        int64_t flat = base_in + in_off;
        float v = to_float<T>(input[flat]);

        if (mode == kPoolMax) {
          if (v > best) {
            best = v;
            best_idx = flat;
          }
        } else if (mode == kPoolLp) {
          sum += powf(fabsf(v), pf);
        } else { // AVERAGE
          sum += v;
          ++valid_count;
        }
      }
    }
  }

  float result;
  if (mode == kPoolMax) {
    result = best;
  } else if (mode == kPoolLp) {
    result = powf(sum, 1.0f / pf);
  } else { // AVERAGE
    // count_include_pad == 1 divides by the full kernel volume (pad cells
    // contributed 0 to the sum); otherwise divide by the in-bounds count.
    int64_t divisor = count_include_pad ? (k0 * k1 * k2) : valid_count;
    result = (divisor > 0) ? (sum / static_cast<float>(divisor)) : 0.0f;
  }

  output[tid] = from_float<T>(result);
  if (mode == kPoolMax && indices) indices[tid] = best_idx;
}

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  void* indices, int mode, int count_include_pad, int p,
                  int spatial_rank,
                  int64_t N, int64_t C,
                  int64_t in0, int64_t in1, int64_t in2,
                  int64_t out0, int64_t out1, int64_t out2,
                  int64_t k0, int64_t k1, int64_t k2,
                  int64_t s0, int64_t s1, int64_t s2,
                  int64_t p0, int64_t p1, int64_t p2,
                  int64_t d0, int64_t d1, int64_t d2) {
  int64_t total = N * C * out0 * out1 * out2;
  if (total <= 0) return 0;
  int block = 256;
  int64_t grid = (total + block - 1) / block;

  hipLaunchKernelGGL(pool_kernel<T>,
                     dim3(static_cast<unsigned>(grid)), dim3(block),
                     0, stream,
                     static_cast<const T*>(input), static_cast<T*>(output),
                     static_cast<int64_t*>(indices),
                     mode, count_include_pad, p,
                     spatial_rank, N, C,
                     in0, in1, in2,
                     out0, out1, out2,
                     k0, k1, k2,
                     s0, s1, s2,
                     p0, p1, p2,
                     d0, d1, d2,
                     total);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_pool launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

} // namespace

extern "C" int hip_pool(void* stream,
                        const void* input,
                        void* output,
                        void* indices,
                        int hip_dtype,
                        int mode,
                        int spatial_rank,
                        int64_t N, int64_t C,
                        int64_t in_d0, int64_t in_d1, int64_t in_d2,
                        int64_t out_d0, int64_t out_d1, int64_t out_d2,
                        int64_t k0, int64_t k1, int64_t k2,
                        int64_t s0, int64_t s1, int64_t s2,
                        int64_t p0, int64_t p1, int64_t p2,
                        int64_t dil0, int64_t dil1, int64_t dil2,
                        int count_include_pad,
                        int p) {
  if (spatial_rank < 1 || spatial_rank > 3) {
    fprintf(stderr,
            "[custom_kernels] hip_pool: unsupported spatial_rank=%d\n",
            spatial_rank);
    return -1;
  }
  if (mode != kPoolAverage && mode != kPoolMax && mode != kPoolLp) {
    fprintf(stderr, "[custom_kernels] hip_pool: unsupported mode=%d\n", mode);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pool: mode=%d dtype=%d spatial_rank=%d "
      "N=%lld C=%lld in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] "
      "k=[%lld,%lld,%lld] s=[%lld,%lld,%lld] p=[%lld,%lld,%lld] "
      "d=[%lld,%lld,%lld] count_include_pad=%d p=%d indices=%s\n",
      mode, hip_dtype, spatial_rank, (long long)N, (long long)C,
      (long long)in_d0, (long long)in_d1, (long long)in_d2,
      (long long)out_d0, (long long)out_d1, (long long)out_d2,
      (long long)k0, (long long)k1, (long long)k2,
      (long long)s0, (long long)s1, (long long)s2,
      (long long)p0, (long long)p1, (long long)p2,
      (long long)dil0, (long long)dil1, (long long)dil2,
      count_include_pad, p, indices ? "yes" : "null");

  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      return launch<float>(hip_stream, input, output, indices, mode,
                           count_include_pad, p, spatial_rank,
                           N, C, in_d0, in_d1, in_d2,
                           out_d0, out_d1, out_d2,
                           k0, k1, k2, s0, s1, s2,
                           p0, p1, p2, dil0, dil1, dil2);
    case HIP_DTYPE_FLOAT16:
      return launch<__half>(hip_stream, input, output, indices, mode,
                            count_include_pad, p, spatial_rank,
                            N, C, in_d0, in_d1, in_d2,
                            out_d0, out_d1, out_d2,
                            k0, k1, k2, s0, s1, s2,
                            p0, p1, p2, dil0, dil1, dil2);
    case HIP_DTYPE_BFLOAT16:
      return launch<hip_bfloat16>(hip_stream, input, output, indices, mode,
                                  count_include_pad, p, spatial_rank, N, C,
                                  in_d0, in_d1, in_d2,
                                  out_d0, out_d1, out_d2,
                                  k0, k1, k2, s0, s1, s2,
                                  p0, p1, p2, dil0, dil1, dil2);
    case HIP_DTYPE_FLOAT64:
      return launch<double>(hip_stream, input, output, indices, mode,
                            count_include_pad, p, spatial_rank,
                            N, C, in_d0, in_d1, in_d2,
                            out_d0, out_d1, out_d2,
                            k0, k1, k2, s0, s1, s2,
                            p0, p1, p2, dil0, dil1, dil2);
    default:
      fprintf(stderr,
              "[custom_kernels] hip_pool: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}
