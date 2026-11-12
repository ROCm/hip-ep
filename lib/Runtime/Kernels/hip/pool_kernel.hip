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
#include <initializer_list>

// =============================================================================
// Pool — MaxPool / AveragePool / LpPool (1D / 2D / 3D)
// =============================================================================
//
// One thread per output element (windows of kWideMinWindow+ elements without
// argmax indices and with 32-bit extents use pool_wide_kernel instead).
// Output is laid out row-major over
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

// Large windows: one output per one-thread-per-output pass leaves few threads,
// each walking the whole window serially. Here a group of kWideGroup lanes
// shares one output, strides the flattened window (consecutive lanes read
// consecutive innermost elements), and combines partials with shuffles.
// Same semantics as pool_kernel; argmax indices are not produced.
//
// All index math is 32-bit (the host checks the extents fit). A lane advances
// its flattened window position by kWideGroup with a carry decode, so the
// loop has no divisions, and out-of-bounds taps are masked rather than
// branched around so the unrolled loop keeps several loads in flight.
constexpr int kWideGroup = 32;
constexpr int64_t kWideMinWindow = 64;

struct WideShape {
  int in0, in1, in2;
  int out0, out1, out2;
  int k0, k1, k2;
  int s0, s1, s2;
  int p0, p1, p2;
  int d0, d1, d2;
  int total;
  // kWideGroup == step0 * k1 * k2 + step1 * k2 + step2 with step1 < k1 and
  // step2 < k2.
  int step0, step1, step2;
};

template <typename T>
__global__ void pool_wide_kernel(const T* __restrict__ input,
                                 T* __restrict__ output, int mode,
                                 int count_include_pad, int p, WideShape s) {
  int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int tid = gid / kWideGroup;
  int lane = static_cast<int>(threadIdx.x % kWideGroup);
  // Every lane stays to the end because the shuffles need the whole group.
  bool active = tid < s.total;
  if (!active) tid = 0;

  int o2 = tid % s.out2;
  int t = tid / s.out2;
  int o1 = t % s.out1;
  t /= s.out1;
  int o0 = t % s.out0;
  int nc = t / s.out0;
  int k12 = s.k1 * s.k2;
  int kvol = s.k0 * k12;

  const T* base = input + static_cast<int64_t>(nc) * s.in0 * s.in1 * s.in2;
  int start0 = o0 * s.s0 - s.p0;
  int start1 = o1 * s.s1 - s.p1;
  int start2 = o2 * s.s2 - s.p2;

  float best = -FLT_MAX;
  float sum = 0.0f;
  int valid_count = 0;
  float pf = static_cast<float>(p);

  int i0 = lane / k12;
  int i1 = (lane - i0 * k12) / s.k2;
  int i2 = lane - i0 * k12 - i1 * s.k2;
#pragma unroll 4
  for (int w = lane; w < kvol; w += kWideGroup) {
    int x0 = start0 + i0 * s.d0;
    int x1 = start1 + i1 * s.d1;
    int x2 = start2 + i2 * s.d2;
    bool in_bounds = static_cast<unsigned>(x0) < static_cast<unsigned>(s.in0) &&
                     static_cast<unsigned>(x1) < static_cast<unsigned>(s.in1) &&
                     static_cast<unsigned>(x2) < static_cast<unsigned>(s.in2);
    int off = in_bounds ? (x0 * s.in1 + x1) * s.in2 + x2 : 0;
    float v = to_float<T>(base[off]);
    if (mode == kPoolMax) {
      best = in_bounds ? fmaxf(best, v) : best;
    } else if (mode == kPoolLp) {
      sum += in_bounds ? powf(fabsf(v), pf) : 0.0f;
    } else {
      sum += in_bounds ? v : 0.0f;
      valid_count += in_bounds ? 1 : 0;
    }
    i2 += s.step2;
    if (i2 >= s.k2) {
      i2 -= s.k2;
      ++i1;
    }
    i1 += s.step1;
    if (i1 >= s.k1) {
      i1 -= s.k1;
      ++i0;
    }
    i0 += s.step0;
  }

  for (int off = kWideGroup / 2; off > 0; off /= 2) {
    best = fmaxf(best, __shfl_xor(best, off, kWideGroup));
    sum += __shfl_xor(sum, off, kWideGroup);
    valid_count += __shfl_xor(valid_count, off, kWideGroup);
  }
  if (!active || lane != 0) return;

  float result;
  if (mode == kPoolMax) {
    result = best;
  } else if (mode == kPoolLp) {
    result = powf(sum, 1.0f / pf);
  } else {
    int64_t divisor = count_include_pad ? kvol : valid_count;
    result = (divisor > 0) ? (sum / static_cast<float>(divisor)) : 0.0f;
  }
  output[tid] = from_float<T>(result);
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

  auto fits = [](std::initializer_list<int64_t> vals) {
    for (int64_t v : vals)
      if (v < INT32_MIN || v > INT32_MAX)
        return false;
    return true;
  };
  if (!indices && k0 * k1 * k2 >= kWideMinWindow &&
      fits({total * kWideGroup + block, in0 * in1 * in2, N * C * in0 * in1 * in2,
            k0 * k1 * k2, out0 * s0 + k0 * d0, out1 * s1 + k1 * d1,
            out2 * s2 + k2 * d2, p0, p1, p2})) {
    WideShape ws;
    ws.in0 = static_cast<int>(in0);
    ws.in1 = static_cast<int>(in1);
    ws.in2 = static_cast<int>(in2);
    ws.out0 = static_cast<int>(out0);
    ws.out1 = static_cast<int>(out1);
    ws.out2 = static_cast<int>(out2);
    ws.k0 = static_cast<int>(k0);
    ws.k1 = static_cast<int>(k1);
    ws.k2 = static_cast<int>(k2);
    ws.s0 = static_cast<int>(s0);
    ws.s1 = static_cast<int>(s1);
    ws.s2 = static_cast<int>(s2);
    ws.p0 = static_cast<int>(p0);
    ws.p1 = static_cast<int>(p1);
    ws.p2 = static_cast<int>(p2);
    ws.d0 = static_cast<int>(d0);
    ws.d1 = static_cast<int>(d1);
    ws.d2 = static_cast<int>(d2);
    ws.total = static_cast<int>(total);
    int k12 = ws.k1 * ws.k2;
    ws.step0 = kWideGroup / k12;
    ws.step1 = (kWideGroup % k12) / ws.k2;
    ws.step2 = kWideGroup % ws.k2;
    int64_t wide_grid = (total * kWideGroup + block - 1) / block;
    (void)hipGetLastError();
    hipLaunchKernelGGL(pool_wide_kernel<T>,
                       dim3(static_cast<unsigned>(wide_grid)), dim3(block),
                       0, stream,
                       static_cast<const T*>(input), static_cast<T*>(output),
                       mode, count_include_pad, p, ws);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
      fprintf(stderr, "[custom_kernels] hip_pool wide launch failed: %s\n",
              hipGetErrorString(err));
    return static_cast<int>(err);
  }

  int64_t grid = (total + block - 1) / block;

  (void)hipGetLastError();
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

// Integer MaxPool. Compare in the storage type so ui8 200 > 100 stays
// unsigned. Average and LP are rejected by the dispatcher; they need a
// float accumulator and are not a quantized-max.
template <typename T>
__global__ void pool_max_int_kernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    int64_t* __restrict__ indices,
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

  int64_t plane = out0 * out1 * out2;
  int64_t channel_stride = plane;
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

  int64_t start0 = o0 * s0 - p0;
  int64_t start1 = o1 * s1 - p1;
  int64_t start2 = o2 * s2 - p2;

  bool found = false;
  T best = 0;
  int64_t best_idx = -1;

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
        T v = input[flat];
        if (!found || v > best) {
          found = true;
          best = v;
          best_idx = flat;
        }
      }
    }
  }

  output[tid] = best;
  if (indices) indices[tid] = best_idx;
  (void)spatial_rank;
  (void)N;
}

template <typename T>
static int launch_max_int(hipStream_t stream, const void* input, void* output,
                          void* indices, int spatial_rank,
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

  hipLaunchKernelGGL(pool_max_int_kernel<T>,
                     dim3(static_cast<unsigned>(grid)), dim3(block),
                     0, stream,
                     static_cast<const T*>(input), static_cast<T*>(output),
                     static_cast<int64_t*>(indices),
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
    fprintf(stderr, "[custom_kernels] hip_pool int max launch failed: %s\n",
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
    case HIP_DTYPE_INT8:
    case HIP_DTYPE_UINT8: {
      if (mode != kPoolMax) {
        fprintf(stderr,
                "[custom_kernels] hip_pool: i8/ui8 only supported for "
                "MaxPool (mode=%d)\n",
                mode);
        return -1;
      }
      if (hip_dtype == HIP_DTYPE_UINT8)
        return launch_max_int<uint8_t>(
            hip_stream, input, output, indices, spatial_rank,
            N, C, in_d0, in_d1, in_d2,
            out_d0, out_d1, out_d2,
            k0, k1, k2, s0, s1, s2,
            p0, p1, p2, dil0, dil1, dil2);
      return launch_max_int<int8_t>(
          hip_stream, input, output, indices, spatial_rank,
          N, C, in_d0, in_d1, in_d2,
          out_d0, out_d1, out_d2,
          k0, k1, k2, s0, s1, s2,
          p0, p1, p2, dil0, dil1, dil2);
    }
    default:
      fprintf(stderr,
              "[custom_kernels] hip_pool: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}
