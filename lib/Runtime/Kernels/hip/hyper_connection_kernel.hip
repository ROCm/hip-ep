/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft HyperConnectionPreMix / HyperConnectionPostMix.
 *
 * Streams are (row, branch, hidden) in memory. A flattened (..., C * H)
 * tensor is that layout with no extra transpose. Gates are either one value
 * per branch or one value per (branch, hidden), already activated by the
 * graph. Products accumulate in FP32 and each output is rounded once, which
 * is the ORT CPU contract.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

__device__ inline float hc_load(const float *p) { return *p; }
__device__ inline float hc_load(const __half *p) { return __half2float(*p); }
__device__ inline void hc_store(float *p, float v) { *p = v; }
__device__ inline void hc_store(__half *p, float v) { *p = __float2half(v); }

template <typename T>
__global__ void hip_hc_pre_mix_kernel(const T *__restrict__ streams,
                                      const T *__restrict__ gate,
                                      T *__restrict__ output, int64_t rows,
                                      int64_t branches, int64_t hidden,
                                      int feature_gate, float reduction_scale) {
  int64_t n = rows * hidden;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t row = i / hidden;
  int64_t h = i - row * hidden;
  float sum = 0.f;
  for (int64_t c = 0; c < branches; ++c) {
    int64_t x_index = (row * branches + c) * hidden + h;
    int64_t g_index = feature_gate ? x_index : row * branches + c;
    sum += hc_load(streams + x_index) * hc_load(gate + g_index);
  }
  hc_store(output + i, sum * reduction_scale);
}

template <typename T>
__global__ void hip_hc_post_mix_kernel(const T *__restrict__ streams,
                                       const T *__restrict__ block_output,
                                       const T *__restrict__ post_mix,
                                       const T *__restrict__ stream_mix,
                                       T *__restrict__ output, int64_t rows,
                                       int64_t branches, int64_t hidden,
                                       int feature_gate) {
  int64_t n = rows * branches * hidden;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t h = i % hidden;
  int64_t tmp = i / hidden;
  int64_t c = tmp % branches;
  int64_t row = tmp / branches;
  float value = 0.f;
  if (stream_mix == nullptr) {
    value = hc_load(streams + i);
  } else {
    for (int64_t source = 0; source < branches; ++source) {
      int64_t mix_index = (row * branches + source) * branches + c;
      int64_t x_index = (row * branches + source) * hidden + h;
      value += hc_load(stream_mix + mix_index) * hc_load(streams + x_index);
    }
  }
  int64_t g_index =
      feature_gate ? (row * branches + c) * hidden + h : row * branches + c;
  value += hc_load(post_mix + g_index) * hc_load(block_output + row * hidden + h);
  hc_store(output + i, value);
}

int launch_count(int64_t n, dim3 &grid) {
  constexpr int kBlock = 256;
  if (n <= 0)
    return 0;
  int64_t blocks = (n + kBlock - 1) / kBlock;
  if (blocks > 0x7fffffff)
    return -1;
  grid = dim3(static_cast<unsigned>(blocks));
  return kBlock;
}

template <typename T>
int launch_pre(hipStream_t stream, const void *streams, const void *gate,
               void *output, int64_t rows, int64_t branches, int64_t hidden,
               int feature_gate, float reduction_scale) {
  dim3 grid;
  int threads = launch_count(rows * hidden, grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_pre_mix_kernel<T>), grid,
                     dim3(threads), 0, stream, static_cast<const T *>(streams),
                     static_cast<const T *>(gate), static_cast<T *>(output),
                     rows, branches, hidden, feature_gate, reduction_scale);
  return 0;
}

template <typename T>
int launch_post(hipStream_t stream, const void *streams,
                const void *block_output, const void *post_mix,
                const void *stream_mix, void *output, int64_t rows,
                int64_t branches, int64_t hidden, int feature_gate) {
  dim3 grid;
  int threads = launch_count(rows * branches * hidden, grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_kernel<T>), grid,
                     dim3(threads), 0, stream, static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(post_mix),
                     static_cast<const T *>(stream_mix),
                     static_cast<T *>(output), rows, branches, hidden,
                     feature_gate);
  return 0;
}

} // namespace

extern "C" int hip_hyper_connection_pre_mix(
    void *stream, const void *streams, const void *pre_mix, void *output,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    float reduction_scale, int hip_dtype) {
  if (!streams || !pre_mix || !output || branches <= 0 || hidden <= 0 ||
      rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_pre<__half>(hip_stream, streams, pre_mix, output, rows,
                            branches, hidden, feature_gate, reduction_scale);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_pre<float>(hip_stream, streams, pre_mix, output, rows, branches,
                           hidden, feature_gate, reduction_scale);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_pre_mix: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_pre_mix launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_hyper_connection_post_mix(
    void *stream, const void *streams, const void *block_output,
    const void *post_mix, const void *stream_mix, void *output, int64_t rows,
    int64_t branches, int64_t hidden, int feature_gate, int hip_dtype) {
  if (!streams || !block_output || !post_mix || !output || branches <= 0 ||
      hidden <= 0 || rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_post<__half>(hip_stream, streams, block_output, post_mix,
                             stream_mix, output, rows, branches, hidden,
                             feature_gate);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_post<float>(hip_stream, streams, block_output, post_mix,
                            stream_mix, output, rows, branches, hidden,
                            feature_gate);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix launch failed: "
            "%s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
