/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft HyperConnectionPreMix / HyperConnectionPostMix.
 *
 * Streams are (row, branch, hidden). A flattened (..., C * H) tensor is
 * that layout with no extra transpose.
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

// Match a separate fp16 elementwise node: compute in FP32, then round to the
// element type before the value is consumed again. f32 is unchanged.
template <typename T>
__device__ float hc_round(float v) {
  if constexpr (sizeof(T) == sizeof(__half))
    return __half2float(__float2half(v));
  return v;
}

template <typename T>
__device__ float hc_sigmoid(float x) {
  return hc_round<T>(1.f / (1.f + expf(-x)));
}

template <typename T>
__global__ void hip_hc_pre_mix_kernel(const T *__restrict__ streams,
                                      const T *__restrict__ gate,
                                      T *__restrict__ output, int64_t rows,
                                      int64_t branches, int64_t hidden,
                                      int feature_gate, int gate_sigmoid,
                                      float reduction_scale) {
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
    float g = hc_load(gate + g_index);
    if (gate_sigmoid)
      g = hc_sigmoid<T>(g);
    sum += hc_load(streams + x_index) * g;
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
               int feature_gate, int gate_sigmoid, float reduction_scale) {
  dim3 grid;
  int threads = launch_count(rows * hidden, grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_pre_mix_kernel<T>), grid,
                     dim3(threads), 0, stream, static_cast<const T *>(streams),
                     static_cast<const T *>(gate), static_cast<T *>(output),
                     rows, branches, hidden, feature_gate, gate_sigmoid,
                     reduction_scale);
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

template <typename T>
__device__ float hc_gate(float raw, int gate_activation) {
  if (!gate_activation)
    return raw;
  return hc_round<T>(hc_sigmoid<T>(hc_round<T>(raw / 4.f)) * 2.f);
}

// One block per (row, branch).
template <typename T>
__global__ void hip_hc_post_mix_norm_kernel(
    const T *__restrict__ streams, const T *__restrict__ block_output,
    const T *__restrict__ gate, const T *__restrict__ stream_mix,
    const T *__restrict__ scale, T *__restrict__ residual, T *__restrict__ normed,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_activation, float epsilon) {
  int64_t branch_row = static_cast<int64_t>(blockIdx.x);
  if (branch_row >= rows * branches)
    return;
  int64_t row = branch_row / branches;
  int64_t c = branch_row - row * branches;
  float branch_gate = 0.f;
  if (!feature_gate)
    branch_gate = hc_gate<T>(hc_load(gate + row * branches + c), gate_activation);

  const T *block_row = block_output + row * hidden;
  const T *scale_row = scale + c * hidden;
  T *residual_row = residual + branch_row * hidden;
  T *normed_row = normed + branch_row * hidden;

  __shared__ float s_reduce[256];
  __shared__ float s_inv_rms;
  float local_sumsq = 0.f;
  for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float base = 0.f;
    if (stream_mix == nullptr) {
      base = hc_load(streams + branch_row * hidden + h);
    } else {
      for (int64_t source = 0; source < branches; ++source) {
        int64_t mix_index = (row * branches + source) * branches + c;
        int64_t x_index = (row * branches + source) * hidden + h;
        base += hc_load(stream_mix + mix_index) * hc_load(streams + x_index);
      }
    }
    float g = branch_gate;
    if (feature_gate) {
      g = hc_gate<T>(hc_load(gate + branch_row * hidden + h), gate_activation);
    }
    float y = hc_round<T>(base + g * hc_load(block_row + h));
    hc_store(residual_row + h, y);
    local_sumsq += y * y;
  }
  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_inv_rms = rsqrtf(s_reduce[0] / static_cast<float>(hidden) + epsilon);
  __syncthreads();
  float inv_rms = s_inv_rms;
  for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float y = hc_load(residual_row + h);
    hc_store(normed_row + h, y * inv_rms * hc_load(scale_row + h));
  }
}

template <typename T>
int launch_post_norm(hipStream_t stream, const void *streams,
                     const void *block_output, const void *gate,
                     const void *stream_mix, const void *scale, void *residual,
                     void *normed, int64_t rows, int64_t branches,
                     int64_t hidden, int feature_gate, int gate_activation,
                     float epsilon) {
  int64_t blocks = rows * branches;
  if (blocks <= 0)
    return 0;
  if (blocks > 0x7fffffff)
    return -1;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_norm_kernel<T>),
                     dim3(static_cast<unsigned>(blocks)), dim3(256), 0, stream,
                     static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(gate),
                     static_cast<const T *>(stream_mix),
                     static_cast<const T *>(scale), static_cast<T *>(residual),
                     static_cast<T *>(normed), rows, branches, hidden,
                     feature_gate, gate_activation, epsilon);
  return 0;
}

} // namespace

extern "C" int hip_hyper_connection_pre_mix(
    void *stream, const void *streams, const void *pre_mix, void *output,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_sigmoid, float reduction_scale, int hip_dtype) {
  if (!streams || !pre_mix || !output || branches <= 0 || hidden <= 0 ||
      rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_pre<__half>(hip_stream, streams, pre_mix, output, rows, branches,
                            hidden, feature_gate, gate_sigmoid, reduction_scale);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_pre<float>(hip_stream, streams, pre_mix, output, rows, branches,
                           hidden, feature_gate, gate_sigmoid, reduction_scale);
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

extern "C" int hip_hyper_connection_post_mix_norm(
    void *stream, const void *streams, const void *block_output,
    const void *gate, const void *stream_mix, const void *scale,
    void *residual, void *normed, int64_t rows, int64_t branches,
    int64_t hidden, int feature_gate, int gate_activation, float epsilon,
    int hip_dtype) {
  if (!streams || !block_output || !gate || !scale || !residual || !normed ||
      branches <= 0 || hidden <= 0 || rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_post_norm<__half>(hip_stream, streams, block_output, gate,
                                  stream_mix, scale, residual, normed, rows,
                                  branches, hidden, feature_gate,
                                  gate_activation, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_post_norm<float>(hip_stream, streams, block_output, gate,
                                 stream_mix, scale, residual, normed, rows,
                                 branches, hidden, feature_gate, gate_activation,
                                 epsilon);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix_norm: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix_norm launch "
            "failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
