/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxStateLen = 32;

__device__ float load_act(__half v) { return __half2float(v); }
__device__ float load_act(float v) { return v; }
__device__ __half store_act(__half /*tag*/, float v) { return __float2half(v); }
__device__ float store_act(float /*tag*/, float v) { return v; }

template <typename T>
__global__ void varlen_causal_conv_kernel(
    const T *__restrict__ input, const T *__restrict__ weight,
    const int32_t *__restrict__ cumulative, const T *__restrict__ bias,
    const T *__restrict__ initial_state, T *__restrict__ output,
    T *__restrict__ present, int total_tokens, int channels, int batch,
    int kernel_size, int dilation, int state_len, int activation) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (c >= channels || b >= batch)
    return;
  if (cumulative[0] != 0 || cumulative[batch] != total_tokens)
    return;

  int start = cumulative[b];
  int end = cumulative[b + 1];
  if (start < 0 || end < start || end > total_tokens || end == start)
    return;

  float window[kMaxStateLen];
  const T *past =
      initial_state + (static_cast<int64_t>(b) * channels + c) * state_len;
  for (int i = 0; i < state_len; ++i)
    window[i] = load_act(past[i]);

  const T *taps = weight + static_cast<int64_t>(c) * kernel_size;
  float bias_v = bias ? load_act(bias[c]) : 0.f;
  int seq = end - start;

  for (int t = 0; t < seq; ++t) {
    float acc = bias_v;
    for (int j = 0; j < kernel_size; ++j) {
      int pos = t - (kernel_size - 1 - j) * dilation;
      float sample;
      if (pos >= 0) {
        sample = load_act(
            input[(static_cast<int64_t>(start + pos) * channels) + c]);
      } else {
        sample = window[state_len + pos - t];
      }
      acc += sample * load_act(taps[j]);
    }
    if (activation)
      acc = acc / (1.f + expf(-acc));
    output[(static_cast<int64_t>(start + t) * channels) + c] =
        store_act(T{}, acc);
    if (state_len > 0) {
      for (int i = 0; i < state_len - 1; ++i)
        window[i] = window[i + 1];
      window[state_len - 1] = load_act(
          input[(static_cast<int64_t>(start + t) * channels) + c]);
    }
  }

  T *next = present + (static_cast<int64_t>(b) * channels + c) * state_len;
  for (int i = 0; i < state_len; ++i)
    next[i] = store_act(T{}, window[i]);
}

template <typename T>
int launch_varlen(hipStream_t stream, const void *input, const void *weight,
                  const void *cumulative, const void *bias,
                  const void *initial_state, void *output, void *present,
                  int total_tokens, int channels, int batch, int kernel_size,
                  int dilation, int state_len, int activation) {
  constexpr int kThreads = 256;
  int blocks = (channels + kThreads - 1) / kThreads;
  (void)hipGetLastError();
  hipLaunchKernelGGL(varlen_causal_conv_kernel<T>, dim3(blocks, batch),
                     dim3(kThreads), 0, stream, static_cast<const T *>(input),
                     static_cast<const T *>(weight),
                     static_cast<const int32_t *>(cumulative),
                     static_cast<const T *>(bias),
                     static_cast<const T *>(initial_state),
                     static_cast<T *>(output), static_cast<T *>(present),
                     total_tokens, channels, batch, kernel_size, dilation,
                     state_len, activation);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_varlen_causal_conv launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

bool fits_int(int64_t value) {
  return value >= 0 && value <= static_cast<int64_t>(INT32_MAX);
}

} // namespace

extern "C" int hip_varlen_causal_conv(
    void *stream_ptr, const void *input, const void *weight,
    const void *cumulative_sequence_length, const void *bias,
    const void *initial_state, void *output, void *present_state,
    int64_t total_tokens, int64_t channels, int64_t batch, int64_t kernel_size,
    int64_t dilation, int64_t state_len, int64_t activation,
    int64_t element_size_bytes) {
  if (!fits_int(total_tokens) || !fits_int(channels) || !fits_int(batch) ||
      !fits_int(kernel_size) || !fits_int(dilation) || !fits_int(state_len))
    return -1;
  auto stream = static_cast<hipStream_t>(stream_ptr);
  if (element_size_bytes == 2) {
    return launch_varlen<__half>(
        stream, input, weight, cumulative_sequence_length, bias, initial_state,
        output, present_state, static_cast<int>(total_tokens),
        static_cast<int>(channels), static_cast<int>(batch),
        static_cast<int>(kernel_size), static_cast<int>(dilation),
        static_cast<int>(state_len), static_cast<int>(activation));
  }
  if (element_size_bytes == 4) {
    return launch_varlen<float>(
        stream, input, weight, cumulative_sequence_length, bias, initial_state,
        output, present_state, static_cast<int>(total_tokens),
        static_cast<int>(channels), static_cast<int>(batch),
        static_cast<int>(kernel_size), static_cast<int>(dilation),
        static_cast<int>(state_len), static_cast<int>(activation));
  }
  return -1;
}
