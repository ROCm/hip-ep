/* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License. */
#include "hip_qsa_gqa.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
namespace {
__global__ void cast_hidden(const __half *input, hip_bfloat16 *out, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = hip_bfloat16(__half2float(input[i]));
}
__global__ void project_decode(const __half *hidden, const hip_bfloat16 *weight,
                               hip_bfloat16 *output) {
  __shared__ float partial[256];
  float sum = 0;
  for (int k = threadIdx.x; k < 4096; k += 256)
    sum += float(hip_bfloat16(__half2float(hidden[k]))) * float(weight[blockIdx.x * 4096 + k]);
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (!threadIdx.x) output[blockIdx.x] = hip_bfloat16(partial[0]);
}
}
extern "C" int hip_qsa_cast_hidden(void *stream, const void *hidden, void *out, int S) {
  if (S < 1 || S > 65536) return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(cast_hidden, dim3((S * 4096 + 255) / 256), dim3(256), 0,
      static_cast<hipStream_t>(stream), static_cast<const __half *>(hidden),
      static_cast<hip_bfloat16 *>(out), S * 4096);
  return int(hipGetLastError());
}
extern "C" int hip_qsa_project_decode(void *stream, const void *hidden,
                                      const void *weight, void *out) {
  hipLaunchKernelGGL(project_decode, dim3(640), dim3(256), 0,
      static_cast<hipStream_t>(stream), static_cast<const __half *>(hidden),
      static_cast<const hip_bfloat16 *>(weight), static_cast<hip_bfloat16 *>(out));
  return int(hipGetLastError());
}
