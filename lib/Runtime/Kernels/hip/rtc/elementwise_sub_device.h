/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Device half of elementwise_sub, split out so the AOT build and hipRTC compile
// the same text; if the two diverge their code objects stop being comparable.
//
// hipRTC injects its own preamble, so <hip/*.h> must not be included here
// (__half and friends are available without them) and neither may host-only
// headers. <cstdint> on the other hand is required: without it int64_t resolves
// only inside __hip_internal and the translation unit fails to compile.

#ifndef HIPDNN_EP_RTC_ELEMENTWISE_SUB_DEVICE_H
#define HIPDNN_EP_RTC_ELEMENTWISE_SUB_DEVICE_H

#if !defined(__HIPCC_RTC__)
  #include <hip/hip_fp16.h>
  #include <hip/hip_runtime.h>
#endif

#include <cstdint>

__global__ void elementwise_sub_i64_kernel(
    const int64_t* __restrict__ lhs,
    const int64_t* __restrict__ rhs,
    int64_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_i32_kernel(
    const int32_t* __restrict__ lhs,
    const int32_t* __restrict__ rhs,
    int32_t* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_f32_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = lhs[idx] - rhs[idx];
  }
}

__global__ void elementwise_sub_f16_kernel(
    const __half* __restrict__ lhs,
    const __half* __restrict__ rhs,
    __half* __restrict__ output,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(__half2float(lhs[idx]) - __half2float(rhs[idx]));
  }
}

#endif  // HIPDNN_EP_RTC_ELEMENTWISE_SUB_DEVICE_H
