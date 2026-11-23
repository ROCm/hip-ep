/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_KQUANT_SWIGLU_DEVICE_H
#define HIP_KQUANT_SWIGLU_DEVICE_H

#include <hip/hip_runtime.h>

// GGML's SwiGLU: gate * sigmoid(gate) * up.
__device__ __forceinline__ float kquant_swiglu_value(float gate, float up) {
  const float sigmoid = 1.0f / (1.0f + expf(-gate));
  return gate * sigmoid * up;
}

#endif // HIP_KQUANT_SWIGLU_DEVICE_H
