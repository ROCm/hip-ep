// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "swiglu_f32.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <type_traits>

namespace {
struct BroadcastShape {
  int64_t output[4], gateStride[4], upStride[4];
};

template <bool Broadcast>
__global__ void swigluF32Kernel(const __half *gate, const __half *up,
                                __half *output, int64_t count,
                                BroadcastShape shape) {
#pragma clang fp contract(off)
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int64_t gi = i, ui = i;
  if constexpr (Broadcast) {
    gi = ui = 0;
    int64_t remaining = i;
#pragma unroll
    for (int axis = 3; axis >= 0; --axis) {
      int64_t coordinate = remaining % shape.output[axis];
      remaining /= shape.output[axis];
      gi += coordinate * shape.gateStride[axis];
      ui += coordinate * shape.upStride[axis];
    }
  }
  float g = __half2float(gate[gi]);
  float sigmoid = 1.0f / (1.0f + expf(-g));
  float activated = __fmul_rn(g, sigmoid);
  float product = __fmul_rn(activated, __half2float(up[ui]));
  // Keep the FP32 product boundary. A mixed multiply-to-FP16 instruction
  // skips this rounding and changes FP16 midpoint cases.
  asm volatile("" : "+v"(product));
  output[i] = __float2half(product);
}
} // namespace

extern "C" HIP_KERNEL_API int
hip_swiglu_f16_f32_bcast(void *stream, const void *gate, const void *up,
                         void *output, const int64_t *gateShape,
                         const int64_t *upShape, const int64_t *outShape) {
  if (!gateShape || !upShape || !outShape)
    return hipErrorInvalidValue;
  bool empty = false, gateBroadcast = false, upBroadcast = false;
  for (int axis = 0; axis < 4; ++axis) {
    int64_t g = gateShape[axis], u = upShape[axis], o = outShape[axis];
    if (g < 0 || u < 0 || o < 0 || (g != u && g != 1 && u != 1) ||
        o != (g == 1 ? u : g))
      return hipErrorInvalidValue;
    empty |= o == 0;
    gateBroadcast |= g != o;
    upBroadcast |= u != o;
  }
  if (empty)
    return hipSuccess;
  if (!gate || !up || !output || (output == gate && gateBroadcast) ||
      (output == up && upBroadcast))
    return hipErrorInvalidValue;
  BroadcastShape shape{};
  int64_t count = 1, gs = 1, us = 1;
  for (int axis = 3; axis >= 0; --axis) {
    if (count > std::numeric_limits<int64_t>::max() / outShape[axis])
      return hipErrorInvalidValue;
    shape.output[axis] = outShape[axis];
    shape.gateStride[axis] = gateShape[axis] == 1 ? 0 : gs;
    shape.upStride[axis] = upShape[axis] == 1 ? 0 : us;
    count *= outShape[axis];
    gs *= gateShape[axis];
    us *= upShape[axis];
  }
  uint64_t blocks = (static_cast<uint64_t>(count) + 255) / 256;
  if (blocks > std::numeric_limits<uint32_t>::max())
    return hipErrorInvalidValue;
  auto launch = [&](auto broadcast) {
    hipLaunchKernelGGL(swigluF32Kernel<decltype(broadcast)::value>,
                       dim3(static_cast<uint32_t>(blocks)), dim3(256), 0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const __half *>(gate),
                       static_cast<const __half *>(up),
                       static_cast<__half *>(output), count, shape);
  };
  if (gateBroadcast || upBroadcast)
    launch(std::true_type{});
  else
    launch(std::false_type{});
  return hipGetLastError();
}
