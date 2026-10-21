// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "rounded_scaled_add.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>

namespace {
struct Shape {
  int64_t output[4], strides[3][4];
};
struct IndexTerms {
  uint32_t shift[4], mask[4], stride[4], count;
  bool flat, innerBroadcast;
};
struct FastShape { IndexTerms input[3]; };

bool powerOfTwo(uint64_t value) { return value && !(value & (value - 1)); }
unsigned log2Exact(uint64_t value) {
  unsigned shift = 0;
  while (value > 1) { value >>= 1; ++shift; }
  return shift;
}
bool buildTerms(const int64_t *input, const int64_t *output, IndexTerms &terms) {
  terms.flat = true;
  terms.innerBroadcast = input[3] == 1;
  for (int axis = 0; axis < 4; ++axis) terms.flat &= input[axis] == output[axis];
  if (terms.flat) return true;
  uint64_t suffix = 1, stride = 1;
  for (int axis = 3; axis >= 0; --axis) {
    if (input[axis] != 1) {
      if (!powerOfTwo(suffix) || !powerOfTwo(output[axis])) return false;
      unsigned n = terms.count++;
      terms.shift[n] = log2Exact(suffix);
      terms.mask[n] = static_cast<uint32_t>(output[axis] - 1);
      terms.stride[n] = static_cast<uint32_t>(stride);
    }
    suffix *= output[axis];
    stride *= input[axis];
  }
  return true;
}
__device__ __forceinline__ float4 loadFour(const float *input, uint32_t i,
                                          const IndexTerms &terms) {
  uint32_t offset = terms.flat ? i : 0;
  if (!terms.flat) {
#pragma unroll
    for (unsigned term = 0; term < 4; ++term)
      if (term < terms.count)
        offset += ((i >> terms.shift[term]) & terms.mask[term]) * terms.stride[term];
  }
  if (terms.innerBroadcast) {
    float v = input[offset];
    return make_float4(v, v, v, v);
  }
  return *reinterpret_cast<const float4 *>(input + offset);
}
__device__ __forceinline__ __half roundedAdd(float base, float x, float y,
                                            bool productOnLhs) {
#pragma clang fp contract(off)
  float product = __fmul_rn(x, y);
  // Keep the FP32 product before conversion. A mixed instruction can skip it.
  asm volatile("" : "+v"(product));
  __half a = __float2half(base), b = __float2half(product);
  return productOnLhs ? __hadd(b, a) : __hadd(a, b);
}
__global__ void roundedScaledAddFast(const float *base, const float *x,
                                     const float *y, __half *output,
                                     uint32_t vectors, FastShape shape,
                                     bool productOnLhs) {
  uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread >= vectors) return;
  uint32_t i = thread * 4;
  float4 b = loadFour(base, i, shape.input[0]);
  float4 xv = loadFour(x, i, shape.input[1]);
  float4 yv = loadFour(y, i, shape.input[2]);
  output[i] = roundedAdd(b.x, xv.x, yv.x, productOnLhs);
  output[i+1] = roundedAdd(b.y, xv.y, yv.y, productOnLhs);
  output[i+2] = roundedAdd(b.z, xv.z, yv.z, productOnLhs);
  output[i+3] = roundedAdd(b.w, xv.w, yv.w, productOnLhs);
}
__global__ void roundedScaledAdd(const float *base, const float *x,
                                 const float *y, __half *output,
                                 int64_t count, Shape shape, bool productOnLhs) {
#pragma clang fp contract(off)
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int64_t offsets[3] = {}, rest = i;
#pragma unroll
  for (int axis = 3; axis >= 0; --axis) {
    int64_t coordinate = rest % shape.output[axis];
    rest /= shape.output[axis];
#pragma unroll
    for (int input = 0; input < 3; ++input)
      offsets[input] += coordinate * shape.strides[input][axis];
  }
  output[i] = roundedAdd(base[offsets[0]], x[offsets[1]], y[offsets[2]], productOnLhs);
}
}

extern "C" HIP_KERNEL_API int hip_rounded_scaled_add_f16(
    void *stream, const void *base, const void *x, const void *y, void *output,
    const int64_t *baseShape, const int64_t *xShape, const int64_t *yShape,
    const int64_t *outShape, int64_t productOnLhs) {
  if (!baseShape || !xShape || !yShape || !outShape ||
      (productOnLhs != 0 && productOnLhs != 1)) return hipErrorInvalidValue;
  bool empty = false;
  for (int axis = 0; axis < 4; ++axis) {
    int64_t b = baseShape[axis], x = xShape[axis], y = yShape[axis], o = outShape[axis];
    if (b < 0 || x < 0 || y < 0 || o < 0 || (x != y && x != 1 && y != 1))
      return hipErrorInvalidValue;
    int64_t product = x == 1 ? y : x;
    if ((b != product && b != 1 && product != 1) || o != (b == 1 ? product : b))
      return hipErrorInvalidValue;
    empty |= o == 0;
  }
  if (empty) return hipSuccess;
  if (!base || !x || !y || !output || output == base || output == x || output == y)
    return hipErrorInvalidValue;
  Shape shape{};
  int64_t count = 1, strides[3] = {1, 1, 1};
  const int64_t *inputs[] = {baseShape, xShape, yShape};
  for (int axis = 3; axis >= 0; --axis) {
    if (count > std::numeric_limits<int64_t>::max() / outShape[axis])
      return hipErrorInvalidValue;
    shape.output[axis] = outShape[axis];
    count *= outShape[axis];
    for (int input = 0; input < 3; ++input) {
      shape.strides[input][axis] = inputs[input][axis] == 1 ? 0 : strides[input];
      strides[input] *= inputs[input][axis];
    }
  }
  uint64_t blocks = (static_cast<uint64_t>(count) + 255) / 256;
  if (blocks > std::numeric_limits<uint32_t>::max()) return hipErrorInvalidValue;
  // Vector loads need whole rows and aligned, contiguous inner dimensions.
  // Terms replace division only when each contributing coordinate is exact.
  FastShape fast{};
  const void *pointers[] = {base, x, y};
  bool useFast = count <= std::numeric_limits<uint32_t>::max() &&
                 outShape[3] % 4 == 0 && reinterpret_cast<uintptr_t>(output) % 8 == 0;
  for (int input = 0; input < 3 && useFast; ++input)
    useFast = (inputs[input][3] == 1 || reinterpret_cast<uintptr_t>(pointers[input]) % 16 == 0) &&
               buildTerms(inputs[input], outShape, fast.input[input]);
  if (useFast) {
    uint32_t vectors = static_cast<uint32_t>(count / 4);
    hipLaunchKernelGGL(roundedScaledAddFast, dim3((vectors + 255) / 256), dim3(256),
        0, static_cast<hipStream_t>(stream), static_cast<const float *>(base),
        static_cast<const float *>(x), static_cast<const float *>(y),
        static_cast<__half *>(output), vectors, fast, bool(productOnLhs));
    return hipGetLastError();
  }
  hipLaunchKernelGGL(roundedScaledAdd, dim3(static_cast<uint32_t>(blocks)), dim3(256),
      0, static_cast<hipStream_t>(stream), static_cast<const float *>(base),
      static_cast<const float *>(x), static_cast<const float *>(y),
      static_cast<__half *>(output), count, shape, bool(productOnLhs));
  return hipGetLastError();
}
