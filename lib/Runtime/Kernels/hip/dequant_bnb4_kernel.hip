/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>

// Index order matches onnxruntime blockwise_quant_block_bnb4.h. The first of
// each packed pair is the high nibble, the opposite of MatMulNBits.
__constant__ float kBnb4Fp4[16] = {
    0.0f,        5.208333333e-03f, 0.66666667f,  1.0f,
    0.33333333f, 0.5f,             0.16666667f,  0.25f,
    -0.0f,       -5.208333333e-03f, -0.66666667f, -1.0f,
    -0.33333333f, -0.5f,           -0.16666667f, -0.25f};

__constant__ float kBnb4Nf4[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

__global__ void dequant_bnb4_kernel(const uint8_t* __restrict__ packed,
                                    const float* __restrict__ absmax,
                                    float* __restrict__ output, int64_t numel,
                                    int64_t block_size, int quant_type) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;
  uint8_t byte = packed[idx >> 1];
  uint8_t code = (idx & 1) ? (byte & 0x0F) : (byte >> 4);
  float value;
  if (quant_type == 0)
    value = kBnb4Fp4[code];
  else
    value = kBnb4Nf4[code];
  output[idx] = value * absmax[idx / block_size];
}

extern "C" int hip_dequant_bnb4(void* stream, const void* packed,
                                const void* absmax, void* output, int64_t n,
                                int64_t k, int64_t block_size,
                                int64_t quant_type) {
  if (n <= 0 || k <= 0 || block_size <= 0 ||
      (quant_type != 0 && quant_type != 1) || !packed || !absmax || !output)
    return -1;
  if (n > std::numeric_limits<int64_t>::max() / k)
    return -1;
  int64_t numel = n * k;
  if (numel <= 0)
    return 0;

  (void)hipGetLastError();

  constexpr int kBlockSize = 256;
  int64_t grid64 = (numel + kBlockSize - 1) / kBlockSize;
  if (grid64 > std::numeric_limits<int>::max())
    return -1;
  hipLaunchKernelGGL(dequant_bnb4_kernel, dim3(static_cast<int>(grid64)),
                     dim3(kBlockSize), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint8_t*>(packed),
                     static_cast<const float*>(absmax),
                     static_cast<float*>(output), numel, block_size,
                     static_cast<int>(quant_type));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_dequant_bnb4 launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
