/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Producer-side int8 activation quantization for the dp4a GEMVs: a kernel
 * that writes an fp16 activation also writes it quantized exactly as
 * hip_matmul_nbits_quant_act_i8 would, saving that launch.
 */
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// Quantizes elements 2p and 2p+1 of a row. A 32-element group is scaled by
// max|x| / 127 and its 8-element chunks store the even elements first. The
// 16 lanes holding pairs 16j .. 16j+15 must call this together, from 16
// consecutive lanes of one wave starting at a multiple of 16.
__device__ __forceinline__ void quant_act_i8_pair(float2 y, int64_t p,
                                                  int8_t *__restrict__ q_row,
                                                  float *__restrict__ scale_row) {
  float m = fmaxf(fabsf(y.x), fabsf(y.y));
  for (int off = 8; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 16));
  const float inv_scale = (m > 0.0f) ? (127.0f / m) : 0.0f;
  const int q0 = max(-127, min(127, __float2int_rn(y.x * inv_scale)));
  const int q1 = max(-127, min(127, __float2int_rn(y.y * inv_scale)));
  const int64_t base = (p >> 2) * 8 + (p & 3);
  q_row[base] = static_cast<int8_t>(q0);
  q_row[base + 4] = static_cast<int8_t>(q1);
  if ((p & 15) == 0)
    scale_row[p >> 4] = (m > 0.0f) ? (m * (1.0f / 127.0f)) : 1.0f;
}
