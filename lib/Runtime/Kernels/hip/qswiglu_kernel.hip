/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Fused per-tensor Q/DQ chains on f32 activations.
//
// Each helper repeats the arithmetic of the standalone kernel it stands in for
// (quantize_one and dequantize_linear_kernel in qdq_kernel.hip,
// elementwise_sigmoid_f32_kernel, qelementwise_nd_kernel), operation for
// operation, so the fused results are bit-identical to the unfused chain.
// Changing one side without the other breaks that equivalence.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int kBlockSize = 256;

__device__ inline float qs_quantize(float x, float scale, int32_t zp, float lo,
                                    float hi) {
  float q = rintf(x / scale) + static_cast<float>(zp);
  q = fmaxf(q, lo);
  q = fminf(q, hi);
  return static_cast<float>(static_cast<int32_t>(q));
}

__device__ inline float qs_dequantize(float code, int32_t zp, float scale) {
  int64_t centered = static_cast<int64_t>(static_cast<int32_t>(code)) -
                     static_cast<int64_t>(zp);
  return static_cast<float>(centered) * scale;
}

__device__ inline float qs_qmul(float a, int32_t za, float b, int32_t zb,
                                float m, int32_t zo) {
  float da = a - static_cast<float>(za);
  float db = b - static_cast<float>(zb);
  float acc = m * (da * db);
  float v = rintf(acc) + static_cast<float>(zo);
  return static_cast<float>(
      static_cast<uint16_t>(fminf(65535.0f, fmaxf(0.0f, v))));
}

struct QSwigluParams {
  float gate_scale;
  int32_t gate_zp;
  float sigmoid_scale;
  int32_t sigmoid_zp;
  float m_silu;
  int32_t silu_zp;
  float up_scale;
  int32_t up_zp;
  float m_mul;
  int32_t mul_zp;
  float dq_scale;
  int32_t dq_zp;
};

__device__ inline float qswiglu_one(float g, float u, const QSwigluParams &p) {
  float qg = qs_quantize(g, p.gate_scale, p.gate_zp, 0.0f, 65535.0f);
  float dg = qs_dequantize(qg, p.gate_zp, p.gate_scale);
  float sig = 1.0f / (1.0f + expf(-dg));
  float qs = qs_quantize(sig, p.sigmoid_scale, p.sigmoid_zp, 0.0f, 65535.0f);
  float qa = qs_qmul(qg, p.gate_zp, qs, p.sigmoid_zp, p.m_silu, p.silu_zp);
  float qu = qs_quantize(u, p.up_scale, p.up_zp, 0.0f, 65535.0f);
  float qo = qs_qmul(qa, p.silu_zp, qu, p.up_zp, p.m_mul, p.mul_zp);
  return qs_dequantize(qo, p.dq_zp, p.dq_scale);
}

__global__ void qswiglu_vec4_kernel(const float4 *__restrict__ gate,
                                    const float4 *__restrict__ up,
                                    float4 *__restrict__ output,
                                    int64_t num_vec, QSwigluParams p) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_vec)
    return;
  float4 g = gate[i];
  float4 u = up[i];
  float4 y;
  y.x = qswiglu_one(g.x, u.x, p);
  y.y = qswiglu_one(g.y, u.y, p);
  y.z = qswiglu_one(g.z, u.z, p);
  y.w = qswiglu_one(g.w, u.w, p);
  output[i] = y;
}

__global__ void qswiglu_scalar_kernel(const float *__restrict__ gate,
                                      const float *__restrict__ up,
                                      float *__restrict__ output,
                                      int64_t num_elements, QSwigluParams p) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_elements)
    output[i] = qswiglu_one(gate[i], up[i], p);
}

__device__ inline float fake_quantize_one(float x, float scale, int32_t zp,
                                          float lo, float hi) {
  return qs_dequantize(qs_quantize(x, scale, zp, lo, hi), zp, scale);
}

__global__ void fake_quantize_vec4_kernel(const float4 *__restrict__ input,
                                          float4 *__restrict__ output,
                                          int64_t num_vec, float scale,
                                          int32_t zp, float lo, float hi) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_vec)
    return;
  float4 x = input[i];
  float4 y;
  y.x = fake_quantize_one(x.x, scale, zp, lo, hi);
  y.y = fake_quantize_one(x.y, scale, zp, lo, hi);
  y.z = fake_quantize_one(x.z, scale, zp, lo, hi);
  y.w = fake_quantize_one(x.w, scale, zp, lo, hi);
  output[i] = y;
}

__global__ void fake_quantize_scalar_kernel(const float *__restrict__ input,
                                            float *__restrict__ output,
                                            int64_t num_elements, float scale,
                                            int32_t zp, float lo, float hi) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_elements)
    output[i] = fake_quantize_one(input[i], scale, zp, lo, hi);
}

bool isVec4Aligned(const void *a, const void *b, const void *c,
                   int64_t num_elements) {
  auto aligned = [](const void *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
  };
  return num_elements % 4 == 0 && aligned(a) && aligned(b) && aligned(c);
}

unsigned gridFor(int64_t work) {
  return static_cast<unsigned>((work + kBlockSize - 1) / kBlockSize);
}

int finishLaunch(const char *name) {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] %s launch failed: %s\n", name,
            hipGetErrorString(err));
  return static_cast<int>(err);
}

} // namespace

extern "C" int hip_qswiglu(void *stream, const float *gate, const float *up,
                           float *output, int64_t num_elements,
                           float gate_scale, int32_t gate_zp,
                           float sigmoid_scale, int32_t sigmoid_zp,
                           float m_silu, int32_t silu_zp, float up_scale,
                           int32_t up_zp, float m_mul, int32_t mul_zp,
                           float dq_scale, int32_t dq_zp) {
  if (!gate || !up || !output) {
    fprintf(stderr, "[custom_kernels] hip_qswiglu: null argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  QSwigluParams p{gate_scale, gate_zp, sigmoid_scale, sigmoid_zp,
                  m_silu,     silu_zp, up_scale,      up_zp,
                  m_mul,      mul_zp,  dq_scale,      dq_zp};
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  if (isVec4Aligned(gate, up, output, num_elements)) {
    int64_t num_vec = num_elements / 4;
    hipLaunchKernelGGL(qswiglu_vec4_kernel, dim3(gridFor(num_vec)),
                       dim3(kBlockSize), 0, s,
                       reinterpret_cast<const float4 *>(gate),
                       reinterpret_cast<const float4 *>(up),
                       reinterpret_cast<float4 *>(output), num_vec, p);
  } else {
    hipLaunchKernelGGL(qswiglu_scalar_kernel, dim3(gridFor(num_elements)),
                       dim3(kBlockSize), 0, s, gate, up, output, num_elements,
                       p);
  }
  return finishLaunch("hip_qswiglu");
}

extern "C" int hip_fake_quantize(void *stream, const float *input,
                                 float *output, int64_t num_elements,
                                 float scale, int32_t zero_point, int32_t qmin,
                                 int32_t qmax) {
  if (!input || !output || qmin > qmax) {
    fprintf(stderr, "[custom_kernels] hip_fake_quantize: invalid argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  float lo = static_cast<float>(qmin);
  float hi = static_cast<float>(qmax);
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  if (isVec4Aligned(input, output, output, num_elements)) {
    int64_t num_vec = num_elements / 4;
    hipLaunchKernelGGL(
        fake_quantize_vec4_kernel, dim3(gridFor(num_vec)), dim3(kBlockSize), 0,
        s, reinterpret_cast<const float4 *>(input),
        reinterpret_cast<float4 *>(output), num_vec, scale, zero_point, lo, hi);
  } else {
    hipLaunchKernelGGL(fake_quantize_scalar_kernel, dim3(gridFor(num_elements)),
                       dim3(kBlockSize), 0, s, input, output, num_elements,
                       scale, zero_point, lo, hi);
  }
  return finishLaunch("hip_fake_quantize");
}
