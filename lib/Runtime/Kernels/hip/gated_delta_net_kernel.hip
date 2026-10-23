/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft GatedDeltaNet: packed QKV, qwen gate, sigmoid beta, L2-normed
 * q/k, gated_delta update rule, fp32 V-major state.
 */

#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kLanes = 32;
constexpr int kBlockSize = 256;

template <typename T> __device__ __forceinline__ float toF32(T x);
template <> __device__ __forceinline__ float toF32(float x) { return x; }
template <> __device__ __forceinline__ float toF32(__half x) {
  return __half2float(x);
}

template <typename T> __device__ __forceinline__ T fromF32(float x);
template <> __device__ __forceinline__ float fromF32(float x) { return x; }
template <> __device__ __forceinline__ __half fromF32(float x) {
  return __float2half(x);
}

template <int WIDTH> __device__ __forceinline__ float groupSum(float x) {
  for (int off = WIDTH / 2; off > 0; off >>= 1) {
    x += __shfl_xor(x, off, WIDTH);
  }
  return x;
}

__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

// TPR consecutive threads own one state row S[v, :] of one (batch, value
// head), E = DK / TPR contiguous floats each, held in registers for the whole
// sequence; the kernel is sequential in t. Keeping a row in at most a few
// lanes is what makes a token step cheap: the two row dot products reduce over
// TPR lanes instead of a full wave. Each row is read before it is written by
// the threads that own it, so an initial_state aliasing final_state is safe.
template <typename T, int DK>
__global__ void __launch_bounds__(kBlockSize) gatedDeltaNetRecurrentKernel(
    const T *__restrict__ qkv, const float *__restrict__ decay,
    const float *__restrict__ beta, const float *initialState,
    const float *__restrict__ aLog, const float *__restrict__ dtBias,
    T *__restrict__ output, float *finalState, int seqLen, int numHeadsQk,
    int numHeadsV, int headSizeV, float scale) {
  constexpr int kTpr = DK >= 64 ? DK / 64 : 1;
  constexpr int kE = DK / kTpr;
  constexpr int kRowsPerBlock = kBlockSize / kTpr;
  constexpr int kQkPerLane = DK / kLanes;
  const int rowGroups = (headSizeV + kRowsPerBlock - 1) / kRowsPerBlock;
  const int rowGroup = static_cast<int>(blockIdx.x) % rowGroups;
  const int hv = (static_cast<int>(blockIdx.x) / rowGroups) % numHeadsV;
  const int b = static_cast<int>(blockIdx.x) / (rowGroups * numHeadsV);
  const int hq = hv * numHeadsQk / numHeadsV;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kLanes;
  const bool leadWave = tid < kLanes;
  const int part = tid % kTpr;
  const int v = rowGroup * kRowsPerBlock + tid / kTpr;
  const bool active = v < headSizeV;

  __shared__ float qs[DK];
  __shared__ float ks[DK];
  __shared__ float gateScale;
  __shared__ float betaScalar;

  const int64_t packed = static_cast<int64_t>(2) * numHeadsQk * DK +
                         static_cast<int64_t>(numHeadsV) * headSizeV;
  const int64_t vOffset = static_cast<int64_t>(2) * numHeadsQk * DK +
                          static_cast<int64_t>(hv) * headSizeV;
  const int64_t rowState =
      ((static_cast<int64_t>(b) * numHeadsV + hv) * headSizeV + v) * DK +
      part * kE;
  const float headDecay = -expf(aLog[hv]);
  const float headBias = dtBias[hv];

  float s[kE];
#pragma unroll
  for (int i = 0; i < kE; ++i) {
    s[i] = active ? initialState[rowState + i] : 0.0f;
  }

  // The step is latency-bound, so token t+1's inputs are fetched while token
  // t is being computed.
  float qv[kQkPerLane];
  float kv[kQkPerLane];
  float rawDecay = 0.0f;
  float rawBeta = 0.0f;
  float vNext = 0.0f;
  auto load = [&](int t) {
    const int64_t tok = static_cast<int64_t>(b) * seqLen + t;
    const T *row = qkv + tok * packed;
    if (leadWave) {
      const T *qh = row + static_cast<int64_t>(hq) * DK;
      const T *kh = row + static_cast<int64_t>(numHeadsQk + hq) * DK;
#pragma unroll
      for (int e = 0; e < kQkPerLane; ++e) {
        qv[e] = toF32(qh[lane + e * kLanes]);
        kv[e] = toF32(kh[lane + e * kLanes]);
      }
      if (lane == 0) {
        rawDecay = decay[tok * numHeadsV + hv];
        rawBeta = beta[tok * numHeadsV + hv];
      }
    }
    vNext = active ? toF32(row[vOffset + v]) : 0.0f;
  };
  load(0);

  for (int t = 0; t < seqLen; ++t) {
    if (leadWave) {
      float qq = 0.0f;
      float kk = 0.0f;
#pragma unroll
      for (int e = 0; e < kQkPerLane; ++e) {
        qq += qv[e] * qv[e];
        kk += kv[e] * kv[e];
      }
      const float qn = rsqrtf(groupSum<kLanes>(qq) + 1e-12f);
      const float kn = rsqrtf(groupSum<kLanes>(kk) + 1e-12f);
#pragma unroll
      for (int e = 0; e < kQkPerLane; ++e) {
        qs[lane + e * kLanes] = qv[e] * qn;
        ks[lane + e * kLanes] = kv[e] * kn;
      }
      if (lane == 0) {
        gateScale = expf(headDecay * softplus(rawDecay + headBias));
        betaScalar = 1.0f / (1.0f + expf(-rawBeta));
      }
    }
    const float vCur = vNext;
    __syncthreads();
    if (t + 1 < seqLen) {
      load(t + 1);
    }

    const float a = gateScale;
    const float* kRow = ks + part * kE;
    const float* qRow = qs + part * kE;
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      s[i] *= a;
      r += s[i] * kRow[i];
    }
    r = groupSum<kTpr>(r);
    const float delta = betaScalar * (vCur - r);
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      s[i] += kRow[i] * delta;
      o += s[i] * qRow[i];
    }
    o = groupSum<kTpr>(o);
    if (active && part == 0) {
      const int64_t tok = static_cast<int64_t>(b) * seqLen + t;
      output[(tok * numHeadsV + hv) * headSizeV + v] = fromF32<T>(scale * o);
    }
    __syncthreads();
  }

  if (active) {
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      finalState[rowState + i] = s[i];
    }
  }
}

template <typename T, int DK>
hipError_t launchRecurrent(hipStream_t stream, const void *qkv,
                           const float *decay, const float *beta,
                           const float *initialState, const float *aLog,
                           const float *dtBias, void *output, float *finalState,
                           int batch, int seqLen, int numHeadsQk,
                           int numHeadsV, int headSizeV, float scale) {
  constexpr int kTpr = DK >= 64 ? DK / 64 : 1;
  constexpr int kRowsPerBlock = kBlockSize / kTpr;
  const int rowGroups = (headSizeV + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 grid(static_cast<unsigned>(batch * numHeadsV * rowGroups));
  hipLaunchKernelGGL((gatedDeltaNetRecurrentKernel<T, DK>), grid,
                     dim3(kBlockSize), 0, stream, static_cast<const T *>(qkv),
                     decay, beta, initialState, aLog, dtBias,
                     static_cast<T *>(output), finalState, seqLen, numHeadsQk,
                     numHeadsV, headSizeV, scale);
  return hipGetLastError();
}

template <typename T>
hipError_t dispatchHeadSize(hipStream_t stream, int64_t headSizeQk,
                            const void *qkv, const float *decay,
                            const float *beta, const float *initialState,
                            const float *aLog, const float *dtBias,
                            void *output, float *finalState, int batch,
                            int seqLen, int numHeadsQk, int numHeadsV,
                            int headSizeV, float scale) {
  switch (headSizeQk) {
  case 32:
    return launchRecurrent<T, 32>(stream, qkv, decay, beta, initialState,
                                  aLog, dtBias, output, finalState, batch,
                                  seqLen, numHeadsQk, numHeadsV, headSizeV,
                                  scale);
  case 64:
    return launchRecurrent<T, 64>(stream, qkv, decay, beta, initialState,
                                  aLog, dtBias, output, finalState, batch,
                                  seqLen, numHeadsQk, numHeadsV, headSizeV,
                                  scale);
  case 128:
    return launchRecurrent<T, 128>(stream, qkv, decay, beta, initialState,
                                   aLog, dtBias, output, finalState, batch,
                                   seqLen, numHeadsQk, numHeadsV, headSizeV,
                                   scale);
  case 256:
    return launchRecurrent<T, 256>(stream, qkv, decay, beta, initialState,
                                   aLog, dtBias, output, finalState, batch,
                                   seqLen, numHeadsQk, numHeadsV, headSizeV,
                                   scale);
  default:
    return hipErrorInvalidValue;
  }
}

} // namespace

extern "C" int hip_gated_delta_net(
    void *stream, const void *qkv, const float *decay, const float *beta,
    const float *initial_state, const float *a_log, const float *dt_bias,
    void *output, float *final_state, int64_t batch, int64_t seq_len,
    int64_t num_heads_qk, int64_t num_heads_v, int64_t head_size_qk,
    int64_t head_size_v, float scale, int element_size_bytes) {
  if (!qkv || !decay || !beta || !initial_state || !a_log || !dt_bias ||
      !output || !final_state) {
    fprintf(stderr, "[custom_kernels] hip_gated_delta_net: null pointer\n");
    return -1;
  }
  if (batch <= 0 || seq_len <= 0 || num_heads_qk <= 0 || num_heads_v <= 0 ||
      head_size_v <= 0 || num_heads_v % num_heads_qk != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gated_delta_net: invalid shape batch=%lld "
            "seq_len=%lld Hqk=%lld Hv=%lld dv=%lld\n",
            (long long)batch, (long long)seq_len, (long long)num_heads_qk,
            (long long)num_heads_v, (long long)head_size_v);
    return -1;
  }
  if (head_size_qk != 32 && head_size_qk != 64 && head_size_qk != 128 &&
      head_size_qk != 256) {
    fprintf(stderr,
            "[custom_kernels] hip_gated_delta_net: head_size_qk=%lld is not "
            "one of 32, 64, 128, 256\n",
            (long long)head_size_qk);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  const int B = static_cast<int>(batch);
  const int L = static_cast<int>(seq_len);
  const int Hqk = static_cast<int>(num_heads_qk);
  const int Hv = static_cast<int>(num_heads_v);
  const int Dv = static_cast<int>(head_size_v);
  hipError_t err;
  if (element_size_bytes == 2) {
    err = dispatchHeadSize<__half>(s, head_size_qk, qkv, decay, beta,
                                   initial_state, a_log, dt_bias, output,
                                   final_state, B, L, Hqk, Hv, Dv, scale);
  } else if (element_size_bytes == 4) {
    err = dispatchHeadSize<float>(s, head_size_qk, qkv, decay, beta,
                                  initial_state, a_log, dt_bias, output,
                                  final_state, B, L, Hqk, Hv, Dv, scale);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_gated_delta_net: unsupported "
            "element_size_bytes=%d (expect 2 or 4)\n",
            element_size_bytes);
    return -1;
  }
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_gated_delta_net: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
