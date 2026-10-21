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
constexpr int kRowsPerBlock = 8;
constexpr int kBlockSize = kLanes * kRowsPerBlock;

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

__device__ __forceinline__ float laneSum(float x) {
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    x += __shfl_xor(x, off, kLanes);
  }
  return x;
}

__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

// One wave per state row S[v, :] (dk floats held EPL per lane), kRowsPerBlock
// rows of one (batch, value head) per block. The row recurrence only couples
// to other rows through q, k and the per-head scalars, so rows are independent
// and the state stays in registers across the whole sequence. Every row is
// read before it is written by the same wave, which makes an initial_state
// that aliases final_state safe.
template <typename T, int EPL>
__global__ void __launch_bounds__(kBlockSize) gatedDeltaNetRecurrentKernel(
    const T *__restrict__ qkv, const float *__restrict__ decay,
    const float *__restrict__ beta, const float *initialState,
    const float *__restrict__ aLog, const float *__restrict__ dtBias,
    T *__restrict__ output, float *finalState, int seqLen, int numHeadsQk,
    int numHeadsV, int headSizeV, float scale) {
  constexpr int kDk = EPL * kLanes;
  const int rowGroups = (headSizeV + kRowsPerBlock - 1) / kRowsPerBlock;
  const int rowGroup = static_cast<int>(blockIdx.x) % rowGroups;
  const int hv = (static_cast<int>(blockIdx.x) / rowGroups) % numHeadsV;
  const int b = static_cast<int>(blockIdx.x) / (rowGroups * numHeadsV);
  const int hq = hv * numHeadsQk / numHeadsV;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int wave = static_cast<int>(threadIdx.x) / kLanes;
  const int v = rowGroup * kRowsPerBlock + wave;
  const bool activeRow = v < headSizeV;

  __shared__ float qs[kDk];
  __shared__ float ks[kDk];
  __shared__ float gateScale;
  __shared__ float betaScalar;

  const int64_t packed = static_cast<int64_t>(2) * numHeadsQk * kDk +
                         static_cast<int64_t>(numHeadsV) * headSizeV;
  const int64_t stateRow =
      ((static_cast<int64_t>(b) * numHeadsV + hv) * headSizeV + v) * kDk;
  const float headDecay = -expf(aLog[hv]);
  const float headBias = dtBias[hv];

  float s[EPL];
#pragma unroll
  for (int e = 0; e < EPL; ++e) {
    s[e] = activeRow ? initialState[stateRow + lane + e * kLanes] : 0.0f;
  }

  for (int t = 0; t < seqLen; ++t) {
    const int64_t tok = static_cast<int64_t>(b) * seqLen + t;
    const T *row = qkv + tok * packed;
    if (wave == 0) {
      const T *qh = row + static_cast<int64_t>(hq) * kDk;
      const T *kh = row + static_cast<int64_t>(numHeadsQk + hq) * kDk;
      float qv[EPL];
      float kv[EPL];
      float qq = 0.0f;
      float kk = 0.0f;
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        qv[e] = toF32(qh[lane + e * kLanes]);
        kv[e] = toF32(kh[lane + e * kLanes]);
        qq += qv[e] * qv[e];
        kk += kv[e] * kv[e];
      }
      const float qn = rsqrtf(laneSum(qq) + 1e-12f);
      const float kn = rsqrtf(laneSum(kk) + 1e-12f);
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        qs[lane + e * kLanes] = qv[e] * qn;
        ks[lane + e * kLanes] = kv[e] * kn;
      }
      if (lane == 0) {
        const int64_t gi = tok * numHeadsV + hv;
        const float g = headDecay * softplus(decay[gi] + headBias);
        gateScale = expf(g);
        betaScalar = 1.0f / (1.0f + expf(-beta[gi]));
      }
    }
    __syncthreads();

    if (activeRow) {
      const float a = gateScale;
      float r = 0.0f;
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        s[e] *= a;
        r += s[e] * ks[lane + e * kLanes];
      }
      r = laneSum(r);
      const float vv = toF32(
          row[static_cast<int64_t>(2) * numHeadsQk * kDk +
              static_cast<int64_t>(hv) * headSizeV + v]);
      const float delta = betaScalar * (vv - r);
      float o = 0.0f;
#pragma unroll
      for (int e = 0; e < EPL; ++e) {
        s[e] += ks[lane + e * kLanes] * delta;
        o += s[e] * qs[lane + e * kLanes];
      }
      o = laneSum(o);
      if (lane == 0) {
        output[(tok * numHeadsV + hv) * headSizeV + v] = fromF32<T>(scale * o);
      }
    }
    __syncthreads();
  }

  if (activeRow) {
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      finalState[stateRow + lane + e * kLanes] = s[e];
    }
  }
}

template <typename T, int EPL>
hipError_t launchRecurrent(hipStream_t stream, const void *qkv,
                           const float *decay, const float *beta,
                           const float *initialState, const float *aLog,
                           const float *dtBias, void *output, float *finalState,
                           int batch, int seqLen, int numHeadsQk,
                           int numHeadsV, int headSizeV, float scale) {
  const int rowGroups = (headSizeV + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 grid(static_cast<unsigned>(batch * numHeadsV * rowGroups));
  hipLaunchKernelGGL((gatedDeltaNetRecurrentKernel<T, EPL>), grid,
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
    return launchRecurrent<T, 1>(stream, qkv, decay, beta, initialState, aLog,
                                 dtBias, output, finalState, batch, seqLen,
                                 numHeadsQk, numHeadsV, headSizeV, scale);
  case 64:
    return launchRecurrent<T, 2>(stream, qkv, decay, beta, initialState, aLog,
                                 dtBias, output, finalState, batch, seqLen,
                                 numHeadsQk, numHeadsV, headSizeV, scale);
  case 128:
    return launchRecurrent<T, 4>(stream, qkv, decay, beta, initialState, aLog,
                                 dtBias, output, finalState, batch, seqLen,
                                 numHeadsQk, numHeadsV, headSizeV, scale);
  case 256:
    return launchRecurrent<T, 8>(stream, qkv, decay, beta, initialState, aLog,
                                 dtBias, output, finalState, batch, seqLen,
                                 numHeadsQk, numHeadsV, headSizeV, scale);
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
