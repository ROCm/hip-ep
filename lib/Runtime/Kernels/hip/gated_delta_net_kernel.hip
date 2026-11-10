/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft GatedDeltaNet: packed QKV, qwen gate, sigmoid beta, L2-normed
 * q/k, gated_delta update rule, fp32 V-major state.
 */

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>

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

typedef float gdnFloat8 __attribute__((ext_vector_type(8)));
#if HIPDNN_HAS_WMMA
typedef _Float16 gdnHalfFrag
    __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
#endif
typedef _Float16 gdnHalf4 __attribute__((ext_vector_type(4)));

constexpr int kChunk = 32;
constexpr int kSliceV = 32;
constexpr int kChunkDk = 128;
constexpr int kWt = 16;
constexpr int kRawStride = kChunkDk + 8;
constexpr int kTileStride = kChunk + 4;
// Largest magnitude a split operand may take: its fp16 hi part must stay
// finite.
constexpr float kSplitSafeMax = 32768.0f;
// Shorter sequences run faster on the recurrent kernel.
constexpr int kChunkedMinSeqLen = 16;

__device__ __forceinline__ int gdnAccRow(int pr, int e) {
#if HIPDNN_HAS_WMMA
  return hipdnn_wmma_acc_row(pr, e);
#else
  return e * 2 + pr;
#endif
}

// acc += A B on one 16x16x16 tile; a(m, kk) and b(kk, n) are in tile
// coordinates. A split operand enters as fp16 hi + lo (the lo * lo term is
// dropped), which keeps about 22 bits of its fp32 mantissa.
template <bool kSplitA, bool kSplitB, typename FA, typename FB>
__device__ __forceinline__ void gdnMma(gdnFloat8 &acc, int lane, FA a, FB b) {
  const int ln = lane % kWt;
  const int pr = lane / kWt;
#if HIPDNN_HAS_WMMA
  const int kOff = hipdnn_wmma_k_off(pr);
  gdnHalfFrag aHi;
  gdnHalfFrag aLo;
  gdnHalfFrag bHi;
  gdnHalfFrag bLo;
#pragma unroll
  for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e) {
    const float av = a(ln, kOff + e);
    const float bv = b(kOff + e, ln);
    aHi[e] = static_cast<_Float16>(av);
    bHi[e] = static_cast<_Float16>(bv);
    if constexpr (kSplitA) {
      aLo[e] = static_cast<_Float16>(av - static_cast<float>(aHi[e]));
    }
    if constexpr (kSplitB) {
      bLo[e] = static_cast<_Float16>(bv - static_cast<float>(bHi[e]));
    }
  }
  acc = HIPDNN_WMMA_F32_16X16X16_F16(aHi, bHi, acc);
  if constexpr (kSplitB) {
    acc = HIPDNN_WMMA_F32_16X16X16_F16(aHi, bLo, acc);
  }
  if constexpr (kSplitA) {
    acc = HIPDNN_WMMA_F32_16X16X16_F16(aLo, bHi, acc);
  }
#else
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float s = 0.0f;
    for (int kk = 0; kk < kWt; ++kk) {
      s += a(e * 2 + pr, kk) * b(kk, ln);
    }
    acc[e] += s;
  }
#endif
}

// Chunked (WY) form of the same recurrence, one block per (batch, value head,
// kSliceV-row slice of the state). Per chunk of kChunk tokens, with G the
// in-chunk cumulative log decay and S0 the chunk-start state [dv, dk]:
//   A = strictLower(beta_i e^{G_i-G_j} k_i.k_j),  T = (I + A)^-1
//   U = T (beta v - beta e^G (S0 k))               (the per-token deltas)
//   O = scale (e^G (S0 q) + lower(e^{G_i-G_j} q_i.k_j) U)
//   S = e^{G_last} S0 + sum_j e^{G_last-G_j} u_j k_j^T
// Products run on the matrix cores with fp32 accumulation. q and k enter as
// their raw fp16 inputs (the L2 norms are applied in fp32 afterwards); fp32
// operands are split into fp16 hi + lo. The state stays in fp32 accumulators
// for the whole sequence. Each block reads its own state slice before writing
// it, so initial_state may alias final_state.
__global__ void __launch_bounds__(kBlockSize) gatedDeltaNetChunkedKernel(
    const __half *__restrict__ qkv, const float *__restrict__ decay,
    const float *__restrict__ beta, const float *initialState,
    const float *__restrict__ aLog, const float *__restrict__ dtBias,
    __half *__restrict__ output, float *finalState, int seqLen,
    int numHeadsQk, int numHeadsV, int headSizeV, float scale) {
  constexpr int DK = kChunkDk;
  constexpr int kWaves = kBlockSize / kLanes;
  constexpr int kTokensPerWave = kChunk / kWaves;
  constexpr int kStateTiles = (kSliceV / kWt) * (DK / kWt) / kWaves;
  const int slices = headSizeV / kSliceV;
  const int slice = static_cast<int>(blockIdx.x) % slices;
  const int hv = (static_cast<int>(blockIdx.x) / slices) % numHeadsV;
  const int b = static_cast<int>(blockIdx.x) / (slices * numHeadsV);
  const int hq = hv * numHeadsQk / numHeadsV;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kLanes;
  const int wave = tid / kLanes;
  const int ln = lane % kWt;
  const int pr = lane / kWt;
  const int side = wave / 4;
  const int ti = (wave % 4) / 2;
  const int tc = (wave % 4) % 2;

  __shared__ __align__(16) _Float16 kRaw[kChunk][kRawStride];
  __shared__ __align__(16) _Float16 qRaw[kChunk][kRawStride];
  __shared__ __align__(16) _Float16 sHi[kSliceV][kRawStride];
  __shared__ __align__(16) _Float16 sLo[kSliceV][kRawStride];
  __shared__ __align__(16) float aTri[kChunk][kTileStride];
  __shared__ __align__(16) float pTri[kChunk][kTileStride];
  __shared__ __align__(16) float rT[kSliceV][kTileStride];
  __shared__ __align__(16) float uT[kSliceV][kTileStride];
  __shared__ float kNorm[kChunk];
  __shared__ float qNorm[kChunk];
  __shared__ float wLast[kChunk];
  __shared__ bool tLargeS;

  const int64_t packed = static_cast<int64_t>(2) * numHeadsQk * DK +
                         static_cast<int64_t>(numHeadsV) * headSizeV;
  const int64_t vOffset = static_cast<int64_t>(2) * numHeadsQk * DK +
                          static_cast<int64_t>(hv) * headSizeV +
                          slice * kSliceV;
  const float headDecay = -expf(aLog[hv]);
  const float headBias = dtBias[hv];
  const int64_t stateBase =
      ((static_cast<int64_t>(b) * numHeadsV + hv) * headSizeV +
       slice * kSliceV) *
      DK;

  gdnFloat8 sAcc[kStateTiles];
  auto storeStateCopy = [&](int q) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int v = (wave / 4) * kWt + gdnAccRow(pr, e);
      const int k = ((wave % 4) * kStateTiles + q) * kWt + ln;
      const _Float16 hi = static_cast<_Float16>(sAcc[q][e]);
      sHi[v][k] = hi;
      sLo[v][k] = static_cast<_Float16>(sAcc[q][e] - static_cast<float>(hi));
    }
  };
#pragma unroll
  for (int q = 0; q < kStateTiles; ++q) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int v = (wave / 4) * kWt + gdnAccRow(pr, e);
      const int k = ((wave % 4) * kStateTiles + q) * kWt + ln;
      sAcc[q][e] = initialState[stateBase + static_cast<int64_t>(v) * DK + k];
    }
    storeStateCopy(q);
  }

  gdnHalf4 pq[kTokensPerWave];
  gdnHalf4 pk[kTokensPerWave];
  float pDecay;
  float pBeta;
  float pv[8];
  auto prefetch = [&](int t0) {
    const int n = min(kChunk, seqLen - t0);
#pragma unroll
    for (int j = 0; j < kTokensPerWave; ++j) {
      const int c = wave + j * kWaves;
      pq[j] = gdnHalf4{};
      pk[j] = gdnHalf4{};
      if (c < n) {
        const int64_t tok = static_cast<int64_t>(b) * seqLen + t0 + c;
        const __half *rowPtr = qkv + tok * packed;
        pq[j] = *reinterpret_cast<const gdnHalf4 *>(
            rowPtr + static_cast<int64_t>(hq) * DK + lane * 4);
        pk[j] = *reinterpret_cast<const gdnHalf4 *>(
            rowPtr + static_cast<int64_t>(numHeadsQk + hq) * DK + lane * 4);
      }
    }
    // Unconditional loads from a clamped row, masked only when consumed: a
    // per-lane guard here would make each load wait for the previous one.
    const int64_t gateTok =
        static_cast<int64_t>(b) * seqLen + t0 + min(lane, n - 1);
    pDecay = decay[gateTok * numHeadsV + hv];
    pBeta = beta[gateTok * numHeadsV + hv];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = ti * kWt + gdnAccRow(pr, e);
      const int64_t tok = static_cast<int64_t>(b) * seqLen + t0 + min(i, n - 1);
      pv[e] = __half2float(qkv[tok * packed + vOffset + tc * kWt + ln]);
    }
  };
  prefetch(0);

  for (int t0 = 0; t0 < seqLen; t0 += kChunk) {
    const int n = min(kChunk, seqLen - t0);

#pragma unroll
    for (int j = 0; j < kTokensPerWave; ++j) {
      const int c = wave + j * kWaves;
      const gdnHalf4 qh = pq[j];
      const gdnHalf4 kh = pk[j];
      float qq = 0.0f;
      float kk = 0.0f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        qq += static_cast<float>(qh[e]) * static_cast<float>(qh[e]);
        kk += static_cast<float>(kh[e]) * static_cast<float>(kh[e]);
      }
      qq = groupSum<kLanes>(qq);
      kk = groupSum<kLanes>(kk);
      *reinterpret_cast<gdnHalf4 *>(&qRaw[c][lane * 4]) = qh;
      *reinterpret_cast<gdnHalf4 *>(&kRaw[c][lane * 4]) = kh;
      if (lane == 0) {
        qNorm[c] = c < n ? rsqrtf(qq + 1e-12f) : 0.0f;
        kNorm[c] = c < n ? rsqrtf(kk + 1e-12f) : 0.0f;
      }
    }
    const bool laneValid = lane < n;
    const float gTok =
        laneValid ? headDecay * softplus(pDecay + headBias) : 0.0f;
    const float betaTok = laneValid ? 1.0f / (1.0f + expf(-pBeta)) : 0.0f;
    float vReg[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = ti * kWt + gdnAccRow(pr, e);
      vReg[e] = side == 0 && i < n ? pv[e] : 0.0f;
    }
    __syncthreads();
    if (t0 + kChunk < seqLen) {
      prefetch(t0 + kChunk);
    }

    float gCum = gTok;
#pragma unroll
    for (int off = 1; off < kLanes; off <<= 1) {
      const float y = __shfl_up(gCum, off, kLanes);
      if (lane >= off) {
        gCum += y;
      }
    }
    const float gLast = __shfl(gCum, kChunk - 1, kLanes);
    if (wave == 0) {
      wLast[lane] = __expf(gLast - gCum) * kNorm[lane];
    }

    _Float16(*lhs)[kRawStride] = side == 0 ? kRaw : qRaw;
    {
      gdnFloat8 acc = {};
      if (ti >= tc) {
        for (int k0 = 0; k0 < DK; k0 += kWt) {
          gdnMma<false, false>(
              acc, lane,
              [&](int m, int kk) {
                return static_cast<float>(lhs[ti * kWt + m][k0 + kk]);
              },
              [&](int kk, int nn) {
                return static_cast<float>(kRaw[tc * kWt + nn][k0 + kk]);
              });
        }
      }
      const int j = tc * kWt + ln;
      const float gj = __shfl(gCum, j, kLanes);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ti * kWt + gdnAccRow(pr, e);
        const float gi = __shfl(gCum, i, kLanes);
        const float bi = __shfl(betaTok, i, kLanes);
        if (side == 0) {
          aTri[i][j] = j < i ? bi * __expf(gi - gj) * kNorm[i] * kNorm[j] *
                                   acc[e]
                             : 0.0f;
        } else {
          pTri[i][j] =
              j <= i ? __expf(gi - gj) * qNorm[i] * kNorm[j] * acc[e] : 0.0f;
        }
      }
    }
    gdnFloat8 oAcc = {};
    {
      gdnFloat8 acc = {};
      for (int k0 = 0; k0 < DK; k0 += kWt) {
        auto aOp = [&](int m, int kk) {
          return static_cast<float>(lhs[ti * kWt + m][k0 + kk]);
        };
        gdnMma<false, false>(acc, lane, aOp, [&](int kk, int nn) {
          return static_cast<float>(sHi[tc * kWt + nn][k0 + kk]);
        });
        gdnMma<false, false>(acc, lane, aOp, [&](int kk, int nn) {
          return static_cast<float>(sLo[tc * kWt + nn][k0 + kk]);
        });
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ti * kWt + gdnAccRow(pr, e);
        const float eg = __expf(__shfl(gCum, i, kLanes));
        const float bi = __shfl(betaTok, i, kLanes);
        if (side == 0) {
          rT[tc * kWt + ln][i] = bi * (vReg[e] - eg * kNorm[i] * acc[e]);
        } else {
          oAcc[e] = eg * qNorm[i] * acc[e];
        }
      }
    }
    __syncthreads();

    if (wave == 0) {
      float tcol[kChunk];
      float tMax = 0.0f;
#pragma unroll
      for (int i = 0; i < kChunk; ++i) {
        float acc = i == lane ? 1.0f : 0.0f;
#pragma unroll
        for (int j = 0; j < i; ++j) {
          acc -= aTri[i][j] * tcol[j];
        }
        tcol[i] = acc;
        tMax = fmaxf(tMax, fabsf(acc));
      }
#pragma unroll
      for (int i = 0; i < kChunk; ++i) {
        aTri[i][lane] = tcol[i];
      }
      for (int off = kLanes / 2; off > 0; off >>= 1) {
        tMax = fmaxf(tMax, __shfl_xor(tMax, off, kLanes));
      }
      if (lane == 0) {
        tLargeS = !(tMax < kSplitSafeMax);
      }
    }
    __syncthreads();

    // Entries of T can grow geometrically along a chunk; past the fp16 range
    // of the split operands, T R runs in fp32.
    if (side == 0 && !tLargeS) {
      gdnFloat8 acc = {};
      for (int tj = 0; tj <= ti; ++tj) {
        gdnMma<true, true>(
            acc, lane,
            [&](int m, int kk) { return aTri[ti * kWt + m][tj * kWt + kk]; },
            [&](int kk, int nn) { return rT[tc * kWt + nn][tj * kWt + kk]; });
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        uT[tc * kWt + ln][ti * kWt + gdnAccRow(pr, e)] = acc[e];
      }
    } else if (side == 0) {
      const float *rRow = rT[tc * kWt + ln];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ti * kWt + gdnAccRow(pr, e);
        float acc = 0.0f;
        for (int j = 0; j < (ti + 1) * kWt; j += 4) {
          const float4 t4 = *reinterpret_cast<const float4 *>(&aTri[i][j]);
          const float4 r4 = *reinterpret_cast<const float4 *>(&rRow[j]);
          acc += t4.x * r4.x + t4.y * r4.y + t4.z * r4.z + t4.w * r4.w;
        }
        uT[tc * kWt + ln][i] = acc;
      }
    }
    __syncthreads();

    if (side == 1) {
      for (int tj = 0; tj <= ti; ++tj) {
        gdnMma<true, true>(
            oAcc, lane,
            [&](int m, int kk) { return pTri[ti * kWt + m][tj * kWt + kk]; },
            [&](int kk, int nn) { return uT[tc * kWt + nn][tj * kWt + kk]; });
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = ti * kWt + gdnAccRow(pr, e);
        if (i < n) {
          const int64_t tok = static_cast<int64_t>(b) * seqLen + t0 + i;
          output[(tok * numHeadsV + hv) * headSizeV + slice * kSliceV +
                 tc * kWt + ln] = __float2half(scale * oAcc[e]);
        }
      }
    }

    const float eLast = __expf(gLast);
#pragma unroll
    for (int q = 0; q < kStateTiles; ++q) {
      sAcc[q] *= eLast;
      const int kBase = ((wave % 4) * kStateTiles + q) * kWt;
      for (int jt = 0; jt < kChunk / kWt; ++jt) {
        gdnMma<true, false>(
            sAcc[q], lane,
            [&](int m, int kk) {
              const int j = jt * kWt + kk;
              return uT[(wave / 4) * kWt + m][j] * wLast[j];
            },
            [&](int kk, int nn) {
              return static_cast<float>(kRaw[jt * kWt + kk][kBase + nn]);
            });
      }
      storeStateCopy(q);
    }
    __syncthreads();
  }

#pragma unroll
  for (int q = 0; q < kStateTiles; ++q) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int v = (wave / 4) * kWt + gdnAccRow(pr, e);
      const int k = ((wave % 4) * kStateTiles + q) * kWt + ln;
      finalState[stateBase + static_cast<int64_t>(v) * DK + k] = sAcc[q][e];
    }
  }
}

hipError_t launchChunked(hipStream_t stream, const void *qkv,
                         const float *decay, const float *beta,
                         const float *initialState, const float *aLog,
                         const float *dtBias, void *output, float *finalState,
                         int batch, int seqLen, int numHeadsQk, int numHeadsV,
                         int headSizeV, float scale) {
  const dim3 grid(
      static_cast<unsigned>(batch * numHeadsV * (headSizeV / kSliceV)));
  hipLaunchKernelGGL(gatedDeltaNetChunkedKernel, grid, dim3(kBlockSize), 0,
                     stream, static_cast<const __half *>(qkv), decay, beta,
                     initialState, aLog, dtBias, static_cast<__half *>(output),
                     finalState, seqLen, numHeadsQk, numHeadsV, headSizeV,
                     scale);
  return hipGetLastError();
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
    if constexpr (std::is_same<T, __half>::value) {
      if (seqLen >= kChunkedMinSeqLen && headSizeV % kSliceV == 0) {
        return launchChunked(stream, qkv, decay, beta, initialState, aLog,
                             dtBias, output, finalState, batch, seqLen,
                             numHeadsQk, numHeadsV, headSizeV, scale);
      }
    }
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
