// SPDX-License-Identifier: MIT
// Native FP16 adaptation of the METIS Ring implementations supplied by the
// contributor. WMMA score ownership and V swizzle derive from hip-ep v8x (MIT).

#include "hip_custom_kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

// Shared by workspace sizing and dispatch; measured for gfx1151 B1/H24/G4/D256.
static int ring_decode_partitions(int total) {
  if (total <= 32)
    return 1;
  if (total <= 128)
    return 4;
  if (total <= 1024)
    return 8;
  return 16;
}

#if HIP_EP_RING_DEVICE_ENABLED
namespace {
struct RingTensor {
  void *data;
  int64_t shape[4], stride[4];
};
struct RingCall {
  RingTensor tensors[4];
  int64_t B, Q, P;
  void *workspace;
  hipStream_t stream;
};
RingCall make_call(hipStream_t stream, const void *q, const void *k,
                   const void *v, void *o, int nq, int past, int capacity,
                   void *workspace) {
  RingCall c{};
  c.B = 1;
  c.Q = nq;
  c.P = past;
  c.workspace = workspace;
  c.stream = stream;
  const void *pointers[] = {q, k, v, o};
  for (int i = 0; i < 4; ++i) {
    auto &t = c.tensors[i];
    t.data = const_cast<void *>(pointers[i]);
    t.shape[0] = 1;
    t.shape[1] = (i == 1 || i == 2) ? 4 : 24;
    t.shape[2] = (i == 1 || i == 2) ? past + nq : nq;
    t.shape[3] = 256;
    t.stride[3] = 1;
    t.stride[2] = (i == 1 || i == 2) ? 256 : 24 * 256;
    t.stride[1] = (i == 1 || i == 2) ? int64_t(capacity) * 256 : 256;
    t.stride[0] = (i == 1 || i == 2) ? int64_t(capacity) * 4 * 256
                                     : int64_t(nq) * 24 * 256;
  }
  return c;
}
namespace prefill {
#if HIP_EP_GQA_IMPL_ID == 1
// FP16 online attention, gfx1151. Natural-log softmax and expf only.
// Single-device Ring attention. Transposed WMMA score ownership
// and V swizzle informed by hip-ep v8x (MIT).

using I = int64_t;
using U = uint16_t;
using V16 = short __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ F8 ring_fp16_wmma(V16 a, V16 b, F8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      __builtin_bit_cast(H16, a), __builtin_bit_cast(H16, b), c);
}
using Pack = U __attribute__((ext_vector_type(16), aligned(16)));
constexpr int ND = 2, HB = 6, KV = 16, D = 256, HQ = 24, HK = 4,
              DT = D / (16 * ND), NT = 32 * ND * HB, KS = D + 8, VS = KV + 8,
              KT = KV / 16;
static_assert(6 % HB == 0 && D % (16 * ND) == 0 && KV % 16 == 0,
              "WMMA geometry");
__device__ U narrow(float x) { return __half_as_ushort(__float2half_rn(x)); }
__device__ I at(const RingTensor &t, int h, I r, int d) {
  return h * t.stride[1] + r * t.stride[2] + d * t.stride[3];
}
__device__ U raw(const RingTensor &t, int h, I r, int d) {
  return static_cast<const U *>(t.data)[at(t, h, r, d)];
}
__device__ int swizzle(int d, int k) {
  return (k & ~15) | (((k ^ d) & 1) << 3) | ((k & 15) >> 1);
}
__global__ __launch_bounds__(NT) void ring_online(RingCall c) {
  int tid = threadIdx.x, lh = __builtin_amdgcn_readfirstlane(tid / (32 * ND)),
      dw = __builtin_amdgcn_readfirstlane((tid / 32) % ND), lane = tid % 16,
      par = (tid % 32) / 16;
  // Longest causal queries first, adjacent heads reuse one KV group in cache.
  int nqtiles = (c.Q + 15) / 16, cpg = 6 / HB, bid = blockIdx.x;
  int group = bid / (nqtiles * cpg), hg = bid % cpg,
      h = group * 6 + hg * HB + lh;
  I q0 = (nqtiles - 1 - (bid / cpg) % nqtiles) * 16, q = q0 + lane,
    C = c.P + c.Q, end = c.P + c.Q;
  I stop = C < c.P + q0 + 16 ? C : c.P + q0 + 16;
  stop = stop < end ? stop : end;
  __shared__ __align__(16) U key[KV][KS], value[D][VS];
  __shared__ F8 partial[HB][ND][KT][32];
  V16 qr[DT];
  F8 out[DT];
  float maximum = -INFINITY, denom = 0;
  _Pragma("unroll") for (int t = 0; t < DT; ++t) {
    out[t] = F8{};
    qr[t] = V16{};
    if (q < c.Q) {
      const U *ptr = static_cast<const U *>(c.tensors[0].data) +
                     at(c.tensors[0], h, q, (dw * DT + t) * 16);
      if (c.tensors[0].stride[3] == 1 && uintptr_t(ptr) % 32 == 0)
        qr[t] = *reinterpret_cast<const V16 *>(ptr);
      else
        _Pragma("unroll") for (int j = 0; j < 16; ++j) qr[t][j] =
            short(raw(c.tensors[0], h, q, (dw * DT + t) * 16 + j));
    }
  }

  constexpr int CHUNKS = KV * (D / 16), PER = (2 * CHUNKS + NT - 1) / NT,
                PF = PER < 2 ? PER : 2;
  Pack pending[PF];
  auto prefetch = [&](I k0, int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      bool isk = z < CHUNKS;
      int j = isk ? z : z - CHUNKS;
      int kr = isk ? j / (D / 16) : j % KV,
          d = (isk ? j % (D / 16) : j / KV) * 16;
      Pack pack{};
      if (z < 2 * CHUNKS && k0 + kr < C && k0 + kr < end) {
        const RingTensor &src = c.tensors[isk ? 1 : 2];

        const U *ptr =
            static_cast<const U *>(src.data) + at(src, group, k0 + kr, d);
        if (src.stride[3] == 1 && uintptr_t(ptr) % 16 == 0)
          pack = *reinterpret_cast<const Pack *>(ptr);
        else
          _Pragma("unroll") for (int e = 0; e < 16; ++e) pack[e] =
              raw(src, group, k0 + kr, d + e);
      }
      pending[r] = pack;
    }
  };
  auto commit = [&](int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      if (z < CHUNKS) {
        _Pragma("unroll") for (int e = 0; e < 16; ++e)
            key[z / (D / 16)][(z % (D / 16)) * 16 + e] = pending[r][e];
      } else if (z < 2 * CHUNKS) {
        int j = z - CHUNKS;
        _Pragma("unroll") for (int e = 0; e < 16; ++e) {
          int d = (j / KV) * 16 + e;
          value[d][swizzle(d, j % KV)] = pending[r][e];
        }
      }
    }
  };
  if (0 < stop)
    prefetch(0, 0, PF);
  for (I k0 = 0; k0 < stop; k0 += KV) {
    commit(0, PF);
    _Pragma("unroll 1") for (int r = PF; r < PER; ++r) {
      prefetch(k0, r, 1);
      commit(r, 1);
    }
    __syncthreads();
    if (k0 + KV < stop)
      prefetch(k0 + KV, 0, PF);
    F8 s[KT];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      s[kt] = F8{};
      _Pragma("unroll") for (int t = 0; t < DT; ++t) {
        V16 kr;
        _Pragma("unroll") for (int j = 0; j < 16; ++j) kr[j] =
            short(key[kt * 16 + lane][(dw * DT + t) * 16 + j]);
        s[kt] = ring_fp16_wmma(kr, qr[t], s[kt]);
      }
      if constexpr (ND > 1)
        partial[lh][dw][kt][tid % 32] = s[kt];
    }
    if constexpr (ND > 1) {
      __syncthreads();
      _Pragma("unroll") for (int w = 0; w < ND; ++w) if (w != dw) {
        _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) s[kt] +=
            partial[lh][w][kt][tid % 32];
      }
    }
    float m = -INFINITY;
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt)
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {
      I k = k0 + kt * 16 + 2 * e + par;
      float x = s[kt][e] * 0.0625f;
      if (k >= end || k >= C || k > c.P + q)
        x = -INFINITY;
      s[kt][e] = x;
      m = fmaxf(m, x);
    }
    m = fmaxf(m, __shfl_xor(m, 16, 32));
    float next = fmaxf(maximum, m),
          alpha = next == maximum
                      ? 1.f
                      : (maximum == -INFINITY ? 0.f : expf(maximum - next)),
          sum = 0;
    V16 prob[KT];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      U own[8];
      _Pragma("unroll") for (int e = 0; e < 8; ++e) {
        float p = next == -INFINITY ? 0.f : expf(s[kt][e] - next);
        sum += p;
        own[e] = narrow(p);
      }
      // Gfx11 WMMA: even/odd keys in separate half waves. Exchange packed
      // pairs.
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {
        unsigned pack = unsigned(own[2 * j]) | (unsigned(own[2 * j + 1]) << 16);
        unsigned other = __shfl_xor(pack, 16, 32);
        prob[kt][2 * j] = short(pack);
        prob[kt][2 * j + 1] = short(pack >> 16);
        prob[kt][8 + 2 * j] = short(other);
        prob[kt][9 + 2 * j] = short(other >> 16);
      }
    }
    sum += __shfl_xor(sum, 16, 32);
    denom = denom * alpha + sum;
    maximum = next;
    if (__any(alpha != 1.f)) {
      _Pragma("unroll") for (int t = 0; t < DT; ++t) out[t] *= alpha;
    }
    _Pragma("unroll") for (int t = 0; t < DT; ++t) {
      _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
        V16 vr;
        _Pragma("unroll") for (int j = 0; j < 16; ++j) vr[j] =
            short(value[(dw * DT + t) * 16 + lane][kt * 16 + j]);
        out[t] = ring_fp16_wmma(vr, prob[kt], out[t]);
      }
    }
    __syncthreads();
  }
  if (q < c.Q) {
    _Pragma("unroll") for (int t = 0; t < DT; ++t)
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {
      int d = (dw * DT + t) * 16 + 2 * e + par;

      static_cast<U *>(c.tensors[3].data)[(q * HQ + h) * D + d] =
          narrow(out[t][e] / denom);
    }
  }
}

#else
// KV-outer Ring: parallel Q-tile consumers share one LDS KV tile. FP16
// online attention, gfx1151. Natural-log softmax and expf only. Transposed WMMA
// score ownership and V swizzle informed by hip-ep v8x (MIT).

constexpr int QT = 2;
using I = int64_t;
using U = uint16_t;
using V16 = short __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ F8 ring_fp16_wmma(V16 a, V16 b, F8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      __builtin_bit_cast(H16, a), __builtin_bit_cast(H16, b), c);
}
using Pack = U __attribute__((ext_vector_type(16), aligned(16)));
constexpr int ND = 2, HB = 6, KV = 16, D = 256, HQ = 24, HK = 4,
              DT = D / (16 * ND), NT = 32 * ND * HB * QT, KS = D + 8,
              VS = KV + 8, KT = KV / 16;
static_assert(6 % HB == 0 && D % (16 * ND) == 0 && KV % 16 == 0,
              "WMMA geometry");
__device__ U narrow(float x) { return __half_as_ushort(__float2half_rn(x)); }
__device__ I at(const RingTensor &t, int h, I r, int d) {
  return h * t.stride[1] + r * t.stride[2] + d * t.stride[3];
}
__device__ U raw(const RingTensor &t, int h, I r, int d) {
  return static_cast<const U *>(t.data)[at(t, h, r, d)];
}
__device__ int swizzle(int d, int k) {
  return (k & ~15) | (((k ^ d) & 1) << 3) | ((k & 15) >> 1);
}
__global__ __launch_bounds__(NT) void ring_kv_outer_parallel(RingCall c) {
  int tid = threadIdx.x,
      lh = __builtin_amdgcn_readfirstlane((tid / (32 * ND)) % HB),
      dw = __builtin_amdgcn_readfirstlane((tid / 32) % ND), lane = tid % 16,
      par = (tid % 32) / 16;
  // Longest causal queries first, adjacent heads reuse one KV group in cache.
  int qi = __builtin_amdgcn_readfirstlane(tid / (32 * ND * HB));
  int nqtiles = (c.Q + 16 * QT - 1) / (16 * QT), cpg = 6 / HB, bid = blockIdx.x;
  int group = bid / (nqtiles * cpg), hg = bid % cpg,
      h = group * 6 + hg * HB + lh;
  I base = (nqtiles - 1 - (bid / cpg) % nqtiles) * 16 * QT, q0 = base + qi * 16,
    q = q0 + lane, C = c.P + c.Q, end = c.P + c.Q;
  I stop = C < c.P + base + 16 * QT ? C : c.P + base + 16 * QT;
  stop = stop < end ? stop : end;
  __shared__ __align__(16) U key[KV][KS], value[D][VS];
  __shared__ F8 partial[QT][HB][ND][KT][32];
  V16 qr[DT];
  F8 out[DT];
  float maximum = -INFINITY, denom = 0;
  _Pragma("unroll") for (int t = 0; t < DT; ++t) {
    out[t] = F8{};
    qr[t] = V16{};
    if (q < c.Q) {
      const U *ptr = static_cast<const U *>(c.tensors[0].data) +
                     at(c.tensors[0], h, q, (dw * DT + t) * 16);
      if (c.tensors[0].stride[3] == 1 && uintptr_t(ptr) % 32 == 0)
        qr[t] = *reinterpret_cast<const V16 *>(ptr);
      else
        _Pragma("unroll") for (int j = 0; j < 16; ++j) qr[t][j] =
            short(raw(c.tensors[0], h, q, (dw * DT + t) * 16 + j));
    }
  }

  constexpr int CHUNKS = KV * (D / 16), PER = (2 * CHUNKS + NT - 1) / NT,
                PF = PER < 2 ? PER : 2;
  Pack pending[PF];
  auto prefetch = [&](I k0, int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      bool isk = z < CHUNKS;
      int j = isk ? z : z - CHUNKS;
      int kr = isk ? j / (D / 16) : j % KV,
          d = (isk ? j % (D / 16) : j / KV) * 16;
      Pack pack{};
      if (z < 2 * CHUNKS && k0 + kr < C && k0 + kr < end) {
        const RingTensor &src = c.tensors[isk ? 1 : 2];

        const U *ptr =
            static_cast<const U *>(src.data) + at(src, group, k0 + kr, d);
        if (src.stride[3] == 1 && uintptr_t(ptr) % 16 == 0)
          pack = *reinterpret_cast<const Pack *>(ptr);
        else
          _Pragma("unroll") for (int e = 0; e < 16; ++e) pack[e] =
              raw(src, group, k0 + kr, d + e);
      }
      pending[r] = pack;
    }
  };
  auto commit = [&](int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      if (z < CHUNKS) {
        _Pragma("unroll") for (int e = 0; e < 16; ++e)
            key[z / (D / 16)][(z % (D / 16)) * 16 + e] = pending[r][e];
      } else if (z < 2 * CHUNKS) {
        int j = z - CHUNKS;
        _Pragma("unroll") for (int e = 0; e < 16; ++e) {
          int d = (j / KV) * 16 + e;
          value[d][swizzle(d, j % KV)] = pending[r][e];
        }
      }
    }
  };
  if (0 < stop)
    prefetch(0, 0, PF);
  for (I k0 = 0; k0 < stop; k0 += KV) {
    commit(0, PF);
    _Pragma("unroll 1") for (int r = PF; r < PER; ++r) {
      prefetch(k0, r, 1);
      commit(r, 1);
    }
    __syncthreads();
    if (k0 + KV < stop)
      prefetch(k0 + KV, 0, PF);
    F8 s[KT];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      s[kt] = F8{};
      _Pragma("unroll") for (int t = 0; t < DT; ++t) {
        V16 kr;
        _Pragma("unroll") for (int j = 0; j < 16; ++j) kr[j] =
            short(key[kt * 16 + lane][(dw * DT + t) * 16 + j]);
        s[kt] = ring_fp16_wmma(kr, qr[t], s[kt]);
      }
      if constexpr (ND > 1)
        partial[qi][lh][dw][kt][tid % 32] = s[kt];
    }
    if constexpr (ND > 1) {
      __syncthreads();
      _Pragma("unroll") for (int w = 0; w < ND; ++w) if (w != dw) {
        _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) s[kt] +=
            partial[qi][lh][w][kt][tid % 32];
      }
    }
    float m = -INFINITY;
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt)
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {
      I k = k0 + kt * 16 + 2 * e + par;
      float x = s[kt][e] * 0.0625f;
      if (k >= end || k >= C || k > c.P + q)
        x = -INFINITY;
      s[kt][e] = x;
      m = fmaxf(m, x);
    }
    m = fmaxf(m, __shfl_xor(m, 16, 32));
    float next = fmaxf(maximum, m),
          alpha = next == maximum
                      ? 1.f
                      : (maximum == -INFINITY ? 0.f : expf(maximum - next)),
          sum = 0;
    V16 prob[KT];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      U own[8];
      _Pragma("unroll") for (int e = 0; e < 8; ++e) {
        float p = next == -INFINITY ? 0.f : expf(s[kt][e] - next);
        sum += p;
        own[e] = narrow(p);
      }
      // Gfx11 WMMA: even/odd keys in separate half waves. Exchange packed
      // pairs.
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {
        unsigned pack = unsigned(own[2 * j]) | (unsigned(own[2 * j + 1]) << 16);
        unsigned other = __shfl_xor(pack, 16, 32);
        prob[kt][2 * j] = short(pack);
        prob[kt][2 * j + 1] = short(pack >> 16);
        prob[kt][8 + 2 * j] = short(other);
        prob[kt][9 + 2 * j] = short(other >> 16);
      }
    }
    sum += __shfl_xor(sum, 16, 32);
    denom = denom * alpha + sum;
    maximum = next;
    if (__any(alpha != 1.f)) {
      _Pragma("unroll") for (int t = 0; t < DT; ++t) out[t] *= alpha;
    }
    _Pragma("unroll") for (int t = 0; t < DT; ++t) {
      _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
        V16 vr;
        _Pragma("unroll") for (int j = 0; j < 16; ++j) vr[j] =
            short(value[(dw * DT + t) * 16 + lane][kt * 16 + j]);
        out[t] = ring_fp16_wmma(vr, prob[kt], out[t]);
      }
    }
    __syncthreads();
  }
  if (q < c.Q) {
    _Pragma("unroll") for (int t = 0; t < DT; ++t)
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {
      int d = (dw * DT + t) * 16 + 2 * e + par;

      static_cast<U *>(c.tensors[3].data)[(q * HQ + h) * D + d] =
          narrow(out[t][e] / denom);
    }
  }
}

#endif
} // namespace prefill
namespace decode {

using I = int64_t;
using U = uint16_t;
using V16 = short __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ F8 ring_fp16_wmma(V16 a, V16 b, F8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      __builtin_bit_cast(H16, a), __builtin_bit_cast(H16, b), c);
}
using Pack = U __attribute__((ext_vector_type(16), aligned(16)));
constexpr int ND = 4, HB = 1, KV = 16, D = 256, HQ = 24, HK = 4,
              DT = D / (16 * ND), NT = 32 * ND * HB, KS = D + 8, VS = KV + 8,
              KT = KV / 16;
static_assert(6 % HB == 0 && D % (16 * ND) == 0 && KV % 16 == 0,
              "WMMA geometry");
__device__ U narrow(float x) { return __half_as_ushort(__float2half_rn(x)); }
__device__ I at(const RingTensor &t, int h, I r, int d) {
  return h * t.stride[1] + r * t.stride[2] + d * t.stride[3];
}
__device__ U raw(const RingTensor &t, int h, I r, int d) {
  return static_cast<const U *>(t.data)[at(t, h, r, d)];
}
__device__ int swizzle(int d, int k) {
  return (k & ~15) | (((k ^ d) & 1) << 3) | ((k & 15) >> 1);
}
// Six query heads reuse one staged KV tile. QK uses WMMA with FP32
// accumulation; probabilities and weighted V accumulation stay in FP32.
__global__ __launch_bounds__(NT) void ring_decode_scan(RingCall c, int parts) {
  int tid = threadIdx.x, lh = __builtin_amdgcn_readfirstlane(tid / (32 * ND)),
      dw = __builtin_amdgcn_readfirstlane((tid / 32) % ND), lane = tid % 16,
      par = (tid % 32) / 16;
  int group=blockIdx.x, h=group*6+lane;
  I C=c.P+1, span=((C+parts-1)/parts+KV-1)/KV*KV;
  I begin=blockIdx.y*span, stop=begin+span<C?begin+span:C, end=stop;
  I q=0;
  __shared__ __align__(16) U key[KV][KS], value[D][VS];
  __shared__ F8 partial[HB][ND][KT][32];
  V16 qr[DT];
  F8 out[DT];
  float maximum = -INFINITY, denom = 0;
  _Pragma("unroll") for (int t = 0; t < DT; ++t) {
    out[t] = F8{};
    qr[t] = V16{};
    if (lane < 6) {
      const U *ptr = static_cast<const U *>(c.tensors[0].data) +
                     at(c.tensors[0], h, q, (dw * DT + t) * 16);
      if (c.tensors[0].stride[3] == 1 && uintptr_t(ptr) % 32 == 0)
        qr[t] = *reinterpret_cast<const V16 *>(ptr);
      else
        _Pragma("unroll") for (int j = 0; j < 16; ++j) qr[t][j] =
            short(raw(c.tensors[0], h, q, (dw * DT + t) * 16 + j));
    }
  }

  constexpr int CHUNKS = KV * (D / 16), PER = (2 * CHUNKS + NT - 1) / NT,
                PF = PER < 2 ? PER : 2;
  Pack pending[PF];
  auto prefetch = [&](I k0, int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      bool isk = z < CHUNKS;
      int j = isk ? z : z - CHUNKS;
      int kr = isk ? j / (D / 16) : j % KV,
          d = (isk ? j % (D / 16) : j / KV) * 16;
      Pack pack{};
      if (z < 2 * CHUNKS && k0 + kr < C && k0 + kr < end) {
        const RingTensor &src = c.tensors[isk ? 1 : 2];

        const U *ptr =
            static_cast<const U *>(src.data) + at(src, group, k0 + kr, d);
        if (src.stride[3] == 1 && uintptr_t(ptr) % 16 == 0)
          pack = *reinterpret_cast<const Pack *>(ptr);
        else
          _Pragma("unroll") for (int e = 0; e < 16; ++e) pack[e] =
              raw(src, group, k0 + kr, d + e);
      }
      pending[r] = pack;
    }
  };
  auto commit = [&](int base, int n) {
    _Pragma("unroll") for (int r = 0; r < n; ++r) {
      int z = tid + (base + r) * NT;
      if (z < CHUNKS) {
        _Pragma("unroll") for (int e = 0; e < 16; ++e)
            key[z / (D / 16)][(z % (D / 16)) * 16 + e] = pending[r][e];
      } else if (z < 2 * CHUNKS) {
        int j = z - CHUNKS;
        _Pragma("unroll") for (int e = 0; e < 16; ++e) {
          int d = (j / KV) * 16 + e;
          value[d][swizzle(d, j % KV)] = pending[r][e];
        }
      }
    }
  };
  if (begin < stop)
    prefetch(begin, 0, PF);
  for (I k0 = begin; k0 < stop; k0 += KV) {
    commit(0, PF);
    _Pragma("unroll 1") for (int r = PF; r < PER; ++r) {
      prefetch(k0, r, 1);
      commit(r, 1);
    }
    __syncthreads();
    if (k0 + KV < stop)
      prefetch(k0 + KV, 0, PF);
    F8 s[KT];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      s[kt] = F8{};
      _Pragma("unroll") for (int t = 0; t < DT; ++t) {
        V16 kr;
        _Pragma("unroll") for (int j = 0; j < 16; ++j) kr[j] =
            short(key[kt * 16 + lane][(dw * DT + t) * 16 + j]);
        s[kt] = ring_fp16_wmma(kr, qr[t], s[kt]);
      }
      if constexpr (ND > 1)
        partial[lh][dw][kt][tid % 32] = s[kt];
    }
    if constexpr (ND > 1) {
      __syncthreads();
      // Every output slice must use the same score and denominator.
      _Pragma("unroll") for(int kt=0;kt<KT;++kt) {
        s[kt]=partial[lh][0][kt][tid%32];
        _Pragma("unroll") for(int w=1;w<ND;++w)
          s[kt]+=partial[lh][w][kt][tid%32];
      }
    }
    float m = -INFINITY;
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt)
        _Pragma("unroll") for (int e = 0; e < 8; ++e) {
      I k = k0 + kt * 16 + 2 * e + par;
      float x = s[kt][e] * 0.0625f;
      if (k >= end || k >= C)
        x = -INFINITY;
      s[kt][e] = x;
      m = fmaxf(m, x);
    }
    m = fmaxf(m, __shfl_xor(m, 16, 32));
    float next = fmaxf(maximum, m),
          alpha = next == maximum
                      ? 1.f
                      : (maximum == -INFINITY ? 0.f : expf(maximum - next)),
          sum = 0;
    float prob[KT][16];
    _Pragma("unroll") for (int kt = 0; kt < KT; ++kt) {
      float own[8];
      _Pragma("unroll") for (int e = 0; e < 8; ++e) {
        float p = next == -INFINITY ? 0.f : expf(s[kt][e] - next);
        sum += p;
        own[e] = p;
      }
      _Pragma("unroll") for(int e=0;e<8;++e) {
        float other=__shfl_xor(own[e],16,32);
        prob[kt][2*e]=par?other:own[e];
        prob[kt][2*e+1]=par?own[e]:other;
      }
    }

    sum += __shfl_xor(sum, 16, 32);
    denom = denom * alpha + sum;
    maximum = next;
    if (__any(alpha != 1.f)) {
      _Pragma("unroll") for (int t = 0; t < DT; ++t) out[t] *= alpha;
    }
    _Pragma("unroll") for(int t=0;t<DT;++t) {
      _Pragma("unroll") for(int e=0;e<8;++e) {
        int d=(dw*DT+t)*16+2*e+par;
        float val=out[t][e];
        _Pragma("unroll") for(int kt=0;kt<KT;++kt)
          _Pragma("unroll") for(int k=0;k<16;++k)
            val+=prob[kt][k]*__half2float(__ushort_as_half(value[d][kt*16+swizzle(d,k)]));
        out[t][e]=val;
      }
    }
    __syncthreads();
  }
  if (lane < 6) {
    float *state=parts>1 ? static_cast<float*>(c.workspace)+(I(h)*parts+blockIdx.y)*(D+2) : nullptr;
    if(parts>1 && dw==0 && par==0) {state[D]=maximum;state[D+1]=denom;}
    _Pragma("unroll") for(int t=0;t<DT;++t)
      _Pragma("unroll") for(int e=0;e<8;++e) {
        int d=(dw*DT+t)*16+2*e+par;
        if(parts==1) static_cast<U*>(c.tensors[3].data)[h*D+d]=narrow(out[t][e]/denom);
        else state[d]=out[t][e];
      }
  }
}
__global__ void ring_decode_merge(RingCall c, int parts) {
  constexpr int O = D / NT;
  int tid = threadIdx.x;
  float *state =
      static_cast<float *>(c.workspace) + I(blockIdx.x) * parts * (D + 2);
  __shared__ float weights[16], denominator;
  if (tid == 0) {
    float maximum = -INFINITY, total = 0.f;
    bool bad = false;
    for (int p = 0; p < parts; ++p) {
      float x = state[p * (D + 2) + D];
      bad |= isnan(x);
      maximum = fmaxf(maximum, x);
    }
    for (int p = 0; p < parts; ++p) {
      float x = state[p * (D + 2) + D];
      float w = x == -INFINITY ? 0.f : expf(x - maximum);
      weights[p] = bad ? NAN : w;
      total += weights[p] * state[p * (D + 2) + D + 1];
    }
    denominator = total;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < O; ++j) {
    float out = 0.f;
    int d = tid + j * NT;
    for (int p = 0; p < parts; ++p)
      out += weights[p] * state[p * (D + 2) + d];
    static_cast<U *>(c.tensors[3].data)[I(blockIdx.x) * D + d] =
        narrow(out / denominator);
  }
}
static int ring_decode_launch(RingCall c) {
  int parts = ring_decode_partitions(int(c.P + c.Q));
  hipLaunchKernelGGL(ring_decode_scan, dim3(4, parts), dim3(NT), 0,
                    c.stream, c, parts);
  if (hipPeekAtLastError() != hipSuccess)
    return int(hipGetLastError());
  if (parts > 1)
    hipLaunchKernelGGL(ring_decode_merge, dim3(24), dim3(NT), 0,
                      c.stream, c, parts);
  return int(hipGetLastError());
}
} // namespace decode
} // namespace
#endif

extern "C" HIP_KERNEL_API int hip_gqa_ring_device_supported() {
#if HIP_EP_RING_DEVICE_ENABLED
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess)
    return 0;
  // Cache per calling thread and active device, not process-wide device zero.
  static thread_local int previous = -1;
  static thread_local bool supported = false;
  if (previous != device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return 0;
    supported = std::strncmp(props.gcnArchName, "gfx1151", 7) == 0 &&
                (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':');
    previous = device;
  }
  return supported;
#else
  return 0;
#endif
}

extern "C" HIP_KERNEL_API size_t hip_gqa_ring_workspace(int nq, int total) {
  if (nq != 1 || total < 1 || total > 262144)
    return 0;
  const int parts = ring_decode_partitions(total);
  return parts == 1 ? 0 : size_t(24) * parts * 258 * sizeof(float);
}

extern "C" HIP_KERNEL_API int
hip_gqa_ring_attention(void *stream_ptr, const void *q, const void *k,
                       const void *v, void *o, int nq, int past, int capacity,
                       void *workspace, size_t workspace_bytes) {
#if HIP_EP_RING_DEVICE_ENABLED
  if (!q || !k || !v || !o || nq < 1 || nq > 131072 || past < 0 ||
      past > 262144 - nq || capacity < past + nq ||
      !hip_gqa_ring_device_supported())
    return int(hipErrorInvalidValue);
  const size_t needed = hip_gqa_ring_workspace(nq, past + nq);
  if (needed && (!workspace || workspace_bytes < needed))
    return int(hipErrorInvalidValue);
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  RingCall c = make_call(stream, q, k, v, o, nq, past, capacity, workspace);
  (void)hipGetLastError();
  if (nq == 1)
    return decode::ring_decode_launch(c);
#if HIP_EP_GQA_IMPL_ID == 1
  hipLaunchKernelGGL(prefill::ring_online, dim3(((nq + 15) / 16) * 4),
                     dim3(prefill::NT), 0, stream, c);
#else
  hipLaunchKernelGGL(prefill::ring_kv_outer_parallel,
                     dim3(((nq + 31) / 32) * 4), dim3(prefill::NT), 0, stream,
                     c);
#endif
  return int(hipGetLastError());
#else
  return int(hipErrorNotSupported);
#endif
}
