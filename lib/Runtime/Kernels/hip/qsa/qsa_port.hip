/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Frozen OLMo selector mathematics adapted from sparse-attn qsa_fused.cu.
 * Wave32 staged launches replace its MI355X cooperative wave64 grid. Backbone
 * Q/K norms and FP32 RoPE remain in the shared hip-ep graph on both paths.
 */
#include "hip_qsa.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
using BF = hip_bfloat16;
constexpr int NT = 256, WAVES = 8, PARTS = 8;
constexpr float SCALE = 0.08838834764831845f;

__device__ float block_sum(float x, float *scratch) {
  for (int s = 16; s; s >>= 1) x += __shfl_down(x, s, 32);
  if (!(threadIdx.x & 31)) scratch[threadIdx.x / 32] = x;
  __syncthreads();
  if (!threadIdx.x) {
    float total = 0;
    for (int w = 0; w < WAVES; ++w) total += scratch[w];
    scratch[0] = total;
  }
  __syncthreads();
  return scratch[0];
}

__device__ int block_scan(int x, int *scratch) {
  int lane = threadIdx.x & 31;
  for (int s = 1; s < 32; s <<= 1) {
    int other = __shfl_up(x, s, 32);
    if (lane >= s) x += other;
  }
  if (lane == 31) scratch[threadIdx.x / 32] = x;
  __syncthreads();
  for (int w = 0; w < int(threadIdx.x) / 32; ++w) x += scratch[w];
  __syncthreads();
  return x;
}

__global__ void append_kv(const __half *k, const __half *v, __half *ks,
                          __half *vs, int B, int S, int past, int C) {
  int64_t i = int64_t(blockIdx.x) * NT + threadIdx.x;
  if (i >= int64_t(B) * S * 4096) return;
  int d = i % 128, h = (i / 128) % 32;
  int s = (i / 4096) % S, b = i / (int64_t(S) * 4096);
  int64_t dest = ((int64_t(b) * 32 + h) * C + past + s) * 128 + d;
  ks[dest] = k[i];
  vs[dest] = v[i];
}

template <bool POOL>
__global__ void make_features(const BF *projected, const BF *weight,
                             const float *cos, const float *sin, BF *output,
                             BF *tail, int S, int past, int P) {
  int t = threadIdx.x, b = blockIdx.z, row = blockIdx.x, head = blockIdx.y;
  int position = POOL ? (past / 4 + row) * 4 : past + row;
  __shared__ float sm[384];
  if (t < 128) {
    if constexpr (POOL) {
      float total = 0;
      for (int j = 0; j < 4; ++j) {
        int pos = position + j;
        if (pos >= past + S) continue;
        BF raw;
        if (pos < past) raw = tail[(int64_t(b) * 4 + j) * 128 + t];
        else raw = projected[(int64_t(b) * S + pos - past) * 640 + 512 + t];
        total += float(raw);
        if ((position / 4) == (past + S - 1) / 4 && pos >= past)
          tail[(int64_t(b) * 4 + j) * 128 + t] = raw;
      }
      sm[t] = float(BF(total * 0.25f));
    } else {
      sm[t] = float(projected[(int64_t(b) * S + row) * 640 + head * 128 + t]);
    }
    sm[128 + t] = sm[t] * sm[t];
  }
  // An incomplete group only updates the <=3-token tail, never a pooled key.
  if constexpr (POOL) if (position + 4 > past + S) return;
  __syncthreads();
  for (int step = 64; step; step >>= 1) {
    if (t < step) sm[128 + t] += sm[128 + t + step];
    __syncthreads();
  }
  if (t < 128)
    sm[256 + t] = float(BF(sm[t] * rsqrtf(sm[128] / 128.0f + 1e-6f) * float(weight[t])));
  __syncthreads();
  if (t < 128) {
    float other = t < 64 ? -sm[256 + t + 64] : sm[256 + t - 64];
    int64_t rotary = int64_t(position) * 64 + t % 64;
    BF value = BF(sm[256 + t] * cos[rotary] + other * sin[rotary]);
    if constexpr (POOL) output[(int64_t(b) * P + position / 4) * 128 + t] = value;
    else output[((int64_t(b) * S + row) * 4 + head) * 128 + t] = value;
  }
}

__global__ void score_select(const BF *features, const BF *pool, float *all_scores,
                             int *all_ids, int S, int past, int first, int chunk,
                             int P) {
  int t = threadIdx.x, row = blockIdx.x, b = blockIdx.y;
  int query = first + row, groups = (past + query + 1) / 4;
  int *ids = all_ids + (int64_t(b) * chunk + row) * 128;
  if (groups <= 128) {
    if (t < 128) ids[t] = t < groups ? t : -1;
    return;
  }
  float *scores = all_scores + (int64_t(b) * chunk + row) * P;
  __shared__ int hist[WAVES * 256 + 16];
  __shared__ float red[WAVES];
  int lane = t % 16;
  for (int g = t / 16; g < groups; g += NT / 16) {
    float dots[4] = {0, 0, 0, 0};
    for (int d = lane; d < 128; d += 16) {
      float key = float(pool[(int64_t(b) * P + g) * 128 + d]);
      for (int h = 0; h < 4; ++h)
        dots[h] += key * float(features[((int64_t(b) * S + query) * 4 + h) * 128 + d]);
    }
    float total = 0;
    for (int h = 0; h < 4; ++h) {
      for (int s = 8; s; s >>= 1) dots[h] += __shfl_down(dots[h], s, 16);
      total += fmaxf(dots[h], 0.0f);
    }
    if (!lane) scores[g] = total * SCALE;
  }
  __syncthreads();
  if (!t) { hist[WAVES * 256] = 0; hist[WAVES * 256 + 1] = 128; }
  __syncthreads();
  uint32_t mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = t; i < WAVES * 256; i += NT) hist[i] = 0;
    __syncthreads();
    uint32_t prefix = uint32_t(hist[WAVES * 256]);
    for (int g = t; g < groups; g += NT) {
      uint32_t bits = __float_as_uint(scores[g]);
      if ((bits & mask) == prefix)
        atomicAdd(&hist[(t / 32) * 256 + ((bits >> shift) & 255)], 1);
    }
    __syncthreads();
    int count = 0;
    for (int w = 0; w < WAVES; ++w) count += hist[w * 256 + t];
    __syncthreads();
    hist[t] = count;
    __syncthreads();
    if (!t) {
      int rank = hist[WAVES * 256 + 1];
      for (int bin = 255; bin >= 0; --bin) {
        if (rank > hist[bin]) rank -= hist[bin];
        else {
          hist[WAVES * 256] = int(prefix | (uint32_t(bin) << shift));
          hist[WAVES * 256 + 1] = rank;
          break;
        }
      }
    }
    __syncthreads();
    mask |= 255u << shift;
  }
  uint32_t threshold = uint32_t(hist[WAVES * 256]);
  int span = (groups + NT - 1) / NT, begin = t * span;
  int end = min(groups, begin + span), eq = 0, gt = 0;
  for (int g = begin; g < end; ++g) {
    uint32_t value = __float_as_uint(scores[g]);
    eq += value == threshold;
    gt += value > threshold;
  }
  int budget = 128 - int(block_sum(float(gt), red));
  __syncthreads();
  int eqbase = block_scan(eq, hist) - eq;
  int take = gt + max(0, min(eq, budget - eqbase));
  int dest = block_scan(take, hist) - take;
  for (int g = begin; g < end; ++g) {
    uint32_t value = __float_as_uint(scores[g]);
    bool tied = value == threshold;
    bool keep = value > threshold || (tied && eqbase < budget);
    eqbase += tied;
    if (keep) ids[dest++] = g;
  }
}

__device__ int selected_token(int slot, const int *ids, int groups) {
  int chosen = min(groups, 128);
  return slot < chosen * 4 ? ids[slot / 4] * 4 + slot % 4
                           : groups * 4 + slot - chosen * 4;
}

__global__ void attend_parts(const __half *query, const __half *keys,
                             const __half *values, const int *all_ids,
                             float *partials, float *stats, int S, int past,
                             int first, int chunk, int C) {
  int t = threadIdx.x, task = blockIdx.x, row = blockIdx.y, b = blockIdx.z;
  int head = task / PARTS, part = task % PARTS, qrow = first + row;
  int position = past + qrow, groups = (position + 1) / 4;
  int slots = min(groups, 128) * 4 + (position + 1) % 4;
  int span = (slots + PARTS - 1) / PARTS, begin = part * span;
  int end = min(slots, begin + span);
  int64_t work = ((int64_t(b) * chunk + row) * 32 + head) * PARTS + part;
  if (begin >= end) {
    if (t < 128) partials[work * 128 + t] = 0;
    if (!t) { stats[work * 2] = -INFINITY; stats[work * 2 + 1] = 0; }
    return;
  }
  const int *ids = all_ids + (int64_t(b) * chunk + row) * 128;
  __shared__ float sm[128 + WAVES * 128];
  __shared__ float red[WAVES];
  int lane = t % 16;
  float q[8];
  for (int j = 0; j < 8; ++j)
    q[j] = __half2float(query[(int64_t(b) * S + qrow) * 4096 + head * 128 + lane * 8 + j]);
  for (int i = t / 16; i < span; i += NT / 16) {
    int slot = begin + i;
    float dot = 0;
    if (slot < end) {
      int token = selected_token(slot, ids, groups);
      for (int j = 0; j < 8; ++j)
        dot += q[j] * __half2float(keys[((int64_t(b) * 32 + head) * C + token) * 128 + lane * 8 + j]);
    }
    for (int s = 8; s; s >>= 1) dot += __shfl_down(dot, s, 16);
    if (!lane) sm[i] = slot < end ? dot * SCALE : -INFINITY;
  }
  __syncthreads();
  float mx = t < span ? sm[t] : -INFINITY;
  for (int s = 16; s; s >>= 1) mx = fmaxf(mx, __shfl_down(mx, s, 32));
  if (!(t & 31)) red[t / 32] = mx;
  __syncthreads();
  if (!t) {
    float m = -INFINITY;
    for (int w = 0; w < WAVES; ++w) m = fmaxf(m, red[w]);
    red[0] = m;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float weight = t < span ? expf(sm[t] - mx) : 0;
  if (t < span) sm[t] = weight;
  float denom = block_sum(weight, red);
  __syncthreads();
  int wave = t / 32, d = (t & 31) * 4;
  float result[4] = {0, 0, 0, 0};
  for (int i = wave; begin + i < end; i += WAVES) {
    int token = selected_token(begin + i, ids, groups);
    for (int j = 0; j < 4; ++j)
      result[j] += sm[i] * __half2float(values[((int64_t(b) * 32 + head) * C + token) * 128 + d + j]);
  }
  for (int j = 0; j < 4; ++j) sm[128 + wave * 128 + d + j] = result[j];
  __syncthreads();
  if (t < 128) {
    float total = 0;
    for (int w = 0; w < WAVES; ++w) total += sm[128 + w * 128 + t];
    partials[work * 128 + t] = total;
  }
  if (!t) { stats[work * 2] = mx; stats[work * 2 + 1] = denom; }
}

__global__ void merge_parts(const float *partial, const float *stats,
                            __half *output, int S, int first, int chunk) {
  int d = threadIdx.x, head = blockIdx.x, row = blockIdx.y, b = blockIdx.z;
  int64_t work = ((int64_t(b) * chunk + row) * 32 + head) * PARTS;
  __shared__ float weights[PARTS + 1];
  if (!d) {
    float mx = -INFINITY;
    for (int p = 0; p < PARTS; ++p) mx = fmaxf(mx, stats[(work + p) * 2]);
    float denom = 0;
    for (int p = 0; p < PARTS; ++p) {
      weights[p] = expf(stats[(work + p) * 2] - mx);
      denom += weights[p] * stats[(work + p) * 2 + 1];
    }
    weights[PARTS] = denom;
  }
  __syncthreads();
  float numerator = 0;
  for (int p = 0; p < PARTS; ++p) numerator += partial[(work + p) * 128 + d] * weights[p];
  output[(int64_t(b) * S + first + row) * 4096 + head * 128 + d] = __float2half(numerator / weights[PARTS]);
}

bool valid(int B, int S, int past) {
  return B == 1 && S > 0 && past >= 0 && (past == 0 || S == 1);
}
} // namespace

extern "C" int hip_qsa_features(void *stream, const void *projected,
    const void *qw, const void *kw, const void *cos, const void *sin,
    void *features, void *pool, void *tail, const void *key, const void *value,
    void *keys, void *values, int B, int S, int past, int C, int P, float eps) {
  if (!valid(B, S, past) || past + S > C || C > 65536 || P < (C + 3) / 4 || eps != 1e-6f)
    return int(hipErrorInvalidValue);
  hipStream_t st = static_cast<hipStream_t>(stream);
  hipGetLastError();
  hipLaunchKernelGGL(append_kv, dim3((int64_t(B) * S * 4096 + NT - 1) / NT), dim3(NT), 0, st,
      static_cast<const __half *>(key), static_cast<const __half *>(value), static_cast<__half *>(keys), static_cast<__half *>(values), B, S, past, C);
  hipError_t error = hipGetLastError(); if (error != hipSuccess) return int(error);
  hipLaunchKernelGGL((make_features<false>), dim3(S, 4, B), dim3(NT), 0, st,
      static_cast<const BF *>(projected), static_cast<const BF *>(qw), static_cast<const float *>(cos), static_cast<const float *>(sin),
      static_cast<BF *>(features), static_cast<BF *>(tail), S, past, P);
  error = hipGetLastError(); if (error != hipSuccess) return int(error);
  int groups = (past + S - 1) / 4 - past / 4 + 1;
  hipLaunchKernelGGL((make_features<true>), dim3(groups, 1, B), dim3(NT), 0, st,
      static_cast<const BF *>(projected), static_cast<const BF *>(kw), static_cast<const float *>(cos), static_cast<const float *>(sin),
      static_cast<BF *>(pool), static_cast<BF *>(tail), S, past, P);
  return int(hipGetLastError());
}

extern "C" int hip_qsa_select(void *stream, const void *features,
    const void *pool, void *scores, void *selected, int B, int S, int past,
    int first, int chunk, int P) {
  if (!valid(B, S, past) || first < 0 || chunk < 1 || first + chunk > S || (past + S) / 4 > P)
    return int(hipErrorInvalidValue);
  hipGetLastError();
  hipLaunchKernelGGL(score_select, dim3(chunk, B), dim3(NT), 0, static_cast<hipStream_t>(stream),
      static_cast<const BF *>(features), static_cast<const BF *>(pool), static_cast<float *>(scores),
      static_cast<int *>(selected), S, past, first, chunk, P);
  return int(hipGetLastError());
}

extern "C" int hip_qsa_attend(void *stream, const void *query, const void *keys,
    const void *values, const void *selected, void *partial, void *stats,
    void *output, int B, int S, int past, int first, int chunk, int C) {
  if (!valid(B, S, past) || past + S > C || first < 0 || chunk < 1 || first + chunk > S)
    return int(hipErrorInvalidValue);
  hipStream_t st = static_cast<hipStream_t>(stream);
  hipGetLastError();
  hipLaunchKernelGGL(attend_parts, dim3(32 * PARTS, chunk, B), dim3(NT), 0, st,
      static_cast<const __half *>(query), static_cast<const __half *>(keys), static_cast<const __half *>(values),
      static_cast<const int *>(selected), static_cast<float *>(partial), static_cast<float *>(stats), S, past, first, chunk, C);
  hipError_t error = hipGetLastError(); if (error != hipSuccess) return int(error);
  hipLaunchKernelGGL(merge_parts, dim3(32, chunk, B), dim3(128), 0, st,
      static_cast<const float *>(partial), static_cast<const float *>(stats), static_cast<__half *>(output), S, first, chunk);
  return int(hipGetLastError());
}
