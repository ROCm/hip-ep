/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Frozen Qwen math, qualified for wave32. Native attention/KV are FP16;
// selector projection/features/pools are BF16 with FP32 reductions.
#include "hip_qwen_qsa.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
using BF = hip_bfloat16;
using NF = __half;
constexpr int Threads = 256, Waves = 8, Budget = 512, Groups = 513, MaxSlots = 2052;
__device__ float wave_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down(value, offset, 32);
  return __shfl(value, 0, 32);
}
__device__ float block_sum(float value, float *scratch) {
  value = wave_sum(value);
  if (!(threadIdx.x & 31)) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  if (!threadIdx.x) {
    float sum = 0;
    for (int i = 0; i < Waves; ++i) sum += scratch[i];
    scratch[0] = sum;
  }
  __syncthreads();
  return scratch[0];
}
__device__ int block_scan(int value, int *scratch) {
  int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
  for (int offset = 1; offset < 32; offset <<= 1) {
    int previous = __shfl_up(value, offset, 32);
    if (lane >= offset) value += previous;
  }
  if (lane == 31) scratch[wave] = value;
  __syncthreads();
  for (int i = 0; i < wave; ++i) value += scratch[i];
  __syncthreads();
  return value;
}
__device__ void validate_position(const int64_t *positions, unsigned *error,
    int B, int S, int b, int j, int past) {
  for (int axis = 0; axis < 3; ++axis)
    if (positions[(int64_t(axis) * B + b) * S + j] != past + j)
      atomicOr(error, 1u);
}
__global__ void cast_hidden(const NF *input, const int64_t *positions,
    BF *output, unsigned *error, int B, int S, int past) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t size = int64_t(B) * S * 5120;
  if (index >= size) return;
  if (index % 5120 == 0) {
    int row = int(index / 5120);
    validate_position(positions, error, B, S, row / S, row % S, past);
  }
  output[index] = BF(__half2float(input[index]));
}
__global__ void project_decode(const NF *hidden, const BF *weight,
    const int64_t *positions, BF *output, unsigned *error, int B, int past) {
  __shared__ float partial[Threads];
  int row = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
  if (!row && !t) validate_position(positions, error, B, 1, b, 0, past);
  float sum = 0;
  for (int k = t; k < 5120; k += Threads)
    sum += float(BF(__half2float(hidden[int64_t(b) * 5120 + k]))) *
           float(weight[int64_t(row) * 5120 + k]);
  partial[t] = sum;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (t < offset) partial[t] += partial[t + offset];
    __syncthreads();
  }
  if (!t) output[b * 640 + row] = BF(partial[0]);
}
__device__ void normalize_four(float *values, const BF *weight, int lane) {
  float square = 0;
  for (int i = 0; i < 4; ++i) square += values[i] * values[i];
  float inverse = rsqrtf(wave_sum(square) / 128.0f + 1e-6f);
  for (int i = 0; i < 4; ++i)
    values[i] = float(BF(values[i] * inverse * (1.0f + float(weight[lane + i * 32]))));
}
__device__ void rotate_four(float *values, const float *cos, const float *sin,
    int position, int lane) {
  float a = values[0], b = values[1];
  float c = cos[int64_t(position) * 32 + lane];
  float s = sin[int64_t(position) * 32 + lane];
  values[0] = float(BF(a * c - b * s));
  values[1] = float(BF(b * c + a * s));
}
__global__ void feature_kernel(BF *projected, const BF *q_norm,
    const BF *token_norm, const BF *pool_norm, const float *cos,
    const float *sin, BF *queries, BF *pool, BF *tail,
    int B, int S, int past, int P) {
  int b = blockIdx.y, j = blockIdx.x, head = threadIdx.x / 32;
  int lane = threadIdx.x & 31, position = past + j;
  int64_t row = (int64_t(b) * S + j) * 640;
  int base = head < 4 ? head * 128 : 512;
  float values[4];
  for (int i = 0; i < 4; ++i) values[i] = float(projected[row + base + lane + i * 32]);
  normalize_four(values, head < 4 ? q_norm : token_norm, lane);
  if (head < 4) {
    rotate_four(values, cos, sin, position, lane);
    for (int i = 0; i < 4; ++i)
      queries[(int64_t(b) * S + j) * 512 + head * 128 + lane + i * 32] = BF(values[i]);
    return;
  }
  for (int i = 0; i < 4; ++i)
    projected[row + 512 + lane + i * 32] = BF(values[i]);
  bool decode = S == 1 && past != 0;
  bool incomplete = !past && S % 4 && j / 4 == S / 4;
  if (decode || incomplete || (S == 1 && !past))
    for (int i = 0; i < 4; ++i)
      tail[(int64_t(b) * 4 + position % 4) * 128 + lane + i * 32] = BF(values[i]);
  if (decode && position % 4 == 3) {
    float average[4];
    for (int i = 0; i < 4; ++i) {
      float sum = 0;
      for (int token = 0; token < 3; ++token)
        sum += float(tail[(int64_t(b) * 4 + token) * 128 + lane + i * 32]);
      sum += values[i];
      average[i] = float(BF(sum * 0.25f));
    }
    normalize_four(average, pool_norm, lane);
    rotate_four(average, cos, sin, position - 3, lane);
    for (int i = 0; i < 4; ++i)
      pool[(int64_t(b) * P + position / 4) * 128 + lane + i * 32] = BF(average[i]);
  }
}
__global__ void pool_prefill(const BF *projected, const BF *weight,
    const float *cos, const float *sin, BF *pool, int S, int P) {
  int group = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;
  float values[4];
  for (int i = 0; i < 4; ++i) {
    float sum = 0;
    for (int token = 0; token < 4; ++token)
      sum += float(projected[(int64_t(b) * S + group * 4 + token) * 640 + 512 + lane + i * 32]);
    values[i] = float(BF(sum * 0.25f));
  }
  normalize_four(values, weight, lane);
  rotate_four(values, cos, sin, group * 4, lane);
  for (int i = 0; i < 4; ++i)
    pool[(int64_t(b) * P + group) * 128 + lane + i * 32] = BF(values[i]);
}
__global__ void pack_kv(const NF *key, const NF *value, NF *cache_key,
    NF *cache_value, int B, int S, int past, int C) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(B) * S * 1024) return;
  int d = index % 256, head = (index / 256) % 4;
  int j = (index / 1024) % S, b = index / (int64_t(S) * 1024);
  int64_t target = ((int64_t(b) * 4 + head) * C + past + j) * 256 + d;
  cache_key[target] = key[index];
  cache_value[target] = value[index];
}
__global__ void score_kernel(const BF *queries, const BF *pool, float *scores,
    int S, int past, int first, int count, int P) {
  int group = blockIdx.x * 16 + threadIdx.x / 16;
  int j = blockIdx.y, b = blockIdx.z, lane = threadIdx.x % 16;
  int historical = (past + first + j) / 4;
  if (group >= historical) return;
  const BF *query = queries + (int64_t(b) * S + first + j) * 512;
  const BF *key = pool + (int64_t(b) * P + group) * 128;
  float sums[4] = {0, 0, 0, 0};
  for (int d = lane; d < 128; d += 16) {
    float value = float(key[d]);
    for (int head = 0; head < 4; ++head) sums[head] += float(query[head * 128 + d]) * value;
  }
  float total = 0;
  for (int head = 0; head < 4; ++head) {
    for (int offset = 8; offset; offset >>= 1) sums[head] += __shfl_down(sums[head], offset, 16);
    total += fmaxf(sums[head], 0.0f);
  }
  if (!lane) scores[(int64_t(b) * count + j) * P + group] = total * 0.08838834764831845f;
}
__global__ void rank_kernel(const float *all_scores, int *all_ids,
    int past, int first, int count, int P) {
  int t = threadIdx.x, j = blockIdx.x, b = blockIdx.y;
  int historical = (past + first + j) / 4;
  const float *scores = all_scores + (int64_t(b) * count + j) * P;
  int *ids = all_ids + (int64_t(b) * count + j) * Groups;
  for (int i = t; i < Groups; i += Threads) ids[i] = -1;
  if (historical <= Budget) {
    for (int group = t; group <= historical; group += Threads) ids[group] = group;
    return;
  }
  __shared__ int histogram[Waves * 256 + 2];
  __shared__ float reduction[Waves];
  if (!t) { histogram[Waves * 256] = 0; histogram[Waves * 256 + 1] = Budget; }
  __syncthreads();
  uint32_t mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = t; i < Waves * 256; i += Threads) histogram[i] = 0;
    __syncthreads();
    uint32_t prefix = uint32_t(histogram[Waves * 256]);
    for (int group = t; group < historical; group += Threads) {
      uint32_t bits = __float_as_uint(scores[group]);
      int bin = (bits & mask) == prefix ? int((bits >> shift) & 255) : 256;
      int first_bin = __shfl(bin, 0, 32);
      bool same = __all(bin == first_bin);
      unsigned active = __popc(static_cast<unsigned>(__ballot(1)));
      if (same) {
        if (!(t & 31) && bin < 256) atomicAdd(&histogram[(t / 32) * 256 + bin], active);
      } else if (bin < 256) atomicAdd(&histogram[(t / 32) * 256 + bin], 1);
    }
    __syncthreads();
    int value = 0;
    for (int wave = 0; wave < Waves; ++wave) value += histogram[wave * 256 + t];
    __syncthreads();
    histogram[t] = value;
    __syncthreads();
    if (!t) {
      int rank = histogram[Waves * 256 + 1];
      for (int bin = 255; bin >= 0; --bin) {
        if (rank > histogram[bin]) rank -= histogram[bin];
        else { histogram[Waves * 256] = int(prefix | (uint32_t(bin) << shift)); histogram[Waves * 256 + 1] = rank; break; }
      }
    }
    __syncthreads();
    mask |= 255u << shift;
  }
  uint32_t threshold = uint32_t(histogram[Waves * 256]);
  int span = (historical + Threads - 1) / Threads;
  int begin = t * span, end = min(historical, begin + span), equal = 0, greater = 0;
  for (int group = begin; group < end; ++group) {
    uint32_t bits = __float_as_uint(scores[group]);
    equal += bits == threshold; greater += bits > threshold;
  }
  int available = Budget - int(block_sum(float(greater), reduction));
  __syncthreads();
  int equal_base = block_scan(equal, histogram) - equal;
  int take = greater + max(0, min(equal, available - equal_base));
  int destination = block_scan(take, histogram) - take;
  for (int group = begin; group < end; ++group) {
    uint32_t bits = __float_as_uint(scores[group]);
    bool tied = bits == threshold;
    bool keep = bits > threshold || (tied && equal_base < available);
    equal_base += tied;
    if (keep) ids[destination++] = group;
  }
  if (!t) ids[Budget] = historical;
}
struct Core {
  const NF *query, *key, *value;
  const int *ids;
  NF *output;
  float *partial, *stats;
  int B, S, past, first, count, C, splits;
};
__global__ void attend_kernel(Core p) {
  int t = threadIdx.x, head = blockIdx.x / p.splits, part = blockIdx.x % p.splits;
  int j = blockIdx.y, b = blockIdx.z, position = p.past + p.first + j;
  int slots = min(position / 4, Budget) * 4 + position % 4 + 1;
  int span = (slots + p.splits - 1) / p.splits;
  int begin = part * span, end = min(slots, begin + span), count = max(0, end - begin);
  const int *ids = p.ids + (int64_t(b) * p.count + j) * Groups;
  const NF *query = p.query + (int64_t(b) * p.S + p.first + j) * 6144 + head * 256;
  const NF *key = p.key + (int64_t(b) * 4 + head / 6) * p.C * 256;
  const NF *value = p.value + (int64_t(b) * 4 + head / 6) * p.C * 256;
  __shared__ float logits[MaxSlots], accum[Waves * 256], reduction[Waves];
  int lane = t % 16;
  float q[16];
  for (int d = 0; d < 16; ++d) q[d] = __half2float(query[lane * 16 + d]);
  for (int i = t / 16; i < count; i += 16) {
    int slot = begin + i, address = ids[slot / 4] * 4 + slot % 4;
    float dot = 0;
    for (int d = 0; d < 16; ++d) dot += q[d] * __half2float(key[int64_t(address) * 256 + lane * 16 + d]);
    for (int offset = 8; offset; offset >>= 1) dot += __shfl_down(dot, offset, 16);
    if (!lane) logits[i] = dot * 0.0625f;
  }
  __syncthreads();
  float maximum = -INFINITY;
  for (int i = t; i < count; i += Threads) maximum = fmaxf(maximum, logits[i]);
  for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
  if (!(t & 31)) reduction[t / 32] = maximum;
  __syncthreads();
  if (!t) { float m = -INFINITY; for (int i = 0; i < Waves; ++i) m = fmaxf(m, reduction[i]); reduction[0] = m; }
  __syncthreads(); maximum = reduction[0]; __syncthreads();
  float sum = 0;
  for (int i = t; i < count; i += Threads) { float weight = expf(logits[i] - maximum); logits[i] = weight; sum += weight; }
  float denominator = block_sum(sum, reduction);
  __syncthreads();
  int wave = t / 32, dimension = (t & 31) * 8;
  float values[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = wave; i < count; i += Waves) {
    int slot = begin + i, address = ids[slot / 4] * 4 + slot % 4;
    float weight = logits[i];
    for (int z = 0; z < 8; ++z) values[z] += weight * __half2float(value[int64_t(address) * 256 + dimension + z]);
  }
  for (int z = 0; z < 8; ++z) accum[wave * 256 + dimension + z] = values[z];
  __syncthreads();
  float result = 0;
  for (int i = 0; i < Waves; ++i) result += accum[i * 256 + t];
  int64_t row = (int64_t(b) * p.count + j) * 24 + head;
  if (p.splits == 1) p.output[(int64_t(b) * p.S + p.first + j) * 6144 + head * 256 + t] = __float2half(result / denominator);
  else { int64_t item = row * p.splits + part; p.partial[item * 256 + t] = result; if (!t) { p.stats[item * 2] = maximum; p.stats[item * 2 + 1] = denominator; } }
}
__global__ void combine_kernel(Core p) {
  int row = blockIdx.x, t = threadIdx.x;
  __shared__ float weights[17];
  if (!t) {
    float maximum = -INFINITY;
    for (int i = 0; i < p.splits; ++i) maximum = fmaxf(maximum, p.stats[(int64_t(row) * p.splits + i) * 2]);
    float denominator = 0;
    for (int i = 0; i < p.splits; ++i) { float weight = expf(p.stats[(int64_t(row) * p.splits + i) * 2] - maximum); weights[i] = weight; denominator += weight * p.stats[(int64_t(row) * p.splits + i) * 2 + 1]; }
    weights[16] = denominator;
  }
  __syncthreads();
  float value = 0;
  for (int i = 0; i < p.splits; ++i) value += p.partial[(int64_t(row) * p.splits + i) * 256 + t] * weights[i];
  int head = row % 24, j = (row / 24) % p.count, b = row / (24 * p.count);
  p.output[(int64_t(b) * p.S + p.first + j) * 6144 + head * 256 + t] = __float2half(value / weights[16]);
}
bool geometry(int B, int S, int past) { return B >= 1 && B <= 8 && S >= 1 && past >= 0 && int64_t(past) + S <= 262144; }
} // namespace

extern "C" int hip_qwen_qsa_cast_hidden(void *stream, const void *hidden,
    const void *positions, void *converted, void *error, int B, int S, int past) {
  if (!geometry(B, S, past) || !hidden || !positions || !converted || !error) return int(hipErrorInvalidValue);
  int64_t size = int64_t(B) * S * 5120;
  hipLaunchKernelGGL(cast_hidden, dim3((size + 255) / 256), dim3(256), 0, static_cast<hipStream_t>(stream), static_cast<const NF *>(hidden), static_cast<const int64_t *>(positions), static_cast<BF *>(converted), static_cast<unsigned *>(error), B, S, past);
  return int(hipGetLastError());
}
extern "C" int hip_qwen_qsa_project_decode(void *stream, const void *hidden,
    const void *weight, const void *positions, void *projected, void *error, int B, int past) {
  if (!geometry(B, 1, past) || !hidden || !weight || !positions || !projected || !error) return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(project_decode, dim3(640, B), dim3(256), 0, static_cast<hipStream_t>(stream), static_cast<const NF *>(hidden), static_cast<const BF *>(weight), static_cast<const int64_t *>(positions), static_cast<BF *>(projected), static_cast<unsigned *>(error), B, past);
  return int(hipGetLastError());
}
extern "C" int hip_qwen_qsa_features(void *stream, void *projected,
    const void *q_norm, const void *token_norm, const void *pool_norm,
    const void *cos, const void *sin, void *queries, void *pool, void *tail,
    int B, int S, int past, int P) {
  if (!geometry(B, S, past) || P < (past + S + 3) / 4 || !projected || !q_norm || !token_norm || !pool_norm || !cos || !sin || !queries || !pool || !tail || (past && S != 1)) return int(hipErrorInvalidValue);
  auto s = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(feature_kernel, dim3(S, B), dim3(160), 0, s, static_cast<BF *>(projected), static_cast<const BF *>(q_norm), static_cast<const BF *>(token_norm), static_cast<const BF *>(pool_norm), static_cast<const float *>(cos), static_cast<const float *>(sin), static_cast<BF *>(queries), static_cast<BF *>(pool), static_cast<BF *>(tail), B, S, past, P);
  if (auto error = hipGetLastError()) return int(error);
  if (!past && S / 4 > 0) hipLaunchKernelGGL(pool_prefill, dim3(S / 4, B), dim3(32), 0, s, static_cast<const BF *>(projected), static_cast<const BF *>(pool_norm), static_cast<const float *>(cos), static_cast<const float *>(sin), static_cast<BF *>(pool), S, P);
  return int(hipGetLastError());
}
extern "C" int hip_qwen_qsa_pack_kv(void *stream, const void *key,
    const void *value, void *cache_key, void *cache_value, int B, int S, int past, int C) {
  if (!geometry(B, S, past) || C < past + S || C > 262144 || !key || !value || !cache_key || !cache_value) return int(hipErrorInvalidValue);
  int64_t size = int64_t(B) * S * 1024;
  hipLaunchKernelGGL(pack_kv, dim3((size + 255) / 256), dim3(256), 0, static_cast<hipStream_t>(stream), static_cast<const NF *>(key), static_cast<const NF *>(value), static_cast<NF *>(cache_key), static_cast<NF *>(cache_value), B, S, past, C);
  return int(hipGetLastError());
}
extern "C" int hip_qwen_qsa_select(void *stream, const void *queries,
    const void *pool, void *scores, void *selected, int B, int S, int past, int first, int count, int P) {
  if (!geometry(B, S, past) || first < 0 || count < 1 || count > 32 || first + count > S || P < (past + S + 3) / 4 || !queries || !pool || !scores || !selected) return int(hipErrorInvalidValue);
  auto s = static_cast<hipStream_t>(stream);
  int maximum = (past + first + count - 1) / 4;
  if (maximum > Budget) hipLaunchKernelGGL(score_kernel, dim3((maximum + 15) / 16, count, B), dim3(Threads), 0, s, static_cast<const BF *>(queries), static_cast<const BF *>(pool), static_cast<float *>(scores), S, past, first, count, P);
  if (auto error = hipGetLastError()) return int(error);
  hipLaunchKernelGGL(rank_kernel, dim3(count, B), dim3(Threads), 0, s, static_cast<const float *>(scores), static_cast<int *>(selected), past, first, count, P);
  return int(hipGetLastError());
}
extern "C" int hip_qwen_qsa_attend(void *stream, const void *query,
    const void *key, const void *value, const void *selected, void *partial,
    void *stats, void *output, int B, int S, int past, int first, int count, int C) {
  if (!geometry(B, S, past) || C < past + S || C > 262144 || first < 0 || count < 1 || count > 32 || first + count > S || !query || !key || !value || !selected || !partial || !stats || !output) return int(hipErrorInvalidValue);
  Core p{static_cast<const NF *>(query), static_cast<const NF *>(key), static_cast<const NF *>(value), static_cast<const int *>(selected), static_cast<NF *>(output), static_cast<float *>(partial), static_cast<float *>(stats), B, S, past, first, count, C, S == 1 ? 16 : 1};
  auto s = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(attend_kernel, dim3(24 * p.splits, count, B), dim3(Threads), 0, s, p);
  if (auto error = hipGetLastError()) return int(error);
  if (p.splits > 1) hipLaunchKernelGGL(combine_kernel, dim3(B * count * 24), dim3(Threads), 0, s, p);
  return int(hipGetLastError());
}
