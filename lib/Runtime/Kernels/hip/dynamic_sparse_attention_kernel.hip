/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft DynamicSparseAttention, attention_mode = selected_only over the
 * main KV cache: every query row attends to its own list of selected cache
 * positions. Numerics follow the ORT CUDA kernel: fp32 logits and online
 * softmax, invalid candidates skipped, empty candidate set -> zero output.
 */

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kTile = 16;
constexpr int kBkv = 32;
constexpr int kMaxSplits = 128;
constexpr int kTargetBlocks = 256;
constexpr int kAppendThreads = 256;
constexpr float kLog2e = 1.4426950408889634f;

typedef _Float16 frag_t
    __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float acc_t __attribute__((ext_vector_type(8)));
typedef unsigned int vec16_t __attribute__((ext_vector_type(4)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

#if HIPDNN_HAS_WMMA
#define DSA_WMMA(a, b, c) HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)
#else
__device__ __forceinline__ acc_t dsa_wmma_unavailable(frag_t, frag_t, acc_t c) {
  __builtin_trap();
  return c;
}
#define DSA_WMMA(a, b, c) dsa_wmma_unavailable((a), (b), (c))
#endif

__device__ __forceinline__ frag_t load_frag(const _Float16 *p, int pair) {
  return *reinterpret_cast<const frag_t *>(p + hipdnn_wmma_k_off(pair));
}

// Writes the new tokens' K/V (BSHD) into the BNSH present cache at
// seqlens_k[b] + 1 - S + s, dropping only the tokens that land outside the
// cache. One thread moves one 16-byte chunk of K and the same chunk of V.
__global__ void __launch_bounds__(kAppendThreads)
dsa_kv_append_kernel(const vec16_t *__restrict__ key,
                     const vec16_t *__restrict__ value,
                     vec16_t *__restrict__ present_key,
                     vec16_t *__restrict__ present_value,
                     const int *__restrict__ seqlens_k, int seq_len,
                     int kv_heads, int chunks_per_head, int cap) {
  const int b = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seq_len * kv_heads * chunks_per_head) {
    return;
  }
  const int c = i % chunks_per_head;
  const int h = (i / chunks_per_head) % kv_heads;
  const int s = i / (chunks_per_head * kv_heads);
  const int dest = seqlens_k[b] + 1 - seq_len + s;
  if (dest < 0 || dest >= cap) {
    return;
  }
  const size_t src =
      (static_cast<size_t>(b * seq_len + s) * kv_heads + h) * chunks_per_head +
      c;
  const size_t dst =
      ((static_cast<size_t>(b) * kv_heads + h) * cap + dest) * chunks_per_head +
      c;
  present_key[dst] = key[src];
  present_value[dst] = value[src];
}

// One block per (query row, KV head, split). The heads of the KV group are the
// 16 WMMA rows (rows >= hpg stay zero and are never written); the ND waves each
// own D/ND of the head dim. Per tile of kBkv candidates: S = Q K^T with K
// fragments read straight from the cache, the ND partial scores summed in LDS,
// online softmax by 2*ND threads per row, O += P V with V staged transposed.
// splits == 1 writes the normalized output, otherwise (m, l, O) partials in
// log2 units for dsa_combine_kernel.
template <int D, int ND>
__global__ void __launch_bounds__(32 * ND)
__attribute__((amdgpu_waves_per_eu(HIPDNN_WAVES_FOR_VGPRS(256))))
dsa_attention_kernel(const _Float16 *__restrict__ query,
                     const _Float16 *__restrict__ present_key,
                     const _Float16 *__restrict__ present_value,
                     const int *__restrict__ selected_indices,
                     const int *__restrict__ selected_counts,
                     const int *__restrict__ seqlens_k,
                     _Float16 *__restrict__ output,
                     float *__restrict__ partials, int seq_len, int num_heads,
                     int kv_heads, int cap, int max_sel, int splits,
                     float scale_log2) {
  constexpr int NT = 32 * ND;
  constexpr int DT = D / kTile / ND;
  constexpr int DS = DT * kTile;
  constexpr int N_TILES = kBkv / kTile;
  constexpr int S_STR = kBkv + 1;
  constexpr int P_STR = kBkv + 8;
  constexpr int VT_STR = kBkv + 8;
  constexpr int TPR = NT / kTile;
  constexpr int CPT = kBkv / TPR;
  constexpr int V_PER_LANE = kBkv * (DS / 8) / 32;
  static_assert(DT * kTile * ND == D, "ND must divide the 16-wide d tiles");
  static_assert(CPT * TPR == kBkv, "softmax threads must tile kBkv");
  static_assert(V_PER_LANE * 32 == kBkv * (DS / 8), "V staging must tile");

  __shared__ __align__(16) _Float16 vt_lds[ND][DS * VT_STR];
  __shared__ __align__(16) _Float16 p_lds[kTile * P_STR];
  __shared__ float s_lds[ND][kTile * S_STR];
  __shared__ int idx_lds[kBkv];
  __shared__ float m_lds[kTile];
  __shared__ float l_lds[kTile];

  const int row = blockIdx.x;
  const int g = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lane = tid % 32;
  const int wl = lane % kTile;
  const int pair = lane / kTile;
  const int hpg = num_heads / kv_heads;
  const int b = row / seq_len;
  const int s = row % seq_len;

  const int total = __builtin_amdgcn_readfirstlane(seqlens_k[b] + 1);
  const bool valid_len = seq_len <= total && total <= cap;
  const int qpos = total - seq_len + s;
  int count = 0;
  if (valid_len) {
    count = __builtin_amdgcn_readfirstlane(selected_counts[row]);
    count = count < 0 ? 0 : (count > max_sel ? max_sel : count);
  }
  int chunk = (count + splits - 1) / splits;
  chunk = (chunk + kBkv - 1) / kBkv * kBkv;
  const int c_lo = split * chunk;
  const int c_hi = c_lo + chunk < count ? c_lo + chunk : count;
  const int num_tiles = c_hi > c_lo ? (c_hi - c_lo + kBkv - 1) / kBkv : 0;

  frag_t q_reg[DT];
  {
    const bool ok = wl < hpg;
    const _Float16 *q_row =
        query +
        (static_cast<size_t>(row) * num_heads + g * hpg + (ok ? wl : 0)) * D +
        wave * DS;
#pragma unroll
    for (int tk = 0; tk < DT; ++tk) {
      q_reg[tk] = ok ? load_frag(q_row + tk * kTile, pair) : frag_t{};
    }
  }

  const size_t kv_base = (static_cast<size_t>(b) * kv_heads + g) * cap * D;
  const _Float16 *k_base = present_key + kv_base + wave * DS;
  const _Float16 *v_base = present_value + kv_base + wave * DS;
  const int *sel_row = selected_indices + static_cast<size_t>(row) * max_sel;
  _Float16 *vt = vt_lds[wave];

  const int sm_row = tid / TPR;
  const int sm_sub = tid % TPR;
  float m_run = -FLT_MAX;
  float l_run = 0.0f;
  acc_t o_acc[DT];
#pragma unroll
  for (int tj = 0; tj < DT; ++tj) {
    o_acc[tj] = acc_t{};
  }

  auto load_idx = [&](int t) {
    const int c = c_lo + t * kBkv + tid;
    int idx = -1;
    if (tid < kBkv && t < num_tiles && c < c_hi) {
      idx = sel_row[c];
      // valid_len implies qpos < total <= cap, so this is the whole ORT
      // filter 0 <= idx < total, idx <= qpos, idx < cap.
      if (idx < 0 || idx > qpos) {
        idx = -1;
      }
    }
    return idx;
  };
  int idx_next = load_idx(0);

  for (int t = 0; t < num_tiles; ++t) {
    if (tid < kBkv) {
      idx_lds[tid] = idx_next;
    }
    __syncthreads();
    idx_next = load_idx(t + 1);

    acc_t s_acc[N_TILES];
#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      const int idx = idx_lds[sj * kTile + wl];
      const _Float16 *k_row =
          k_base + static_cast<size_t>(idx < 0 ? 0 : idx) * D;
      s_acc[sj] = acc_t{};
#pragma unroll
      for (int tk = 0; tk < DT; ++tk) {
        const frag_t kf =
            idx >= 0 ? load_frag(k_row + tk * kTile, pair) : frag_t{};
        s_acc[sj] = DSA_WMMA(q_reg[tk], kf, s_acc[sj]);
      }
    }

    // Skipped candidates stage zero V rows: a masked p of 0 must not meet an
    // uninitialized cache row (0 * NaN).
#pragma unroll
    for (int i = 0; i < V_PER_LANE; ++i) {
      const int q = lane + i * 32;
      const int r = q % kBkv;
      const int ch = q / kBkv;
      const int idx = idx_lds[r];
      vec16_t v = vec16_t{};
      if (idx >= 0) {
        v = *reinterpret_cast<const vec16_t *>(
            v_base + static_cast<size_t>(idx) * D + ch * 8);
      }
      const half8_t ve = __builtin_bit_cast(half8_t, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        vt[(ch * 8 + j) * VT_STR + r] = ve[j];
      }
    }

#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        s_lds[wave][hipdnn_wmma_acc_row(pair, e) * S_STR + sj * kTile + wl] =
            s_acc[sj][e];
      }
    }
    __syncthreads();

    {
      float sv[CPT];
      float tmax = -INFINITY;
#pragma unroll
      for (int j = 0; j < CPT; ++j) {
        const int col = sm_sub + j * TPR;
        float x = 0.0f;
#pragma unroll
        for (int w = 0; w < ND; ++w) {
          x += s_lds[w][sm_row * S_STR + col];
        }
        x = idx_lds[col] >= 0 ? x * scale_log2 : -INFINITY;
        sv[j] = x;
        tmax = fmaxf(tmax, x);
      }
#pragma unroll
      for (int off = 1; off < TPR; off <<= 1) {
        tmax = fmaxf(tmax, __shfl_xor(tmax, off));
      }
      const float m_new = fmaxf(m_run, tmax);
      const float corr = exp2f(m_run - m_new);
      float tsum = 0.0f;
#pragma unroll
      for (int j = 0; j < CPT; ++j) {
        const float p = exp2f(sv[j] - m_new);
        tsum += p;
        p_lds[sm_row * P_STR + sm_sub + j * TPR] = static_cast<_Float16>(p);
      }
#pragma unroll
      for (int off = 1; off < TPR; off <<= 1) {
        tsum += __shfl_xor(tsum, off);
      }
      l_run = l_run * corr + tsum;
      m_run = m_new;
      if (sm_sub == 0) {
        m_lds[sm_row] = corr;
      }
    }
    __syncthreads();

#pragma unroll
    for (int tj = 0; tj < DT; ++tj) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        o_acc[tj][e] *= m_lds[hipdnn_wmma_acc_row(pair, e)];
      }
    }
#pragma unroll
    for (int tk = 0; tk < N_TILES; ++tk) {
      const frag_t pf = load_frag(p_lds + wl * P_STR + tk * kTile, pair);
#pragma unroll
      for (int tj = 0; tj < DT; ++tj) {
        const frag_t vf =
            load_frag(vt + (tj * kTile + wl) * VT_STR + tk * kTile, pair);
        o_acc[tj] = DSA_WMMA(pf, vf, o_acc[tj]);
      }
    }
    __syncthreads();
  }

  if (sm_sub == 0) {
    m_lds[sm_row] = m_run;
    l_lds[sm_row] = l_run;
  }
  __syncthreads();

  const int d0 = wave * DS + wl;
  if (splits == 1) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = hipdnn_wmma_acc_row(pair, e);
      if (r < hpg) {
        const float l = l_lds[r];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        _Float16 *o_row =
            output + (static_cast<size_t>(row) * num_heads + g * hpg + r) * D;
#pragma unroll
        for (int tj = 0; tj < DT; ++tj) {
          o_row[d0 + tj * kTile] = static_cast<_Float16>(o_acc[tj][e] * inv);
        }
      }
    }
    return;
  }

#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = hipdnn_wmma_acc_row(pair, e);
    if (r < hpg) {
      float *pd = partials +
                  ((static_cast<size_t>(row) * num_heads + g * hpg + r) *
                       splits +
                   split) *
                      (D + 2);
#pragma unroll
      for (int tj = 0; tj < DT; ++tj) {
        pd[2 + d0 + tj * kTile] = o_acc[tj][e];
      }
    }
  }
  if (tid < hpg) {
    float *pd = partials +
                ((static_cast<size_t>(row) * num_heads + g * hpg + tid) *
                     splits +
                 split) *
                    (D + 2);
    pd[0] = m_lds[tid];
    pd[1] = l_lds[tid];
  }
}

template <bool IS_MAX>
__device__ __forceinline__ float blockReduce(float x, float *red) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float y = __shfl_xor(x, off);
    x = IS_MAX ? fmaxf(x, y) : x + y;
  }
  const int wave = threadIdx.x / 32;
  const int waves = blockDim.x / 32;
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    red[wave] = x;
  }
  __syncthreads();
  x = red[0];
  for (int w = 1; w < waves; ++w) {
    x = IS_MAX ? fmaxf(x, red[w]) : x + red[w];
  }
  return x;
}

// One block per (query row, head). Thread k < splits owns split k's (m, l);
// thread d then merges column d of every split.
template <int D>
__global__ void __launch_bounds__(D)
dsa_combine_kernel(const float *__restrict__ partials,
                   _Float16 *__restrict__ output, int splits) {
  static_assert(D >= kMaxSplits, "one thread per split");
  __shared__ float w_lds[kMaxSplits];
  __shared__ float red[D / 32];
  const float *base =
      partials + static_cast<size_t>(blockIdx.x) * splits * (D + 2);
  const int t = threadIdx.x;
  const bool owner = t < splits;
  const float m_k = owner ? base[static_cast<size_t>(t) * (D + 2)] : -FLT_MAX;
  const float l_k = owner ? base[static_cast<size_t>(t) * (D + 2) + 1] : 0.0f;
  const float gm = blockReduce<true>(m_k, red);
  const float w = owner ? exp2f(m_k - gm) : 0.0f;
  if (owner) {
    w_lds[t] = w;
  }
  const float l = blockReduce<false>(w * l_k, red);
  float o = 0.0f;
#pragma unroll 4
  for (int k = 0; k < splits; ++k) {
    o += w_lds[k] * base[static_cast<size_t>(k) * (D + 2) + 2 + t];
  }
  output[static_cast<size_t>(blockIdx.x) * D + t] =
      static_cast<_Float16>(l > 0.0f ? o / l : 0.0f);
}

int numSplits(int64_t rows, int64_t kv_heads, int64_t max_selected) {
  const int64_t blocks = rows * kv_heads;
  if (blocks >= kTargetBlocks || max_selected <= kBkv) {
    return 1;
  }
  int64_t splits = (kTargetBlocks + blocks - 1) / blocks;
  const int64_t useful = (max_selected + kBkv - 1) / kBkv;
  if (splits > useful) {
    splits = useful;
  }
  if (splits > kMaxSplits) {
    splits = kMaxSplits;
  }
  return static_cast<int>(splits);
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <int D, int ND>
hipError_t launchAttention(hipStream_t stream, const void *query,
                           const void *present_key, const void *present_value,
                           const int32_t *selected_indices,
                           const int32_t *selected_counts,
                           const int32_t *seqlens_k, void *output,
                           float *partials, int rows, int seq_len,
                           int num_heads, int kv_heads, int cap, int max_sel,
                           int splits, float scale) {
  dsa_attention_kernel<D, ND>
      <<<dim3(rows, kv_heads, splits), 32 * ND, 0, stream>>>(
          static_cast<const _Float16 *>(query),
          static_cast<const _Float16 *>(present_key),
          static_cast<const _Float16 *>(present_value), selected_indices,
          selected_counts, seqlens_k, static_cast<_Float16 *>(output),
          partials, seq_len, num_heads, kv_heads, cap, max_sel, splits,
          scale * kLog2e);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess || splits == 1) {
    return err;
  }
  dsa_combine_kernel<D><<<rows * num_heads, D, 0, stream>>>(
      partials, static_cast<_Float16 *>(output), splits);
  return hipGetLastError();
}

} // namespace

extern "C" size_t hip_dynamic_sparse_attention_workspace_size(
    int64_t batch, int64_t seq_len, int64_t num_heads, int64_t kv_num_heads,
    int64_t head_size, int64_t max_selected) {
  const int splits = numSplits(batch * seq_len, kv_num_heads, max_selected);
  if (splits == 1) {
    return 0;
  }
  return static_cast<size_t>(batch * seq_len * num_heads) * splits *
         (head_size + 2) * sizeof(float);
}

extern "C" int hip_dynamic_sparse_attention(
    void *stream, const void *query, const void *key, const void *value,
    const void *past_key, const void *past_value,
    const int32_t *selected_indices, const int32_t *selected_counts,
    const int32_t *seqlens_k, void *output, void *present_key,
    void *present_value, void *workspace, size_t workspace_bytes,
    int64_t batch, int64_t seq_len, int64_t num_heads, int64_t kv_num_heads,
    int64_t head_size, int64_t capacity, int64_t max_selected, float scale) {
  if (!query || !key || !value || !past_key || !past_value ||
      !selected_counts || !seqlens_k || !output || !present_key ||
      !present_value || (max_selected > 0 && !selected_indices)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: null pointer\n");
    return -1;
  }
  if (batch <= 0 || seq_len <= 0 || kv_num_heads <= 0 || capacity <= 0 ||
      max_selected < 0 || num_heads % kv_num_heads != 0 ||
      num_heads / kv_num_heads < 1 || num_heads / kv_num_heads > kTile ||
      (head_size != 128 && head_size != 256)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: unsupported shape "
            "B=%lld S=%lld N=%lld kvN=%lld D=%lld cap=%lld max_selected=%lld\n",
            (long long)batch, (long long)seq_len, (long long)num_heads,
            (long long)kv_num_heads, (long long)head_size, (long long)capacity,
            (long long)max_selected);
    return -1;
  }
  if (!aligned16(query) || !aligned16(key) || !aligned16(value) ||
      !aligned16(past_key) || !aligned16(past_value) ||
      !aligned16(present_key) || !aligned16(present_value)) {
    fprintf(stderr, "[custom_kernels] hip_dynamic_sparse_attention: Q/K/V and "
                    "caches must be 16-byte aligned\n");
    return -1;
  }
  const int64_t rows = batch * seq_len;
  const int splits = numSplits(rows, kv_num_heads, max_selected);
  const size_t needed = hip_dynamic_sparse_attention_workspace_size(
      batch, seq_len, num_heads, kv_num_heads, head_size, max_selected);
  if (needed > workspace_bytes || (needed > 0 && !workspace)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: workspace %zu "
            "bytes, need %zu\n",
            workspace_bytes, needed);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const size_t cache_bytes = static_cast<size_t>(batch * kv_num_heads) *
                             capacity * head_size * sizeof(_Float16);
  hipError_t err = hipSuccess;
  if (past_key != present_key) {
    err = hipMemcpyAsync(present_key, past_key, cache_bytes,
                         hipMemcpyDeviceToDevice, s);
  }
  if (err == hipSuccess && past_value != present_value) {
    err = hipMemcpyAsync(present_value, past_value, cache_bytes,
                         hipMemcpyDeviceToDevice, s);
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: past copy failed "
            "(%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  const int chunks_per_head = static_cast<int>(head_size / 8);
  const int64_t per_batch = seq_len * kv_num_heads * chunks_per_head;
  dsa_kv_append_kernel<<<dim3(static_cast<unsigned>(
                                  (per_batch + kAppendThreads - 1) /
                                  kAppendThreads),
                              static_cast<unsigned>(batch)),
                         kAppendThreads, 0, s>>>(
      static_cast<const vec16_t *>(key), static_cast<const vec16_t *>(value),
      static_cast<vec16_t *>(present_key), static_cast<vec16_t *>(present_value),
      seqlens_k, static_cast<int>(seq_len), static_cast<int>(kv_num_heads),
      chunks_per_head, static_cast<int>(capacity));
  err = hipGetLastError();

  if (err == hipSuccess) {
    float *partials = static_cast<float *>(workspace);
    if (head_size == 256) {
      err = launchAttention<256, 4>(
          s, query, present_key, present_value, selected_indices,
          selected_counts, seqlens_k, output, partials, static_cast<int>(rows),
          static_cast<int>(seq_len), static_cast<int>(num_heads),
          static_cast<int>(kv_num_heads), static_cast<int>(capacity),
          static_cast<int>(max_selected), splits, scale);
    } else {
      err = launchAttention<128, 2>(
          s, query, present_key, present_value, selected_indices,
          selected_counts, seqlens_k, output, partials, static_cast<int>(rows),
          static_cast<int>(seq_len), static_cast<int>(num_heads),
          static_cast<int>(kv_num_heads), static_cast<int>(capacity),
          static_cast<int>(max_selected), splits, scale);
    }
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: launch failed "
            "(%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
