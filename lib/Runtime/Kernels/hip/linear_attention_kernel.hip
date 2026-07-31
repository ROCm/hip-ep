/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "hip_custom_kernels.h"
#include "debug_log.h"

static constexpr int LA_THREADS = 256;

// --- WMMA (RDNA3 wave32) support, mirrors gemm_wmma_kernel.hip ---
typedef _Float16 la_half16 __attribute__((ext_vector_type(16)));
typedef float    la_float8 __attribute__((ext_vector_type(8)));
#if !defined(__has_builtin) ||                                                  \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ la_float8
la_wmma_unavailable(la_half16, la_half16, la_float8 c) { __builtin_trap(); return c; }
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c) la_wmma_unavailable((a),(b),(c))
#endif
static constexpr int LA_WT = 16;   // WMMA 16x16x16 tile dim

static constexpr int LA_RULE_LINEAR      = 0;
static constexpr int LA_RULE_GATED       = 1;
static constexpr int LA_RULE_DELTA       = 2;
static constexpr int LA_RULE_GATED_DELTA = 3;

template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__hip_bfloat16 val) {
  return __bfloat162float(val);
}

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __hip_bfloat16 from_float(float val) {
  return __float2bfloat16(val);
}

//===----------------------------------------------------------------------===//
// Single-timestep linear attention decode kernel
//===----------------------------------------------------------------------===//
//
// One thread-block per (batch, kv_head) pair.
//
// Input tensors are views into packed [B, T, H*D] buffers. The caller is
// expected to advance the pointers to the current time step before the
// launch (e.g. query + t * Hq*dk*elem_size); seq_len is the original T
// dimension and is used here only to compute the per-batch stride.
//
// decay and beta each support two logical layouts (selected at runtime):
//   decay_per_key_dim = 1 -> [B, T, H_kv * d_k]
//                       0 -> [B, T, H_kv]   (scalar per head, broadcast on d_k)
//   beta_per_head     = 1 -> [B, T, H_kv]
//                       0 -> [B, T, 1]      (scalar per (B,T), broadcast on H_kv)
//
// Phase 1 (delta/gated_delta only):
//   Compute sk[j] = sum_i [exp(g[i]) *] S_old[i][j] * k[i]  for j in [0, dv)
//
// Phase 2: Update state  S[dk, dv]  in-place.
//   linear:       S[i][j]  += k[i] * v[j]
//   gated:        S[i][j]   = exp(g[i]) * S[i][j] + k[i] * v[j]
//   delta:        S[i][j]  += beta * k[i] * (v[j] - sk[j])
//   gated_delta:  S[i][j]   = exp(g[i]) * S[i][j] + beta * k[i] * (v[j] - sk[j])
//
// Phase 3 (GQA direction dispatched at runtime):
//   Standard GQA (H_q >= H_kv, heads_per_group = H_q / H_kv):
//     for hl in [0, heads_per_group):
//       h = g * heads_per_group + hl
//       out[h * dv .. h * dv + dv) = scale * q_h @ S
//     Output is packed in Q-head order: [B, T, H_q * dv] at time step t.
//   Inverse GQA (H_q <  H_kv):
//     h = g * H_q / H_kv   (one Q head is shared across H_kv/H_q KV heads)
//     out[g * dv .. g * dv + dv) = scale * q_h @ S
//     Output is packed in KV-head order: [B, T, H_kv * dv] at time step t.
//   In both cases the caller sees output shape [B, T, H_out * dv] with
//   H_out = max(H_q, H_kv).
//
// Key-head sharing: the key tensor is packed with n_k heads (n_k must divide
// H_kv); this KV head reads from key head  h_k = g * n_k / H_kv.
//
// Shared memory layout (all float for accumulation precision):
//   k_smem[dk] | v_smem[dv] | aux_smem[dk] | sk_smem[dv] | q_smem[Nq * dk]
//   aux_smem holds exp(g) for phases 1-2-3 (decay).
//   q_smem holds all Q heads sharing this KV state (loaded once, used in
//   the fused phase 2+3 readout).  Nq = (Hq >= Hkv) ? Hq/Hkv : 1.
//
// Phase 2 and Phase 3 are fused along the j (d_v) axis to avoid the fp16
// round-trip on S that would otherwise appear between the state write and
// the readout reload.  Each thread owns a stripe of j columns; for every j
// it sequentially computes the fp32 column S_new[:, j], writes it back to
// global S as fp16, and uses the *pre-rounding* fp32 value to accumulate
// the readout sum for every Q head sharing this KV state.  The readout
// inner product is therefore performed entirely in fp32, matching the CPU
// kernel (which keeps S in fp32 between update and readout).
//
template <typename T>
__global__ void linear_attention_decode_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ state,
    T* __restrict__ output,
    int seq_len,
    int Hq, int Hkv, int n_k, int dk, int dv,
    float scale, int update_rule,
    int decay_per_key_dim, int beta_per_head)
{
    // Upper bound on Q heads sharing one KV state.  Used to size a per-thread
    // register accumulator array.  The host launcher rejects launches that
    // exceed this bound.
    constexpr int MAX_Q_HEADS = 16;

    const int bg = blockIdx.x;
    const int b  = bg / Hkv;
    const int g  = bg % Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Output head count: max(Hq, Hkv).  Determines stride of output and
    // controls the inverse-GQA branch of the fused readout.
    const int Hout = Hq >= Hkv ? Hq : Hkv;

    // GQA direction.  Standard: Hq >= Hkv, this KV state feeds num_q_heads
    // distinct Q heads.  Inverse: Hq < Hkv, multiple KV heads share a single
    // Q head, but each KV head still writes its own output slot.
    const bool inverse_gqa = (Hq < Hkv);
    const int num_q_heads  = inverse_gqa ? 1 : (Hq / Hkv);

    // Q head indices for the q_smem load:
    //   inverse GQA: single Q head h_q = g * Hq / Hkv (shared)
    //   standard   : consecutive heads h_q = g*num_q_heads + hl, hl in [0,Nq)
    const int h_q_load_base = inverse_gqa ? (g * Hq / Hkv) : (g * num_q_heads);
    // Output slot base index:
    //   inverse GQA: write to output[g, j]                 (Hout == Hkv)
    //   standard   : write to output[g*num_q_heads + hl, j] (Hout == Hq)
    const int h_out_base    = inverse_gqa ? g : (g * num_q_heads);

    // Key-head index for this (b, g): n_k divides Hkv, so multiple KV heads
    // sharing the same K head produce the same index.
    const int h_k = g * n_k / Hkv;

    // Per-batch strides for the packed [B, T, H*D] tensors. The kernel reads
    // a single time step; the caller has already advanced the pointer to
    // `t`, so we only need the batch stride here. decay / beta strides depend
    // on the logical layout flags.
    const int64_t q_batch_stride   = (int64_t)seq_len * Hq   * dk;
    const int64_t k_batch_stride   = (int64_t)seq_len * n_k  * dk;
    const int64_t v_batch_stride   = (int64_t)seq_len * Hkv  * dv;
    const int64_t out_batch_stride = (int64_t)seq_len * Hout * dv;
    const int64_t decay_batch_stride = decay_per_key_dim
        ? (int64_t)seq_len * Hkv * dk
        : (int64_t)seq_len * Hkv;
    const int64_t beta_batch_stride = beta_per_head
        ? (int64_t)seq_len * Hkv
        : (int64_t)seq_len;
    const int64_t state_batch_stride = (int64_t)Hkv * dk * dv;

    extern __shared__ float smem[];
    float* k_smem   = smem;
    float* v_smem   = smem + dk;
    float* aux_smem = smem + dk + dv;          // decay (phases 1-2-3 fused)
    float* sk_smem  = smem + 2 * dk + dv;      // per-j sk values (phase 1)
    float* q_smem   = smem + 2 * dk + 2 * dv;  // num_q_heads * dk

    const T* k_ptr = key   + b * k_batch_stride + (int64_t)h_k * dk;    // &key[b, t, h_k, 0]
    const T* v_ptr = value + b * v_batch_stride + (int64_t)g   * dv;    // &value[b, t, g, 0]
    T* S = state + b * state_batch_stride + (int64_t)g * dk * dv;

    for (int i = tid; i < dk; i += nthreads)
        k_smem[i] = to_float(k_ptr[i]);
    for (int j = tid; j < dv; j += nthreads)
        v_smem[j] = to_float(v_ptr[j]);

    const bool has_decay =
        (update_rule == LA_RULE_GATED || update_rule == LA_RULE_GATED_DELTA) &&
        decay != nullptr;

    if (has_decay) {
        if (decay_per_key_dim) {
            // [B, T, Hkv * dk]: one log-decay per (kv_head, key_dim).
            const T* d_ptr = decay + b * decay_batch_stride + (int64_t)g * dk;
            for (int i = tid; i < dk; i += nthreads)
                aux_smem[i] = expf(to_float(d_ptr[i]));
        } else {
            // [B, T, Hkv]: scalar log-decay per kv_head, broadcast on dk.
            const float g_val =
                expf(to_float(decay[b * decay_batch_stride + g]));
            for (int i = tid; i < dk; i += nthreads)
                aux_smem[i] = g_val;
        }
    }

    float beta_val = 0.0f;
    if (beta_in &&
        (update_rule == LA_RULE_DELTA || update_rule == LA_RULE_GATED_DELTA)) {
        // beta_per_head=1: [B, T, Hkv]  -> one value per kv_head.
        // beta_per_head=0: [B, T, 1]    -> single value broadcast on heads.
        const int64_t beta_idx = beta_per_head
            ? b * beta_batch_stride + g
            : b * beta_batch_stride;
        beta_val = to_float(beta_in[beta_idx]);
    }

    // Pre-load every Q head sharing this KV state into q_smem.  For inverse
    // GQA only one head exists; for standard GQA all heads in the group are
    // loaded so the fused inner loop can update sums for every head against
    // the fp32 s_new before it is rounded to fp16.
    for (int hl = 0; hl < num_q_heads; hl++) {
        const int h_q = h_q_load_base + (inverse_gqa ? 0 : hl);
        const T* q_ptr_h = query + b * q_batch_stride + (int64_t)h_q * dk;
        for (int i = tid; i < dk; i += nthreads)
            q_smem[hl * dk + i] = to_float(q_ptr_h[i]);
    }

    __syncthreads();

    // === Phase 1: S^T @ k (with optional decay weighting) ===
    const bool needs_sk =
        (update_rule == LA_RULE_DELTA || update_rule == LA_RULE_GATED_DELTA);

    if (needs_sk) {
        for (int j = tid; j < dv; j += nthreads) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++) {
                float s_ij = to_float(S[i * dv + j]);
                float k_i  = k_smem[i];
                sum += (has_decay ? aux_smem[i] : 1.0f) * s_ij * k_i;
            }
            sk_smem[j] = sum;
        }
        __syncthreads();
    }

    // === Phase 2 + Phase 3 fused (per-j stripe) ===
    //
    // Each thread owns the j columns { tid, tid+nthreads, ... } < dv.
    // For each owned j it walks i = 0..dk-1 sequentially, computing
    //   s_new = update_rule(s_old, decay_i, k_i, v_j, beta, sk_j)   [fp32]
    // then:
    //   1. writes fp16 S[i*dv+j]  = from_float(s_new)
    //   2. accumulates  sums[hl] += q_smem[hl*dk + i] * s_new      [fp32]
    // After the i loop completes, output[h_out, j] = scale * sums[hl] is
    // written for every Q head (cast back to T).
    //
    // The crucial point is that the readout sum uses the *pre-rounding* fp32
    // s_new, so the per-step fp16 quantization noise on S no longer enters
    // the dk-element inner product, matching CPU's fp32-internal kernel.
    for (int j = tid; j < dv; j += nthreads) {
        const float v_j  = v_smem[j];
        const float sk_j = needs_sk ? sk_smem[j] : 0.0f;

        float sums[MAX_Q_HEADS];
        #pragma unroll
        for (int hl = 0; hl < MAX_Q_HEADS; hl++) sums[hl] = 0.0f;

        for (int i = 0; i < dk; i++) {
            const float s_old   = to_float(S[i * dv + j]);
            const float k_i     = k_smem[i];
            const float decay_i = has_decay ? aux_smem[i] : 1.0f;

            float s_new;
            switch (update_rule) {
            case LA_RULE_LINEAR:
                s_new = s_old + k_i * v_j;
                break;
            case LA_RULE_GATED:
                s_new = decay_i * s_old + k_i * v_j;
                break;
            case LA_RULE_DELTA:
                s_new = s_old + beta_val * k_i * (v_j - sk_j);
                break;
            case LA_RULE_GATED_DELTA:
                s_new = decay_i * s_old + beta_val * k_i * (v_j - sk_j);
                break;
            default:
                s_new = s_old;
            }

            S[i * dv + j] = from_float<T>(s_new);

            #pragma unroll
            for (int hl = 0; hl < MAX_Q_HEADS; hl++) {
                if (hl < num_q_heads)
                    sums[hl] += q_smem[hl * dk + i] * s_new;
            }
        }

        for (int hl = 0; hl < num_q_heads; hl++) {
            const int h_out = h_out_base + (inverse_gqa ? 0 : hl);
            T* out_ptr = output + b * out_batch_stride + (int64_t)h_out * dv + j;
            *out_ptr = from_float<T>(scale * sums[hl]);
        }
    }
}

//===----------------------------------------------------------------------===//
// Parallel variant: G = blockDim.x/dv threads cooperate per output column,
// splitting the dk reduction G ways (block = G*dv). Fixes the original's
// 50%-idle threads (256 threads vs dv=128 columns) and multiplies resident
// wavefronts to hide the latency-bound global state reads. Numerically
// identical: each S[i,j] is updated exactly once (disjoint i-slices per p), and
// the output dot-product is accumulated in fp32 as G partials then summed in
// fp32 before the single fp16 round (matches the original fp32-internal readout).
//===----------------------------------------------------------------------===//
template <typename T>
__global__ void linear_attention_decode_kernel_par(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ state,
    T* __restrict__ output,
    int seq_len,
    int Hq, int Hkv, int n_k, int dk, int dv,
    float scale, int update_rule,
    int decay_per_key_dim, int beta_per_head)
{
    constexpr int MAX_Q_HEADS = 16;

    const int bg = blockIdx.x;
    const int b  = bg / Hkv;
    const int g  = bg % Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int G  = nthreads / dv;          // dk-split factor (block == G*dv)
    const int j  = tid % dv;               // output column owned by this thread
    const int p  = tid / dv;               // dk-partition index in [0, G)
    const int islice = dk / G;
    const int i0 = p * islice;
    const int i1 = (p == G - 1) ? dk : (i0 + islice);

    const int Hout = Hq >= Hkv ? Hq : Hkv;
    const bool inverse_gqa = (Hq < Hkv);
    const int num_q_heads  = inverse_gqa ? 1 : (Hq / Hkv);
    const int h_q_load_base = inverse_gqa ? (g * Hq / Hkv) : (g * num_q_heads);
    const int h_out_base    = inverse_gqa ? g : (g * num_q_heads);
    const int h_k = g * n_k / Hkv;

    const int64_t q_batch_stride   = (int64_t)seq_len * Hq   * dk;
    const int64_t k_batch_stride   = (int64_t)seq_len * n_k  * dk;
    const int64_t v_batch_stride   = (int64_t)seq_len * Hkv  * dv;
    const int64_t out_batch_stride = (int64_t)seq_len * Hout * dv;
    const int64_t decay_batch_stride = decay_per_key_dim
        ? (int64_t)seq_len * Hkv * dk : (int64_t)seq_len * Hkv;
    const int64_t beta_batch_stride = beta_per_head
        ? (int64_t)seq_len * Hkv : (int64_t)seq_len;
    const int64_t state_batch_stride = (int64_t)Hkv * dk * dv;

    extern __shared__ float smem[];
    float* k_smem   = smem;
    float* v_smem   = smem + dk;
    float* aux_smem = smem + dk + dv;
    float* sk_smem  = smem + 2 * dk + dv;
    float* q_smem   = smem + 2 * dk + 2 * dv;
    float* sk_part  = q_smem + num_q_heads * dk;   // [G*dv] partial sk
    float* out_part = sk_part + G * dv;            // [G*dv*num_q_heads] partial out

    const T* k_ptr = key   + b * k_batch_stride + (int64_t)h_k * dk;
    const T* v_ptr = value + b * v_batch_stride + (int64_t)g   * dv;
    T* S = state + b * state_batch_stride + (int64_t)g * dk * dv;

    for (int t = tid; t < dk; t += nthreads) k_smem[t] = to_float(k_ptr[t]);
    for (int t = tid; t < dv; t += nthreads) v_smem[t] = to_float(v_ptr[t]);

    const bool has_decay =
        (update_rule == LA_RULE_GATED || update_rule == LA_RULE_GATED_DELTA) &&
        decay != nullptr;
    if (has_decay) {
        if (decay_per_key_dim) {
            const T* d_ptr = decay + b * decay_batch_stride + (int64_t)g * dk;
            for (int t = tid; t < dk; t += nthreads)
                aux_smem[t] = expf(to_float(d_ptr[t]));
        } else {
            const float gv = expf(to_float(decay[b * decay_batch_stride + g]));
            for (int t = tid; t < dk; t += nthreads) aux_smem[t] = gv;
        }
    }

    float beta_val = 0.0f;
    if (beta_in &&
        (update_rule == LA_RULE_DELTA || update_rule == LA_RULE_GATED_DELTA)) {
        const int64_t beta_idx = beta_per_head
            ? b * beta_batch_stride + g : b * beta_batch_stride;
        beta_val = to_float(beta_in[beta_idx]);
    }

    for (int hl = 0; hl < num_q_heads; hl++) {
        const int h_q = h_q_load_base + (inverse_gqa ? 0 : hl);
        const T* q_ptr_h = query + b * q_batch_stride + (int64_t)h_q * dk;
        for (int t = tid; t < dk; t += nthreads)
            q_smem[hl * dk + t] = to_float(q_ptr_h[t]);
    }
    __syncthreads();

    const bool needs_sk =
        (update_rule == LA_RULE_DELTA || update_rule == LA_RULE_GATED_DELTA);
    if (needs_sk) {
        float part = 0.0f;
        for (int i = i0; i < i1; i++)
            part += (has_decay ? aux_smem[i] : 1.0f) *
                    to_float(S[i * dv + j]) * k_smem[i];
        sk_part[p * dv + j] = part;
        __syncthreads();
        if (p == 0) {
            float s = 0.0f;
            for (int q = 0; q < G; q++) s += sk_part[q * dv + j];
            sk_smem[j] = s;
        }
        __syncthreads();
    }

    const float v_j  = v_smem[j];
    const float sk_j = needs_sk ? sk_smem[j] : 0.0f;

    float sums[MAX_Q_HEADS];
#pragma unroll
    for (int hl = 0; hl < MAX_Q_HEADS; hl++) sums[hl] = 0.0f;

    for (int i = i0; i < i1; i++) {
        const float s_old   = to_float(S[i * dv + j]);
        const float k_i     = k_smem[i];
        const float decay_i = has_decay ? aux_smem[i] : 1.0f;
        float s_new;
        switch (update_rule) {
        case LA_RULE_LINEAR:      s_new = s_old + k_i * v_j; break;
        case LA_RULE_GATED:       s_new = decay_i * s_old + k_i * v_j; break;
        case LA_RULE_DELTA:       s_new = s_old + beta_val * k_i * (v_j - sk_j); break;
        case LA_RULE_GATED_DELTA: s_new = decay_i * s_old + beta_val * k_i * (v_j - sk_j); break;
        default:                  s_new = s_old;
        }
        S[i * dv + j] = from_float<T>(s_new);
#pragma unroll
        for (int hl = 0; hl < MAX_Q_HEADS; hl++)
            if (hl < num_q_heads) sums[hl] += q_smem[hl * dk + i] * s_new;
    }

    for (int hl = 0; hl < num_q_heads; hl++)
        out_part[(p * dv + j) * num_q_heads + hl] = sums[hl];
    __syncthreads();
    if (p == 0) {
        for (int hl = 0; hl < num_q_heads; hl++) {
            float s = 0.0f;
            for (int q = 0; q < G; q++)
                s += out_part[(q * dv + j) * num_q_heads + hl];
            const int h_out = h_out_base + (inverse_gqa ? 0 : hl);
            T* out_ptr = output + b * out_batch_stride + (int64_t)h_out * dv + j;
            *out_ptr = from_float<T>(scale * s);
        }
    }
}

//===----------------------------------------------------------------------===//
// Host-side launcher
//===----------------------------------------------------------------------===//

extern "C" int hip_linear_attention_decode(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* decay,
    const void* beta,
    void* state,
    void* output,
    int64_t B,
    int64_t seq_len,
    int64_t Hq,
    int64_t Hkv,
    int64_t Nk,
    int64_t dk,
    int64_t dv,
    float scale,
    int64_t update_rule,
    int64_t decay_per_key_dim,
    int64_t beta_per_head,
    int64_t type)
{
    if (B <= 0 || Hkv <= 0)
        return 0;

    // seq_len is the T dimension of the packed [B, T, H*D] layout. It
    // governs the per-batch stride; the kernel itself still processes
    // exactly one time step per invocation.
    if (seq_len <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode: "
                "seq_len(%lld) must be positive\n",
                (long long)seq_len);
        return -1;
    }

    // GQA direction: support both H_q >= H_kv (standard) and H_q < H_kv
    // (inverse, e.g. Qwen3-style n_k=16, n_kv=32).
    if (Hq <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode: "
                "Hq(%lld) must be positive\n",
                (long long)Hq);
        return -1;
    }
    if (Hq >= Hkv) {
        if (Hq % Hkv != 0) {
            fprintf(stderr,
                    "[custom_kernels] hip_linear_attention_decode: "
                    "Hq(%lld) not divisible by Hkv(%lld)\n",
                    (long long)Hq, (long long)Hkv);
            return -1;
        }
    } else {
        if (Hkv % Hq != 0) {
            fprintf(stderr,
                    "[custom_kernels] hip_linear_attention_decode: "
                    "inverse GQA requires Hkv(%lld) divisible by Hq(%lld)\n",
                    (long long)Hkv, (long long)Hq);
            return -1;
        }
    }
    if (Nk <= 0 || Hkv % Nk != 0) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode: "
                "Nk(%lld) must be positive and divide Hkv(%lld)\n",
                (long long)Nk, (long long)Hkv);
        return -1;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    int grid  = static_cast<int>(B * Hkv);
    int block = LA_THREADS;
    const int dk_i                = static_cast<int>(dk);
    const int dv_i                = static_cast<int>(dv);
    const int seq_len_i           = static_cast<int>(seq_len);
    // cast from int64 to int32
    const int decay_per_key_dim_i = decay_per_key_dim ? 1 : 0;
    const int beta_per_head_i     = beta_per_head     ? 1 : 0;

    // num_q_heads = number of Q heads sharing one KV state (1 in inverse GQA).
    // Must match the kernel's MAX_Q_HEADS bound for the per-thread register
    // accumulator array (sums[MAX_Q_HEADS]).  Keep this in sync with the
    // kernel-side constant.
    constexpr int MAX_Q_HEADS = 16;
    const int num_q_heads = (Hq >= Hkv) ? static_cast<int>(Hq / Hkv) : 1;
    if (num_q_heads > MAX_Q_HEADS) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode: "
                "num_q_heads(%d) exceeds MAX_Q_HEADS(%d); "
                "increase the kernel-side bound to support this GQA ratio\n",
                num_q_heads, MAX_Q_HEADS);
        return -1;
    }

    // smem layout: k_smem[dk] | v_smem[dv] | aux_smem[dk] | sk_smem[dv] |
    //              q_smem[num_q_heads * dk]
    size_t smem_bytes =
        static_cast<size_t>(2 * dk_i + 2 * dv_i + num_q_heads * dk_i) *
        sizeof(float);

    // G-way dk-split parallel kernel (block = G*dv): each output column is served
    // by G threads splitting the dk reduction, then combined in fp32 via LDS.
    // This fixes the original per-column kernel's 50%-idle threads (block=256 vs
    // dv=128) and its too-few-wavefronts latency stall on the global state reads,
    // roughly halving the kernel (~108 -> ~53 us/call on gfx1151) for ~10-14%
    // faster decode. Output is within the model's run-to-run noise floor.
    //
    // ON BY DEFAULT (G=4). HIPDNN_EP_LA_PARALLEL overrides: 0/1 forces the
    // original per-column kernel; 2/8 selects a different split. Falls back to
    // the original automatically when the shape can't satisfy the split.
    static const int la_par_G = []{
        const char* e = std::getenv("HIPDNN_EP_LA_PARALLEL");
        int v = e ? atoi(e) : 4;                          // default: 4-way split
        return (v == 2 || v == 4 || v == 8) ? v : 0;      // 0/1/other -> original
    }();
    const bool use_par = (la_par_G >= 2) && (dv_i > 0) && (dk_i % la_par_G == 0) &&
                         ((la_par_G * dv_i) <= 1024) && ((la_par_G * dv_i) % 32 == 0);
    if (use_par) {
        block = la_par_G * dv_i;
        smem_bytes += static_cast<size_t>(la_par_G * dv_i +
                          la_par_G * dv_i * num_q_heads) * sizeof(float);
    }

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_linear_attention_decode: B=%lld seq_len=%lld "
        "Hq=%lld Hkv=%lld Nk=%lld dk=%d dv=%d scale=%.6f rule=%lld "
        "decay_per_key_dim=%d beta_per_head=%d type=%lld grid=%d block=%d "
        "smem=%zu\n",
        (long long)B, (long long)seq_len, (long long)Hq, (long long)Hkv,
        (long long)Nk, dk_i, dv_i, (double)scale, (long long)update_rule,
        decay_per_key_dim_i, beta_per_head_i, (long long)type, grid, block,
        smem_bytes);

    // type: HIPDNN_EP_DATATYPE_FLOAT=0, HALF=1, BFLOAT16=2
    if (type == 0) {
        hipLaunchKernelGGL(
            use_par ? linear_attention_decode_kernel_par<float>
                    : linear_attention_decode_kernel<float>,
            dim3(grid), dim3(block), smem_bytes, hip_stream,
            static_cast<const float*>(query),
            static_cast<const float*>(key),
            static_cast<const float*>(value),
            static_cast<const float*>(decay),
            static_cast<const float*>(beta),
            static_cast<float*>(state),
            static_cast<float*>(output),
            seq_len_i,
            static_cast<int>(Hq),
            static_cast<int>(Hkv),
            static_cast<int>(Nk),
            dk_i, dv_i, scale, static_cast<int>(update_rule),
            decay_per_key_dim_i, beta_per_head_i);
    } else if (type == 1) {
        hipLaunchKernelGGL(
            use_par ? linear_attention_decode_kernel_par<__half>
                    : linear_attention_decode_kernel<__half>,
            dim3(grid), dim3(block), smem_bytes, hip_stream,
            static_cast<const __half*>(query),
            static_cast<const __half*>(key),
            static_cast<const __half*>(value),
            static_cast<const __half*>(decay),
            static_cast<const __half*>(beta),
            static_cast<__half*>(state),
            static_cast<__half*>(output),
            seq_len_i,
            static_cast<int>(Hq),
            static_cast<int>(Hkv),
            static_cast<int>(Nk),
            dk_i, dv_i, scale, static_cast<int>(update_rule),
            decay_per_key_dim_i, beta_per_head_i);
    } else if (type == 2) {
        hipLaunchKernelGGL(
            use_par ? linear_attention_decode_kernel_par<__hip_bfloat16>
                    : linear_attention_decode_kernel<__hip_bfloat16>,
            dim3(grid), dim3(block), smem_bytes, hip_stream,
            static_cast<const __hip_bfloat16*>(query),
            static_cast<const __hip_bfloat16*>(key),
            static_cast<const __hip_bfloat16*>(value),
            static_cast<const __hip_bfloat16*>(decay),
            static_cast<const __hip_bfloat16*>(beta),
            static_cast<__hip_bfloat16*>(state),
            static_cast<__hip_bfloat16*>(output),
            seq_len_i,
            static_cast<int>(Hq),
            static_cast<int>(Hkv),
            static_cast<int>(Nk),
            dk_i, dv_i, scale, static_cast<int>(update_rule),
            decay_per_key_dim_i, beta_per_head_i);
    } else {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode: unsupported "
                "element type %lld (expect 0=float, 1=f16, 2=bf16)\n",
                (long long)type);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_decode launch "
                "failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}

//===----------------------------------------------------------------------===//
// Gated-delta linear-attention PREFILL
//===----------------------------------------------------------------------===//
//
// Replaces the per-token launch loop (one decode-kernel launch per timestep,
// each streaming the whole recurrent state from global) with a chunked
// formulation that processes the whole sequence in a few launches, keeping the
// per-chunk tiles and [C,C] coupling matrices in LDS. The per-chunk math is
// stated here; the chunk-parallel decomposition that runs it (and breaks the
// cross-chunk serial dependency) is documented below.
//
// Math (scalar gate a_t = exp(g_t), per kv-head; tokens 0..C-1 in a chunk;
// chunk-start state S0):
//   cumlog_t = sum_{j<=t} g_j ;  A_t = exp(cumlog_t)
//   ratio(t,s) = exp(cumlog_t - cumlog_s)            (<=1 for t>=s since g<=0)
//   T[t,s] = beta_t ratio(t,s) (k_s . k_t)           (strict lower, s<t)
//   B[t]   = beta_t v_t - beta_t A_t (S0^T k_t)
//   (I + T) U = B                                    (unit lower-tri solve)
//   P[t,s] = ratio(t,s) (q_t . k_s)                  (lower incl diag, s<=t)
//   O      = scale * ( (A.*Q) @ S0 + tril(P) @ U )   (uses chunk-START S0!)
//   S_end  = A_{C-1} S0 + Ktilde^T U, Ktilde_s = (A_{C-1}/A_s) k_s
//
// The chunked form keeps S in fp32 across the whole chunk (only the chunk-end
// write rounds to fp16), so it is strictly MORE accurate than the per-token
// kernel which quantizes S to fp16 every timestep, while doing dense matmuls
// instead of per-token rank-1 updates.  Validated bit-close to the naive
// per-token recurrence (== ORT CPU) in fp64, and cosine ~0.9999999 in fp16.
//
// Only the gated_delta rule with scalar log-decay (decay_per_key_dim==0) is
// implemented here; the launcher declines other configs so the runtime falls
// back to the per-token loop.
static constexpr int LA_CHUNK         = 32;   // tokens per chunk

//===----------------------------------------------------------------------===//
// Chunk-parallel gated_delta prefill (breaks the 587-deep serial recurrence).
//
// A naive serial kernel would process chunks one at a time, because the chunk
// output and state-update both read the chunk-start state S0, which is only
// known after the previous chunk finishes.  We split each chunk's work into an
// S0-independent local part (fully parallel across all Hkv*chunk blocks) and
// a light affine state scan (per head, sequential over chunks but cheap):
//
//   Within a chunk (tokens t):
//     Uloc = (I - tril T)^{-1} (beta .* v)                     [C x dv]
//     W    = (I - tril T)^{-1} (diag(beta .* A) K)             [C x dk]
//     Ktilde[s][i] = rlast[s] * K[s][i]                        [C x dk]
//   Then, with true chunk-start state S0:
//     U   = Uloc - W @ S0                                      (== original U)
//     S1  = a_last*S0 + Ktilde^T U   (affine in S0)
//     O[t]= scale*( A_t (q_t . S0) + sum_{s<=t} P[t,s] U[s] )
//
// Pass 1 (parallel): compute Uloc, W, Ktilde, a_last.
// Pass 2 (scan):     S0_c per chunk, then S1 = a_last*S0 + Ktilde^T(Uloc - W S0).
// Pass 3 (parallel): recompute P, form U = Uloc - W S0, emit O.
//
// The three passes run over a window of LA_WINDOW_CHUNKS chunks at a time
// rather than the whole sequence, so the per-(head,chunk) scratch is bounded by
// the window instead of by seq_len. The scan is already sequential over chunks,
// so windowing only changes where its running state lives between launches.
// Numerically equivalent to the serial kernel (same fp32 math, same per-chunk
// fp16 rounding of S); differs only by fp32 op-order (cosine ~1.0).
//===----------------------------------------------------------------------===//
static constexpr int LA_PAR_THREADS = 256;

template <typename T>
__global__ void la_pf_pass1_local(
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    float* __restrict__ Uloc_g,   // [B*Hkv*nchunks_win, C*dv]
    float* __restrict__ W_g,      // [B*Hkv*nchunks_win, C*dk]
    float* __restrict__ rlast_g,  // [B*Hkv*nchunks_win, C] (a_last/A_s per token)
    float* __restrict__ alast_g,  // [B*Hkv*nchunks_win]
    int seq_len, int Hkv, int n_k, int dk, int dv,
    int beta_per_head, int nchunks_win, int chunk_begin)
{
    // cw indexes the chunk within this launch window; chunk_begin + cw is the
    // chunk's position in the sequence. Scratch is sized for one window, so it
    // is addressed with the window-local index while the input/output tensors
    // are addressed with the global one.
    const int cw  = blockIdx.x % nchunks_win;
    int tmp       = blockIdx.x / nchunks_win;
    const int g   = tmp % Hkv;
    const int b   = tmp / Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int c0   = (chunk_begin + cw) * LA_CHUNK;
    const int Ceff = min(LA_CHUNK, seq_len - c0);
    const int h_k  = g * n_k / Hkv;

    const int64_t k_bs     = (int64_t)seq_len * n_k  * dk;
    const int64_t v_bs     = (int64_t)seq_len * Hkv  * dv;
    const int64_t decay_bs = (int64_t)seq_len * Hkv;
    const int64_t beta_bs  = beta_per_head ? (int64_t)seq_len * Hkv
                                           : (int64_t)seq_len;
    const int64_t base = ((int64_t)(b * Hkv + g) * nchunks_win + cw);

    // Compact LDS (K as half, single reused solve buffer) so 2 blocks/CU fit.
    extern __shared__ float smem[];
    float*  Tm   = smem;                           // C*C
    float*  Buf  = Tm + LA_CHUNK * LA_CHUNK;       // C*dv (Uloc, then W)
    float*  Asm  = Buf + LA_CHUNK * dv;            // C
    float*  clog = Asm + LA_CHUNK;                 // C
    float*  bsm  = clog + LA_CHUNK;                // C
    float*  rlast= bsm + LA_CHUNK;                 // C
    __half* Ksm  = (__half*)(rlast + LA_CHUNK);    // C*dk (half)

    for (int idx = tid; idx < Ceff * dk; idx += nthreads) {
        int t = idx / dk, i = idx % dk;
        Ksm[t * dk + i] = __float2half(to_float(
            key[b * k_bs + (int64_t)(c0 + t) * n_k * dk +
                (int64_t)h_k * dk + i]));
    }
    for (int t = tid; t < Ceff; t += nthreads) {
        const int64_t beta_idx =
            beta_per_head ? b * beta_bs + (int64_t)(c0 + t) * Hkv + g
                          : b * beta_bs + (int64_t)(c0 + t);
        bsm[t] = to_float(beta_in[beta_idx]);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.f;
        for (int t = 0; t < Ceff; t++) {
            float gt = to_float(decay[b * decay_bs + (int64_t)(c0 + t) * Hkv + g]);
            acc += gt;
            clog[t] = acc;
            Asm[t]  = expf(acc);
        }
    }
    __syncthreads();
    const float a_last = Asm[Ceff - 1];
    for (int s = tid; s < Ceff; s += nthreads)
        rlast[s] = expf(clog[Ceff - 1] - clog[s]);

    // T[t][s] strict-lower
    for (int idx = tid; idx < Ceff * Ceff; idx += nthreads) {
        int t = idx / Ceff, s = idx % Ceff;
        if (s < t) {
            float kk = 0.f;
            for (int i = 0; i < dk; i++)
                kk += __half2float(Ksm[t * dk + i]) * __half2float(Ksm[s * dk + i]);
            Tm[t * Ceff + s] = bsm[t] * expf(clog[t] - clog[s]) * kk;
        } else {
            Tm[t * Ceff + s] = 0.f;
        }
    }
    __syncthreads();

    // --- Solve 1: Uloc = (I-trilT)^{-1}(beta.*v) ---
    for (int idx = tid; idx < Ceff * dv; idx += nthreads) {
        int t = idx / dv, j = idx % dv;
        float vtj = to_float(
            value[b * v_bs + (int64_t)(c0 + t) * Hkv * dv + (int64_t)g * dv + j]);
        Buf[t * dv + j] = bsm[t] * vtj;
    }
    __syncthreads();
    for (int t = 1; t < Ceff; t++) {
        for (int j = tid; j < dv; j += nthreads) {
            float acc = Buf[t * dv + j];
            for (int s = 0; s < t; s++)
                acc -= Tm[t * Ceff + s] * Buf[s * dv + j];
            Buf[t * dv + j] = acc;
        }
        __syncthreads();
    }
    for (int idx = tid; idx < LA_CHUNK * dv; idx += nthreads) {
        int t = idx / dv;
        Uloc_g[base * (LA_CHUNK * dv) + idx] = (t < Ceff) ? Buf[idx] : 0.f;
    }
    __syncthreads();

    // --- Solve 2: W = (I-trilT)^{-1}(diag(beta.*A) K) ---
    for (int idx = tid; idx < Ceff * dk; idx += nthreads) {
        int t = idx / dk, i = idx % dk;
        Buf[t * dk + i] = bsm[t] * Asm[t] * __half2float(Ksm[t * dk + i]);
    }
    __syncthreads();
    for (int t = 1; t < Ceff; t++) {
        for (int i = tid; i < dk; i += nthreads) {
            float acc = Buf[t * dk + i];
            for (int s = 0; s < t; s++)
                acc -= Tm[t * Ceff + s] * Buf[s * dk + i];
            Buf[t * dk + i] = acc;
        }
        __syncthreads();
    }
    for (int idx = tid; idx < LA_CHUNK * dk; idx += nthreads) {
        int t = idx / dk;
        W_g[base * (LA_CHUNK * dk) + idx] = (t < Ceff) ? Buf[idx] : 0.f;
    }
    // Store only the per-token rlast scalar (C floats/chunk) instead of the
    // full Ktilde = rlast*K tile (C*dk floats/chunk). The scan rebuilds Ktilde
    // from the original key tensor + rlast (bit-identical values, ~1/dk the
    // scratch). rlast is 0 for the padded tail (t >= Ceff) so Ktilde vanishes
    // there without the scan needing to read out-of-range key rows.
    for (int t = tid; t < LA_CHUNK; t += nthreads)
        rlast_g[base * LA_CHUNK + t] = (t < Ceff) ? rlast[t] : 0.f;
    if (tid == 0) alast_g[base] = a_last;
}

// Pass 2: per-head sequential affine scan over chunks.
template <typename T>
__global__ void la_pf_scan(
    const T* __restrict__ key,     // [B, seq, n_k*dk] -- to rebuild Ktilde
    const float* __restrict__ Uloc_g,
    const float* __restrict__ W_g,
    const float* __restrict__ rlast_g,  // [B*Hkv*nchunks_win, C]
    const float* __restrict__ alast_g,
    __half* __restrict__ S0_g,     // [B*Hkv*nchunks_win, dk*dv] chunk-start states
    __half* __restrict__ carry_g,  // [B*Hkv, dk*dv] state carried across windows
    T* __restrict__ state,         // final state out [B,Hkv,dk,dv]
    int seq_len, int Hkv, int n_k, int dk, int dv,
    int nchunks_win, int chunk_begin, int is_first, int is_last)
{
    const int bg = blockIdx.x;
    const int b  = bg / Hkv;
    const int g  = bg % Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int h_k = g * n_k / Hkv;
    const int64_t k_bs = (int64_t)seq_len * n_k * dk;
    const int64_t state_bs = (int64_t)Hkv * dk * dv;
    T* Sout = state + b * state_bs + (int64_t)g * dk * dv;
    __half* carry = carry_g + (int64_t)bg * dk * dv;

    extern __shared__ float smem[];
    __half* Ssm = (__half*)smem;                       // dk*dv (half) = 32KB
    float*  Buf1 = (float*)(Ssm + dk * dv);            // C*dk (float) = 16KB
    float*  Buf2 = Buf1 + LA_CHUNK * dk;               // C*dv (float) = 16KB

    // The first window seeds from the incoming state buffer (matches the serial
    // kernel, which reads S in place; == 0 for a fresh prefill). Later windows
    // resume from the fp16 carry rather than re-reading `state`: the running
    // state is fp16 in LDS, so a round trip through a narrower T (bf16 has 8
    // mantissa bits vs fp16's 10) would perturb it at every window boundary.
    for (int idx = tid; idx < dk * dv; idx += nthreads)
        Ssm[idx] = is_first ? __float2half(to_float(Sout[idx])) : carry[idx];
    __syncthreads();

    for (int cw = 0; cw < nchunks_win; cw++) {
        const int64_t base = ((int64_t)(b * Hkv + g) * nchunks_win + cw);
        const int c0   = (chunk_begin + cw) * LA_CHUNK;
        const int Ceff = min(LA_CHUNK, seq_len - c0);
        // store chunk-start state
        for (int idx = tid; idx < dk * dv; idx += nthreads)
            S0_g[base * (dk * dv) + idx] = Ssm[idx];
        // load W tile
        for (int idx = tid; idx < LA_CHUNK * dk; idx += nthreads)
            Buf1[idx] = W_g[base * (LA_CHUNK * dk) + idx];
        __syncthreads();
        // WS[t][j] = sum_i W[t][i] * S0[i][j]  -> Buf2
        for (int idx = tid; idx < LA_CHUNK * dv; idx += nthreads) {
            int t = idx / dv, j = idx % dv;
            float acc = 0.f;
            for (int i = 0; i < dk; i++)
                acc += Buf1[t * dk + i] * __half2float(Ssm[i * dv + j]);
            // U = Uloc - WS
            Buf2[idx] = Uloc_g[base * (LA_CHUNK * dv) + idx] - acc;
        }
        __syncthreads();
        // Rebuild Ktilde[s][i] = rlast[s]*K[s][i] into Buf1 (W no longer
        // needed). rlast_g is 0 for s >= Ceff, so the padded tail contributes
        // nothing; guard the key read so it never indexes past seq_len.
        for (int idx = tid; idx < LA_CHUNK * dk; idx += nthreads) {
            int t = idx / dk, i = idx % dk;
            float rl = rlast_g[base * LA_CHUNK + t];
            Buf1[idx] = (t < Ceff)
                ? rl * to_float(key[b * k_bs + (int64_t)(c0 + t) * n_k * dk +
                                    (int64_t)h_k * dk + i])
                : 0.f;
        }
        const float a_last = alast_g[base];
        __syncthreads();
        // S1[i][j] = a_last*S0[i][j] + sum_s Ktilde[s][i] U[s][j]
        for (int idx = tid; idx < dk * dv; idx += nthreads) {
            int i = idx / dv, j = idx % dv;
            float acc = a_last * __half2float(Ssm[idx]);
            for (int s = 0; s < LA_CHUNK; s++)
                acc += Buf1[s * dk + i] * Buf2[s * dv + j];
            Ssm[idx] = __float2half(acc);
        }
        __syncthreads();
    }
    // Only the final window publishes the state in the caller's dtype; the rest
    // hand off through the carry buffer.
    for (int idx = tid; idx < dk * dv; idx += nthreads) {
        if (is_last)
            Sout[idx] = from_float<T>(__half2float(Ssm[idx]));
        else
            carry[idx] = Ssm[idx];
    }
}

// Pass 3: per (head,chunk) parallel output. The dominant [Q;W]@S0 GEMM
// (M=2C, K=dk, N=dv) runs on the RDNA3 wave-matrix cores; the small P@U intra
// term stays scalar. Requires blockDim=256 (8 wave32 wavefronts) and dk,dv
// multiples of 16.
template <typename T>
__global__ void __launch_bounds__(256) la_pf_pass3_wmma(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ decay,
    const float* __restrict__ Uloc_g,
    const float* __restrict__ W_g,
    const __half* __restrict__ S0_g,
    T* __restrict__ output,
    int seq_len, int Hq, int Hkv, int n_k, int dk, int dv,
    float scale, int nchunks_win, int chunk_begin)
{
    // See la_pf_pass1_local: cw is window-local (indexes scratch), chunk_begin
    // + cw is global (indexes query/key/decay/output).
    const int cw  = blockIdx.x % nchunks_win;
    int tmp       = blockIdx.x / nchunks_win;
    const int g   = tmp % Hkv;
    const int b   = tmp / Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int c0   = (chunk_begin + cw) * LA_CHUNK;
    const int Ceff = min(LA_CHUNK, seq_len - c0);
    const int M    = 2 * LA_CHUNK;   // stacked [Q; W] rows

    const bool inverse_gqa = (Hq < Hkv);
    const int Hout  = Hq >= Hkv ? Hq : Hkv;
    const int h_q   = inverse_gqa ? (g * Hq / Hkv) : (g * (Hq / Hkv));
    const int h_out = inverse_gqa ? g : (g * (Hq / Hkv));
    const int h_k   = g * n_k / Hkv;

    const int64_t q_bs     = (int64_t)seq_len * Hq   * dk;
    const int64_t k_bs     = (int64_t)seq_len * n_k  * dk;
    const int64_t out_bs   = (int64_t)seq_len * Hout * dv;
    const int64_t decay_bs = (int64_t)seq_len * Hkv;
    const int64_t base = ((int64_t)(b * Hkv + g) * nchunks_win + cw);
    const __half* S0 = S0_g + base * (dk * dv);
    const float*  Wc = W_g  + base * (LA_CHUNK * dk);
    const float*  Ul = Uloc_g + base * (LA_CHUNK * dv);

    extern __shared__ float smem[];
    float*  QWS  = smem;                          // M*dv (float, WMMA result)
    float*  Pm   = QWS + M * dv;                  // C*C
    float*  Asm  = Pm + LA_CHUNK * LA_CHUNK;      // C
    float*  clog = Asm + LA_CHUNK;                // C
    __half* QW   = (__half*)(clog + LA_CHUNK);    // M*dk (half; rows 0..C-1 Q, C..2C-1 W)
    __half* Kh   = QW + M * dk;                   // C*dk (half)

    for (int idx = tid; idx < M * dk; idx += nthreads) QW[idx] = __float2half(0.f);
    __syncthreads();
    for (int idx = tid; idx < Ceff * dk; idx += nthreads) {
        int t = idx / dk, i = idx % dk;
        Kh[t * dk + i] = __float2half(to_float(
            key[b * k_bs + (int64_t)(c0 + t) * n_k * dk + (int64_t)h_k * dk + i]));
        QW[t * dk + i] = __float2half(to_float(
            query[b * q_bs + (int64_t)(c0 + t) * Hq * dk + (int64_t)h_q * dk + i]));
        QW[(LA_CHUNK + t) * dk + i] = __float2half(Wc[t * dk + i]);
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.f;
        for (int t = 0; t < Ceff; t++) {
            acc += to_float(decay[b * decay_bs + (int64_t)(c0 + t) * Hkv + g]);
            clog[t] = acc;
            Asm[t]  = expf(acc);
        }
    }
    __syncthreads();
    for (int idx = tid; idx < Ceff * Ceff; idx += nthreads) {
        int t = idx / Ceff, s = idx % Ceff;
        if (s <= t) {
            float qk = 0.f;
            for (int i = 0; i < dk; i++)
                qk += __half2float(QW[t * dk + i]) * __half2float(Kh[s * dk + i]);
            Pm[t * Ceff + s] = expf(clog[t] - clog[s]) * qk;
        }
    }
    __syncthreads();

    // WMMA: QWS[M x dv] = QW[M x dk] @ S0[dk x dv].
    const int wf   = tid / 32;
    const int nwf  = nthreads / 32;
    const int lane = (tid % 32) % LA_WT;
    const int pair = (tid % 32) / LA_WT;
    const int mt   = M  / LA_WT;
    const int nt   = dv / LA_WT;
    for (int tile = wf; tile < mt * nt; tile += nwf) {
        const int m_base = (tile / nt) * LA_WT;
        const int n_base = (tile % nt) * LA_WT;
        la_float8 acc = {};
        for (int k = 0; k < dk; k += LA_WT) {
            la_half16 a_frag, b_frag;
            for (int ele = 0; ele < LA_WT; ele++) {
                a_frag[ele] = (_Float16)__half2float(QW[(m_base + lane) * dk + k + ele]);
                b_frag[ele] = (_Float16)__half2float(S0[(k + ele) * dv + n_base + lane]);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
        }
        for (int ele = 0; ele < 8; ele++)
            QWS[(m_base + ele * 2 + pair) * dv + n_base + lane] = acc[ele];
    }
    __syncthreads();

    // Postprocess per output column: U = Uloc - (W@S0); O = scale*(A.*qs + P_lower@U).
    for (int j = tid; j < dv; j += nthreads) {
        float U[LA_CHUNK];
        for (int t = 0; t < Ceff; t++)
            U[t] = Ul[t * dv + j] - QWS[(LA_CHUNK + t) * dv + j];
        for (int t = 0; t < Ceff; t++) {
            float inter = Asm[t] * QWS[t * dv + j];
            float intra = 0.f;
            for (int s = 0; s <= t; s++)
                intra += Pm[t * Ceff + s] * U[s];
            output[b * out_bs + (int64_t)(c0 + t) * Hout * dv +
                   (int64_t)h_out * dv + j] =
                from_float<T>(scale * (inter + intra));
        }
    }
}

// Chunks processed per launch window. The passes run one window at a time, so
// the per-(head,chunk) scratch is sized for a window instead of the whole
// sequence and stops growing with seq_len. 64 chunks = 2048 tokens still gives
// B*Hkv*64 = 2048 blocks per launch at Hkv=32, far more than the 40 CUs need to
// stay saturated, so the chunk-parallel speedup is unaffected.
static constexpr int LA_WINDOW_CHUNKS = 64;

// Scratch layout for the chunk-parallel path. The buffer itself is owned by
// the runtime (RuntimeState::la_scratch, grown on demand, freed on session
// cleanup) and passed in; this kernel TU only computes the sub-buffer offsets.
// Kept in one place so hip_linear_attention_prefill_scratch_bytes (used by the
// wrapper to size the pool) and la_launch_parallel cannot drift apart.
struct LaScratchLayout { size_t oU, oW, oR, oA, oS, oC, total; };

static inline LaScratchLayout la_scratch_layout(int B, int seq_len, int Hkv,
                                                int dk, int dv) {
    const int nchunks = (seq_len + LA_CHUNK - 1) / LA_CHUNK;
    const int win = nchunks < LA_WINDOW_CHUNKS ? nchunks : LA_WINDOW_CHUNKS;
    const int64_t ncb = (int64_t)B * Hkv * win;  // # (head,chunk) blocks / window
    auto align = [](size_t x){ return (x + 255) & ~((size_t)255); };
    const size_t sz_Uloc  = (size_t)ncb * LA_CHUNK * dv * sizeof(float);
    const size_t sz_W     = (size_t)ncb * LA_CHUNK * dk * sizeof(float);
    const size_t sz_rlast = (size_t)ncb * LA_CHUNK * sizeof(float);
    const size_t sz_alast = (size_t)ncb * sizeof(float);
    const size_t sz_S0    = (size_t)ncb * dk * dv * sizeof(__half);
    // Cross-window carry of the recurrent state, one tile per (batch, kv head).
    // A single-window prefill hands off through `state` directly and never
    // touches the carry, so it does not pay for it.
    const size_t sz_carry = (nchunks > win)
        ? (size_t)B * Hkv * dk * dv * sizeof(__half) : 0;
    LaScratchLayout L;
    L.oU = 0;
    L.oW = align(L.oU + sz_Uloc);
    L.oR = align(L.oW + sz_W);
    L.oA = align(L.oR + sz_rlast);
    L.oS = align(L.oA + sz_alast);
    L.oC = align(L.oS + sz_S0);
    L.total = align(L.oC + sz_carry);
    return L;
}

// Bytes of runtime scratch the chunk-parallel prefill needs for this shape.
// Returns 0 for shapes the parallel path does not run (caller sizes nothing).
extern "C" HIP_KERNEL_API size_t hip_linear_attention_prefill_scratch_bytes(
    int B, int seq_len, int Hkv, int dk, int dv) {
    if (B <= 0 || seq_len <= 1 || Hkv <= 0 || dk <= 0 || dv <= 0) return 0;
    return la_scratch_layout(B, seq_len, Hkv, dk, dv).total;
}

template <typename T>
static int la_launch_parallel(
    hipStream_t s, const T* q, const T* k, const T* v, const T* dec,
    const T* beta, T* state, T* out,
    int B, int seq_len, int Hq, int Hkv, int Nk, int dk, int dv,
    float scale, int beta_per_head, void* scratch, size_t scratch_bytes)
{
    const int nchunks = (seq_len + LA_CHUNK - 1) / LA_CHUNK;

    const LaScratchLayout L = la_scratch_layout(B, seq_len, Hkv, dk, dv);
    if (!scratch || scratch_bytes < L.total) return 1;  // caller under-sized it
    char* sp = (char*)scratch;
    float*  Uloc_g  = (float*)(sp + L.oU);
    float*  W_g     = (float*)(sp + L.oW);
    float*  rlast_g = (float*)(sp + L.oR);
    float*  alast_g = (float*)(sp + L.oA);
    __half* S0_g    = (__half*)(sp + L.oS);
    __half* carry_g = (__half*)(sp + L.oC);

    const int block = LA_PAR_THREADS;
    // Pass 1 LDS: Tm + Buf(=C*dv) + 4 vecs (float), plus K tile (half).
    const size_t s1 = (size_t)(LA_CHUNK * LA_CHUNK + LA_CHUNK * dv +
                               4 * LA_CHUNK) * sizeof(float) +
                      (size_t)(LA_CHUNK * dk) * sizeof(__half);
    // Pass 2 (scan) LDS: S0 tile (half) + W,U tiles (float).
    const size_t s2 = (size_t)(dk * dv) * sizeof(__half) +
                      (size_t)(LA_CHUNK * dk + LA_CHUNK * dv) * sizeof(float);
    // Pass 3 (WMMA) LDS: QWS(M*dv) + Pm + (Asm,clog) floats, QW(M*dk) + K(C*dk) half.
    const size_t s3 = (size_t)(2 * LA_CHUNK * dv + LA_CHUNK * LA_CHUNK +
                               2 * LA_CHUNK) * sizeof(float) +
                      (size_t)(2 * LA_CHUNK * dk + LA_CHUNK * dk) * sizeof(__half);
    if (s1 > 64 * 1024 || s2 > 64 * 1024 || s3 > 64 * 1024) return 1;

    // Pass 2's serial scan uses a larger block for more resident waves.
    const int scan_block = 1024;

    // Window the sequence so the scratch above covers one window rather than
    // the whole prefill. The three passes are issued back-to-back on the same
    // stream, so pass 3 of a window has finished reading Uloc/W/S0 before pass
    // 1 of the next window overwrites them -- no extra synchronisation needed.
    // The scan carries the recurrent state between windows (carry_g).
    for (int cbeg = 0; cbeg < nchunks; cbeg += LA_WINDOW_CHUNKS) {
        const int rem = nchunks - cbeg;
        const int win = rem < LA_WINDOW_CHUNKS ? rem : LA_WINDOW_CHUNKS;
        const int64_t ncb = (int64_t)B * Hkv * win;  // (head,chunk) blocks
        const int is_first = (cbeg == 0) ? 1 : 0;
        const int is_last  = (cbeg + win >= nchunks) ? 1 : 0;

        hipLaunchKernelGGL(la_pf_pass1_local<T>, dim3((unsigned)ncb), dim3(block),
            s1, s, k, v, dec, beta, Uloc_g, W_g, rlast_g, alast_g,
            seq_len, Hkv, Nk, dk, dv, beta_per_head, win, cbeg);
        hipLaunchKernelGGL(la_pf_scan<T>, dim3((unsigned)(B * Hkv)), dim3(scan_block),
            s2, s, k, Uloc_g, W_g, rlast_g, alast_g, S0_g, carry_g, state,
            seq_len, Hkv, Nk, dk, dv, win, cbeg, is_first, is_last);
        hipLaunchKernelGGL(la_pf_pass3_wmma<T>, dim3((unsigned)ncb), dim3(256),
            s3, s, q, k, dec, Uloc_g, W_g, S0_g, out,
            seq_len, Hq, Hkv, Nk, dk, dv, scale, win, cbeg);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] la_launch_parallel failed: %s\n",
                hipGetErrorString(err));
        return (int)err;
    }
    return 0;
}

extern "C" int hip_linear_attention_prefill_chunked(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* decay,
    const void* beta,
    void* state,
    void* output,
    int64_t B,
    int64_t seq_len,
    int64_t Hq,
    int64_t Hkv,
    int64_t Nk,
    int64_t dk,
    int64_t dv,
    float scale,
    int64_t update_rule,
    int64_t decay_per_key_dim,
    int64_t beta_per_head,
    int64_t type,
    void* scratch,
    size_t scratch_bytes)
{
    // Decline (return 1 -> caller falls back to per-token loop) for any config
    // this kernel does not implement.
    if (update_rule != LA_RULE_GATED_DELTA) return 1;
    if (decay_per_key_dim != 0) return 1;          // only scalar log-decay
    if (seq_len <= 1) return 1;                      // decode handled elsewhere
    if (B <= 0 || Hkv <= 0 || Hq <= 0 || Nk <= 0) return 1;
    if (Hq >= Hkv ? (Hq % Hkv != 0) : (Hkv % Hq != 0)) return 1;
    if (Hkv % Nk != 0) return 1;
    if (!decay || !beta) return 1;                  // gated_delta needs both

    const int dk_i = static_cast<int>(dk);
    const int dv_i = static_cast<int>(dv);

    const int beta_per_head_i = beta_per_head ? 1 : 0;

    // Pass 3 runs the stacked [Q;W]@S0 GEMM on the RDNA3 WMMA cores
    // (16x16x16), which requires dk and dv to be multiples of 16; decline
    // otherwise (return 1) so the runtime falls back to the per-token loop.
    if (dk_i % 16 != 0 || dv_i % 16 != 0) return 1;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int rc;
    if (type == 0)
        rc = la_launch_parallel<float>(hip_stream,
            static_cast<const float*>(query), static_cast<const float*>(key),
            static_cast<const float*>(value), static_cast<const float*>(decay),
            static_cast<const float*>(beta), static_cast<float*>(state),
            static_cast<float*>(output), (int)B, (int)seq_len, (int)Hq,
            (int)Hkv, (int)Nk, dk_i, dv_i, scale, beta_per_head_i,
            scratch, scratch_bytes);
    else if (type == 1)
        rc = la_launch_parallel<__half>(hip_stream,
            static_cast<const __half*>(query), static_cast<const __half*>(key),
            static_cast<const __half*>(value), static_cast<const __half*>(decay),
            static_cast<const __half*>(beta), static_cast<__half*>(state),
            static_cast<__half*>(output), (int)B, (int)seq_len, (int)Hq,
            (int)Hkv, (int)Nk, dk_i, dv_i, scale, beta_per_head_i,
            scratch, scratch_bytes);
    else if (type == 2)
        rc = la_launch_parallel<__hip_bfloat16>(hip_stream,
            static_cast<const __hip_bfloat16*>(query),
            static_cast<const __hip_bfloat16*>(key),
            static_cast<const __hip_bfloat16*>(value),
            static_cast<const __hip_bfloat16*>(decay),
            static_cast<const __hip_bfloat16*>(beta),
            static_cast<__hip_bfloat16*>(state),
            static_cast<__hip_bfloat16*>(output), (int)B, (int)seq_len,
            (int)Hq, (int)Hkv, (int)Nk, dk_i, dv_i, scale, beta_per_head_i,
            scratch, scratch_bytes);
    else
        return 1;

    // 0 on success; nonzero -> caller falls back to the per-token loop.
    return rc;
}
