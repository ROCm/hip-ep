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
// Chunked-parallel gated-delta PREFILL kernel
//===----------------------------------------------------------------------===//
//
// Replaces the per-token launch loop (one decode-kernel launch per timestep,
// each streaming the whole recurrent state from global) with a SINGLE launch
// that processes the entire sequence.  One block per (batch, kv_head); each
// block loops over chunks of LA_CHUNK tokens, keeping the recurrent state S in
// GLOBAL (read once / written once per chunk) and the per-chunk tiles + the
// [C,C] coupling matrices in LDS.
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
static constexpr int LA_PREFILL_THREADS = 256;

template <typename T, int CHUNK>
__global__ void linear_attention_prefill_chunked_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ state,        // [B, Hkv, dk, dv], pre-initialized
    T* __restrict__ output,       // [B, seq_len, Hout*dv]
    int seq_len, int Hq, int Hkv, int n_k, int dk, int dv,
    float scale, int beta_per_head, int G)
{
    // dv-split: grid = B*Hkv*G, ordered bg = (b*Hkv + g)*G + split.  Each block
    // owns the contiguous output-column stripe [j0, j0+dvw) of the dv (value)
    // axis.  Columns are fully independent across splits -- the recurrent state
    // S[:,j], the intra-chunk U[:,j] and the output O[:,j] all factor per-j, so
    // no cross-block communication is required.  The per-head [C,C] coupling
    // matrices T/P (functions of q.k / k.k over dk, independent of j) and the K
    // tile are recomputed by every split (cheap C*C*dk redundancy) in exchange
    // for G-fold more resident workgroups + a smaller per-block LDS footprint.
    const int bg    = blockIdx.x;
    const int split = bg % G;
    const int bh    = bg / G;
    const int b     = bh / Hkv;
    const int g     = bh % Hkv;
    const int dvw   = dv / G;              // stripe width (host guarantees dv%G==0)
    const int j0    = split * dvw;         // first global dv column owned here
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const bool inverse_gqa = (Hq < Hkv);
    const int Hout  = Hq >= Hkv ? Hq : Hkv;
    const int h_q   = inverse_gqa ? (g * Hq / Hkv) : (g * (Hq / Hkv));
    const int h_out = inverse_gqa ? g : (g * (Hq / Hkv));
    const int h_k   = g * n_k / Hkv;

    const int64_t q_bs   = (int64_t)seq_len * Hq   * dk;
    const int64_t k_bs   = (int64_t)seq_len * n_k  * dk;
    const int64_t v_bs   = (int64_t)seq_len * Hkv  * dv;
    const int64_t out_bs = (int64_t)seq_len * Hout * dv;
    const int64_t decay_bs = (int64_t)seq_len * Hkv;          // scalar per head
    const int64_t beta_bs  = beta_per_head ? (int64_t)seq_len * Hkv
                                           : (int64_t)seq_len;
    const int64_t state_bs = (int64_t)Hkv * dk * dv;

    T* S = state + b * state_bs + (int64_t)g * dk * dv;        // [dk,dv]

    extern __shared__ float smem[];
    float* Ksm = smem;                              // [CHUNK*dk]
    float* Usm = Ksm + CHUNK * dk;                  // [CHUNK*dvw] : B -> U
    float* Tm  = Usm + CHUNK * dvw;                 // [CHUNK*CHUNK]
    float* Pm  = Tm + CHUNK * CHUNK;                // [CHUNK*CHUNK]
    float* Asm = Pm + CHUNK * CHUNK;                // [CHUNK]
    float* clog = Asm + CHUNK;                      // [CHUNK]
    float* bsm = clog + CHUNK;                      // [CHUNK]
    float* rlast = bsm + CHUNK;                     // [CHUNK] a_last/A_s

    for (int c0 = 0; c0 < seq_len; c0 += CHUNK) {
        const int Ceff = min(CHUNK, seq_len - c0);

        // load K tile, beta; build cumlog/A (single-thread prefix scan)
        for (int idx = tid; idx < Ceff * dk; idx += nthreads) {
            int t = idx / dk, i = idx % dk;
            Ksm[t * dk + i] = to_float(
                key[b * k_bs + (int64_t)(c0 + t) * n_k * dk +
                    (int64_t)h_k * dk + i]);
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
                float gt = to_float(
                    decay[b * decay_bs + (int64_t)(c0 + t) * Hkv + g]);
                acc += gt;
                clog[t] = acc;
                Asm[t] = expf(acc);
            }
        }
        __syncthreads();
        const float a_last = Asm[Ceff - 1];
        // Precompute the per-s state-update scale (A_{C-1}/A_s) ONCE (Ceff exps
        // total) instead of recomputing it inside the dk*dv phase-6 loop below.
        // Compute it as exp(clog_last - clog_s), NOT as the ratio a_last/A_s:
        // under strong intra-chunk decay clog_s can be very negative, so both
        // A_s = exp(clog_s) AND a_last = exp(clog_last) underflow to 0 in fp32
        // and the ratio becomes 0/0 = NaN (which then poisons S and every
        // subsequent chunk -> all-garbage output). The difference clog_last -
        // clog_s is bounded (<= 0 for log-decay g <= 0), so exp() of it is in
        // (0, 1] and always finite. This also matches the exp(clog_t - clog_s)
        // form already used for the P/T ratios above (internal consistency).
        for (int s = tid; s < Ceff; s += nthreads)
            rlast[s] = expf(clog[Ceff - 1] - clog[s]);
        __syncthreads();

        // B[t][j] = beta_t v_t[j] - beta_t A_t (S0^T k_t)[j]  -> Usm
        // Register-blocked over t: read each S[i,j] from GLOBAL once and fold it
        // into all Ceff partial sums sk[t], instead of re-reading the whole S
        // column once per t (LA_CHUNK x fewer global S loads).
        for (int jj = tid; jj < dvw; jj += nthreads) {
            const int j = j0 + jj;
            float sk[CHUNK];
            for (int t = 0; t < Ceff; t++)
                sk[t] = 0.f;
            for (int i = 0; i < dk; i++) {
                float s = to_float(S[i * dv + j]);
                for (int t = 0; t < Ceff; t++)
                    sk[t] += s * Ksm[t * dk + i];
            }
            for (int t = 0; t < Ceff; t++) {
                float vtj = to_float(
                    value[b * v_bs + (int64_t)(c0 + t) * Hkv * dv +
                          (int64_t)g * dv + j]);
                Usm[t * dvw + jj] = bsm[t] * vtj - bsm[t] * Asm[t] * sk[t];
            }
        }

        // T[t][s] (strict lower), P[t][s] (lower incl diag)
        for (int idx = tid; idx < Ceff * Ceff; idx += nthreads) {
            int t = idx / Ceff, s = idx % Ceff;
            if (s < t) {
                float kk = 0.f;
                for (int i = 0; i < dk; i++)
                    kk += Ksm[t * dk + i] * Ksm[s * dk + i];
                Tm[t * Ceff + s] = bsm[t] * expf(clog[t] - clog[s]) * kk;
            } else if (s == t) {
                Tm[t * Ceff + s] = 0.f;
            }
            if (s <= t) {
                float qk = 0.f;
                for (int i = 0; i < dk; i++)
                    qk += to_float(
                              query[b * q_bs + (int64_t)(c0 + t) * Hq * dk +
                                    (int64_t)h_q * dk + i]) *
                          Ksm[s * dk + i];
                Pm[t * Ceff + s] = expf(clog[t] - clog[s]) * qk;
            }
        }
        __syncthreads();

        // forward substitution: U[t] = B[t] - sum_{s<t} T[t,s] U[s] (serial t)
        for (int t = 1; t < Ceff; t++) {
            for (int jj = tid; jj < dvw; jj += nthreads) {
                float acc = Usm[t * dvw + jj];
                for (int s = 0; s < t; s++)
                    acc -= Tm[t * Ceff + s] * Usm[s * dvw + jj];
                Usm[t * dvw + jj] = acc;
            }
            __syncthreads();
        }

        // O[t][j] = scale*( A_t (q_t . S0[:,j]) + sum_{s<=t} P[t,s] U[s,j] )
        // uses chunk-START S0 (global, not yet updated)
        // Register-blocked over t: read each S0[i,j] from GLOBAL once and fold
        // it into all Ceff inter-chunk partial sums qs[t] (the (A.*Q) @ S0 term),
        // mirroring the phase-2 blocking (LA_CHUNK x fewer global S0 loads).
        for (int jj = tid; jj < dvw; jj += nthreads) {
            const int j = j0 + jj;
            float qs[CHUNK];
            for (int t = 0; t < Ceff; t++)
                qs[t] = 0.f;
            for (int i = 0; i < dk; i++) {
                float s = to_float(S[i * dv + j]);
                for (int t = 0; t < Ceff; t++)
                    qs[t] += to_float(
                                 query[b * q_bs + (int64_t)(c0 + t) * Hq * dk +
                                       (int64_t)h_q * dk + i]) *
                             s;
            }
            for (int t = 0; t < Ceff; t++) {
                float inter = Asm[t] * qs[t];
                float intra = 0.f;
                for (int s = 0; s <= t; s++)
                    intra += Pm[t * Ceff + s] * Usm[s * dvw + jj];
                output[b * out_bs + (int64_t)(c0 + t) * Hout * dv +
                       (int64_t)h_out * dv + j] =
                    from_float<T>(scale * (inter + intra));
            }
        }
        __syncthreads();   // all reads of S0 done before the update below

        // S = a_last*S + Ktilde^T U,  Ktilde[s][i] = (a_last/A_s) Ksm[s][i]
        for (int idx = tid; idx < dk * dvw; idx += nthreads) {
            int i = idx / dvw, jj = idx % dvw;
            int j = j0 + jj;
            float acc = a_last * to_float(S[i * dv + j]);
            for (int s = 0; s < Ceff; s++)
                acc += rlast[s] * Ksm[s * dk + i] * Usm[s * dvw + jj];
            S[i * dv + j] = from_float<T>(acc);
        }
        __syncthreads();   // S update complete before next chunk reads it
    }
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
    int64_t type)
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

    // --- Tunables (env). --------------------------------------------------
    // Defaults were chosen from a measured sweep on the Qwen3.5 gated-delta
    // prefill shape (B=1, Hkv=32, dk=dv=128) on gfx1151:
    //   * chunk=16 is the dominant lever: it cuts the per-chunk O(C^2*dk) T/P
    //     matmuls (C^2 halves overall vs chunk=32), the serial tri-solve
    //     length, and the per-thread sk[C]/qs[C] register arrays -> ~2.4x
    //     faster kernel, ~16-17% lower end-to-end TTFT. (chunk=32 leaves it
    //     ~2.4x slower; chunk=8 over-splits; chunk>=24 regresses to ~chunk=32.)
    //   * G=2 (dv-split) then adds a further ~2-4%: grid B*Hkv=32 underfills
    //     the ~40-CU GPU, and splitting the (independent) dv columns into 2
    //     workgroups per head fills it. Larger G costs more than it gains
    //     (T/P is recomputed per split). Only auto-enabled for small grids so
    //     shapes that already saturate the GPU aren't burdened with the
    //     redundant T/P recompute.
    //   HIPDNN_EP_LA_PREFILL_G:     override dv-split factor (>=1). Must divide
    //       dv; else clamped to 1. (Unset => auto, see below.)
    //   HIPDNN_EP_LA_PREFILL_CHUNK: override tokens per chunk
    //       (8/16/24/32/48/64; default 16).
    static const int la_G_env = [] {
        const char* e = std::getenv("HIPDNN_EP_LA_PREFILL_G");
        return e ? atoi(e) : 0;                  // 0 => auto
    }();
    static const int la_chunk = [] {
        const char* e = std::getenv("HIPDNN_EP_LA_PREFILL_CHUNK");
        int v = e ? atoi(e) : 16;                // measured optimum
        return (v == 8 || v == 16 || v == 24 || v == 32 || v == 48 || v == 64)
                   ? v : 16;
    }();

    const int base_grid = static_cast<int>(B * Hkv);
    int G;
    if (la_G_env >= 1)                            // explicit override
        G = (dv_i % la_G_env == 0) ? la_G_env : 1;
    else                                          // auto: split only small grids
        G = (dv_i % 2 == 0 && base_grid <= 48) ? 2 : 1;
    int dvw = dv_i / G;

    // LDS budget: Ksm[chunk*dk] + Usm[chunk*dvw] + Tm+Pm[2*chunk*chunk] +
    // 4 vectors of length chunk (Asm, clog, bsm, rlast).
    auto smem_of = [&](int c) -> size_t {
        return static_cast<size_t>(c * dk_i + c * dvw + 2 * c * c + 4 * c) *
               sizeof(float);
    };
    size_t smem_bytes = smem_of(la_chunk);
    if (smem_bytes > 64 * 1024) {                   // stripe still too big?
        G = 1;                                       // fall back to un-split
        dvw = dv_i;
        smem_bytes = smem_of(la_chunk);
        if (smem_bytes > 64 * 1024) return 1;        // give up -> per-token loop
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int grid  = static_cast<int>(B * Hkv) * G;
    const int block = LA_PREFILL_THREADS;
    const int beta_per_head_i = beta_per_head ? 1 : 0;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_linear_attention_prefill_chunked: B=%lld "
        "seq_len=%lld Hq=%lld Hkv=%lld Nk=%lld dk=%d dv=%d scale=%.6f "
        "chunk=%d G=%d dvw=%d grid=%d block=%d smem=%zu\n",
        (long long)B, (long long)seq_len, (long long)Hq, (long long)Hkv,
        (long long)Nk, dk_i, dv_i, (double)scale, la_chunk, G, dvw, grid,
        block, smem_bytes);

#define LA_PREFILL_LAUNCH(TYPE, CHUNK)                                        \
    hipLaunchKernelGGL(                                                       \
        (linear_attention_prefill_chunked_kernel<TYPE, CHUNK>),              \
        dim3(grid), dim3(block), smem_bytes, hip_stream,                     \
        static_cast<const TYPE*>(query), static_cast<const TYPE*>(key),      \
        static_cast<const TYPE*>(value), static_cast<const TYPE*>(decay),    \
        static_cast<const TYPE*>(beta), static_cast<TYPE*>(state),           \
        static_cast<TYPE*>(output), static_cast<int>(seq_len),              \
        static_cast<int>(Hq), static_cast<int>(Hkv), static_cast<int>(Nk),   \
        dk_i, dv_i, scale, beta_per_head_i, G)

#define LA_PREFILL_DISPATCH(TYPE)                                             \
    do {                                                                     \
        switch (la_chunk) {                                                  \
        case 8:  LA_PREFILL_LAUNCH(TYPE, 8);  break;                          \
        case 16: LA_PREFILL_LAUNCH(TYPE, 16); break;                         \
        case 24: LA_PREFILL_LAUNCH(TYPE, 24); break;                         \
        case 48: LA_PREFILL_LAUNCH(TYPE, 48); break;                         \
        case 64: LA_PREFILL_LAUNCH(TYPE, 64); break;                         \
        default: LA_PREFILL_LAUNCH(TYPE, 32); break;                         \
        }                                                                    \
    } while (0)

    if (type == 0) {
        LA_PREFILL_DISPATCH(float);
    } else if (type == 1) {
        LA_PREFILL_DISPATCH(__half);
    } else if (type == 2) {
        LA_PREFILL_DISPATCH(__hip_bfloat16);
    } else {
        return 1;
    }

#undef LA_PREFILL_DISPATCH
#undef LA_PREFILL_LAUNCH

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_linear_attention_prefill_chunked launch "
                "failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);   // negative-ish; caller treats !=0,!=1
    }
    return 0;
}
