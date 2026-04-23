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
//   k_smem[dk] | v_smem[dv] | aux_smem[dk] | sk_smem[dv]
//   aux_smem holds exp(g) during phases 1-2, then q during phase 3.
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
    const int bg = blockIdx.x;
    const int b  = bg / Hkv;
    const int g  = bg % Hkv;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int state_size = dk * dv;

    // Output head count: max(Hq, Hkv).  Determines stride of output and
    // controls which branch of Phase 3 runs.
    const int Hout = Hq >= Hkv ? Hq : Hkv;

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

    extern __shared__ float smem[];
    float* k_smem   = smem;
    float* v_smem   = smem + dk;
    float* aux_smem = smem + dk + dv;
    float* sk_smem  = smem + 2 * dk + dv;
    const T* k_ptr = key   + b * k_batch_stride + (int64_t)h_k * dk;    // k_ptr = &key[b, t, h_k, 0]
    const T* v_ptr = value + b * v_batch_stride + (int64_t)g   * dv;    // v_ptr = &value[b, t, g, 0]
    T* S = state + ((int64_t)b * Hkv + g) * dk * dv;

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

    // === Phase 2: state update ===
    for (int idx = tid; idx < state_size; idx += nthreads) {
        const int i = idx / dv;
        const int j = idx % dv;

        float s_val = to_float(S[idx]);
        const float k_i = k_smem[i];
        const float v_j = v_smem[j];

        switch (update_rule) {
        case LA_RULE_LINEAR:
            s_val += k_i * v_j;
            break;
        case LA_RULE_GATED:
            s_val = aux_smem[i] * s_val + k_i * v_j;
            break;
        case LA_RULE_DELTA:
            s_val += beta_val * k_i * (v_j - sk_smem[j]);
            break;
        case LA_RULE_GATED_DELTA:
            s_val = aux_smem[i] * s_val + beta_val * k_i * (v_j - sk_smem[j]);
            break;
        }

        S[idx] = from_float<T>(s_val);
    }
    __syncthreads();

    // === Phase 3: output = scale * S^T @ q  (per query head) ===
    // Dispatches on GQA direction:
    //   Standard (Hq >= Hkv): loop over Hq/Hkv Q heads sharing this KV state,
    //                          write each to output[h_q * dv] (Hout == Hq).
    //   Inverse  (Hq <  Hkv): single Q head shared across Hkv/Hq KV heads;
    //                          write to output[g * dv] (Hout == Hkv).
    float* q_smem = aux_smem;

    auto readout = [&](const T* q_ptr, T* out_ptr) {
        for (int i = tid; i < dk; i += nthreads)
            q_smem[i] = to_float(q_ptr[i]);
        __syncthreads();

        for (int j = tid; j < dv; j += nthreads) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++)
                sum += q_smem[i] * to_float(S[i * dv + j]);
            out_ptr[j] = from_float<T>(scale * sum);
        }
        __syncthreads();
    };

    if (Hq >= Hkv) {
        const int heads_per_group = Hq / Hkv;
        for (int hl = 0; hl < heads_per_group; hl++) {
            const int h = g * heads_per_group + hl;
            const T* q_ptr = query  + b * q_batch_stride   + (int64_t)h * dk;
            T* out_ptr     = output + b * out_batch_stride + (int64_t)h * dv;
            readout(q_ptr, out_ptr);
        }
    } else {
        // Inverse GQA: this KV head maps to a single Q head (shared across
        // Hkv / Hq consecutive KV heads), but each KV head writes its own
        // output slot -> Hout = Hkv output heads.
        const int h_q = g * Hq / Hkv;
        const T* q_ptr = query  + b * q_batch_stride   + (int64_t)h_q * dk;
        T* out_ptr     = output + b * out_batch_stride + (int64_t)g   * dv;
        readout(q_ptr, out_ptr);
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
    // Normalise to canonical 0/1 so the kernel can rely on the invariant.
    const int decay_per_key_dim_i = decay_per_key_dim ? 1 : 0;
    const int beta_per_head_i     = beta_per_head     ? 1 : 0;

    size_t smem_bytes = static_cast<size_t>(2 * dk_i + 2 * dv_i) * sizeof(float);

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
            linear_attention_decode_kernel<float>,
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
            linear_attention_decode_kernel<__half>,
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
            linear_attention_decode_kernel<__hip_bfloat16>,
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
