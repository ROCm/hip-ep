/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "hip_custom_kernels.h"

static constexpr int GQA_THREADS = 256;
static constexpr int WAVE_SIZE = 32;  // RDNA wave32; not portable to CDNA/wave64
static constexpr float LOG2E = 1.4426950408889634f;

//===----------------------------------------------------------------------===//
// Algorithm: ONNX com.microsoft.GroupQueryAttention
//===----------------------------------------------------------------------===//
//
// This module implements the device kernels for the ONNX Runtime
// GroupQueryAttention (GQA) contrib operator as specified in
// docs/ContribOperators.md. The operator performs multi-head attention
// with grouped key/value heads (GQA), optional rotary positional
// embeddings (RoPE), and KV cache management.
//
// ONNX OPERATOR SPECIFICATION
// -----------------------------------------------------------------------
//   Attributes:
//     num_heads       (H)  — number of query heads             (e.g. 32)
//     kv_num_heads    (G)  — number of key/value heads         (e.g. 8)
//     scale                — attention scaling factor           (1/sqrt(d))
//     do_rotary            — apply RoPE inside the operator    (0 or 1)
//     rotary_interleaved   — RoPE pairing mode                 (0=half-rotated)
//     softcap              — soft capping for logits            (0=disabled)
//     local_window_size    — sliding window size for local attention (-1=full)
//     smooth_softmax       — use smooth softmax                (0 or 1)
//
//   Inputs (separate Q/K/V path):
//     [0] query       : (B, S, H*d)      — FP16, BSNH layout (3D)
//     [1] key         : (B, S, G*d)      — FP16, BSNH layout (3D)
//     [2] value       : (B, S, G*d)      — FP16, BSNH layout (3D)
//
//   Inputs (packed QKV path — key and value are absent):
//     [0] query       : (B, S, (H+2G)*d) — FP16, packed Q|K|V (3D)
//
//   Common inputs:
//     [3] past_key    : (B, G, P, d)     — FP16, BNSH layout (4D), P=past buf len
//     [4] past_value  : (B, G, P, d)     — FP16, BNSH layout (4D)
//     [5] seqlens_k   : (B,)             — INT32, seqlens_k[b] = total_seqlen - 1
//     [6] total_seqlen : scalar           — INT32
//     [7] cos_cache   : (max_pos, d/2)   — optional, when do_rotary=1
//     [8] sin_cache   : (max_pos, d/2)   — optional, when do_rotary=1
//     [9] position_ids: (B,) or (B,S)    — optional, not yet implemented
//    [10] attn_bias   : (B,H,S,T) etc.   — optional, not yet implemented
//    [11] head_sink   : (H,)             — optional, per-head smooth softmax factor
//
//   Outputs:
//     [0] output      : (B, S, H*d)      — FP16, BSNH layout (3D)
//     [1] present_key : (B, G, T, d)     — FP16, BNSH layout (4D), T=total_seqlen
//     [2] present_value: (B, G, T, d)    — FP16, BNSH layout (4D)
//
//   Layout equivalence:
//     ONNX 3D BSNH [B, S, H*d] == our 4D BSHD [B, S, H, d] (identical memory)
//     ONNX 4D BNSH [B, G, T, d] == our 4D BNSD [B, G, T, d] (identical memory)
//
// ALGORITHM — 12-STEP PIPELINE (Step 0 + Steps 1-11)
// -----------------------------------------------------------------------
// Given: B=batch, S=seq_q, T=total_seqlen(=past_len+S), H=num_heads,
//        G=kv_num_heads, d=head_dim, HPG=H/G (heads per group)
//
// Step 0: Split packed QKV (when key/value are absent)
//   Split packed input [B, S, (H+2G)*d] into Q [B*S, H*d], K [B*S, G*d],
//   V [B*S, G*d].  Skipped when separate Q/K/V inputs are provided.
//
// Step 1-2: RoPE (when do_rotary=1, skipped when do_rotary=0)
//   For each position s in [0, S), compute pos = past_len + s.
//   Half-rotated (non-interleaved) formula per head:
//     Q_out[d_i]          = Q[d_i]*cos[pos,d_i] - Q[d_i+half]*sin[pos,d_i]
//     Q_out[d_i+half]     = Q[d_i+half]*cos[pos,d_i] + Q[d_i]*sin[pos,d_i]
//   where half = d/2, d_i in [0, half). Same formula for K.
//   cos/sin cache: precomputed as cos(pos * theta_i), sin(pos * theta_i)
//   where theta_i = 1 / (base^(2*i/d)), base typically 10000.
//   Ref: onnxruntime/core/mlas/lib/rotary_embedding.cpp (MlasRotaryEmbedOneRow)
//
// Step 3: Q Transpose — BSHD [B,S,H,d] -> BNSD [B,H,S,d]
//   Transposes the middle two dimensions for batched GEMM compatibility.
//
// Step 4-5: KV Cache Update
//   Write new K (after RoPE) from BSHD [B,S,G,d] into BNSD cache [B,G,present_seq,d]
//   at positions [past_len, past_len+S). Same for V (V is never RoPE'd).
//   Fuses BSHD->BNSD transpose with the cache write.
//   Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h (ConcatStateChunkGQA)
//
// Step 6-7: KV Expand — G groups -> H heads
//   For each query head h, copy data from KV group g = h / HPG.
//   Produces K_exp, V_exp each of shape [B*H, T, d] (flattened for batched GEMM).
//   Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h line 238
//        (k = K + kv_input_chunk_length * (i / kv_num_heads_factor))
//
// Step 8: Score GEMM — S = K_exp^T * Q_trans * scale
//   hipBLASLt batched GEMM over B*H heads:
//     A = K_exp  [d, T]  col-major, transA=T -> op(A) = [T, d]
//     B = Q_trans [d, S]  col-major, transB=N -> op(B) = [d, S]
//     D = S      [T, S]  col-major
//   alpha = 1/sqrt(d), beta = 0.
//   Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h line 259
//        (math::GemmEx Q * K^T with alpha=scale)
//
// Step 9: Causal Mask + Softmax
//   Causal mask (prefill only, skipped when S==1):
//     S[k, q] = -inf  where k > past_len + q  (future tokens)
//   When local_window_size > 0, also masks distant past:
//     S[k, q] = -inf  where k < past_len + q - local_window_size + 1
//   This makes each query attend only to a sliding window of tokens.
//   Equivalent to ORT's approach of zeroing future positions and running
//   softmax on the valid window only (gqa_attention_base.h line 337).
//   Then column-wise softmax: for each query column, compute
//     softmax(S[:,q]) = exp(S[:,q] - max) / sum(exp(S[:,q] - max))
//   Smooth softmax (when head_sink is provided or smooth_softmax=1):
//     softmax_i = exp(x_i - m) / (exp(s - m) + sum_j exp(x_j - m))
//     where s = head_sink[h] (per-head) or 0 (ORT default when no sink).
//   Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h lines 297-354
//
// Step 10: Value GEMM — O = V_exp * softmax(S)
//   hipBLASLt batched GEMM over B*H heads:
//     A = V_exp  [d, T]  col-major, transA=N -> op(A) = [d, T]
//     B = S      [T, S]  col-major, transB=N -> op(B) = [T, S]
//     D = O      [d, S]  col-major
//   alpha = 1, beta = 0.
//   Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h line 458
//        (math::GemmEx attn_probs * V)
//
// Step 11: O Transpose — BNSD [B,H,S,d] -> BSHD [B,S,H,d]
//   Transposes back to match the ONNX output layout [B, S, H*d].
//   Ref: onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h line 414
//        (memcpy loop transposing BNSH -> BSNH)
//
// ONNX RUNTIME REFERENCE FILES
// -----------------------------------------------------------------------
//   Operator spec:       docs/ContribOperators.md (com.microsoft.GroupQueryAttention)
//   CPU GQA compute:     contrib_ops/cpu/bert/gqa_attention_base.h
//   CPU GQA entry:       contrib_ops/cpu/bert/group_query_attention.cc
//   Input validation:    contrib_ops/cpu/bert/group_query_attention_helper.h
//   RoPE (CPU, MLAS):    core/mlas/lib/rotary_embedding.cpp
//   RoPE (CUDA):         contrib_ops/cuda/bert/rotary_embedding_impl.cu
//   CUDA GQA:            contrib_ops/cuda/bert/group_query_attention_impl.cu
//   CUDA fused QKV+RoPE: contrib_ops/cuda/bert/group_query_attention_qkv.cuh
//
// DATA FLOW (GPU buffer names)
// -----------------------------------------------------------------------
//  Packed QKV path (when key/value absent):
//    query --[Step 0: Split]--> d_Q, d_K, d_V
//  Separate Q/K/V path:
//    d_Q = query, d_K = key, d_V = value
//
//  d_Q --[Step 1: RoPE]--> d_Qroped --[Step 3: Transpose]--> d_Qtrans
//  d_K --[Step 2: RoPE]--> d_Kroped --[Step 4: Cache]-------> d_Kcache
//  d_V --------------------------------[Step 5: Cache]-------> d_Vcache
//
//  d_Kcache --[Step 6: Expand]--> d_Kexp --+
//  d_Vcache --[Step 7: Expand]--> d_Vexp --+--> GEMMs
//  d_Qtrans ----------------------------->-+
//
//  d_Kexp, d_Qtrans --[Step 8:  Score GEMM]-----------> d_S
//  d_S ---------------[Step 9:  Causal Mask + Softmax]-> d_S (in-place)
//  d_Vexp, d_S -------[Step 10: Value GEMM]-----------> d_O
//  d_O ---------------[Step 11: Transpose]-------------> d_Obshd (output)
//
//===----------------------------------------------------------------------===//

// -------------------------------------------------------------------------
// Step 1-2: RoPE -- Rotary Positional Embedding (half-rotated)
// -------------------------------------------------------------------------
// Applies to Q (Step 1) and K (Step 2) independently.
// Layout: BSHD [B, seq, num_heads, head_dim]
//
// For position s, head h, dimension d_i (d_i < half_rot):
//   pos = past_len + s
//   out[d_i]      = in[d_i]*cos[pos,d_i] - in[d_i+half]*sin[pos,d_i]
//   out[d_i+half] = in[d_i+half]*cos[pos,d_i] + in[d_i]*sin[pos,d_i]
//
// Thread: one per (b,s,h,d_i) where d_i in [0, half_rot). blockIdx.y = batch.
// Ref: onnxruntime/core/mlas/lib/rotary_embedding.cpp (MlasRotaryEmbedOneRow)
// -------------------------------------------------------------------------

__global__ void rope_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ cos_cache,
    const __half* __restrict__ sin_cache,
    int seq_len, int num_heads, int head_dim,
    int half_rot, int past_len,
    const int* __restrict__ seqlens_k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  // When seqlens_k is provided, read past_len from device memory
  // (ORT convention: seqlens_k[b] = total_tokens - 1, so past_len = total - sq).
  // All threads read the same address; hardware coalesces into a broadcast.
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - seq_len) : past_len;
  int elems = seq_len * num_heads * half_rot;
  if (idx >= elems) return;

  int d = idx % half_rot;
  int remaining = idx / half_rot;
  int h = remaining % num_heads;
  int s = remaining / num_heads;
  if (s >= seq_len) return;

  int pos = eff_past_len + s;
  int base = ((batch_idx * seq_len + s) * num_heads + h) * head_dim;

  float x1 = __half2float(input[base + d]);
  float x2 = __half2float(input[base + d + half_rot]);
  float c  = __half2float(cos_cache[pos * half_rot + d]);
  float sn = __half2float(sin_cache[pos * half_rot + d]);

  output[base + d]            = __float2half(x1 * c - x2 * sn);
  output[base + d + half_rot] = __float2half(x2 * c + x1 * sn);
}

// -------------------------------------------------------------------------
// Step 3 / Step 11: Middle-dimension transpose
// -------------------------------------------------------------------------
// Transposes dims 1 and 2 of a 4D tensor [B, dim1, dim2, D]:
//   Step 3  (Q):  BSHD [B,S,H,d] -> BNSD [B,H,S,d]  (dim1=S, dim2=H)
//   Step 11 (O):  BNSD [B,H,S,d] -> BSHD [B,S,H,d]  (dim1=H, dim2=S)
//
// Thread: one per element in dim1*dim2*D sub-tensor. blockIdx.y = batch.
// -------------------------------------------------------------------------

__global__ void transpose_mid_dims_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int dim1, int dim2, int D) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int per_batch = dim1 * dim2 * D;
  if (idx >= per_batch) return;

  int di = idx % D;
  int remaining = idx / D;
  int d2 = remaining % dim2;
  int d1 = remaining / dim2;

  int src_off = batch_idx * per_batch + idx;
  int dst_off = batch_idx * per_batch + (d2 * dim1 + d1) * D + di;
  dst[dst_off] = src[src_off];
}

// -------------------------------------------------------------------------
// Step 4-5: KV Cache Append (fused transpose + scatter)
// -------------------------------------------------------------------------
// Appends new K (Step 4) or V (Step 5) tokens into an existing BNSD cache.
//   Src: BSHD [B, sq, G, d]              (new tokens from current input)
//   Dst: BNSD [B, G, present_seq, d]     (pre-allocated KV cache)
//   Write offset: [past_len .. past_len + sq)
//
// present_seq is the sequence dimension of the present buffer (its stride),
// which may be larger than past_len + sq if the buffer is pre-allocated.
//
// Fuses the BSHD -> BNSD layout transpose with the cache scatter,
// avoiding a separate transpose pass.  Past data is assumed to already be
// at the correct positions (same-buffer / aliased KV cache).
// Thread: one per (b,s,g,d_i) element. blockIdx.y = batch.
// Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h (ConcatStateChunkGQA)
// -------------------------------------------------------------------------

__global__ void kv_cache_append_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ cache,
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + di;
  cache[dst_idx] = src[src_idx];
}

// -------------------------------------------------------------------------
// Step 4-5 (separate-buffer variant): KV Cache Concat
// -------------------------------------------------------------------------
// Concatenates past KV data and new tokens into a fresh present buffer.
// Used when past and present are separate allocations with different strides.
//   - Positions [0, past_len): copy from past BNSD [B, G, past_seq, d]
//     (different stride: past has stride past_seq*d, present has present_seq*d)
//   - Positions [past_len, past_len+sq): transpose new tokens from
//     BSHD [B, sq, G, d] into BNSD present buffer
//
// past_seq and present_seq are the actual sequence dimensions (strides) of
// the respective buffers, which may differ from the active token counts.
//
// Performs both the strided past copy and the BSHD->BNSD new-token transpose
// in a single kernel launch, avoiding host-side loop overhead.
// Thread: one per (b, s_present, g, d_i) element. blockIdx.y = batch.
// -------------------------------------------------------------------------

__global__ void kv_cache_concat_kernel(
    const __half* __restrict__ past,
    const __half* __restrict__ current,
    __half* __restrict__ present,
    int past_len, int sq, int G, int d,
    int past_seq, int present_seq) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_seq = past_len + sq;
  int total_per_batch = total_seq * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;

  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + di;

  if (s < past_len) {
    int src_idx = ((batch_idx * G + g) * past_seq + s) * d + di;
    present[dst_idx] = past[src_idx];
  } else {
    int cs = s - past_len;
    int src_idx = ((batch_idx * sq + cs) * G + g) * d + di;
    present[dst_idx] = current[src_idx];
  }
}

// -------------------------------------------------------------------------
// Step 6-7: KV Group Expansion (G groups -> H heads)
// -------------------------------------------------------------------------
// Replicates each KV group for HPG = H/G query heads.
//   Src: [B*G groups, present_seq*d each]   (KV cache)
//   Dst: [B*H heads,  skv*d each]       (expanded for batched GEMM)
//   For head h: source group g = h / HPG
//
// blockIdx.x = head (0..B*H-1)
// blockIdx.y * blockDim.x + threadIdx.x = element within the skv*d slice
// Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h line 238
// -------------------------------------------------------------------------

__global__ void expand_kv_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int heads_per_group,
    int src_stride,
    int dst_stride,
    int copy_elems) {
  const int h = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int g = __builtin_amdgcn_readfirstlane(h / heads_per_group);
  int idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (idx < copy_elems) {
    dst[(size_t)h * dst_stride + idx] =
        src[(size_t)g * src_stride + idx];
  }
}

// -------------------------------------------------------------------------
// Step 0: Split packed QKV into separate Q, K, V buffers
// -------------------------------------------------------------------------
// When GQA receives a single packed query tensor [B*S, (H + 2*G)*d],
// this kernel splits it into:
//   Q [B*S, H*d],  K [B*S, G*d],  V [B*S, G*d]
//
// Thread: one per element in the packed tensor.
// -------------------------------------------------------------------------

__global__ void split_qkv_kernel(
    const __half* __restrict__ packed,
    __half* __restrict__ Q,
    __half* __restrict__ K,
    __half* __restrict__ V,
    int BS, int D_total, int Q_dim, int K_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * D_total) return;

  int row = idx / D_total;
  int col = idx % D_total;
  __half val = packed[idx];

  if (col < Q_dim)
    Q[row * Q_dim + col] = val;
  else if (col < Q_dim + K_dim)
    K[row * K_dim + (col - Q_dim)] = val;
  else
    V[row * K_dim + (col - Q_dim - K_dim)] = val;
}

// -------------------------------------------------------------------------
// Step 9a: Causal Mask (prefill only, skipped when seq_q == 1)
// -------------------------------------------------------------------------
// Masks future positions so each query attends only to past + itself.
//   Score matrix: col-major [skv, sq] per head
//   S[k, q] = -inf where k > past_len + q
//
// When local_window_size > 0, also masks distant past positions:
//   S[k, q] = -inf where k < past_len + q - local_window_size + 1
//
// blockIdx.y = head (0..B*H-1)
//
// Templated on T ∈ {__half, float}. The mask value is -65504 for fp16
// (largest finite negative) and -INFINITY for fp32.
// -------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T causal_mask_fill_value();

template <>
__device__ __forceinline__ __half causal_mask_fill_value<__half>() {
  return __float2half(-65504.0f);
}

template <>
__device__ __forceinline__ float causal_mask_fill_value<float>() {
  return -INFINITY;
}

template <typename T>
__global__ void causal_mask_kernel_impl(
    T* __restrict__ S,
    int skv, int sq, int batch_stride, int past_len,
    int local_window_size) {
  const int head = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = skv * sq;
  if (idx >= total) return;

  int k = idx % skv;
  int q = idx / skv;
  int abs_q = past_len + q;
  const T fill = causal_mask_fill_value<T>();
  if (k > abs_q) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] = fill;
  } else if (local_window_size > 0 && k < abs_q - local_window_size + 1) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] = fill;
  }
}

// -------------------------------------------------------------------------
// Step 9b: Column-wise Softmax (numerically stable, in-place)
// -------------------------------------------------------------------------
// Per column q: softmax(S[:,q]) = exp(S[:,q]-max) / sum(exp(S[:,q]-max))
//
// Smooth softmax (matching ORT behaviour):
//   Activated when head_sink is non-null OR use_smooth_softmax is set.
//   softmax_i = exp(x_i - m) / (exp(s - m) + sum_j exp(x_j - m))
//   where s = head_sink[h] when provided, or 0 when head_sink is null.
//
// One threadblock per (head, query) pair.
// Uses shared memory for parallel max-reduction and sum-reduction.
// -------------------------------------------------------------------------

// Warp-level reduction helpers (RDNA wave32)

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor(val, 16));
  val = fmaxf(val, __shfl_xor(val, 8));
  val = fmaxf(val, __shfl_xor(val, 4));
  val = fmaxf(val, __shfl_xor(val, 2));
  val = fmaxf(val, __shfl_xor(val, 1));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  val += __shfl_xor(val, 16);
  val += __shfl_xor(val, 8);
  val += __shfl_xor(val, 4);
  val += __shfl_xor(val, 2);
  val += __shfl_xor(val, 1);
  return val;
}

// Column-wise Softmax in-place
// One threadblock per (head, query) pair.
// Uses warp shuffles for intra-wave reductions and LDS only for cross-wave.
// Uses exp2f(x * LOG2E) instead of expf(x) for native HW instruction.

__global__ void softmax_inplace_kernel(
    __half* data, int rows, int cols, int batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int head = __builtin_amdgcn_readfirstlane(linear / cols);
  const int col  = __builtin_amdgcn_readfirstlane(linear % cols);
  __half* col_ptr = data + (size_t)head * batch_stride + (size_t)col * rows;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: find max
  float local_max = -INFINITY;
  for (int i = tid; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, __half2float(col_ptr[i]));

  local_max = warp_reduce_max(local_max);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_max;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : -INFINITY;
    v = warp_reduce_max(v);
    smem[0] = v;
  }
  __syncthreads();
  float max_val = smem[0];

  // Pass 2: exp2f + sum
  float local_sum = 0.0f;
  for (int i = tid; i < rows; i += blockDim.x) {
    float e = exp2f((__half2float(col_ptr[i]) - max_val) * LOG2E);
    col_ptr[i] = __float2half(e);
    local_sum += e;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_sum;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : 0.0f;
    v = warp_reduce_sum(v);
    smem[0] = v;
  }
  __syncthreads();
  float sum_val = smem[0];

  // Smooth softmax: add exp(s - max) to the denominator.
  if (head_sink != nullptr) {
    int actual_head = head % num_heads;
    float s = __half2float(head_sink[actual_head]);
    sum_val += expf(s - max_val);
  } else if (use_smooth_softmax) {
    sum_val += expf(0.0f - max_val);
  }

  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < rows; i += blockDim.x)
    col_ptr[i] = __float2half(__half2float(col_ptr[i]) * inv_sum);
}

//===----------------------------------------------------------------------===//
// Extern "C" Launchers (thin wrappers, ordered by pipeline step)
//===----------------------------------------------------------------------===//

// Step 1-2: RoPE
extern "C" int hip_gqa_rope(
    void* stream_ptr, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len,
    const void* seqlens_k) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int elems = seq_len * num_heads * half_rot;
  int blocks = (elems + GQA_THREADS - 1) / GQA_THREADS;
  rope_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(input),
      static_cast<__half*>(output),
      static_cast<const __half*>(cos_cache),
      static_cast<const __half*>(sin_cache),
      seq_len, num_heads, head_dim, half_rot, past_len,
      static_cast<const int*>(seqlens_k));
  return hipSuccess;
}

// Step 3 / Step 11: Transpose middle dims
extern "C" int hip_gqa_transpose_mid_dims(
    void* stream_ptr, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int per_batch = dim1 * dim2 * D;
  int blocks = (per_batch + GQA_THREADS - 1) / GQA_THREADS;
  transpose_mid_dims_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst),
      dim1, dim2, D);
  return hipSuccess;
}

// Step 4-5: KV cache append (same-buffer)
extern "C" int hip_gqa_kv_cache_append(
    void* stream_ptr, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = sq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_append_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(cache),
      sq, G, d, present_seq, past_len,
      static_cast<const int*>(seqlens_k));
  return hipSuccess;
}

// Step 4-5: KV cache concat (separate-buffer)
extern "C" int hip_gqa_kv_cache_concat(
    void* stream_ptr, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total_seq = past_len + sq;
  int total = total_seq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_concat_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(past),
      static_cast<const __half*>(current),
      static_cast<__half*>(present),
      past_len, sq, G, d, past_seq, present_seq);
  return hipSuccess;
}

// Step 6-7: KV group expansion
extern "C" int hip_gqa_expand_kv(
    void* stream_ptr, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int blocksY = (copy_elems + GQA_THREADS - 1) / GQA_THREADS;
  expand_kv_kernel<<<dim3(total_heads, blocksY), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst),
      heads_per_group, src_stride, dst_stride, copy_elems);
  return hipSuccess;
}

// Step 0: Split packed QKV
extern "C" int hip_gqa_split_qkv(
    void* stream_ptr, const void* packed, void* Q, void* K, void* V,
    int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_dim) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int BS = batch_size * seq_len;
  int Q_dim = num_heads * head_dim;
  int K_dim = kv_num_heads * head_dim;
  int D_total = Q_dim + 2 * K_dim;
  int total_elems = BS * D_total;
  int blocks = (total_elems + GQA_THREADS - 1) / GQA_THREADS;
  split_qkv_kernel<<<blocks, GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(packed),
      static_cast<__half*>(Q),
      static_cast<__half*>(K),
      static_cast<__half*>(V),
      BS, D_total, Q_dim, K_dim);
  return hipSuccess;
}

// Step 9a: Causal mask launchers (fp16 and fp32 instantiations)

extern "C" int hip_gqa_causal_mask(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  causal_mask_kernel_impl<__half><<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<__half*>(S),
      skv, sq, batch_stride, past_len, local_window_size);
  return hipSuccess;
}

extern "C" int hip_gqa_causal_mask_f32(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  causal_mask_kernel_impl<float><<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<float*>(S),
      skv, sq, batch_stride, past_len, local_window_size);
  return hipSuccess;
}

// Step 9b (fp32->fp16 variant): Softmax reading fp32 scores, writing fp16 output
// Avoids inf-inf=NaN by operating entirely in fp32 before the final fp16 store.

__global__ void softmax_f32_to_f16_kernel(
    const float* input, __half* output,
    int rows, int cols, int input_batch_stride, int output_batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int head = __builtin_amdgcn_readfirstlane(linear / cols);
  const int col  = __builtin_amdgcn_readfirstlane(linear % cols);
  const float* in_col = input + (size_t)head * input_batch_stride + (size_t)col * rows;
  __half* out_col = output + (size_t)head * output_batch_stride + (size_t)col * rows;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: find max (fp32, no overflow risk)
  float local_max = -INFINITY;
  for (int i = tid; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, in_col[i]);

  local_max = warp_reduce_max(local_max);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_max;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : -INFINITY;
    v = warp_reduce_max(v);
    smem[0] = v;
  }
  __syncthreads();
  float max_val = smem[0];

  // Pass 2: exp2f + sum (fp32 throughout, values recomputed in Pass 3)
  float local_sum = 0.0f;
  for (int i = tid; i < rows; i += blockDim.x)
    local_sum += exp2f((in_col[i] - max_val) * LOG2E);

  local_sum = warp_reduce_sum(local_sum);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_sum;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : 0.0f;
    v = warp_reduce_sum(v);
    smem[0] = v;
  }
  __syncthreads();
  float sum_val = smem[0];

  if (head_sink != nullptr) {
    int actual_head = head % num_heads;
    float s = __half2float(head_sink[actual_head]);
    sum_val += expf(s - max_val);
  } else if (use_smooth_softmax) {
    sum_val += expf(0.0f - max_val);
  }

  // Pass 3: normalize and write fp16 output
  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < rows; i += blockDim.x) {
    float e = exp2f((in_col[i] - max_val) * LOG2E);
    out_col[i] = __float2half(e * inv_sum);
  }
}

extern "C" int hip_gqa_softmax_f32_to_f16(
    void* stream_ptr, const void* input_f32, void* output_f16,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int threads = 32;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  int num_waves = threads / WAVE_SIZE;
  softmax_f32_to_f16_kernel<<<total_head_queries, threads,
                              num_waves * sizeof(float), stream>>>(
      static_cast<const float*>(input_f32),
      static_cast<__half*>(output_f16),
      rows, cols, input_batch_stride, output_batch_stride,
      static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax);
  return hipSuccess;
}

// Step 9b: Softmax
extern "C" int hip_gqa_softmax_inplace(
    void* stream_ptr, void* data,
    int total_head_queries, int rows, int cols,
    int batch_stride, const void* head_sink, int num_heads,
    int use_smooth_softmax) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int threads = 32;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  int num_waves = threads / WAVE_SIZE;
  softmax_inplace_kernel<<<total_head_queries, threads,
                           num_waves * sizeof(float), stream>>>(
      static_cast<__half*>(data),
      rows, cols, batch_stride,
      static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// WMMA types for fused kernels (RDNA 3 / 3.5, wave32)
//===----------------------------------------------------------------------===//

static constexpr int kWmmaTile = 16;  // RDNA 3 WMMA tile dimension (16x16x16)

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));
#define HALF16(pointer) (reinterpret_cast<half16*>((void *)&(pointer))[0])

//===----------------------------------------------------------------------===//
// Fused GQA Decode Kernel (sq == 1, d in {64, 128, 256})
//===----------------------------------------------------------------------===//
//
// Single-token decode attention: one block per (batch, head_q), D threads.
// Each thread owns one element of the d-dimensional head.
//
// Algorithm:
//   For each KV cache position s in [0, skv):
//     1. Cooperative dot product Q[tid] * K[s,tid] via warp shuffle reduction
//        + cross-wave LDS reduction (NUM_WAVES partial sums)
//     2. Online softmax in log2e space: score *= scale * LOG2E, then exp2f
//     3. Accumulate weighted V: acc += softmax_weight * V[s,tid]
//
// Assumes RDNA wave32 (__shfl_xor with masks 16,8,4,2,1).
// Reads Q in BSHD layout, KV cache in BNSD layout.

template <int D>
__global__ void __launch_bounds__(D)
fused_gqa_decode_kernel(
    const __half* __restrict__ Q_roped,
    const __half* __restrict__ Kcache,
    const __half* __restrict__ Vcache,
    __half* __restrict__ O,
    int H, int G, int d, int skv, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k)
{
    constexpr int NUM_WAVES = D / WAVE_SIZE;

    const int head_q  = __builtin_amdgcn_readfirstlane(blockIdx.x % H);
    const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.x / H);
    const int head_kv = __builtin_amdgcn_readfirstlane(head_q / (H / G));
    const int tid     = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;

    // When seqlens_k is provided, total_seq = seqlens_k[batch] + 1
    const int eff_skv = seqlens_k
        ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1) : skv;

    __shared__ float smem[NUM_WAVES];

    const float scale_log2e = scale * LOG2E;

    const float Q_val = __half2float(Q_roped[(batch * H + head_q) * d + tid]);

    const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
        (unsigned)((batch * G + head_kv) * max_seq)) * d;

    float m = -INFINITY;
    float l = 0.0f;
    float O_acc = 0.0f;

    for (int t = 0; t < eff_skv; ++t) {
        float dot = Q_val * __half2float(Kcache[kv_base + (size_t)t * d + tid]);

        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);

        if (lane_id == 0)
            smem[wave_id] = dot;
        __syncthreads();

        float score = 0.0f;
        for (int w = 0; w < NUM_WAVES; ++w)
            score += smem[w];
        score *= scale_log2e;

        __syncthreads();

        float m_new = fmaxf(m, score);
        float correction = exp2f(m - m_new);
        float p = exp2f(score - m_new);
        l = correction * l + p;

        float V_val = __half2float(Vcache[kv_base + (size_t)t * d + tid]);
        O_acc = correction * O_acc + p * V_val;
        m = m_new;
    }

    O_acc /= fmaxf(l, 1e-6f);
    O[(batch * H + head_q) * d + tid] = __float2half(O_acc);
}

//===----------------------------------------------------------------------===//
// Fused GQA Prefill Kernel (sq > 1, Flash Attention 2 with WMMA)
//===----------------------------------------------------------------------===//
// Grid: (num_q_tiles * B, H). Block: 128 threads = 4 waves of 32.
// WMMA 16x16x16 tile GEMMs, online softmax with exp2f, double-buffered KV.

static constexpr int PREFILL_BQ   = 32;
static constexpr int PREFILL_D    = 128;
static constexpr int PREFILL_DSTR = 130;

template<int BKV_T>
__global__ void __launch_bounds__(128)
fused_gqa_prefill_kernel(
    const __half* __restrict__ Q_roped,
    const __half* __restrict__ Kcache,
    const __half* __restrict__ Vcache,
    __half* __restrict__ O,
    int B_count, int H, int G, int sq, int skv, int max_seq, int past_len,
    float scale)
{
    constexpr int BQ    = PREFILL_BQ;
    constexpr int BKV   = BKV_T;
    constexpr int D     = PREFILL_D;
    constexpr int DSTR  = PREFILL_DSTR;
    constexpr int S_STR = BKV + 1;
    constexpr int P_STR = BKV + 2;

    const int num_q_tiles = (sq + BQ - 1) / BQ;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (H / G));
    const int q_start     = q_tile_idx * BQ;

    const int tid     = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int wmma_lane = lane_id % kWmmaTile;
    const int pair      = lane_id / kWmmaTile;

    extern __shared__ char smem_prefill[];
    _Float16* Q_lds    = (_Float16*)(smem_prefill);
    _Float16* KV_lds0  = (_Float16*)(smem_prefill + BQ * DSTR * 2);
    _Float16* KV_lds1  = (_Float16*)(smem_prefill + (BQ + BKV) * DSTR * 2);
    float*    S_lds    = (float*)(smem_prefill + (BQ + 2 * BKV) * DSTR * 2);
    _Float16* P_lds    = (_Float16*)(smem_prefill + (BQ + 2 * BKV) * DSTR * 2 + BQ * S_STR * 4);
    float*    ml_lds   = (float*)(smem_prefill + (BQ + 2 * BKV) * DSTR * 2 + BQ * S_STR * 4 + BQ * P_STR * 2);
    _Float16* KV_bufs[2] = { KV_lds0, KV_lds1 };

    const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
        (unsigned)((batch * G + head_kv) * max_seq)) * D;

    for (int i = tid; i < BQ * D; i += 128) {
        int row = i / D, col = i % D;
        int q_pos = q_start + row;
        _Float16 val = 0;
        if (q_pos < sq)
            val = (_Float16)Q_roped[((size_t)(batch * sq + q_pos) * H + head_q) * D + col];
        Q_lds[row * DSTR + col] = val;
    }

    if (tid < BQ) {
        ml_lds[tid * 3]     = -INFINITY;
        ml_lds[tid * 3 + 1] = 0.0f;
        ml_lds[tid * 3 + 2] = 0.0f;
    }

    const int o_tile_i = wave_id / 2;
    const int o_tile_j_base = (wave_id % 2) * 4;
    float8 O_frag[4] = {};

    __syncthreads();

    {
        for (int i = tid; i < BKV * D; i += 128) {
            int row = i / D, col = i % D;
            _Float16 val = 0;
            if (row < skv)
                val = (_Float16)Kcache[kv_base + (size_t)row * D + col];
            KV_bufs[0][row * DSTR + col] = val;
        }
    }
    __syncthreads();

    int kv_iter = 0;
    for (int kv_start = 0; kv_start < skv; kv_start += BKV, ++kv_iter) {
        if (sq > 1 && kv_start > past_len + q_start + BQ - 1) break;

        _Float16* K_buf = KV_bufs[kv_iter & 1];
        _Float16* V_buf = KV_bufs[1 - (kv_iter & 1)];

        {
            constexpr int S_TILES_J = BKV / 16;
            constexpr int TILES_PER_WAVE = S_TILES_J / 2;
            const int s_tile_i = wave_id / 2;
            const int s_tile_j_base = (wave_id % 2) * TILES_PER_WAVE;

            for (int tj_off = 0; tj_off < TILES_PER_WAVE; ++tj_off) {
                int sj = s_tile_j_base + tj_off;
                float8 s_frag = {};
                for (int tk = 0; tk < D / 16; tk += 2) {
                    half16 q0 = HALF16(Q_lds[(s_tile_i * 16 + wmma_lane) * DSTR + tk * 16]);
                    half16 k0 = HALF16(K_buf[(sj * 16 + wmma_lane) * DSTR + tk * 16]);
                    half16 q1 = HALF16(Q_lds[(s_tile_i * 16 + wmma_lane) * DSTR + (tk + 1) * 16]);
                    half16 k1 = HALF16(K_buf[(sj * 16 + wmma_lane) * DSTR + (tk + 1) * 16]);
                    s_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q0, k0, s_frag);
                    s_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q1, k1, s_frag);
                }

                for (int ele = 0; ele < 8; ++ele) {
                    int r = s_tile_i * 16 + ele * 2 + pair;
                    int c = sj * 16 + wmma_lane;
                    S_lds[r * S_STR + c] = s_frag[ele] * scale;
                }
            }
        }
        __syncthreads();

        for (int i = tid; i < BKV * D; i += 128) {
            int row = i / D, col = i % D;
            int kv_pos = kv_start + row;
            _Float16 val = 0;
            if (kv_pos < skv)
                val = (_Float16)Vcache[kv_base + (size_t)kv_pos * D + col];
            V_buf[row * DSTR + col] = val;
        }

        {
            const int row = tid / 4;
            const int col_grp = tid % 4;
            const int col_start = col_grp * (BKV / 4);
            const int q_pos = q_start + row;

            const bool tile_fully_visible =
                (kv_start + BKV - 1 <= past_len + q_start);

            if (!tile_fully_visible) {
                for (int c = 0; c < BKV / 4; ++c) {
                    int kv_pos = kv_start + col_start + c;
                    if (kv_pos >= skv || (sq > 1 && kv_pos > past_len + q_pos))
                        S_lds[row * S_STR + col_start + c] = -INFINITY;
                }
            }

            float local_max = -INFINITY;
            for (int c = 0; c < BKV / 4; ++c)
                local_max = fmaxf(local_max, S_lds[row * S_STR + col_start + c]);
            local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
            float row_max = local_max;

            float m_old = ml_lds[row * 3];
            float l_old = ml_lds[row * 3 + 1];
            float m_new = fmaxf(m_old, row_max);
            float correction = exp2f(m_old - m_new);

            float local_sum = 0.0f;
            for (int c = 0; c < BKV / 4; ++c) {
                float p_val = exp2f(S_lds[row * S_STR + col_start + c] - m_new);
                local_sum += p_val;
                P_lds[row * P_STR + col_start + c] = (_Float16)p_val;
            }
            local_sum += __shfl_xor(local_sum, 1);
            local_sum += __shfl_xor(local_sum, 2);

            float l_new = correction * l_old + local_sum;
            if (col_grp == 0) {
                ml_lds[row * 3]     = m_new;
                ml_lds[row * 3 + 1] = l_new;
                ml_lds[row * 3 + 2] = correction;
            }
        }
        __syncthreads();

        for (int t = 0; t < 4; ++t) {
            for (int ele = 0; ele < 8; ++ele) {
                int r = o_tile_i * 16 + ele * 2 + pair;
                O_frag[t][ele] *= ml_lds[r * 3 + 2];
            }
        }

        {
            for (int tj_off = 0; tj_off < 4; ++tj_off) {
                int tj = o_tile_j_base + tj_off;

                for (int tk = 0; tk < BKV / 16; tk += 2) {
                    half16 p0 = HALF16(P_lds[(o_tile_i * 16 + wmma_lane) * P_STR + tk * 16]);
                    half16 v0;
                    for (int e = 0; e < 16; ++e)
                        v0[e] = V_buf[(tk * 16 + e) * DSTR + tj * 16 + wmma_lane];

                    half16 p1 = HALF16(P_lds[(o_tile_i * 16 + wmma_lane) * P_STR + (tk + 1) * 16]);
                    half16 v1;
                    for (int e = 0; e < 16; ++e)
                        v1[e] = V_buf[((tk + 1) * 16 + e) * DSTR + tj * 16 + wmma_lane];

                    O_frag[tj_off] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p0, v0, O_frag[tj_off]);
                    O_frag[tj_off] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p1, v1, O_frag[tj_off]);
                }
            }
        }

        __syncthreads();

        {
            int next_kv = kv_start + BKV;
            bool need_next = (next_kv < skv) &&
                !(sq > 1 && next_kv > past_len + q_start + BQ - 1);
            if (need_next) {
                for (int i = tid; i < BKV * D; i += 128) {
                    int row = i / D, col = i % D;
                    int kv_pos = next_kv + row;
                    _Float16 val = 0;
                    if (kv_pos < skv)
                        val = (_Float16)Kcache[kv_base + (size_t)kv_pos * D + col];
                    V_buf[row * DSTR + col] = val;
                }
            }
        }
        __syncthreads();
    }

    for (int t = 0; t < 4; ++t) {
        int tj = o_tile_j_base + t;
        for (int ele = 0; ele < 8; ++ele) {
            int r = o_tile_i * 16 + ele * 2 + pair;
            int c = tj * 16 + wmma_lane;
            int q_pos = q_start + r;
            if (q_pos < sq && c < D) {
                float l_val = fmaxf(ml_lds[r * 3 + 1], 1e-6f);
                float out_val = O_frag[t][ele] / l_val;
                O[((size_t)(batch * sq + q_pos) * H + head_q) * D + c] = __float2half(out_val);
            }
        }
    }
}

// Fused decode launcher -- dispatches to the template matching d
extern "C" int hip_gqa_fused_decode(
    void* stream_ptr, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int d, int skv, int max_seq,
    float scale, const void* seqlens_k) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  dim3 grid(B * H, 1);
  const __half* hQ = static_cast<const __half*>(Q);
  const __half* hK = static_cast<const __half*>(Kcache);
  const __half* hV = static_cast<const __half*>(Vcache);
  __half* hO = static_cast<__half*>(O);
  const int* iSeqlensK = static_cast<const int*>(seqlens_k);
  switch (d) {
    case 64:
      fused_gqa_decode_kernel<64><<<grid, 64, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    case 128:
      fused_gqa_decode_kernel<128><<<grid, 128, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    case 256:
      fused_gqa_decode_kernel<256><<<grid, 256, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    default:
      fprintf(stderr, "hip_gqa_fused_decode: unsupported d=%d "
              "(must be 64, 128, or 256)\n", d);
      return -1;
  }
  return hipSuccess;
}

// Fused prefill launcher
extern "C" int hip_gqa_fused_prefill(
    void* stream_ptr, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int sq, int skv, int max_seq, int past_len,
    float scale) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  constexpr int BKV = 32;
  int num_q_tiles = (sq + PREFILL_BQ - 1) / PREFILL_BQ;
  dim3 grid(num_q_tiles * B, H);
  size_t smem = (PREFILL_BQ + 2 * BKV) * PREFILL_DSTR * 2
              + PREFILL_BQ * (BKV + 1) * 4
              + PREFILL_BQ * (BKV + 2) * 2
              + PREFILL_BQ * 3 * 4;
  fused_gqa_prefill_kernel<BKV><<<grid, 128, smem, stream>>>(
      static_cast<const __half*>(Q),
      static_cast<const __half*>(Kcache),
      static_cast<const __half*>(Vcache),
      static_cast<__half*>(O),
      B, H, G, sq, skv, max_seq, past_len, scale);
  return hipGetLastError();
}
