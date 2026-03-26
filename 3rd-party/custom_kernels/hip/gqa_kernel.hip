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

// =========================================================================
// Algorithm: ONNX com.microsoft.GroupQueryAttention
// =========================================================================
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
// =========================================================================

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
    int half_rot, int past_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;
  int elems = seq_len * num_heads * half_rot;
  if (idx >= elems) return;

  int d = idx % half_rot;
  int remaining = idx / half_rot;
  int h = remaining % num_heads;
  int s = remaining / num_heads;
  if (s >= seq_len) return;

  int pos = past_len + s;
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
  int batch_idx = blockIdx.y;
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
    int sq, int G, int d, int present_seq, int past_len) {
  int batch_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * present_seq + (past_len + s)) * d + di;
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
  int batch_idx = blockIdx.y;
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
  int h = blockIdx.x;
  int g = h / heads_per_group;
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
//   S[k, q] = -65504 (FP16 -inf) where k > past_len + q
//
// When local_window_size > 0, also masks distant past positions:
//   S[k, q] = -65504 where k < past_len + q - local_window_size + 1
//
// blockIdx.y = head (0..B*H-1)
// -------------------------------------------------------------------------

__global__ void causal_mask_kernel(
    __half* __restrict__ S,
    int skv, int sq, int batch_stride, int past_len,
    int local_window_size) {
  int head = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = skv * sq;
  if (idx >= total) return;

  int k = idx % skv;
  int q = idx / skv;
  int abs_q = past_len + q;
  if (k > abs_q) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] =
        __float2half(-65504.0f);
  } else if (local_window_size > 0 && k < abs_q - local_window_size + 1) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] =
        __float2half(-65504.0f);
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

__global__ void softmax_inplace_kernel(
    __half* data, int rows, int cols, int batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  int linear = blockIdx.x;
  int head = linear / cols;
  int col = linear % cols;
  __half* col_ptr = data + (size_t)head * batch_stride + (size_t)col * rows;

  extern __shared__ float smem[];

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, __half2float(col_ptr[i]));
  smem[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < (unsigned)s)
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  float max_val = smem[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < rows; i += blockDim.x) {
    float e = expf(__half2float(col_ptr[i]) - max_val);
    col_ptr[i] = __float2half(e);
    local_sum += e;
  }
  smem[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < (unsigned)s)
      smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  float sum_val = smem[0];
  __syncthreads();

  // Smooth softmax: add exp(s - max) to the denominator.
  // When head_sink is provided, s = head_sink[h] (per-head factor).
  // When head_sink is null but use_smooth_softmax is set, s = 0 (ORT default).
  if (head_sink != nullptr) {
    int actual_head = head % num_heads;
    float s = __half2float(head_sink[actual_head]);
    sum_val += expf(s - max_val);
  } else if (use_smooth_softmax) {
    sum_val += expf(0.0f - max_val);
  }

  float inv_sum = 1.0f / sum_val;
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    col_ptr[i] = __float2half(__half2float(col_ptr[i]) * inv_sum);
}

// =============================================================================
// Extern "C" Launchers (thin wrappers, ordered by pipeline step)
// =============================================================================

// Step 1-2: RoPE
extern "C" int hip_gqa_rope(
    void* stream_ptr, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int elems = seq_len * num_heads * half_rot;
  int blocks = (elems + GQA_THREADS - 1) / GQA_THREADS;
  rope_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(input),
      static_cast<__half*>(output),
      static_cast<const __half*>(cos_cache),
      static_cast<const __half*>(sin_cache),
      seq_len, num_heads, head_dim, half_rot, past_len);
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
    int batch_size, int sq, int G, int d, int present_seq, int past_len) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = sq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_append_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(cache),
      sq, G, d, present_seq, past_len);
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

// Step 9a: Causal mask
extern "C" int hip_gqa_causal_mask(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  causal_mask_kernel<<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<__half*>(S),
      skv, sq, batch_stride, past_len, local_window_size);
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
  softmax_inplace_kernel<<<total_head_queries, threads,
                           threads * sizeof(float), stream>>>(
      static_cast<__half*>(data),
      rows, cols, batch_stride,
      static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax);
  return hipSuccess;
}
