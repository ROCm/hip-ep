/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "hip_custom_kernels.h"
#include "debug_log.h"

static constexpr int GQA_THREADS = 256;
static constexpr int WAVE_SIZE = 32;  // RDNA wave32; not portable to CDNA/wave64
static constexpr float LOG2E = 1.4426950408889634f;

// Element-size dispatch helper for the pure data-movement kernels (transpose,
// kv-cache append/concat, kv-expand, split-qkv). These kernels only copy bytes
// of the element type, so templating them on T and selecting __half (2 bytes)
// vs float (4 bytes) is sufficient to fp32-enable the decomposed GQA path
// (Whisper no_causal). 2 == fp16, 4 == fp32; anything else is an error.
#define GQA_DISPATCH_BY_ELEM_SIZE(elem_size, FN, ...)                          \
  do {                                                                         \
    /* Clear any stale error so a launch failure is attributed to THIS         \
     * kernel, not blamed on the next op that happens to call                  \
     * hipGetLastError() (was misreported as "hip_layer_norm launch           \
     * failed" on the Whisper path). */                                        \
    (void)hipGetLastError();                                                   \
    if ((elem_size) == 2) {                                                    \
      FN<__half>(__VA_ARGS__);                                                 \
    } else if ((elem_size) == 4) {                                            \
      FN<float>(__VA_ARGS__);                                                  \
    } else {                                                                   \
      fprintf(stderr, "%s: unsupported element_size_bytes=%d (must be 2 or 4)\n", \
              __func__, (int)(elem_size));                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Report (and return) the launch status of the kernel issued just above.
// hipGetLastError() returns + clears the most recent error, so pairing this
// with the clear in GQA_DISPATCH_BY_ELEM_SIZE (or an explicit clear before a
// raw <<<>>> launch) attributes errors to the correct kernel.
#define GQA_RETURN_LAUNCH_STATUS()                                             \
  do {                                                                         \
    hipError_t _gqa_err = hipGetLastError();                                   \
    if (_gqa_err != hipSuccess)                                                \
      fprintf(stderr, "[custom_kernels] %s launch failed: %s\n", __func__,     \
              hipGetErrorString(_gqa_err));                                     \
    return _gqa_err;                                                           \
  } while (0)

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

// Scalar load/store helpers that promote the element type to float for math
// and narrow back on store. Specialized for __half (HW convert) and float
// (identity) so the data-movement / RoPE kernels can be templated on T.
template <typename T>
__device__ __forceinline__ float gqa_to_float(T v);
template <>
__device__ __forceinline__ float gqa_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float gqa_to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T gqa_from_float(float v);
template <>
__device__ __forceinline__ __half gqa_from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ float gqa_from_float<float>(float v) {
  return v;
}

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
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

  float x1 = gqa_to_float<T>(input[base + d]);
  float x2 = gqa_to_float<T>(input[base + d + half_rot]);
  float c  = gqa_to_float<T>(cos_cache[pos * half_rot + d]);
  float sn = gqa_to_float<T>(sin_cache[pos * half_rot + d]);

  output[base + d]            = gqa_from_float<T>(x1 * c - x2 * sn);
  output[base + d + half_rot] = gqa_from_float<T>(x2 * c + x1 * sn);
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

template <typename T>
__global__ void transpose_mid_dims_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
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

template <typename T>
__global__ void kv_cache_append_kernel(
    const T* __restrict__ src,
    T* __restrict__ cache,
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  // Skip malformed batches rather than write past this slice.
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
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

template <typename T>
__global__ void kv_cache_concat_kernel(
    const T* __restrict__ past,
    const T* __restrict__ current,
    T* __restrict__ present,
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
// INT8 KV cache: quantized append / concat + dequant (per-channel symmetric)
// -------------------------------------------------------------------------
// For the quantized-KV GroupQueryAttention variant (k/v_quant_type=PER_CHANNEL,
// kv_cache_bit_width=8): the KV cache is stored as symmetric INT8 with a static
// per-channel fp32 scale [G, d] (no zero point). These mirror the fp16
// append/concat above, but fold in QuantizeLinear on the incoming fp16 tokens
// (q = clamp(round(x / scale[g*d+di]), -128, 127)) and a matching DequantizeLinear
// used to rebuild an fp16 view for the compute-bound prefill (x = q * scale).

// Quantize-append (in-place cache): fp16 BSHD [B,sq,G,d] -> INT8 BNSD
// [B,G,present_seq,d] at offset past_len (or seqlens_k-derived).
__global__ void kv_cache_append_quant_i8_kernel(
    const _Float16* __restrict__ src,     // BSHD [B, sq, G, d]
    int8_t* __restrict__ cache,           // BNSD [B, G, present_seq, d]
    const float* __restrict__ scale,      // [G, d]
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;
  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;
  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + di;
  float inv = 1.0f / scale[g * d + di];
  float q = rintf((float)src[src_idx] * inv);
  q = fminf(fmaxf(q, -128.0f), 127.0f);
  cache[dst_idx] = (int8_t)q;
}

// Quantize-concat (separate past/present buffers): copy past INT8 [0,past_len)
// then quantize new fp16 tokens [past_len, past_len+sq) into present INT8.
__global__ void kv_cache_concat_quant_i8_kernel(
    const int8_t* __restrict__ past,      // BNSD int8 [B, G, past_seq, d]
    const _Float16* __restrict__ current, // BSHD fp16 [B, sq, G, d]
    int8_t* __restrict__ present,         // BNSD int8 [B, G, present_seq, d]
    const float* __restrict__ scale,      // [G, d]
    int past_len, int sq, int G, int d, int past_seq, int present_seq) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_seq = past_len + sq;
  if (idx >= total_seq * G * d) return;
  int di = idx % d;
  int r = idx / d;
  int g = r % G;
  int s = r / G;
  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + di;
  if (s < past_len) {
    present[dst_idx] = past[((batch_idx * G + g) * past_seq + s) * d + di];
  } else {
    int cs = s - past_len;
    float inv = 1.0f / scale[g * d + di];
    float q = rintf((float)current[((batch_idx * sq + cs) * G + g) * d + di] * inv);
    q = fminf(fmaxf(q, -128.0f), 127.0f);
    present[dst_idx] = (int8_t)q;
  }
}

// Dequant INT8 BNSD [B,G,src_seq,d] -> fp16 BNSD [B,G,dst_seq,d] over the first
// total_seq positions (rebuilds an fp16 cache view for the fp16 prefill kernel).
__global__ void dequant_kv_i8_to_fp16_kernel(
    const int8_t* __restrict__ src,   // BNSD int8 [B, G, src_seq, d]
    _Float16* __restrict__ dst,       // BNSD fp16 [B, G, dst_seq, d]
    const float* __restrict__ scale,  // [G, d]
    int total_seq, int G, int d, int src_seq, int dst_seq) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_seq * G * d) return;
  int di = idx % d;
  int r = idx / d;
  int g = r % G;
  int s = r / G;
  int src_idx = ((batch_idx * G + g) * src_seq + s) * d + di;
  int dst_idx = ((batch_idx * G + g) * dst_seq + s) * d + di;
  dst[dst_idx] = (_Float16)((float)src[src_idx] * scale[g * d + di]);
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

template <typename T>
__global__ void expand_kv_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
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

template <typename T>
__global__ void split_qkv_kernel(
    const T* __restrict__ packed,
    T* __restrict__ Q,
    T* __restrict__ K,
    T* __restrict__ V,
    int BS, int D_total, int Q_dim, int K_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * D_total) return;

  int row = idx / D_total;
  int col = idx % D_total;
  T val = packed[idx];

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

// Templated launch helpers (one per data-movement kernel). The extern "C"
// launchers below dispatch to <__half> or <float> via element_size_bytes.

template <typename T>
static void launch_rope(
    hipStream_t stream, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len, const void* seqlens_k) {
  int elems = seq_len * num_heads * half_rot;
  int blocks = (elems + GQA_THREADS - 1) / GQA_THREADS;
  rope_kernel<T><<<dim3(blocks, batch_size), GQA_THREADS, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(output),
      static_cast<const T*>(cos_cache), static_cast<const T*>(sin_cache),
      seq_len, num_heads, head_dim, half_rot, past_len,
      static_cast<const int*>(seqlens_k));
}

template <typename T>
static void launch_transpose_mid_dims(
    hipStream_t stream, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D) {
  int per_batch = dim1 * dim2 * D;
  int blocks = (per_batch + GQA_THREADS - 1) / GQA_THREADS;
  transpose_mid_dims_kernel<T><<<dim3(blocks, batch_size), GQA_THREADS, 0,
                                 stream>>>(
      static_cast<const T*>(src), static_cast<T*>(dst), dim1, dim2, D);
}

template <typename T>
static void launch_kv_cache_append(
    hipStream_t stream, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k) {
  int total = sq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_append_kernel<T><<<dim3(blocks, batch_size), GQA_THREADS, 0,
                              stream>>>(
      static_cast<const T*>(src), static_cast<T*>(cache),
      sq, G, d, present_seq, past_len, static_cast<const int*>(seqlens_k));
}

template <typename T>
static void launch_kv_cache_concat(
    hipStream_t stream, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq) {
  int total_seq = past_len + sq;
  int total = total_seq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  kv_cache_concat_kernel<T><<<dim3(blocks, batch_size), GQA_THREADS, 0,
                              stream>>>(
      static_cast<const T*>(past), static_cast<const T*>(current),
      static_cast<T*>(present), past_len, sq, G, d, past_seq, present_seq);
}

template <typename T>
static void launch_expand_kv(
    hipStream_t stream, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems) {
  int blocksY = (copy_elems + GQA_THREADS - 1) / GQA_THREADS;
  expand_kv_kernel<T><<<dim3(total_heads, blocksY), GQA_THREADS, 0, stream>>>(
      static_cast<const T*>(src), static_cast<T*>(dst),
      heads_per_group, src_stride, dst_stride, copy_elems);
}

template <typename T>
static void launch_split_qkv(
    hipStream_t stream, const void* packed, void* Q, void* K, void* V,
    int BS, int D_total, int Q_dim, int K_dim) {
  int total_elems = BS * D_total;
  int blocks = (total_elems + GQA_THREADS - 1) / GQA_THREADS;
  split_qkv_kernel<T><<<blocks, GQA_THREADS, 0, stream>>>(
      static_cast<const T*>(packed), static_cast<T*>(Q),
      static_cast<T*>(K), static_cast<T*>(V), BS, D_total, Q_dim, K_dim);
}

// Step 1-2: RoPE
extern "C" int hip_gqa_rope(
    void* stream_ptr, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len,
    const void* seqlens_k, int element_size_bytes) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_rope, stream, input,
                            output, cos_cache, sin_cache, batch_size, seq_len,
                            num_heads, head_dim, half_rot, past_len, seqlens_k);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 3 / Step 11: Transpose middle dims
extern "C" int hip_gqa_transpose_mid_dims(
    void* stream_ptr, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D, int element_size_bytes) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_transpose_mid_dims,
                            stream, src, dst, batch_size, dim1, dim2, D);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 4-5: KV cache append (same-buffer). `kv_dtype` selects the format:
//   HIP_KV_DTYPE_FP16 -> plain elem-size copy/transpose (fp16 or fp32 cache).
//   HIP_KV_DTYPE_INT8 -> symmetric per-channel INT8 quant (fp16 src, [G,d] scale
//                        -> int8 cache). The format decision lives here so the
//                        host only calls one append entry.
extern "C" int hip_gqa_kv_cache_append(
    void* stream_ptr, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k, int element_size_bytes,
    int kv_dtype, const void* scale) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  if (kv_dtype == HIP_KV_DTYPE_INT8) {
    if (scale == nullptr) {
      fprintf(stderr, "hip_gqa_kv_cache_append: INT8 requires scale\n");
      return -1;
    }
    int total = sq * G * d;
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    (void)hipGetLastError();
    kv_cache_append_quant_i8_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0,
                                      stream>>>(
        static_cast<const _Float16*>(src), static_cast<int8_t*>(cache),
        static_cast<const float*>(scale), sq, G, d, present_seq, past_len,
        static_cast<const int*>(seqlens_k));
    GQA_RETURN_LAUNCH_STATUS();
  }
  if (kv_dtype != HIP_KV_DTYPE_FP16) {
    fprintf(stderr, "hip_gqa_kv_cache_append: unsupported kv_dtype=%d\n",
            kv_dtype);
    return -1;
  }
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_kv_cache_append, stream,
                            src, cache, batch_size, sq, G, d, present_seq,
                            past_len, seqlens_k);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 4-5: KV cache concat (separate-buffer). `kv_dtype` selects the format, as
// in hip_gqa_kv_cache_append (FP16 -> plain copy, INT8 -> INT8 quant).
extern "C" int hip_gqa_kv_cache_concat(
    void* stream_ptr, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int element_size_bytes,
    int kv_dtype, const void* scale) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  if (kv_dtype == HIP_KV_DTYPE_INT8) {
    if (scale == nullptr) {
      fprintf(stderr, "hip_gqa_kv_cache_concat: INT8 requires scale\n");
      return -1;
    }
    int total_seq = past_len + sq;
    int total = total_seq * G * d;
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    (void)hipGetLastError();
    kv_cache_concat_quant_i8_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0,
                                      stream>>>(
        static_cast<const int8_t*>(past),
        static_cast<const _Float16*>(current), static_cast<int8_t*>(present),
        static_cast<const float*>(scale), past_len, sq, G, d, past_seq,
        present_seq);
    GQA_RETURN_LAUNCH_STATUS();
  }
  if (kv_dtype != HIP_KV_DTYPE_FP16) {
    fprintf(stderr, "hip_gqa_kv_cache_concat: unsupported kv_dtype=%d\n",
            kv_dtype);
    return -1;
  }
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_kv_cache_concat, stream,
                            past, current, present, batch_size, past_len, sq, G,
                            d, past_seq, present_seq);
  GQA_RETURN_LAUNCH_STATUS();
}

// INT8 KV cache: dequant to fp16 view (for the compute-bound fp16 prefill)
extern "C" int hip_gqa_dequant_kv_i8_to_fp16(
    void* stream_ptr, const void* src, void* dst, const void* scale,
    int batch_size, int total_seq, int G, int d, int src_seq, int dst_seq) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = total_seq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  (void)hipGetLastError();
  dequant_kv_i8_to_fp16_kernel<<<dim3(blocks, batch_size), GQA_THREADS, 0,
                                 stream>>>(
      static_cast<const int8_t*>(src), static_cast<_Float16*>(dst),
      static_cast<const float*>(scale), total_seq, G, d, src_seq, dst_seq);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 6-7: KV group expansion
extern "C" int hip_gqa_expand_kv(
    void* stream_ptr, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems, int element_size_bytes) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_expand_kv, stream, src,
                            dst, total_heads, heads_per_group, src_stride,
                            dst_stride, copy_elems);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 0: Split packed QKV
extern "C" int hip_gqa_split_qkv(
    void* stream_ptr, const void* packed, void* Q, void* K, void* V,
    int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_dim,
    int element_size_bytes) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int BS = batch_size * seq_len;
  int Q_dim = num_heads * head_dim;
  int K_dim = kv_num_heads * head_dim;
  int D_total = Q_dim + 2 * K_dim;
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_split_qkv, stream, packed,
                            Q, K, V, BS, D_total, Q_dim, K_dim);
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 9a: Causal mask launchers (fp16 and fp32 instantiations)

extern "C" int hip_gqa_causal_mask(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  (void)hipGetLastError();  // clear stale error for correct attribution
  causal_mask_kernel_impl<__half><<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<__half*>(S),
      skv, sq, batch_stride, past_len, local_window_size);
  GQA_RETURN_LAUNCH_STATUS();
}

extern "C" int hip_gqa_causal_mask_f32(
    void* stream_ptr, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int total = skv * sq;
  int blocksX = (total + GQA_THREADS - 1) / GQA_THREADS;
  (void)hipGetLastError();  // clear stale error for correct attribution
  causal_mask_kernel_impl<float><<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
      static_cast<float*>(S),
      skv, sq, batch_stride, past_len, local_window_size);
  GQA_RETURN_LAUNCH_STATUS();
}

template <typename TBias>
__global__ void add_attention_bias_kernel(
    float* __restrict__ scores, const TBias* __restrict__ bias,
    int total_heads, int num_heads, int bias_batch, int bias_heads, int sq,
    int total_seq, int score_batch_stride) {
  int head = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = sq * total_seq;
  if (tid >= n)
    return;
  int row = tid / total_seq;
  int col = tid % total_seq;
  int b = head / num_heads;
  int h = head % num_heads;
  int bias_b = (bias_batch == 1) ? 0 : b;
  int bias_h = (bias_heads == 1) ? 0 : h;
  size_t bias_idx = static_cast<size_t>(bias_b) * bias_heads * sq * total_seq +
                    static_cast<size_t>(bias_h) * sq * total_seq +
                    static_cast<size_t>(row) * total_seq + col;
  scores[static_cast<size_t>(head) * score_batch_stride + tid] +=
      static_cast<float>(bias[bias_idx]);
}

extern "C" int hip_gqa_add_attention_bias_f32(
    void* stream_ptr, void* scores, const void* bias, int total_heads,
    int num_heads, int bias_batch, int bias_heads, int sq, int total_seq,
    int score_batch_stride, int bias_element_size_bytes) {
  if (!scores || !bias)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  int n = sq * total_seq;
  int blocksX = (n + GQA_THREADS - 1) / GQA_THREADS;
  (void)hipGetLastError();
  if (bias_element_size_bytes == 2) {
    add_attention_bias_kernel<__half>
        <<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
            static_cast<float*>(scores), static_cast<const __half*>(bias),
            total_heads, num_heads, bias_batch, bias_heads, sq, total_seq,
            score_batch_stride);
  } else if (bias_element_size_bytes == 4) {
    add_attention_bias_kernel<float>
        <<<dim3(blocksX, total_heads), GQA_THREADS, 0, stream>>>(
            static_cast<float*>(scores), static_cast<const float*>(bias),
            total_heads, num_heads, bias_batch, bias_heads, sq, total_seq,
            score_batch_stride);
  } else {
    fprintf(stderr,
            "hip_gqa_add_attention_bias_f32: unsupported bias elem size %d\n",
            bias_element_size_bytes);
    return -1;
  }
  GQA_RETURN_LAUNCH_STATUS();
}

// Step 9b (fp32->fp16 variant): Softmax reading fp32 scores, writing fp16 output
// Avoids inf-inf=NaN by operating entirely in fp32 before the final fp16 store.

// Output-type-templated: reads fp32 scores, writes TOut ∈ {__half, float}.
// fp16 output (TOut=__half) feeds the fp16 Value GEMM; fp32 output (TOut=float)
// feeds the fp32 Value GEMM on the Whisper no_causal fp32 decomposed path.
template <typename TOut>
__global__ void softmax_f32_to_out_kernel(
    const float* input, TOut* output,
    int rows, int cols, int input_batch_stride, int output_batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int head = __builtin_amdgcn_readfirstlane(linear / cols);
  const int col  = __builtin_amdgcn_readfirstlane(linear % cols);
  const float* in_col = input + (size_t)head * input_batch_stride + (size_t)col * rows;
  TOut* out_col = output + (size_t)head * output_batch_stride + (size_t)col * rows;

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

  // Pass 3: normalize and write output (fp16 or fp32 per TOut).
  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < rows; i += blockDim.x) {
    float e = exp2f((in_col[i] - max_val) * LOG2E);
    out_col[i] = gqa_from_float<TOut>(e * inv_sum);
  }
}

template <typename TOut>
static void launch_softmax_f32_to_out(
    hipStream_t stream, const void* input_f32, void* output,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax) {
  int threads = 32;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  int num_waves = threads / WAVE_SIZE;
  softmax_f32_to_out_kernel<TOut><<<total_head_queries, threads,
                                    num_waves * sizeof(float), stream>>>(
      static_cast<const float*>(input_f32), static_cast<TOut*>(output),
      rows, cols, input_batch_stride, output_batch_stride,
      static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax);
}

extern "C" int hip_gqa_softmax_f32_to_f16(
    void* stream_ptr, const void* input_f32, void* output_f16,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();  // clear stale error for correct attribution
  launch_softmax_f32_to_out<__half>(
      stream, input_f32, output_f16, total_head_queries, rows, cols,
      input_batch_stride, output_batch_stride, head_sink, num_heads,
      use_smooth_softmax);
  GQA_RETURN_LAUNCH_STATUS();
}

// fp32-output variant: fp32 scores -> fp32 probabilities, feeding the fp32
// Value GEMM on the Whisper no_causal fp32 decomposed path. output_batch_stride
// is in float elements (same units as the fp16 variant's half elements).
extern "C" int hip_gqa_softmax_f32_to_f32(
    void* stream_ptr, const void* input_f32, void* output_f32,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  (void)hipGetLastError();  // clear stale error for correct attribution
  launch_softmax_f32_to_out<float>(
      stream, input_f32, output_f32, total_head_queries, rows, cols,
      input_batch_stride, output_batch_stride, head_sink, num_heads,
      use_smooth_softmax);
  GQA_RETURN_LAUNCH_STATUS();
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
  (void)hipGetLastError();  // clear stale error for correct attribution
  softmax_inplace_kernel<<<total_head_queries, threads,
                           num_waves * sizeof(float), stream>>>(
      static_cast<__half*>(data),
      rows, cols, batch_stride,
      static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax);
  GQA_RETURN_LAUNCH_STATUS();
}

//===----------------------------------------------------------------------===//
// WMMA types for fused kernels (RDNA 3 / 3.5, wave32)
//===----------------------------------------------------------------------===//

static constexpr int kWmmaTile = 16;  // RDNA 3 WMMA tile dimension (16x16x16)

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));
#define HALF16(pointer) (reinterpret_cast<half16*>((void *)&(pointer))[0])

// The WMMA builtin below is RDNA3-family (wave32) only. Where __has_builtin
// reports it absent, trap instead so this file still compiles and a WMMA
// launch on an unsupported arch aborts loudly rather than returning garbage.
#if !defined(__has_builtin) ||                                                  \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_w32_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)                     \
  hipdnn_wmma_f32_16x16x16_f16_w32_unavailable((a), (b), (c))
#endif

// [removed] fused_gqa_decode_kernel / hip_gqa_fused_decode: the legacy
// one-block-per-(batch,head_q) decode kernel has been retired. Decode is now a
// single path -- hip_gqa_flash_decode_v2 (FA-2 split-K, scalar|WMMA selected by
// geometry) -- which serves GQA and MHA (HpG==1) alike. See gqa.cpp decode
// dispatch and the SCALAR/WMMA kernels above.

//===----------------------------------------------------------------------===//
// FA-2 Split-K GQA Decode Kernel (Phase 1)
//===----------------------------------------------------------------------===//
// Goal: 4-15x speedup over hip_gqa_fused_decode at high KV depth (skv >= 256)
// by exploiting:
//   1. GQA-aware KV reuse: one block per (batch, head_kv) instead of per
//      (batch, head_q). HPG=4 query heads share K/V loaded once into LDS,
//      reducing global KV bandwidth by 4x.
//   2. Split-K parallelism: K_SPLITS blocks per (batch, head_kv) cover
//      different KV ranges, raising occupancy for long sequences.
//   3. LDS-staged K/V tiles (BKV=64): K[t] is loaded once and reused HPG
//      times for dot products across 4 query heads.
//
// Algorithm:
//   Each block computes partial (m, l, O) for its slice [kv_start, kv_end)
//   using online softmax in log2e space. Partials written to workspace
//   [B, H, K_SPLITS, D+2] floats. A second kernel (gqa_flash_decode_reduce)
//   merges the K_SPLITS partials per (batch, head_q) into the final output.
//
// Block geometry:
//   Block = HPG waves of 32 = HPG * 32 threads. Each wave handles ONE query
//   head out of HPG. Thread `lane` in wave handles d-elements
//   [lane * EPT, (lane+1) * EPT) where EPT = D / WAVE_SIZE.
//
// Grid:
//   gridX = B * G        - one block per (batch, head_kv)
//   gridY = K_SPLITS     - split-K dimension
//
// Supported HPG specializations (template param):
//   - HPG=4  (Llama-3.x family at d=64 and d=128)
//   - HPG=8  (gpt-oss-20b at d=64; 64 q heads / 8 kv heads)
//
// Constraints (asserted at launcher):
//   - D must be 64, 128 or 256 (EPT = D/32 in 1..8; D=256 -> BKV=16 to keep
//     the LDS tile ~16KB). The scalar kernel is otherwise D-generic.
//   - RDNA wave32 (uses __shfl_xor across 32 lanes).
//===----------------------------------------------------------------------===//

static constexpr int FLASH_DECODE_BKV     = 64;

// Partial layout per (b, h_q, k_split): {m: float, l: float, O[D]: float}
__device__ __forceinline__ size_t flash_decode_partial_offset(
    int b, int h_q, int k_split, int H, int K_SPLITS, int D) {
  return ((size_t)((b * H + h_q) * K_SPLITS + k_split)) * (size_t)(D + 2);
}

// KV-cache element format for the unified decode kernels. Extensible: to add a
// new quantized cache (INT4, FP8, per-tensor int8, ...) add an enum value here
// plus the matching `if constexpr` branch inside the kernels -- no new template
// parameter is needed (a bool per format would not scale).
enum class KvDtype { kF16, kI8 };

// FMT==kF16: fp16 K/V cache (k_scale/v_scale ignored, pass nullptr).
// FMT==kI8 : symmetric per-channel INT8 K/V cache with fp32 scales [G, D]:
//   k_fp16 = k_i8 * k_scale[head_kv*D + channel]  (folded into Q here)
//   v_fp16 = v_i8 * v_scale[head_kv*D + channel]  (deferred to the epilogue)
// The int8 read stays 1 byte/elem, halving the fp16 path's DRAM traffic on the
// bandwidth-bound decode. All divergent code is a compile-time `if constexpr`
// branch, so FMT==kF16 lowers to byte-identical SASS as the original fp16
// kernel. Both paths emit the SAME [B, H, K_SPLITS, D+2] partials.
template <int D, int HPG, KvDtype FMT = KvDtype::kF16>
__global__ void __launch_bounds__(HPG * WAVE_SIZE)
gqa_flash_decode_kernel(
    const __half* __restrict__ Q_roped,    // [B, 1, H, D] (BSHD with sq=1)
    const std::conditional_t<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ Kcache,  // [B, G, max_seq, D]
    const std::conditional_t<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ Vcache,  // [B, G, max_seq, D]
    const float* __restrict__ k_scale,     // [G, D] per-channel; FMT==kI8 only
    const float* __restrict__ v_scale,     // [G, D] per-channel; FMT==kI8 only
    float* __restrict__ partials,          // [B, H, K_SPLITS, D+2]
    int H, int G, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  constexpr bool KV_I8 = (FMT == KvDtype::kI8);
  using KVt = std::conditional_t<KV_I8, int8_t, __half>;
  // Keep the K+V LDS tile at ~16KB/block (4 blocks resident per 64KB-LDS CU)
  // regardless of D, so DRAM latency stays hidden. At BKV=64,D=128 the tile
  // would be 32KB -> only 2 resident blocks -> latency-bound (decode stalls
  // ~1.6x). 2*BKV*D*2 bytes: D=256 -> BKV=16 (16KB), D=128 -> BKV=32 (16KB),
  // D=64 -> BKV=64 (16KB).
  constexpr int BKV = (D >= 256) ? 16 : (D == 128) ? 32 : FLASH_DECODE_BKV;
  constexpr int THREADS = HPG * WAVE_SIZE;  // total threads in block
  constexpr int EPT = D / WAVE_SIZE;  // elements-per-thread along d

  static_assert(D % WAVE_SIZE == 0, "D must be a multiple of WAVE_SIZE");
  static_assert(EPT >= 1 && EPT <= 8, "EPT must be 1..8");

  const int batch     = __builtin_amdgcn_readfirstlane(blockIdx.x / G);
  const int head_kv   = __builtin_amdgcn_readfirstlane(blockIdx.x % G);
  const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int tid       = threadIdx.x;
  const int wave_id   = tid / WAVE_SIZE;       // [0, HPG)
  const int lane_id   = tid % WAVE_SIZE;
  const int head_q    = head_kv * HPG + wave_id;

  // total_seq = seqlens_k[batch] + 1, clamped to [0, max_seq].
  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;

  // Sliding-window: at decode (sq==1) the query position is total_seq-1, so
  // it attends to KV positions [max(0, total_seq - window), total_seq). When
  // window <= 0 the attention is full (kv_lo = 0). gpt-oss-20b alternates
  // sliding (local_window_size=128) and full layers; clamping kv_lo here lets
  // sliding layers skip the full KV cache scan and process only ~window slots.
  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;

  // Partition KV range into K_SPLITS roughly-equal chunks.
  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split_idx;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  // Output partials destination for THIS wave's query head.
  float* partial_dst = partials +
      flash_decode_partial_offset(batch, head_q, split_idx, H, K_SPLITS, D);

  // Empty slice: write neutral identity so the merge ignores this split.
  // exp2(-INFINITY - global_m) == 0, so l/O contributions vanish.
  if (kv_start >= kv_end) {
    if (lane_id == 0) {
      partial_dst[0] = -INFINITY;  // m
      partial_dst[1] = 0.0f;       // l
    }
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = 0.0f;
    return;
  }

  // LDS tiles for K/V (shared across HPG=4 query heads for dedup). 16B-aligned
  // so the cooperative load can move 8 fp16 per float4 (see staging below).
  static_assert(EPT % 2 == 0, "EPT must be even for __half2 vectorized loads");
  // HpG==1 (MHA) runs a single wave per block, so the K/V tile in LDS is never
  // reused across waves -- staging is pure overhead (32KB LDS + a __syncthreads
  // every tile). Stream straight from global instead (kStage=false): the LDS
  // arrays shrink to 1 elem and the inner loop reads the global tile directly.
  // GQA (HpG>1) keeps the shared tile so HpG query heads load K/V once. The
  // per-lane inner math is byte-identical either way (OPTIMIZATION.md §3.1).
  constexpr bool kStage = (HPG > 1);
  constexpr int kShElems = kStage ? (BKV * D) : 1;
  __shared__ __align__(16) KVt K_lds[kShElems];
  __shared__ __align__(16) KVt V_lds[kShElems];

  // Load this wave's Q into registers, pre-converted to fp32 (the per-key dot
  // is fp32; converting once here avoids EPT half->float casts per key). For
  // INT8 the per-channel K scale is folded into Q here (dot = sum_e
  // (Q[e]*Ksc[e]) * K_i8[e]) so the inner loop is a plain int8*fp32 MAC and the
  // K-scale registers are freed; the per-channel V scale is deferred to the
  // epilogue (loaded once there instead of held live across the KV loop).
  float Q_f[EPT];
  {
    const __half* Q_ptr = Q_roped + (size_t)(batch * H + head_q) * D;
    if constexpr (KV_I8) {
      const float* ks = k_scale + (size_t)head_kv * D;
      #pragma unroll
      for (int e = 0; e < EPT; ++e)
        Q_f[e] = __half2float(Q_ptr[lane_id * EPT + e]) * ks[lane_id * EPT + e];
    } else {
      #pragma unroll
      for (int e = 0; e < EPT; ++e)
        Q_f[e] = __half2float(Q_ptr[lane_id * EPT + e]);
    }
  }

  // KV cache base offset for this (batch, head_kv).
  const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
      (unsigned)((batch * G + head_kv) * max_seq)) * (size_t)D;

  const float scale_log2e = scale * LOG2E;

  // Online softmax accumulators (per query head, in registers).
  float m_acc = -INFINITY;
  float l_acc = 0.0f;
  float O_acc[EPT];
  #pragma unroll
  for (int e = 0; e < EPT; ++e) O_acc[e] = 0.0f;

  // Iterate KV tiles within this block's split.
  for (int tile_start = kv_start; tile_start < kv_end; tile_start += BKV) {
    const int tile_n = (tile_start + BKV <= kv_end)
        ? BKV : (kv_end - tile_start);

    // Stage the K/V tile into LDS (GQA: reused across the HpG waves) or point
    // straight at the global tile (MHA HpG==1: single wave, no reuse).
    const KVt* Kt;
    const KVt* Vt;
    if constexpr (kStage) {
      if constexpr (KV_I8) {
        // INT8 tile: tile_n*D is a multiple of 16 bytes (D % 16 == 0), so a
        // single int4 (16 int8) vectorized copy tiles it exactly -- no tail.
        const int total_vec = (tile_n * D) / 16;
        const int4* Kv = reinterpret_cast<const int4*>(
            Kcache + kv_base + (size_t)tile_start * D);
        const int4* Vv = reinterpret_cast<const int4*>(
            Vcache + kv_base + (size_t)tile_start * D);
        int4* Ksv = reinterpret_cast<int4*>(K_lds);
        int4* Vsv = reinterpret_cast<int4*>(V_lds);
        for (int i = tid; i < total_vec; i += THREADS) {
          Ksv[i] = Kv[i];
          Vsv[i] = Vv[i];
        }
      } else {
        // Cooperative vectorized LDS load: one float4 moves 8 fp16 (D % 8 == 0,
        // tile base kv_base+tile_start*D is 16B-aligned since tile_start*D*2 is a
        // multiple of D*2 >= 128B), cutting load instructions 8x vs per-element.
        const int total_vec = (tile_n * D) / 8;
        const float4* Kv = reinterpret_cast<const float4*>(
            Kcache + kv_base + (size_t)tile_start * D);
        const float4* Vv = reinterpret_cast<const float4*>(
            Vcache + kv_base + (size_t)tile_start * D);
        float4* Ksv = reinterpret_cast<float4*>(K_lds);
        float4* Vsv = reinterpret_cast<float4*>(V_lds);
        for (int i = tid; i < total_vec; i += THREADS) {
          Ksv[i] = Kv[i];
          Vsv[i] = Vv[i];
        }
      }
      __syncthreads();
      Kt = K_lds;
      Vt = V_lds;
    } else {
      // &Kt[t*D + ...] == Kcache[kv_base + (tile_start + t)*D + ...].
      Kt = Kcache + kv_base + (size_t)tile_start * D;
      Vt = Vcache + kv_base + (size_t)tile_start * D;
    }

    // Each wave processes the tile against its own query head.
    for (int t = 0; t < tile_n; ++t) {
      // Per-lane partial dot. fp16: __half2 reads (2 fp16/load + half22float2).
      // int8: K scale is pre-folded into Q_f, so this is a plain int8*fp32 MAC;
      // when EPT % 4 == 0 a single char4 (32-bit) LDS load beats two 16-bit reads.
      float dot_partial = 0.0f;
      if constexpr (KV_I8) {
        if constexpr (EPT % 4 == 0) {
          #pragma unroll
          for (int e = 0; e < EPT; e += 4) {
            const char4 kk =
                *reinterpret_cast<const char4*>(&Kt[t * D + lane_id * EPT + e]);
            dot_partial += Q_f[e]     * (float)kk.x + Q_f[e + 1] * (float)kk.y
                         + Q_f[e + 2] * (float)kk.z + Q_f[e + 3] * (float)kk.w;
          }
        } else {
          #pragma unroll
          for (int e = 0; e < EPT; e += 2) {
            const char2 kk =
                *reinterpret_cast<const char2*>(&Kt[t * D + lane_id * EPT + e]);
            dot_partial += Q_f[e] * (float)kk.x + Q_f[e + 1] * (float)kk.y;
          }
        }
      } else {
        #pragma unroll
        for (int e = 0; e < EPT; e += 2) {
          const __half2 kk =
              *reinterpret_cast<const __half2*>(&Kt[t * D + lane_id * EPT + e]);
          const float2 kf = __half22float2(kk);
          dot_partial += Q_f[e] * kf.x + Q_f[e + 1] * kf.y;
        }
      }
      // Warp reduce across 32 lanes -> all lanes hold the full dot.
      dot_partial += __shfl_xor(dot_partial, 16);
      dot_partial += __shfl_xor(dot_partial, 8);
      dot_partial += __shfl_xor(dot_partial, 4);
      dot_partial += __shfl_xor(dot_partial, 2);
      dot_partial += __shfl_xor(dot_partial, 1);

      const float score = dot_partial * scale_log2e;
      const float m_new = fmaxf(m_acc, score);
      const float correction = exp2f(m_acc - m_new);
      const float p = exp2f(score - m_new);
      l_acc = correction * l_acc + p;

      // fp16: dequant-on-read via __half2. int8: accumulate raw p*V_i8; the
      // per-channel V scale is applied once in the epilogue.
      if constexpr (KV_I8) {
        if constexpr (EPT % 4 == 0) {
          #pragma unroll
          for (int e = 0; e < EPT; e += 4) {
            const char4 vv =
                *reinterpret_cast<const char4*>(&Vt[t * D + lane_id * EPT + e]);
            O_acc[e]     = correction * O_acc[e]     + p * (float)vv.x;
            O_acc[e + 1] = correction * O_acc[e + 1] + p * (float)vv.y;
            O_acc[e + 2] = correction * O_acc[e + 2] + p * (float)vv.z;
            O_acc[e + 3] = correction * O_acc[e + 3] + p * (float)vv.w;
          }
        } else {
          #pragma unroll
          for (int e = 0; e < EPT; e += 2) {
            const char2 vv =
                *reinterpret_cast<const char2*>(&Vt[t * D + lane_id * EPT + e]);
            O_acc[e]     = correction * O_acc[e]     + p * (float)vv.x;
            O_acc[e + 1] = correction * O_acc[e + 1] + p * (float)vv.y;
          }
        }
      } else {
        #pragma unroll
        for (int e = 0; e < EPT; e += 2) {
          const __half2 vv =
              *reinterpret_cast<const __half2*>(&Vt[t * D + lane_id * EPT + e]);
          const float2 vf = __half22float2(vv);
          O_acc[e]     = correction * O_acc[e]     + p * vf.x;
          O_acc[e + 1] = correction * O_acc[e + 1] + p * vf.y;
        }
      }
      m_acc = m_new;
    }
    if constexpr (kStage) __syncthreads();
  }

  // Write partial (m, l, O) for this wave's query head.
  if (lane_id == 0) {
    partial_dst[0] = m_acc;
    partial_dst[1] = l_acc;
  }
  if constexpr (KV_I8) {
    // Apply the deferred per-channel V scale on the way out (loaded here, not
    // held in registers across the KV loop).
    const float* vs = v_scale + (size_t)head_kv * D;
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = O_acc[e] * vs[lane_id * EPT + e];
  } else {
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = O_acc[e];
  }
}

//===----------------------------------------------------------------------===//
// WMMA Split-K Decode Kernel (sq == 1) -- optimized TPS path
//===----------------------------------------------------------------------===//
// Drop-in producer of the SAME [B, H, K_SPLITS, D+2] partials as the scalar
// gqa_flash_decode_kernel above, so it shares the identical FA-2 reduce (and
// therefore seqlens_k / sliding-window / head-sink / smooth-softmax handling).
//
// Why WMMA: the scalar kernel's per-key Q.K dot ends in a 5-step __shfl_xor
// cross-lane reduction that is a serialized VALU latency chain (~41% of the
// decode runtime for high-GQA D=64, see OPTIMIZATION.md ch.8). This kernel
// instead packs the HPG query heads of one KV group into the M=16 rows of a
// WMMA 16x16x16 tile and runs the QK / PV GEMMs on the matrix unit, removing
// the reduction entirely. A software pipeline keeps the NEXT KV tile's global
// loads in flight (in registers) across the current tile's GEMMs to hide DRAM
// latency on the single resident wave.
//
// Grid: (G, K_SPLITS, B). Block: 32 (one wave). M=16 rows == HPG query heads
// (rows >= HPG are inert / never written). Split math mirrors the scalar
// kernel exactly (eff_skv, kv_lo, chunk).
//===----------------------------------------------------------------------===//
// FMT==kF16: fp16 K/V cache (k_scale/v_scale ignored, pass nullptr).
// FMT==kI8 : symmetric per-channel INT8 K/V cache -- the int8 tile is loaded
// at 1 byte/elem (HALF the fp16 DRAM traffic) and dequantized to fp16 only on
// the LDS write, so the 16x16x16 WMMA score/value GEMMs stay byte-identical to
// the fp16 path. All divergent code is a compile-time `if constexpr` branch, so
// FMT==kF16 lowers to the same SASS as the original fp16 kernel. Both paths
// emit the SAME [B, H, K_SPLITS, D+2] partials.
template <int D, int HPG, int BKV, KvDtype FMT = KvDtype::kF16>
__global__ void __launch_bounds__(32)
gqa_flash_decode_wmma_kernel(
    const __half* __restrict__ Q_in,       // [B, 1, H, D] (sq=1) == [B, H, D]
    const std::conditional_t<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ K_in,  // [B, G, max_seq, D]
    const std::conditional_t<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ V_in,  // [B, G, max_seq, D]
    const float* __restrict__ k_scale,     // [G, D] per-channel; FMT==kI8 only
    const float* __restrict__ v_scale,     // [G, D] per-channel; FMT==kI8 only
    float* __restrict__ partials,          // [B, H, K_SPLITS, D+2]
    int H, int G, int max_seq, float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  constexpr bool KV_I8 = (FMT == KvDtype::kI8);
  using KVt = std::conditional_t<KV_I8, int8_t, __half>;
  constexpr int DSTR    = D + 2;
  constexpr int S_STR   = BKV + 1;
  constexpr int P_STR   = BKV + 2;
  constexpr int Q_TILES = D / kWmmaTile;
  constexpr int O_TILES = D / kWmmaTile;
  constexpr int N_TILES = BKV / kWmmaTile;

  const _Float16* Q = reinterpret_cast<const _Float16*>(Q_in);
  // fp16: KVt==_Float16 (reinterpret of __half*); int8: KVt==int8_t (identity).
  const KVt* Ksrc = reinterpret_cast<const KVt*>(K_in);
  const KVt* Vsrc = reinterpret_cast<const KVt*>(V_in);

  const int head_kv = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int split   = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.z);

  const int lane      = threadIdx.x;
  const int wmma_lane = lane % kWmmaTile;
  const int pair      = lane / kWmmaTile;

  // total_seq = seqlens_k[batch] + 1, clamped to [0, max_seq] (mirrors scalar).
  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;
  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;
  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  // Empty slice: emit identity partials so the reduce ignores this split.
  if (kv_start >= kv_end) {
    if (lane < HPG) {
      const int head_q = head_kv * HPG + lane;
      float* pd = partials +
          flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
      pd[0] = -INFINITY;  // m
      pd[1] = 0.0f;       // l
      #pragma unroll
      for (int c = 0; c < D; ++c) pd[2 + c] = 0.0f;
    }
    return;
  }

  extern __shared__ char smem_dw[];
  float*    S_lds = reinterpret_cast<float*>(smem_dw);
  _Float16* K_lds = reinterpret_cast<_Float16*>(smem_dw + 16 * S_STR * sizeof(float));
  _Float16* V_lds = K_lds + BKV * DSTR;
  _Float16* P_lds = V_lds + BKV * DSTR;
  float*    ml    = reinterpret_cast<float*>(P_lds + 16 * P_STR);  // [16*3]
  // INT8 only: this KV head's per-channel scales, staged into LDS once. The
  // fp16 launcher does NOT reserve this trailing region, so it is untouched.
  float* ksc_sh = nullptr;
  float* vsc_sh = nullptr;
  if constexpr (KV_I8) {
    ksc_sh = ml + 16 * 3;   // [D]
    vsc_sh = ksc_sh + D;    // [D]
    const float* ks = k_scale + (size_t)head_kv * D;
    const float* vs = v_scale + (size_t)head_kv * D;
    for (int c = lane; c < D; c += 32) { ksc_sh[c] = ks[c]; vsc_sh[c] = vs[c]; }
  }

  const size_t kv_base = (static_cast<size_t>(batch) * G + head_kv)
                         * static_cast<size_t>(max_seq) * D;

  // Q fragments: lane wmma_lane holds row wmma_lane (head_kv*HPG + wmma_lane).
  half16 Q_reg[Q_TILES];
  {
    const bool valid = wmma_lane < HPG;
    const int head = head_kv * HPG + wmma_lane;
    const size_t q_row = (static_cast<size_t>(batch) * H + head) * D;
    #pragma unroll
    for (int tk = 0; tk < Q_TILES; ++tk)
      Q_reg[tk] = valid ? HALF16(Q[q_row + tk * 16]) : half16{};
  }

  if (lane < 16) {
    ml[lane * 3]     = -INFINITY;
    ml[lane * 3 + 1] = 0.0f;
    ml[lane * 3 + 2] = 1.0f;
  }
  if constexpr (KV_I8) __syncthreads();  // ml + scale tables visible to all lanes

  float8 O_frag[O_TILES] = {};

  const float scale_log2e = scale * LOG2E;
  const int num_tiles = (kv_end - kv_start + BKV - 1) / BKV;

  // Software pipeline: prefetch the next tile's K/V global loads into registers
  // while the current tile's GEMMs run (see gqa_flash_decode_wmma.hip rationale).
  // Only viable when the per-lane register footprint is small (D=64/BKV=16).
  // For INT8 the prefetch registers stay int8 (KVt) and dequant to fp16 happens
  // on the LDS write below, keeping the global read at 1 byte/elem.
  constexpr int kPET = (BKV * D) / 32;          // elements per lane per tile
  constexpr bool kPrefetch = (kPET <= 32);

  KVt k_pf[kPrefetch ? kPET : 1];
  KVt v_pf[kPrefetch ? kPET : 1];
  if (kPrefetch) {
    #pragma unroll
    for (int j = 0; j < kPET; ++j) {
      const int i = lane + j * 32;
      const int row = i / D, col = i % D;
      const int kv_pos = kv_start + row;          // tile 0
      const bool ok = kv_pos < kv_end;
      k_pf[j] = ok ? Ksrc[kv_base + static_cast<size_t>(kv_pos) * D + col] : (KVt)0;
      v_pf[j] = ok ? Vsrc[kv_base + static_cast<size_t>(kv_pos) * D + col] : (KVt)0;
    }
  }

  for (int t = 0; t < num_tiles; ++t) {
    const int kv0 = kv_start + t * BKV;

    if (kPrefetch) {
      // Write the prefetched tile to LDS (int8: dequant to fp16 on the way in).
      #pragma unroll
      for (int j = 0; j < kPET; ++j) {
        const int i = lane + j * 32;
        const int row = i / D, col = i % D;
        if constexpr (KV_I8) {
          K_lds[row * DSTR + col] = (_Float16)((float)k_pf[j] * ksc_sh[col]);
          V_lds[row * DSTR + col] = (_Float16)((float)v_pf[j] * vsc_sh[col]);
        } else {
          K_lds[row * DSTR + col] = k_pf[j];
          V_lds[row * DSTR + col] = v_pf[j];
        }
      }
      __syncthreads();
      const int kv0n = kv_start + (t + 1) * BKV;
      if (t + 1 < num_tiles) {
        #pragma unroll
        for (int j = 0; j < kPET; ++j) {
          const int i = lane + j * 32;
          const int row = i / D, col = i % D;
          const int kv_pos = kv0n + row;
          const bool ok = kv_pos < kv_end;
          k_pf[j] = ok ? Ksrc[kv_base + static_cast<size_t>(kv_pos) * D + col] : (KVt)0;
          v_pf[j] = ok ? Vsrc[kv_base + static_cast<size_t>(kv_pos) * D + col] : (KVt)0;
        }
      }
    } else {
      for (int i = lane; i < BKV * D; i += 32) {
        const int row = i / D, col = i % D;
        const int kv_pos = kv0 + row;
        if (kv_pos < kv_end) {
          const size_t off = kv_base + static_cast<size_t>(kv_pos) * D + col;
          if constexpr (KV_I8) {
            K_lds[row * DSTR + col] = (_Float16)((float)Ksrc[off] * ksc_sh[col]);
            V_lds[row * DSTR + col] = (_Float16)((float)Vsrc[off] * vsc_sh[col]);
          } else {
            K_lds[row * DSTR + col] = Ksrc[off];
            V_lds[row * DSTR + col] = Vsrc[off];
          }
        } else {
          K_lds[row * DSTR + col] = (_Float16)0;
          V_lds[row * DSTR + col] = (_Float16)0;
        }
      }
      __syncthreads();
    }

    // scoreGEMM: S[16, BKV] = Q . K^T (scaled, log2 domain) -> S_lds.
    #pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      float8 s_frag = {};
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk) {
        half16 kf = HALF16(K_lds[(sj * 16 + wmma_lane) * DSTR + tk * 16]);
        s_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Q_reg[tk], kf, s_frag);
      }
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele) {
        const int r = ele * 2 + pair;
        const int c = sj * 16 + wmma_lane;
        S_lds[r * S_STR + c] = s_frag[ele] * scale_log2e;
      }
    }
    __syncthreads();

    // Online softmax over the BKV columns (no causal mask at decode).
    {
      const int row  = lane / 2;
      const int half = lane & 1;
      const int col_start = half * (BKV / 2);

      float local_max = -INFINITY;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const int kv_pos = kv0 + col_start + c;
        float s = (kv_pos < kv_end) ? S_lds[row * S_STR + col_start + c]
                                    : -INFINITY;
        S_lds[row * S_STR + col_start + c] = s;
        local_max = fmaxf(local_max, s);
      }
      local_max = fmaxf(local_max, __shfl_xor(local_max, 1));

      const float m_old = ml[row * 3];
      const float l_old = ml[row * 3 + 1];
      const float m_new = fmaxf(m_old, local_max);
      const float correction = exp2f(m_old - m_new);

      float local_sum = 0.0f;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const float p = exp2f(S_lds[row * S_STR + col_start + c] - m_new);
        local_sum += p;
        P_lds[row * P_STR + col_start + c] = static_cast<_Float16>(p);
      }
      local_sum += __shfl_xor(local_sum, 1);

      if (half == 0) {
        ml[row * 3]     = m_new;
        ml[row * 3 + 1] = correction * l_old + local_sum;
        ml[row * 3 + 2] = correction;
      }
    }
    __syncthreads();

    // valueGEMM: rescale running O by correction, then O += P . V.
    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj)
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele)
        O_frag[tj][ele] *= ml[(ele * 2 + pair) * 3 + 2];

    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj) {
      #pragma unroll
      for (int tk = 0; tk < N_TILES; ++tk) {
        half16 p = HALF16(P_lds[wmma_lane * P_STR + tk * 16]);
        half16 v;
        #pragma unroll
        for (int e = 0; e < 16; ++e)
          v[e] = V_lds[(tk * 16 + e) * DSTR + tj * 16 + wmma_lane];
        O_frag[tj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p, v, O_frag[tj]);
      }
    }
    __syncthreads();
  }

  // Epilogue: write UN-normalized partials (m, l, O) per query head row; the
  // shared reduce kernel performs the /l normalization + sink terms.
  #pragma unroll
  for (int tj = 0; tj < O_TILES; ++tj)
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
      const int r = ele * 2 + pair;
      const int c = tj * 16 + wmma_lane;
      if (r < HPG) {
        const int head_q = head_kv * HPG + r;
        float* pd = partials +
            flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
        pd[2 + c] = O_frag[tj][ele];
      }
    }
  if (lane < HPG) {
    const int head_q = head_kv * HPG + lane;
    float* pd = partials +
        flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
    pd[0] = ml[lane * 3];
    pd[1] = ml[lane * 3 + 1];
  }
}

//===----------------------------------------------------------------------===//
// FA-2 Split-K Reduction
//===----------------------------------------------------------------------===//
// Combines K_SPLITS partial (m, l, O) into the final output per (B, head_q).
//
//   global_m = max_k m_k
//   global_l = sum_k (exp2(m_k - global_m) * l_k)
//   global_O = sum_k (exp2(m_k - global_m) * O_k) / global_l
//
// Empty splits write m=-INFINITY, l=0, O=0; exp2(-INF - global_m) = 0 so
// those splits contribute nothing.
//
// Smooth softmax (attention sink): when head_sink is non-null OR
// use_smooth_softmax is set, an extra term is added to the denominator only:
//   global_l += exp(s - global_m_natural)
// where s is in natural (qk*scale) units. The partials are in log2-scaled
// space (m and l use exp2/LOG2E), so the natural identity becomes
//   exp(s - m_nat) = exp2((s - m_nat) * LOG2E) = exp2(s*LOG2E - global_m).
// The sink contributes to L only — there is no V term for it (gpt-oss-20b /
// Mistral-style attention sink).
//
// Grid: (B * H, 1) - one block per query head.
// Block: D threads. Each thread reduces one element of the d-dimension.
//===----------------------------------------------------------------------===//

template <int D>
__global__ void __launch_bounds__(D)
gqa_flash_decode_reduce_kernel(
    const float* __restrict__ partials,    // [B, H, K_SPLITS, D+2]
    __half* __restrict__ O,                // [B, 1, H, D] (BSHD with sq=1)
    int H,
    const __half* __restrict__ head_sink,  // [num_heads] or null
    int num_heads,
    int use_smooth_softmax,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int batch  = __builtin_amdgcn_readfirstlane(linear / H);
  const int head_q = __builtin_amdgcn_readfirstlane(linear % H);
  const int tid    = threadIdx.x;

  // Pass 1: find global_m. K_SPLITS is small (8..64) - serial per thread is fine.
  float global_m = -INFINITY;
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    global_m = fmaxf(global_m, m_k);
  }

  // When all splits are -INF (eff_skv == 0) but a sink is configured, the
  // sink term still defines a valid (zero) output: softmax over empty +
  // sink ⇒ all weight on the sink, which contributes 0 to V. We therefore
  // write a zero output here either way.
  if (!isfinite(global_m)) {
    O[(batch * H + head_q) * D + tid] = __float2half(0.0f);
    return;
  }

  // Pass 2: accumulate l and O contributions.
  float l_sum = 0.0f;
  float O_sum = 0.0f;
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    const float l_k = p[1];
    const float scale_k = exp2f(m_k - global_m);
    l_sum += scale_k * l_k;
    O_sum += scale_k * p[2 + tid];
  }

  // Smooth softmax: add the sink term to the denominator only. global_m is
  // in log2-scaled units (it's the max of dot*scale*LOG2E across splits), so
  // the natural-space exp(s - m_nat) becomes exp2(s*LOG2E - global_m).
  if (head_sink != nullptr) {
    const int actual_head = head_q % num_heads;
    const float s = __half2float(head_sink[actual_head]);
    l_sum += exp2f(s * (float)LOG2E - global_m);
  } else if (use_smooth_softmax) {
    // s = 0 in natural units; exp2(0*LOG2E - global_m) = exp2(-global_m).
    l_sum += exp2f(-global_m);
  }

  const float inv_l = 1.0f / fmaxf(l_sum, 1e-6f);
  O[(batch * H + head_q) * D + tid] = __float2half(O_sum * inv_l);
}

//===----------------------------------------------------------------------===//
// FA-2 split-K decode: runtime split count + (scalar|WMMA) autotune cache
//
// The three decode kernels above take K_SPLITS as a *runtime* argument, so the
// launcher is free to pick any split count up to the workspace capacity the
// caller allocated. Production previously locked K_SPLITS=8 (one template
// instantiation + a fixed B*H*8*(D+2) partials workspace). On the 20-CU gfx1151
// that under-fills the GPU at long context: split-K decode launches only
// B*G*K_SPLITS thread blocks, so with B=1, G=8, K_SPLITS=8 just 64 blocks share
// 20 CUs and the kernel is launch/occupancy-bound, not bandwidth-bound. Raising
// the split count at large skv fills the CUs and is the dominant decode-TPS win.
//
// Following matmul_nbits' GEMV autotuner (matmul_nbits_kernel.hip): the first
// call for each (B,H,G,d, skv-bucket) shape benchmarks a small candidate set of
// (impl, split-count) pairs with hipEvent timing and caches the fastest config
// in a process-wide map; every later call for that shape just looks it up (no
// re-timing). skv is bucketed to the next power of two so the map holds
// O(log max_seq) entries per layer-shape rather than one per decode step (skv
// grows by 1 every token). The map stores exactly the shapes that are fed.
//===----------------------------------------------------------------------===//

// Workspace capacity / autotune ceiling. Splitting decode into >64 chunks on a
// 20-CU part yields diminishing returns and inflates the partials workspace and
// the reduce's per-head serial scan, so cap here.
static constexpr int kFlashDecodeMaxSplits = 64;

struct FlashDecodeCfg { bool use_wmma; int splits; };

// WMMA decode packs the HpG query heads of a KV group into a 16-row tile, so it
// only pays off (and is only instantiated) for these geometries:
//   (HpG==4, d in {64,128})  and  (HpG==8, d==64).
// Every other geometry -- including MHA (HpG==1) and HpG in {2,3,16} -- runs the
// HpG-generic scalar decode kernel.
static inline bool flash_decode_wmma_supported(int d, int hpg) {
  if (hpg == 4) return d == 64 || d == 128;
  if (hpg == 8) return d == 64;
  // NOTE: HpG=5 (Qwen2.5-14B 40:8) deliberately has NO WMMA path. Packing only
  // 5 of the 16 M-tile rows wastes 11 lanes and measured 3-4x SLOWER than the
  // scalar kernel across skv (even losing to the legacy decode at skv=2k), so
  // HpG=5 runs the HpG-generic scalar decode exclusively.
  return false;
}

// Launch one (impl, splits) configuration: split-K decode + FA-2 reduce. Both
// impls emit identical [B,H,splits,D+2] partials, so the reduce is shared.
static inline void launchFlashDecodeConfig(
    const FlashDecodeCfg& cfg, hipStream_t stream,
    const __half* hQ, const __half* hK, const __half* hV, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int max_seq, float scale,
    const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax) {
  const int splits = cfg.splits;
  const bool use_wmma = cfg.use_wmma && flash_decode_wmma_supported(d, hpg);
  if (use_wmma) {
    // BKV: D=64 -> 16 (small tile, more resident -> better latency hiding);
    // D=128 -> 32 (direct staging path). Grid (G, splits, B), one wave/block.
    auto smem_bytes = [](int D_, int bkv) -> size_t {
      const int DSTR = D_ + 2;
      return (size_t)16 * (bkv + 1) * sizeof(float)
           + (size_t)2 * bkv * DSTR * sizeof(_Float16)
           + (size_t)16 * (bkv + 2) * sizeof(_Float16)
           + (size_t)16 * 3 * sizeof(float);
    };
    dim3 grid_wmma((unsigned)G, (unsigned)splits, (unsigned)B);
    #define LAUNCH_DW(D_, HPG_, BKV_)                                          \
      gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kF16>             \
          <<<grid_wmma, 32, smem_bytes(D_, BKV_), stream>>>(                   \
              hQ, hK, hV, nullptr, nullptr, hP, H, G, max_seq, scale, iSeq,    \
              local_window_size, splits)
    if (hpg == 4) {
      if (d == 128) LAUNCH_DW(128, 4, 32); else LAUNCH_DW(64, 4, 16);
    } else { /* hpg == 8, d == 64 */
      LAUNCH_DW(64, 8, 16);
    }
    #undef LAUNCH_DW
  } else {
    dim3 grid_main(B * G, splits);
    // Scalar decode: one wave (32 lanes) per query head, HpG waves per block.
    // The kernel body is HpG-generic (each wave owns one query head and reads
    // the KV cache with the max_seq stride), so a single template covers GQA
    // and MHA (HpG==1) alike; we just instantiate every supported HpG.
    #define LAUNCH_SCALAR(HPG_)                                                \
      do {                                                                     \
        if (d == 256)                                                          \
          gqa_flash_decode_kernel<256, HPG_, KvDtype::kF16>                    \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                  \
                  hQ, hK, hV, nullptr, nullptr, hP, H, G, max_seq, scale,      \
                  iSeq, local_window_size, splits);                            \
        else if (d == 128)                                                     \
          gqa_flash_decode_kernel<128, HPG_, KvDtype::kF16>                   \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                  \
                  hQ, hK, hV, nullptr, nullptr, hP, H, G, max_seq, scale,      \
                  iSeq, local_window_size, splits);                            \
        else                                                                   \
          gqa_flash_decode_kernel<64, HPG_, KvDtype::kF16>                    \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                  \
                  hQ, hK, hV, nullptr, nullptr, hP, H, G, max_seq, scale,      \
                  iSeq, local_window_size, splits);                            \
      } while (0)
    switch (hpg) {
      case 1:  LAUNCH_SCALAR(1);  break;
      case 2:  LAUNCH_SCALAR(2);  break;
      case 3:  LAUNCH_SCALAR(3);  break;
      case 4:  LAUNCH_SCALAR(4);  break;
      case 5:  LAUNCH_SCALAR(5);  break;
      case 8:  LAUNCH_SCALAR(8);  break;
      case 16: LAUNCH_SCALAR(16); break;
      default: break;  // unsupported HpG is rejected in hip_gqa_flash_decode_v2
    }
    #undef LAUNCH_SCALAR
  }

  // Reduce: one block per query head, block dim D. seqlens_k / sliding-window /
  // head_sink / smooth_softmax are all folded in here (not the partials).
  dim3 grid_reduce(B * H);
  if (d == 256)
    gqa_flash_decode_reduce_kernel<256><<<grid_reduce, 256, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
  else if (d == 128)
    gqa_flash_decode_reduce_kernel<128><<<grid_reduce, 128, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
  else
    gqa_flash_decode_reduce_kernel<64><<<grid_reduce, 64, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
}

// Candidate split counts, clamped to [1, cap] and to the useful range for skv
// (no point launching more splits than there are 16-key tiles). Deduplicated.
static void flashDecodeCandidateSplits(int skv, int max_splits,
                                       int (&out)[7], int& n) {
  // Small counts {2,4} let short contexts pick a leaner reduce (fewer partial
  // rows -> smaller reduce launch). But a small split at LARGE skv makes each
  // block serial-scan skv/splits keys, which is a hard cliff (measured at
  // skv=2048: splits=2 -> 0.175ms, splits=4 -> 0.0075ms, vs splits>=8 ->
  // ~0.0045ms). The autotune only times 10 iters, so at us scale it can noisily
  // mis-pick a catastrophic small split. Guard it structurally: never offer a
  // split whose per-block KV scan exceeds kMaxKeysPerSplit. This keeps {2,4}
  // available exactly where they help (short contexts) and also raises the
  // floor at huge skv so the tuner can't land on an under-filled config.
  static const int kBase[7] = {2, 4, 8, 16, 32, 48, 64};
  constexpr int kMaxKeysPerSplit = 256;
  int cap = max_splits < kFlashDecodeMaxSplits ? max_splits : kFlashDecodeMaxSplits;
  if (cap < 1) cap = 1;
  int eff = skv > 0 ? skv : 1;
  int max_useful = (eff + 15) / 16; if (max_useful < 1) max_useful = 1;
  int min_splits = (eff + kMaxKeysPerSplit - 1) / kMaxKeysPerSplit;
  if (min_splits < 1) min_splits = 1;
  if (min_splits > cap) min_splits = cap;
  n = 0;
  for (int s : kBase) {
    int v = s; if (v > cap) v = cap; if (v > max_useful) v = max_useful;
    if (v < min_splits) continue;  // too few splits for this skv -> skip
    bool dup = false;
    for (int i = 0; i < n; ++i) if (out[i] == v) dup = true;
    if (!dup) out[n++] = v;
  }
  if (n == 0) { out[0] = min_splits; n = 1; }  // safety: always >=1 candidate
}

static std::unordered_map<uint64_t, FlashDecodeCfg> g_flash_decode_cache;
static std::mutex g_flash_decode_mutex;

// Packed integer key (no heap alloc on the hot path -- the old std::string key
// did 5x to_string + concat + a malloc on EVERY decode token). NOTE: this is
// host-CPU hygiene only -- it does NOT move the dispatch-bench numbers, because
// that bench enqueues thousands of async calls and the host runs far ahead, so
// the per-call CPU cost overlaps GPU execution. It matters in *host-bound* real
// decode (per-token host sync / sampling between steps drains the queue, so the
// CPU cost is on the critical path). skv is bucketed to its next-power-of-two
// exponent. Layout: B[40..51] H[28..39] G[16..27] d[6..15] log2bucket[0..5].
// d gets a full 10 bits so d=256 (Qwen3 16:4) does not spill into G.
static uint64_t flashDecodeKey(int B, int H, int G, int d, int skv) {
  int blog = 0;
  uint32_t bucket = 1;
  while (bucket < (uint32_t)skv) { bucket <<= 1; ++blog; }
  return ((uint64_t)(uint32_t)B << 40) | ((uint64_t)(uint32_t)H << 28) |
         ((uint64_t)(uint32_t)G << 16) | ((uint64_t)(uint32_t)d << 6) |
         (uint64_t)(uint32_t)blog;
}

// Benchmark all (impl, splits) candidates on the actual device buffers and pick
// the fastest. Leaves the output buffers holding the *last* candidate's result;
// the caller relaunches the winner afterwards to produce the correct output.
static FlashDecodeCfg tuneFlashDecode(
    hipStream_t stream,
    const __half* hQ, const __half* hK, const __half* hV, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int skv, int max_seq, int max_splits,
    float scale, const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax) {
  int split_set[7]; int n_splits = 0;
  flashDecodeCandidateSplits(skv, max_splits, split_set, n_splits);
  const bool impls[2] = {false, true};  // {scalar, WMMA}; geometry is identical

  hipEvent_t ev0, ev1;
  hipEventCreate(&ev0);
  hipEventCreate(&ev1);
  constexpr int WARMUP = 2;
  constexpr int ITERS  = 10;

  const bool wmma_ok = flash_decode_wmma_supported(d, hpg);
  FlashDecodeCfg best{wmma_ok && (d == 64), split_set[0]};
  float best_ms = 1e30f;

  for (int ii = 0; ii < 2; ++ii) {
    if (impls[ii] && !wmma_ok) continue;  // skip WMMA where it isn't templated
    for (int si = 0; si < n_splits; ++si) {
      FlashDecodeCfg cfg{impls[ii], split_set[si]};
      for (int w = 0; w < WARMUP; ++w)
        launchFlashDecodeConfig(cfg, stream, hQ, hK, hV, hO, hP, B, H, G, d, hpg,
                                max_seq, scale, iSeq, local_window_size, hSink,
                                use_smooth_softmax);
      hipEventRecord(ev0, stream);
      for (int it = 0; it < ITERS; ++it)
        launchFlashDecodeConfig(cfg, stream, hQ, hK, hV, hO, hP, B, H, G, d, hpg,
                                max_seq, scale, iSeq, local_window_size, hSink,
                                use_smooth_softmax);
      hipEventRecord(ev1, stream);
      hipEventSynchronize(ev1);
      float ms = 0.0f;
      hipEventElapsedTime(&ms, ev0, ev1);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels]   flash_decode cand impl=%-6s splits=%2d : %.4f ms\n",
          cfg.use_wmma ? "wmma" : "scalar", cfg.splits, ms / ITERS);
      if (ms < best_ms) { best_ms = ms; best = cfg; }
    }
  }

  hipEventDestroy(ev0);
  hipEventDestroy(ev1);
  return best;
}

// INT8 decode autotune state + helpers (defined below). Forward-declared here so
// the unified hip_gqa_flash_decode_v2 can dispatch the int8 path when scales are
// supplied; the launcher/tuner bodies live after the fp16 entry to keep the two
// autotune skeletons side by side.
static std::unordered_map<uint64_t, FlashDecodeCfg> g_flash_decode_i8_cache;
static std::mutex g_flash_decode_i8_mutex;
static inline void launchFlashDecodeConfigI8(
    const FlashDecodeCfg& cfg, hipStream_t stream,
    const __half* hQ, const int8_t* hK, const int8_t* hV,
    const float* kSc, const float* vSc, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int max_seq, float scale,
    const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax);
static FlashDecodeCfg tuneFlashDecodeI8(
    hipStream_t stream,
    const __half* hQ, const int8_t* hK, const int8_t* hV,
    const float* kSc, const float* vSc, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int skv, int max_seq, int max_splits,
    float scale, const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax);

// Launcher: dispatches FA-2 split-K decode + reduction.
//   partials_workspace: float scratch of size B*H*max_splits*(d+2)*4 bytes.
//   skv:        current total KV length (host-known) used only for autotune
//               keying + candidate clamping; the kernels read seqlens_k for the
//               exact per-batch length at runtime.
//   max_splits: workspace capacity in splits (autotune picks an actual count
//               <= this). Pass <= kFlashDecodeMaxSplits.
// Impl / split selection is autotuned per (B,H,G,d, skv-bucket) and cached.
// Env overrides (skip autotune entirely):
//   HIPDNN_GQA_DECODE_SCALAR=1 / _WMMA=1 -> force impl
//   HIPDNN_GQA_DECODE_SPLITS=N           -> force split count
//   HIPDNN_GQA_DECODE_NOAUTOTUNE=1       -> fixed heuristic (wmma@d64, splits=8)
//
// KV-cache format is chosen by kv_dtype (hip_kv_dtype_t), not a separate entry:
//   HIP_KV_DTYPE_FP16 -> fp16 K/V cache (Kcache/Vcache are __half).
//   HIP_KV_DTYPE_INT8 -> symmetric per-channel INT8 K/V cache
//     (Kcache/Vcache are int8 [B,G,max_seq,d]; k_scale/v_scale fp32 [G,d]).
// The int8 read is 1 byte/elem (half the DRAM traffic on the bandwidth-bound
// decode). Both formats share this autotune/env skeleton and emit identical
// [B,H,splits,d+2] partials, so only the tune/launch tail differs. Adding INT4/
// FP8 = one enumerator (hip_kv_dtype_t) + a kernel specialization + a branch.
extern "C" int hip_gqa_flash_decode_v2(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int skv, int max_seq, int max_splits,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax,
    int kv_dtype,
    const void* k_scale,
    const void* v_scale) {
  // HPG = heads-per-group (Q heads sharing one KV head). The kernels are
  // instantiated for HPG=4 (Llama-3.x family at d=64 and d=128), HPG=5
  // (Qwen2.5-14B's 40:8 at d=128) and HPG=8 (gpt-oss-20b at d=64). HPG=8 with
  // d=128 is rejected to keep template bloat bounded; no current model needs it.
  if (G <= 0 || H % G != 0) {
    fprintf(stderr, "hip_gqa_flash_decode_v2: H=%d not divisible by G=%d\n", H, G);
    return -1;
  }
  const int hpg = H / G;
  // Scalar decode is templated for these HpG (GQA + MHA); WMMA is layered on
  // top where flash_decode_wmma_supported() says so. Anything else is rejected.
  if (hpg != 1 && hpg != 2 && hpg != 3 && hpg != 4 && hpg != 5 && hpg != 8 &&
      hpg != 16) {
    fprintf(stderr,
            "hip_gqa_flash_decode_v2: unsupported HPG=%d (must be "
            "1/2/3/4/5/8/16; H=%d, G=%d)\n",
            hpg, H, G);
    return -1;
  }
  if (d != 64 && d != 128 && d != 256) {
    fprintf(stderr,
            "hip_gqa_flash_decode_v2: unsupported d=%d (must be 64/128/256)\n",
            d);
    return -1;
  }
  // Cache format is chosen by kv_dtype (extensible: add a case for INT4/FP8).
  if (kv_dtype != HIP_KV_DTYPE_FP16 && kv_dtype != HIP_KV_DTYPE_INT8) {
    fprintf(stderr, "hip_gqa_flash_decode_v2: unsupported kv_dtype=%d\n",
            kv_dtype);
    return -1;
  }
  const bool kv_i8 = (kv_dtype == HIP_KV_DTYPE_INT8);
  if (kv_i8 && (k_scale == nullptr || v_scale == nullptr)) {
    fprintf(stderr,
            "hip_gqa_flash_decode_v2: INT8 KV requires k_scale and v_scale\n");
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  const __half* hQ = static_cast<const __half*>(Q);
  __half* hO = static_cast<__half*>(O);
  float* hP = static_cast<float*>(partials_workspace);
  const int* iSeq = static_cast<const int*>(seqlens_k);
  const __half* hSink = static_cast<const __half*>(head_sink);

  int cap = max_splits;
  if (cap < 1) cap = 1;
  if (cap > kFlashDecodeMaxSplits) cap = kFlashDecodeMaxSplits;
  const int eff_skv = skv > 0 ? skv : max_seq;  // host-side length for tuning
  // Effective KV range each query actually scans. Sliding-window layers clamp
  // to `local_window_size`, so their optimal split count is driven by the
  // window (e.g. 128), NOT the full context -- they MUST key/tune separately
  // from full-attention layers of the same (B,H,G,d,skv) or they would reuse a
  // large split count and pay reduce/empty-split overhead over a tiny range.
  const int tune_len = (local_window_size > 0 && eff_skv > local_window_size)
                           ? local_window_size
                           : eff_skv;

  // --- Config selection: env override > autotune cache > one-shot tune ---
  // Heuristic default favours WMMA at d==64, but only where it is templated;
  // every other geometry (incl. MHA) starts from the scalar kernel.
  FlashDecodeCfg cfg{flash_decode_wmma_supported(d, hpg) && (d == 64),
                     cap < 8 ? cap : 8};

  bool forced = false;
  if (const char* e = getenv("HIPDNN_GQA_DECODE_SCALAR")) { if (atoi(e) != 0) { cfg.use_wmma = false; forced = true; } }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_WMMA"))   { if (atoi(e) != 0) { cfg.use_wmma = true;  forced = true; } }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_SPLITS")) {
    int v = atoi(e); if (v >= 1) { cfg.splits = v > cap ? cap : v; forced = true; }
  }
  bool no_autotune = false;
  if (const char* e = getenv("HIPDNN_GQA_DECODE_NOAUTOTUNE")) no_autotune = (atoi(e) != 0);

  // INT8 path: cache is int8, scales are per-channel fp32. Autotune keeps its
  // own cache/mutex (identical geometry, but the int8 kernels time differently).
  if (kv_i8) {
    const int8_t* hK  = static_cast<const int8_t*>(Kcache);
    const int8_t* hV  = static_cast<const int8_t*>(Vcache);
    const float*  kSc = static_cast<const float*>(k_scale);
    const float*  vSc = static_cast<const float*>(v_scale);
    if (!forced && !no_autotune) {
      const uint64_t key = flashDecodeKey(B, H, G, d, tune_len);
      std::lock_guard<std::mutex> lock(g_flash_decode_i8_mutex);
      auto it = g_flash_decode_i8_cache.find(key);
      if (it != g_flash_decode_i8_cache.end()) {
        cfg = it->second;
      } else {
        cfg = tuneFlashDecodeI8(stream, hQ, hK, hV, kSc, vSc, hO, hP, B, H, G, d,
                                hpg, tune_len, max_seq, cap, scale, iSeq,
                                local_window_size, hSink, use_smooth_softmax);
        g_flash_decode_i8_cache[key] = cfg;
      }
    }
    launchFlashDecodeConfigI8(cfg, stream, hQ, hK, hV, kSc, vSc, hO, hP, B, H, G,
                              d, hpg, max_seq, scale, iSeq, local_window_size,
                              hSink, use_smooth_softmax);
    return hipGetLastError();
  }

  const __half* hK = static_cast<const __half*>(Kcache);
  const __half* hV = static_cast<const __half*>(Vcache);
  if (!forced && !no_autotune) {
    const uint64_t key = flashDecodeKey(B, H, G, d, tune_len);
    std::lock_guard<std::mutex> lock(g_flash_decode_mutex);
    auto it = g_flash_decode_cache.find(key);
    if (it != g_flash_decode_cache.end()) {
      cfg = it->second;
    } else {
      cfg = tuneFlashDecode(stream, hQ, hK, hV, hO, hP, B, H, G, d, hpg,
                            tune_len, max_seq, cap, scale, iSeq,
                            local_window_size, hSink, use_smooth_softmax);
      g_flash_decode_cache[key] = cfg;
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] flash_decode autotune: key=0x%llx -> impl=%s splits=%d "
          "(cache size=%zu)\n",
          (unsigned long long)key, cfg.use_wmma ? "wmma" : "scalar", cfg.splits,
          g_flash_decode_cache.size());
    }
  }

  // Produce the correct output with the selected config (tuning above left the
  // buffers holding the last candidate's result).
  launchFlashDecodeConfig(cfg, stream, hQ, hK, hV, hO, hP, B, H, G, d, hpg,
                          max_seq, scale, iSeq, local_window_size, hSink,
                          use_smooth_softmax);

  return hipGetLastError();
}

//===----------------------------------------------------------------------===//
// INT8 decode launcher: same autotune structure as hip_gqa_flash_decode_v2, but
// carries the k_scale/v_scale per-channel tables and dispatches the int8 kernels.
//===----------------------------------------------------------------------===//
static inline void launchFlashDecodeConfigI8(
    const FlashDecodeCfg& cfg, hipStream_t stream,
    const __half* hQ, const int8_t* hK, const int8_t* hV,
    const float* kSc, const float* vSc, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int max_seq, float scale,
    const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax) {
  const int splits = cfg.splits;
  const bool use_wmma = cfg.use_wmma && flash_decode_wmma_supported(d, hpg);
  if (use_wmma) {
    auto smem_bytes = [](int D_, int bkv) -> size_t {
      const int DSTR = D_ + 2;
      return (size_t)16 * (bkv + 1) * sizeof(float)
           + (size_t)2 * bkv * DSTR * sizeof(_Float16)
           + (size_t)16 * (bkv + 2) * sizeof(_Float16)
           + (size_t)16 * 3 * sizeof(float)
           + (size_t)2 * D_ * sizeof(float);  // ksc_sh + vsc_sh
    };
    dim3 grid_wmma((unsigned)G, (unsigned)splits, (unsigned)B);
    #define LAUNCH_DW_I8(D_, HPG_, BKV_)                                        \
      gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kI8>                \
          <<<grid_wmma, 32, smem_bytes(D_, BKV_), stream>>>(                    \
              hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,             \
              local_window_size, splits)
    if (hpg == 4) {
      if (d == 128) LAUNCH_DW_I8(128, 4, 32); else LAUNCH_DW_I8(64, 4, 16);
    } else { /* hpg == 8, d == 64 */
      LAUNCH_DW_I8(64, 8, 16);
    }
    #undef LAUNCH_DW_I8
  } else {
    dim3 grid_main(B * G, splits);
    #define LAUNCH_SCALAR_I8(HPG_)                                              \
      do {                                                                      \
        if (d == 256)                                                           \
          gqa_flash_decode_kernel<256, HPG_, KvDtype::kI8>                      \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                    \
                  hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,         \
                  local_window_size, splits);                                   \
        else if (d == 128)                                                      \
          gqa_flash_decode_kernel<128, HPG_, KvDtype::kI8>                     \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                    \
                  hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,         \
                  local_window_size, splits);                                   \
        else                                                                    \
          gqa_flash_decode_kernel<64, HPG_, KvDtype::kI8>                      \
              <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                    \
                  hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,         \
                  local_window_size, splits);                                   \
      } while (0)
    switch (hpg) {
      case 1:  LAUNCH_SCALAR_I8(1);  break;
      case 2:  LAUNCH_SCALAR_I8(2);  break;
      case 3:  LAUNCH_SCALAR_I8(3);  break;
      case 4:  LAUNCH_SCALAR_I8(4);  break;
      case 5:  LAUNCH_SCALAR_I8(5);  break;
      case 8:  LAUNCH_SCALAR_I8(8);  break;
      case 16: LAUNCH_SCALAR_I8(16); break;
      default: break;
    }
    #undef LAUNCH_SCALAR_I8
  }

  dim3 grid_reduce(B * H);
  if (d == 256)
    gqa_flash_decode_reduce_kernel<256><<<grid_reduce, 256, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
  else if (d == 128)
    gqa_flash_decode_reduce_kernel<128><<<grid_reduce, 128, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
  else
    gqa_flash_decode_reduce_kernel<64><<<grid_reduce, 64, 0, stream>>>(
        hP, hO, H, hSink, H, use_smooth_softmax, splits);
}

static FlashDecodeCfg tuneFlashDecodeI8(
    hipStream_t stream,
    const __half* hQ, const int8_t* hK, const int8_t* hV,
    const float* kSc, const float* vSc, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int skv, int max_seq, int max_splits,
    float scale, const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax) {
  int split_set[7]; int n_splits = 0;
  flashDecodeCandidateSplits(skv, max_splits, split_set, n_splits);
  const bool impls[2] = {false, true};

  hipEvent_t ev0, ev1;
  hipEventCreate(&ev0);
  hipEventCreate(&ev1);
  constexpr int WARMUP = 2;
  constexpr int ITERS  = 10;

  const bool wmma_ok = flash_decode_wmma_supported(d, hpg);
  FlashDecodeCfg best{wmma_ok && (d == 64), split_set[0]};
  float best_ms = 1e30f;

  for (int ii = 0; ii < 2; ++ii) {
    if (impls[ii] && !wmma_ok) continue;
    for (int si = 0; si < n_splits; ++si) {
      FlashDecodeCfg cfg{impls[ii], split_set[si]};
      for (int w = 0; w < WARMUP; ++w)
        launchFlashDecodeConfigI8(cfg, stream, hQ, hK, hV, kSc, vSc, hO, hP,
                                  B, H, G, d, hpg, max_seq, scale, iSeq,
                                  local_window_size, hSink, use_smooth_softmax);
      hipEventRecord(ev0, stream);
      for (int it = 0; it < ITERS; ++it)
        launchFlashDecodeConfigI8(cfg, stream, hQ, hK, hV, kSc, vSc, hO, hP,
                                  B, H, G, d, hpg, max_seq, scale, iSeq,
                                  local_window_size, hSink, use_smooth_softmax);
      hipEventRecord(ev1, stream);
      hipEventSynchronize(ev1);
      float ms = 0.0f;
      hipEventElapsedTime(&ms, ev0, ev1);
      if (ms < best_ms) { best_ms = ms; best = cfg; }
    }
  }
  hipEventDestroy(ev0);
  hipEventDestroy(ev1);
  return best;
}

//===----------------------------------------------------------------------===//
// Fused FA-2 WMMA prefill kernels (v5 d64 / v7 d128 / v8 d256)
//
// fp16 in/out, fp32 accumulate, causal, GQA. Q [B,sq,Hq,D]; K/V cache
// [B,G,max_seq,D]. Each launcher self-tunes its config per (d,sq,skv,Hq,G)
// shape on first use. The unified hip_gqa_flash_prefill_v2 entry selects:
//   d==64  -> v5 (warp-private, register-resident)
//   d==128 -> v7 (M-register-blocked, K streamed from global)
//   d==256 -> v8 (output-D split across ND waves + secondary score
//                 reduction; halves per-wave Q_reg/O_h to lift occupancy)
//===----------------------------------------------------------------------===//
namespace {
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));
#ifndef HALF16_LOAD
#define HALF16_LOAD(ptr) (*reinterpret_cast<const half16_t*>(ptr))
#endif




// ---------------------------------------------------------------------
// Warp-private prefill kernel (v5). Grid: (num_q_tiles*B, Hq). Block: 32.
// Each wave owns M_TILES 16-row q-subtiles x all BKV columns. Only LDS is
// P_lds (C->A transpose); only sync is single-wave around it.
// ---------------------------------------------------------------------
template <int M_TILES, int BKV, int D>
__global__ void __launch_bounds__(32)
gqa_flash_prefill_v5_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate,
    const _Float16* __restrict__ head_sink,  // [num_heads] or null
    int num_heads, int smooth_softmax)
{
    // ablation bitmask (perf diagnostics only; wrong numerics):
    //   bit0 skip softmax, bit1 skip valueGEMM, bit2 skip scoreGEMM,
    //   bit3 skip global loads (K frags + V_lds staging).
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = M_TILES * kWmmaTile;
    constexpr int S_TILES_J = BKV / kWmmaTile;   // 16-col score tiles
    constexpr int D_TILES   = D / kWmmaTile;     // 16-col D tiles
    constexpr int P_STR     = BKV + 2;
    constexpr int V_STR     = D + 2;             // V_lds row stride (pad)
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));
    const int q_start     = q_tile_idx * ROWS;

    const int lane_id   = threadIdx.x;          // 0..31
    const int wmma_lane = lane_id % kWmmaTile;   // 0..15
    const int pair      = lane_id / kWmmaTile;   // 0 -> even rows, 1 -> odd

    // Per-wave LDS: P scratch for the WMMA C->A transpose of P, and a V tile
    // staged with coalesced HALF16 loads (the direct-from-global V fragment is
    // a stride-D gather that wrecks D=128; K stays direct since it is already
    // coalesced HALF16 along D).
    extern __shared__ char smem_v5[];
    _Float16* P_lds = reinterpret_cast<_Float16*>(smem_v5);
    _Float16* V_lds = reinterpret_cast<_Float16*>(
        smem_v5 + ROWS * P_STR * sizeof(_Float16));

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: Q_reg[mi][tk], row = q_start + mi*16 + wmma_lane ----
    half16_t Q_reg[M_TILES][D_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        const int q_pos = q_start + mi * 16 + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < D_TILES; ++tk)
            Q_reg[mi][tk] = valid ? HALF16_LOAD(&Q[q_row_base + tk * 16]) : half16_t{};
    }

    // ---- running state in registers (per owned row) ----
    float8_t O_frag[M_TILES][D_TILES] = {};
    float    m_reg[M_TILES][8];
    float    l_reg[M_TILES][8];
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
        #pragma unroll
        for (int e = 0; e < 8; ++e) { m_reg[mi][e] = -INFINITY; l_reg[mi][e] = 0.0f; }

    // Score GEMM for one kv tile -> S_reg (registers), scaled by scale*log2e.
    // K is loaded straight from global as B-fragments (reused across mi).
    auto scoreGEMM = [&](int kv0, float8_t S_reg[M_TILES][S_TILES_J]) {
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mi][sj] = float8_t{};
        if (abl & 4) return;  // skip scoreGEMM (also its K loads)
        #pragma unroll
        for (int sj = 0; sj < S_TILES_J; ++sj) {
            const int key = kv0 + sj * 16 + wmma_lane;
            const bool kvalid = key < skv;
            const size_t k_row = kv_base + static_cast<size_t>(kvalid ? key : 0) * D;
            #pragma unroll
            for (int tk = 0; tk < D_TILES; ++tk) {
                half16_t k_frag = (kvalid && !(abl & 8))
                                         ? HALF16_LOAD(&Kcache[k_row + tk * 16])
                                         : half16_t{};
                #pragma unroll
                for (int mi = 0; mi < M_TILES; ++mi)
                    S_reg[mi][sj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        Q_reg[mi][tk], k_frag, S_reg[mi][sj]);
            }
        }
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    S_reg[mi][sj][e] *= scale * kLog2e;
    };

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // Stage V tile into LDS with coalesced HALF16 loads along D.
        if (!(abl & 8)) {
            #pragma unroll
            for (int i = lane_id; i < BKV * D_TILES; i += 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                half16_t vv = (kv_pos < skv)
                    ? HALF16_LOAD(&Vcache[kv_base + (size_t)kv_pos * D + dt * 16])
                    : half16_t{};
                *reinterpret_cast<half16_t*>(&V_lds[kv_local * V_STR + dt * 16]) = vv;
            }
        }

        float8_t S_reg[M_TILES][S_TILES_J];
        scoreGEMM(kv0, S_reg);

        // ---- online softmax, fully intra-wave; writes P_lds (row-major) ----
        float corr_reg[M_TILES][8];
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int e = 0; e < 8; ++e) corr_reg[mi][e] = 1.0f;
        if (!(abl & 1))
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row    = mi * 16 + e * 2 + pair;        // local q row
                const int q_pos  = q_start + row;
                // causal mask + per-row partial max over this lane's columns
                float rmax = -INFINITY;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const int kv_pos = kv0 + sj * 16 + wmma_lane;
                    if (kv_pos >= skv || kv_pos > past_len + q_pos)
                        S_reg[mi][sj][e] = -INFINITY;
                    rmax = fmaxf(rmax, S_reg[mi][sj][e]);
                }
                rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                const float m_old = m_reg[mi][e];
                const float m_new = fmaxf(m_old, rmax);
                const float corr  = exp2f(m_old - m_new);
                corr_reg[mi][e]   = corr;

                float rsum = 0.0f;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const float p = exp2f(S_reg[mi][sj][e] - m_new);
                    rsum += p;
                    P_lds[row * P_STR + sj * 16 + wmma_lane] = static_cast<_Float16>(p);
                }
                rsum += __shfl_xor(rsum, 1);
                rsum += __shfl_xor(rsum, 2);
                rsum += __shfl_xor(rsum, 4);
                rsum += __shfl_xor(rsum, 8);

                m_reg[mi][e] = m_new;
                l_reg[mi][e] = corr * l_reg[mi][e] + rsum;
            }
        }
        __syncthreads();   // single-wave: P_lds + V_lds visible for valueGEMM

        // ---- value GEMM: rescale O by correction, O += P . V (V from global) ----
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int tj = 0; tj < D_TILES; ++tj)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    O_frag[mi][tj][e] *= corr_reg[mi][e];

        if (!(abl & 2))
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                // V as B-fragment from LDS: v[e=kv] for column d = tj*16+wmma_lane
                half16_t v_frag;
                #pragma unroll
                for (int e = 0; e < 16; ++e)
                    v_frag[e] = V_lds[(tk * 16 + e) * V_STR + tj * 16 + wmma_lane];
                #pragma unroll
                for (int mi = 0; mi < M_TILES; ++mi) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(mi * 16 + wmma_lane) * P_STR + tk * 16]);
                    O_frag[mi][tj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p_frag, v_frag, O_frag[mi][tj]);
                }
            }
        }
        __syncthreads();   // WAR: P_lds reused by next tile's softmax
    }

    // ---- Attention sink: one extra term in the softmax denominator ----
    // m_reg / l_reg are in log2 space (scores were premultiplied by
    // scale * kLog2e above and exp2f is used for both), so m_reg == m_nat *
    // kLog2e and the natural-units sink term exp(s - m_nat) becomes
    // exp2(s * kLog2e - m_reg) -- the same identity the decode reduce kernel
    // documents. Row-invariant in the D dimension, so it is folded into l once
    // here rather than inside the D_TILES loop below.
    //
    // smooth_softmax with a null sink means s = 0, matching
    // softmax_f32_to_out_kernel. A row where every key was masked keeps
    // m_reg == -INFINITY; it must be skipped, because the exponent would
    // otherwise be +inf and the whole row would come out zero instead of
    // sink-only.
    float l_final[M_TILES][8];
    const bool apply_sink = (head_sink != nullptr) || (smooth_softmax != 0);
    float sink_logit = 0.0f;
    if (head_sink != nullptr)
        sink_logit = static_cast<float>(
            head_sink[num_heads > 0 ? (head_q % num_heads) : head_q]);
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float l = l_reg[mi][e];
            if (apply_sink && m_reg[mi][e] > -INFINITY)
                l += exp2f(sink_logit * kLog2e - m_reg[mi][e]);
            l_final[mi][e] = fmaxf(l, 1e-6f);
        }
    }

    // ---- Epilogue: O / l ----
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row   = mi * 16 + e * 2 + pair;
                const int c     = tj * 16 + wmma_lane;
                const int q_pos = q_start + row;
                if (q_pos < sq && c < D) {
                    const float out_val = O_frag[mi][tj][e] / l_final[mi][e];
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Host side: LDS sizing, autotune (M_TILES x BKV), launcher.
// ---------------------------------------------------------------------
static inline size_t prefillV5SmemBytes(int m_tiles, int bkv, int d) {
    const size_t p_lds = (size_t)(m_tiles * 16) * (bkv + 2) * sizeof(_Float16);
    const size_t v_lds = (size_t)bkv * (d + 2) * sizeof(_Float16);
    return p_lds + v_lds;
}
static constexpr size_t kPrefillV5MaxSmem = 65536;

static int g_v5_ablate = 0;  // perf-diagnostics ablation bitmask (default off)

static inline void dispatchPrefillV5(
    int m_tiles, int bkv, int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale,
    const _Float16* head_sink, int num_heads, int smooth_softmax)
{
    const int rows = m_tiles * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV5SmemBytes(m_tiles, bkv, d);
    const int ablate = g_v5_ablate;

    #define LAUNCH_V5(MT_, BKV_, D_)                                            \
        gqa_flash_prefill_v5_kernel<MT_, BKV_, D_>                              \
            <<<grid, 32, smem, stream>>>(                                       \
                Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,     \
                scale, ablate, head_sink, num_heads, smooth_softmax)

    if (d == 64) {
        if (m_tiles == 2 && bkv == 64)      LAUNCH_V5(2, 64, 64);
        else if (m_tiles == 2)              LAUNCH_V5(2, 32, 64);
        else if (bkv == 64)                 LAUNCH_V5(1, 64, 64);
        else                                LAUNCH_V5(1, 32, 64);
    } else {
        if (m_tiles == 2 && bkv == 64)      LAUNCH_V5(2, 64, 128);
        else if (m_tiles == 2)              LAUNCH_V5(2, 32, 128);
        else if (bkv == 64)                 LAUNCH_V5(1, 64, 128);
        else                                LAUNCH_V5(1, 32, 128);
    }
    #undef LAUNCH_V5
}

struct PrefillV5Cfg { int m_tiles; int bkv; };
static inline int prefillV5Candidates(int d, PrefillV5Cfg* out) {
    static const PrefillV5Cfg kAll[] = {
        {1, 32}, {1, 64}, {2, 32}, {2, 64}};
    int n = 0;
    for (const PrefillV5Cfg& c : kAll)
        if (prefillV5SmemBytes(c.m_tiles, c.bkv, d) <= kPrefillV5MaxSmem) out[n++] = c;
    return n;
}

static std::unordered_map<std::string, int> g_v5_cfg_cache;
static std::mutex g_v5_cfg_mutex;
static char g_v5_tune_info[128] = "gqa_flash_prefill_v5 cfg=?";

static int tunePrefillV5Cfg(
    int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale)
{
    PrefillV5Cfg cands[8];
    const int nc = prefillV5Candidates(d, cands);
    constexpr int kWarmup = 30, kRounds = 4, kIters = 30;

    // Tuning launches carry no sink: it is a single fused-multiply-add per row in
    // the epilogue, so it cannot shift which (M_TILES, BKV) wins, and leaving it
    // out keeps tuned configs comparable with the non-sink case.
    auto launch_cfg = [&](const PrefillV5Cfg& c) {
        dispatchPrefillV5(c.m_tiles, c.bkv, d, stream, Q, Kcache, Vcache, O,
                          B, Hq, G, sq, skv, max_seq, past_len, scale,
                          nullptr, Hq, 0);
    };

    for (int w = 0; w < kWarmup; ++w) launch_cfg(cands[w % nc]);
    (void)hipDeviceSynchronize();

    hipEvent_t ev0, ev1;
    (void)hipEventCreate(&ev0);
    (void)hipEventCreate(&ev1);
    float best_ms[8];
    for (int i = 0; i < nc; ++i) best_ms[i] = 1.0e30f;
    for (int r = 0; r < kRounds; ++r) {
        for (int idx = 0; idx < nc; ++idx) {
            launch_cfg(cands[idx]);
            (void)hipEventRecord(ev0, stream);
            for (int i = 0; i < kIters; ++i) launch_cfg(cands[idx]);
            (void)hipEventRecord(ev1, stream);
            (void)hipEventSynchronize(ev1);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, ev0, ev1);
            if (ms < best_ms[idx]) best_ms[idx] = ms;
        }
    }
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);

    if (getenv("HIPDNN_PREFILL_TUNE_DEBUG")) {
        for (int i = 0; i < nc; ++i)
            fprintf(stderr,
                "[prefill-v5-tune] d=%d sq=%d skv=%d  M_TILES=%-2d BKV=%-3d "
                "num_q_tiles=%-4d : %.5f ms/call\n",
                d, sq, skv, cands[i].m_tiles, cands[i].bkv,
                (sq + cands[i].m_tiles * 16 - 1) / (cands[i].m_tiles * 16),
                best_ms[i] / kIters);
    }

    int best = 0;
    for (int i = 1; i < nc; ++i) if (best_ms[i] < best_ms[best]) best = i;
    return cands[best].m_tiles * 1000 + cands[best].bkv;
}





// ---------------------------------------------------------------------
// M-register-blocked shared-KV prefill kernel (v7). Grid: (num_q_tiles*B, Hq).
// Block: NW*32 threads. Each wave owns MT 16-row q-subtiles; K/V tiles are
// staged once into block-shared LDS, and within a wave each K/V fragment feeds
// all MT M-tiles -> the LDS read of K/V is amortized MT x.
// ---------------------------------------------------------------------
template <int NW, int BKV, int D, int MT>
__global__ void __launch_bounds__(NW * 32)
gqa_flash_prefill_v7_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate)
{
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = NW * MT * kWmmaTile;   // query rows per block
    constexpr int S_TILES_J = BKV / kWmmaTile;
    constexpr int D_TILES   = D / kWmmaTile;
    constexpr int P_STR     = BKV + 2;
    constexpr int KV_STR    = D + 2;
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));

    const int tid       = threadIdx.x;
    const int wave_id   = tid / 32;
    const int lane_id   = tid % 32;
    const int wmma_lane = lane_id % kWmmaTile;
    const int pair      = lane_id / kWmmaTile;       // 0 -> even rows, 1 -> odd

    const int q_start   = q_tile_idx * ROWS;
    const int row_base  = q_start + wave_id * MT * kWmmaTile;  // wave's first row
    const int pbase     = wave_id * MT * kWmmaTile;           // wave's P_lds offset

    // K streams from global (cache-resident) straight into WMMA fragments; only
    // V (stride-D gather) and the per-wave P transpose need LDS.
    extern __shared__ char smem_v7[];
    _Float16* V_lds = reinterpret_cast<_Float16*>(smem_v7);
    _Float16* P_lds = reinterpret_cast<_Float16*>(
        smem_v7 + (size_t)BKV * KV_STR * sizeof(_Float16));

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: Q_reg[mt][tk], row = row_base + mt*16 + wmma_lane.
    //      (Tried Q->LDS to free VGPRs -- net loss: occupancy didn't rise but the
    //       per-WMMA LDS re-reads hit the LDS port; registers win here. Also
    //       tried streaming Q from global at D=256 to shed the resident
    //       Q_reg[MT][16] -- also a net loss: the per-KV-tile Q reloads (xnum_tiles)
    //       cost more than the spill they avoid.) ----
    half16_t Q_reg[MT][D_TILES];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int q_pos = row_base + mt * kWmmaTile + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < D_TILES; ++tk)
            Q_reg[mt][tk] = valid ? HALF16_LOAD(&Q[q_row_base + tk * 16]) : half16_t{};
    }

    // ---- running state: O stored compactly as half8 (fp16 carry) ----
    half8_t O_h[MT][D_TILES] = {};
    float   m_reg[MT][8];
    float   l_reg[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 8; ++e) { m_reg[mt][e] = -INFINITY; l_reg[mt][e] = 0.0f; }

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // ---- cooperative V staging into shared LDS (K streams from global) ----
        if (!(abl & 8)) {
            #pragma unroll
            for (int i = tid; i < BKV * D_TILES; i += NW * 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                const bool kvalid  = kv_pos < skv;
                const size_t off   = kv_base + (size_t)(kvalid ? kv_pos : 0) * D + dt * 16;
                half16_t vv = kvalid ? HALF16_LOAD(&Vcache[off]) : half16_t{};
                *reinterpret_cast<half16_t*>(&V_lds[kv_local * KV_STR + dt * 16]) = vv;
            }
        }
        // NB: no barrier here. v7 streams K from global and softmax only touches
        // S_reg/P_lds -- nothing between here and B2 reads V_lds, so the V staging
        // overlaps the scoreGEMM+softmax work and is made visible by B2.
        // (The previous B1 was a v6 leftover from when scoreGEMM read K from LDS.)

        // ---- score GEMM: K streamed from global (cache-resident), reused / MT ----
        float8_t S_reg[MT][S_TILES_J];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mt][sj] = float8_t{};
        if (!(abl & 4)) {
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) {
                const int key = kv0 + sj * 16 + wmma_lane;
                const bool kvalid = key < skv;
                const size_t k_row = kv_base + (size_t)(kvalid ? key : 0) * D;
                for (int tk = 0; tk < D_TILES; ++tk) {
                    half16_t k_frag = (kvalid && !(abl & 16))
                        ? HALF16_LOAD(&Kcache[k_row + tk * 16]) : half16_t{};
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        S_reg[mt][sj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            Q_reg[mt][tk], k_frag, S_reg[mt][sj]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) S_reg[mt][sj][e] *= scale * kLog2e;
        }

        // ---- online softmax (per M-tile, intra-wave); fold corr into O_h ----
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float corr_e[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row    = e * 2 + pair;
                    const int q_pos  = row_base + mt * kWmmaTile + row;
                    float rmax = -INFINITY;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int kv_pos = kv0 + sj * 16 + wmma_lane;
                        if (kv_pos >= skv || kv_pos > past_len + q_pos)
                            S_reg[mt][sj][e] = -INFINITY;
                        rmax = fmaxf(rmax, S_reg[mt][sj][e]);
                    }
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                    const float m_old = m_reg[mt][e];
                    const float m_new = fmaxf(m_old, rmax);
                    const float corr  = exp2f(m_old - m_new);
                    corr_e[e]         = corr;

                    float rsum = 0.0f;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const float p = exp2f(S_reg[mt][sj][e] - m_new);
                        rsum += p;
                        P_lds[(pbase + mt * kWmmaTile + row) * P_STR + sj * 16 + wmma_lane] =
                            static_cast<_Float16>(p);
                    }
                    rsum += __shfl_xor(rsum, 1);
                    rsum += __shfl_xor(rsum, 2);
                    rsum += __shfl_xor(rsum, 4);
                    rsum += __shfl_xor(rsum, 8);

                    m_reg[mt][e] = m_new;
                    l_reg[mt][e] = corr * l_reg[mt][e] + rsum;
                }
                // rescale this M-tile's running O by correction (fp16 carry)
                #pragma unroll
                for (int tj = 0; tj < D_TILES; ++tj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        O_h[mt][tj][e] =
                            static_cast<_Float16>((float)O_h[mt][tj][e] * corr_e[e]);
            }
        }
        __syncthreads();   // B2

        // ---- value GEMM: V gathered once per (tj,tk), reused across MT ----
        if (!(abl & 2))
        for (int tj = 0; tj < D_TILES; ++tj) {
            float8_t Of[MT];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int e = 0; e < 8; ++e) Of[mt][e] = (float)O_h[mt][tj][e];
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                half16_t v_frag;
                #pragma unroll
                for (int e = 0; e < 16; ++e)
                    v_frag[e] = V_lds[(tk * 16 + e) * KV_STR + tj * 16 + wmma_lane];
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(pbase + mt * kWmmaTile + wmma_lane) * P_STR + tk * 16]);
                    Of[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p_frag, v_frag, Of[mt]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    O_h[mt][tj][e] = static_cast<_Float16>(Of[mt][e]);
        }
        __syncthreads();   // B3: WAR -- V_lds reused next tile
    }

    // ---- Epilogue: O / l ----
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row   = e * 2 + pair;
                const int c     = tj * 16 + wmma_lane;
                const int q_pos = row_base + mt * kWmmaTile + row;
                if (q_pos < sq && c < D) {
                    const float l_val   = fmaxf(l_reg[mt][e], 1e-6f);
                    const float out_val = (float)O_h[mt][tj][e] / l_val;
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
}

// ---------------------------------------------------------------------
// Host side: LDS sizing, autotune (NW x BKV at MT=2), launcher.
// ---------------------------------------------------------------------
static inline size_t prefillV7SmemBytes(int nw, int bkv, int d, int mt) {
    const size_t v_lds = (size_t)bkv * (d + 2) * sizeof(_Float16);   // K streams from global
    const size_t p_lds = (size_t)(nw * mt * 16) * (bkv + 2) * sizeof(_Float16);
    return v_lds + p_lds;
}
static constexpr size_t kPrefillV7MaxSmem = 65536;

static int g_v7_ablate = 0;

static inline void dispatchPrefillV7(
    int nw, int bkv, int mt, int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale)
{
    const int rows = nw * mt * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV7SmemBytes(nw, bkv, d, mt);
    const int ablate = g_v7_ablate;

    #define LAUNCH_V7(NW_, BKV_, D_, MT_)                                       \
        gqa_flash_prefill_v7_kernel<NW_, BKV_, D_, MT_>                         \
            <<<grid, NW_ * 32, smem, stream>>>(                                 \
                Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,     \
                scale, ablate)
    #define DISPATCH_V7_D(D_)                                                   \
        do {                                                                    \
            if (mt == 1) {                                                      \
                if      (nw == 4 && bkv == 64) LAUNCH_V7(4, 64, D_, 1);         \
                else if (nw == 4)              LAUNCH_V7(4, 32, D_, 1);         \
                else if (nw == 2 && bkv == 64) LAUNCH_V7(2, 64, D_, 1);         \
                else if (nw == 2)              LAUNCH_V7(2, 32, D_, 1);         \
                else if (bkv == 64)            LAUNCH_V7(1, 64, D_, 1);         \
                else                           LAUNCH_V7(1, 32, D_, 1);         \
            } else {                                                            \
                if      (nw == 4 && bkv == 64) LAUNCH_V7(4, 64, D_, 2);         \
                else if (nw == 4)              LAUNCH_V7(4, 32, D_, 2);         \
                else if (nw == 2 && bkv == 64) LAUNCH_V7(2, 64, D_, 2);         \
                else if (nw == 2)              LAUNCH_V7(2, 32, D_, 2);         \
                else if (bkv == 64)            LAUNCH_V7(1, 64, D_, 2);         \
                else                           LAUNCH_V7(1, 32, D_, 2);         \
            }                                                                   \
        } while (0)

    if (d == 64)       DISPATCH_V7_D(64);
    else if (d == 128) DISPATCH_V7_D(128);
    else               DISPATCH_V7_D(256);
    #undef DISPATCH_V7_D
    #undef LAUNCH_V7
}

struct PrefillV7Cfg { int nw; int bkv; int mt; };
static inline int prefillV7Candidates(int d, PrefillV7Cfg* out) {
    // MT=1 cuts register-resident Q in half (key for D=128 where MT=2 spills
    // ~358 VGPR dwords to scratch). K streams from global (cache-resident) so
    // losing the MT-fold K reuse is nearly free -- worth tuning per shape.
    static const PrefillV7Cfg kAll[] = {
        {1, 32, 1}, {1, 64, 1}, {2, 32, 1}, {2, 64, 1}, {4, 32, 1}, {4, 64, 1},
        {1, 32, 2}, {1, 64, 2}, {2, 32, 2}, {2, 64, 2}, {4, 32, 2}, {4, 64, 2}};
    int n = 0;
    for (const PrefillV7Cfg& c : kAll)
        if (prefillV7SmemBytes(c.nw, c.bkv, d, c.mt) <= kPrefillV7MaxSmem) out[n++] = c;
    return n;
}

static std::unordered_map<std::string, int> g_v7_cfg_cache;
static std::mutex g_v7_cfg_mutex;
static char g_v7_tune_info[128] = "gqa_flash_prefill_v7 cfg=?";

static int tunePrefillV7Cfg(
    int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale)
{
    PrefillV7Cfg cands[12];
    const int nc = prefillV7Candidates(d, cands);
    constexpr int kWarmup = 30, kRounds = 4, kIters = 30;

    auto launch_cfg = [&](const PrefillV7Cfg& c) {
        dispatchPrefillV7(c.nw, c.bkv, c.mt, d, stream, Q, Kcache, Vcache, O,
                          B, Hq, G, sq, skv, max_seq, past_len, scale);
    };

    for (int w = 0; w < kWarmup; ++w) launch_cfg(cands[w % nc]);
    (void)hipDeviceSynchronize();

    hipEvent_t ev0, ev1;
    (void)hipEventCreate(&ev0);
    (void)hipEventCreate(&ev1);
    float best_ms[12];
    for (int i = 0; i < nc; ++i) best_ms[i] = 1.0e30f;
    for (int r = 0; r < kRounds; ++r) {
        for (int idx = 0; idx < nc; ++idx) {
            launch_cfg(cands[idx]);
            (void)hipEventRecord(ev0, stream);
            for (int i = 0; i < kIters; ++i) launch_cfg(cands[idx]);
            (void)hipEventRecord(ev1, stream);
            (void)hipEventSynchronize(ev1);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, ev0, ev1);
            if (ms < best_ms[idx]) best_ms[idx] = ms;
        }
    }
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);

    if (getenv("HIPDNN_PREFILL_TUNE_DEBUG")) {
        for (int i = 0; i < nc; ++i)
            fprintf(stderr,
                "[prefill-v7-tune] d=%d sq=%d skv=%d  NW=%-2d BKV=%-3d MT=%d "
                "num_q_tiles=%-4d : %.5f ms/call\n",
                d, sq, skv, cands[i].nw, cands[i].bkv, cands[i].mt,
                (sq + cands[i].nw * cands[i].mt * 16 - 1) /
                    (cands[i].nw * cands[i].mt * 16),
                best_ms[i] / kIters);
    }

    int best = 0;
    for (int i = 1; i < nc; ++i) if (best_ms[i] < best_ms[best]) best = i;
    return cands[best].mt * 1000000 + cands[best].nw * 1000 + cands[best].bkv;
}

// =====================================================================
// Prefill v8: D=256-specialized "output-D split + score reduction".
//
// A block runs ND waves that share the SAME query rows but each owns
// D_TILES/ND output-D tiles. The score needs the full-D contraction, so
// each wave computes a PARTIAL score over its own D slice; the partials
// are summed across waves through LDS (the secondary reduction) to form
// the full score. Every wave then runs an identical online softmax and a
// value GEMM for its own (disjoint) output-D tiles. Halving each wave's
// Q_reg/O_h footprint is what lifts occupancy at D=256 (the bottleneck
// that capped v7); the smaller ROWS also spawns more blocks, filling the
// GPU better on short prompts.
// Grid: (num_q_tiles*B, Hq). Block: ND*32.
// =====================================================================
template <int ND, int MT, int BKV, int D>
__global__ void __launch_bounds__(ND * 32)
gqa_flash_prefill_v8_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate)
{
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = MT * kWmmaTile;
    constexpr int S_TILES_J = BKV / kWmmaTile;
    constexpr int D_TILES   = D / kWmmaTile;
    constexpr int DPG       = D_TILES / ND;    // D-tiles owned per wave
    constexpr int P_STR     = BKV + 2;
    constexpr int KV_STR    = D + 2;
    constexpr int SP_STR    = BKV;             // partial-score LDS row stride
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));

    const int tid       = threadIdx.x;
    const int group     = __builtin_amdgcn_readfirstlane(tid / 32);   // 0..ND-1
    const int lane_id   = tid % 32;
    const int wmma_lane = lane_id % kWmmaTile;
    const int pair      = lane_id / kWmmaTile;
    const int dt_base   = group * DPG;         // this wave's first D-tile
    const int q_start   = q_tile_idx * ROWS;

    extern __shared__ char smem_v8[];
    _Float16* V_lds  = reinterpret_cast<_Float16*>(smem_v8);
    float*    Sp_lds = reinterpret_cast<float*>(
        smem_v8 + (size_t)BKV * KV_STR * sizeof(_Float16));
    _Float16* P_lds  = reinterpret_cast<_Float16*>(
        Sp_lds + (size_t)ND * ROWS * SP_STR);

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: only THIS wave's D slice (DPG tiles) ----
    half16_t Q_reg[MT][DPG];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int q_pos = q_start + mt * kWmmaTile + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < DPG; ++tk)
            Q_reg[mt][tk] = valid
                ? HALF16_LOAD(&Q[q_row_base + (dt_base + tk) * 16]) : half16_t{};
    }

    half8_t O_h[MT][DPG] = {};
    float   m_reg[MT][8];
    float   l_reg[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 8; ++e) { m_reg[mt][e] = -INFINITY; l_reg[mt][e] = 0.0f; }

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // ---- cooperative V staging (all ND waves), row-major V[kv][d] ----
        if (!(abl & 8)) {
            for (int i = tid; i < BKV * D_TILES; i += ND * 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                const bool kvalid  = kv_pos < skv;
                const size_t off   = kv_base + (size_t)(kvalid ? kv_pos : 0) * D + dt * 16;
                half16_t vv = kvalid ? HALF16_LOAD(&Vcache[off]) : half16_t{};
                *reinterpret_cast<half16_t*>(&V_lds[kv_local * KV_STR + dt * 16]) = vv;
            }
        }

        // ---- partial score GEMM over THIS wave's D slice (K from global) ----
        float8_t S_reg[MT][S_TILES_J];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mt][sj] = float8_t{};
        if (!(abl & 4)) {
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) {
                const int key = kv0 + sj * 16 + wmma_lane;
                const bool kvalid = key < skv;
                const size_t k_row = kv_base + (size_t)(kvalid ? key : 0) * D;
                #pragma unroll
                for (int tk = 0; tk < DPG; ++tk) {
                    half16_t k_frag = (kvalid && !(abl & 16))
                        ? HALF16_LOAD(&Kcache[k_row + (dt_base + tk) * 16]) : half16_t{};
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        S_reg[mt][sj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            Q_reg[mt][tk], k_frag, S_reg[mt][sj]);
                }
            }
        }

        // ---- secondary reduction: publish partials, then sum across waves ----
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj)
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rl  = e * 2 + pair;
                    const int col = sj * 16 + wmma_lane;
                    Sp_lds[((size_t)group * ROWS + mt * 16 + rl) * SP_STR + col] =
                        S_reg[mt][sj][e];
                }
        __syncthreads();   // b_a: partials + V staging visible
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj)
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rl  = e * 2 + pair;
                    const int col = sj * 16 + wmma_lane;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int g = 0; g < ND; ++g)
                        acc += Sp_lds[((size_t)g * ROWS + mt * 16 + rl) * SP_STR + col];
                    S_reg[mt][sj][e] = acc * scale * kLog2e;
                }

        // ---- online softmax on full S (identical across waves) ----
        // Only group 0 stores P (all waves would write identical values);
        // b_b then publishes it to every wave for the value GEMM.
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float corr_e[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rl    = e * 2 + pair;
                    const int q_pos = q_start + mt * 16 + rl;
                    float rmax = -INFINITY;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int kv_pos = kv0 + sj * 16 + wmma_lane;
                        if (kv_pos >= skv || kv_pos > past_len + q_pos)
                            S_reg[mt][sj][e] = -INFINITY;
                        rmax = fmaxf(rmax, S_reg[mt][sj][e]);
                    }
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                    const float m_old = m_reg[mt][e];
                    const float m_new = fmaxf(m_old, rmax);
                    const float corr  = exp2f(m_old - m_new);
                    corr_e[e]         = corr;

                    float rsum = 0.0f;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const float p = exp2f(S_reg[mt][sj][e] - m_new);
                        rsum += p;
                        if (group == 0)
                            P_lds[(mt * 16 + rl) * P_STR + sj * 16 + wmma_lane] =
                                static_cast<_Float16>(p);
                    }
                    rsum += __shfl_xor(rsum, 1);
                    rsum += __shfl_xor(rsum, 2);
                    rsum += __shfl_xor(rsum, 4);
                    rsum += __shfl_xor(rsum, 8);

                    m_reg[mt][e] = m_new;
                    l_reg[mt][e] = corr * l_reg[mt][e] + rsum;
                }
                // rescale this wave's owned O tiles by the correction
                #pragma unroll
                for (int tj = 0; tj < DPG; ++tj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        O_h[mt][tj][e] =
                            static_cast<_Float16>((float)O_h[mt][tj][e] * corr_e[e]);
            }
        }
        __syncthreads();   // b_b: P_lds visible for the value GEMM

        // ---- value GEMM: this wave's owned D tiles only ----
        if (!(abl & 2))
        for (int tj = 0; tj < DPG; ++tj) {
            const int d_tile = dt_base + tj;
            float8_t Of[MT];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int e = 0; e < 8; ++e) Of[mt][e] = (float)O_h[mt][tj][e];
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                half16_t v_frag;
                #pragma unroll
                for (int e = 0; e < 16; ++e)
                    v_frag[e] = V_lds[(tk * 16 + e) * KV_STR + d_tile * 16 + wmma_lane];
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(mt * 16 + wmma_lane) * P_STR + tk * 16]);
                    Of[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p_frag, v_frag, Of[mt]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    O_h[mt][tj][e] = static_cast<_Float16>(Of[mt][e]);
        }
        __syncthreads();   // b_c: WAR -- V_lds/Sp_lds/P_lds reused next tile
    }

    // ---- Epilogue: this wave writes its owned D tiles ----
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int tj = 0; tj < DPG; ++tj) {
            const int c = (dt_base + tj) * 16 + wmma_lane;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int rl    = e * 2 + pair;
                const int q_pos = q_start + mt * 16 + rl;
                if (q_pos < sq && c < D) {
                    const float l_val   = fmaxf(l_reg[mt][e], 1e-6f);
                    const float out_val = (float)O_h[mt][tj][e] / l_val;
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
}

// ---------------------------------------------------------------------
// v8 host: LDS sizing, dispatch, autotune (ND x MT x BKV at D=256).
// ---------------------------------------------------------------------
static inline size_t prefillV8SmemBytes(int nd, int mt, int bkv, int d) {
    const int rows = mt * 16;
    const size_t v_lds  = (size_t)bkv * (d + 2) * sizeof(_Float16);
    const size_t sp_lds = (size_t)nd * rows * bkv * sizeof(float);
    const size_t p_lds  = (size_t)rows * (bkv + 2) * sizeof(_Float16);
    return v_lds + sp_lds + p_lds;
}
static constexpr size_t kPrefillV8MaxSmem = 65536;
static int g_v8_ablate = 0;

static inline void dispatchPrefillV8(
    int nd, int mt, int bkv, int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale)
{
    const int rows = mt * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV8SmemBytes(nd, mt, bkv, d);
    const int ablate = g_v8_ablate;

    #define LAUNCH_V8(ND_, MT_, BKV_)                                           \
        gqa_flash_prefill_v8_kernel<ND_, MT_, BKV_, 256>                        \
            <<<grid, ND_ * 32, smem, stream>>>(                                 \
                Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,     \
                scale, ablate)

    if      (nd == 2 && mt == 1 && bkv == 32) LAUNCH_V8(2, 1, 32);
    else if (nd == 2 && mt == 2 && bkv == 32) LAUNCH_V8(2, 2, 32);
    else if (nd == 2 && mt == 1 && bkv == 64) LAUNCH_V8(2, 1, 64);
    else if (nd == 2 && mt == 2 && bkv == 64) LAUNCH_V8(2, 2, 64);
    else if (nd == 4 && mt == 1 && bkv == 32) LAUNCH_V8(4, 1, 32);
    else if (nd == 4 && mt == 2 && bkv == 32) LAUNCH_V8(4, 2, 32);
    else                                      LAUNCH_V8(2, 1, 32);
    #undef LAUNCH_V8
}

struct PrefillV8Cfg { int nd; int mt; int bkv; };
static inline int prefillV8Candidates(int d, PrefillV8Cfg* out) {
    static const PrefillV8Cfg kAll[] = {
        {2, 1, 32}, {2, 2, 32}, {2, 1, 64}, {2, 2, 64}, {4, 1, 32}, {4, 2, 32}};
    int n = 0;
    for (const PrefillV8Cfg& c : kAll)
        if (prefillV8SmemBytes(c.nd, c.mt, c.bkv, d) <= kPrefillV8MaxSmem)
            out[n++] = c;
    return n;
}

static std::unordered_map<std::string, int> g_v8_cfg_cache;
static std::mutex g_v8_cfg_mutex;
static char g_v8_tune_info[128] = "gqa_flash_prefill_v8 cfg=?";

static int tunePrefillV8Cfg(
    int d, hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int max_seq,
    int past_len, float scale)
{
    PrefillV8Cfg cands[8];
    const int nc = prefillV8Candidates(d, cands);
    constexpr int kWarmup = 30, kRounds = 4, kIters = 30;

    auto launch_cfg = [&](const PrefillV8Cfg& c) {
        dispatchPrefillV8(c.nd, c.mt, c.bkv, d, stream, Q, Kcache, Vcache, O,
                          B, Hq, G, sq, skv, max_seq, past_len, scale);
    };

    for (int w = 0; w < kWarmup; ++w) launch_cfg(cands[w % nc]);
    (void)hipDeviceSynchronize();

    hipEvent_t ev0, ev1;
    (void)hipEventCreate(&ev0);
    (void)hipEventCreate(&ev1);
    float best_ms[8];
    for (int i = 0; i < nc; ++i) best_ms[i] = 1.0e30f;
    for (int r = 0; r < kRounds; ++r) {
        for (int idx = 0; idx < nc; ++idx) {
            launch_cfg(cands[idx]);
            (void)hipEventRecord(ev0, stream);
            for (int i = 0; i < kIters; ++i) launch_cfg(cands[idx]);
            (void)hipEventRecord(ev1, stream);
            (void)hipEventSynchronize(ev1);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, ev0, ev1);
            if (ms < best_ms[idx]) best_ms[idx] = ms;
        }
    }
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);

    if (getenv("HIPDNN_PREFILL_TUNE_DEBUG")) {
        for (int i = 0; i < nc; ++i)
            fprintf(stderr,
                "[prefill-v8-tune] d=%d sq=%d skv=%d  ND=%d MT=%d BKV=%-3d : "
                "%.5f ms/call\n",
                d, sq, skv, cands[i].nd, cands[i].mt, cands[i].bkv,
                best_ms[i] / kIters);
    }

    int best = 0;
    for (int i = 1; i < nc; ++i) if (best_ms[i] < best_ms[best]) best = i;
    return cands[best].nd * 1000000 + cands[best].mt * 1000 + cands[best].bkv;
}

// Namespace-internal entry: cache + tune + dispatch. Called by the unified
// prefill_v2 entry for D=256.
static int runPrefillV8(
    hipStream_t stream,
    const _Float16* Q, const _Float16* Kcache, const _Float16* Vcache,
    _Float16* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale)
{
    if (d != 256) return -1;
    std::string key = std::to_string(d) + "_" + std::to_string(sq) + "_" +
                      std::to_string(skv) + "_" + std::to_string(Hq) + "_" +
                      std::to_string(G);
    int cfg;
    {
        std::lock_guard<std::mutex> lk(g_v8_cfg_mutex);
        auto it = g_v8_cfg_cache.find(key);
        if (it != g_v8_cfg_cache.end()) {
            cfg = it->second;
        } else {
            cfg = tunePrefillV8Cfg(d, stream, Q, Kcache, Vcache, O,
                                   B, Hq, G, sq, skv, max_seq, past_len, scale);
            g_v8_cfg_cache[key] = cfg;
            snprintf(g_v8_tune_info, sizeof(g_v8_tune_info),
                     "gqa_flash_prefill_v8 cfg=ND%d_MT%d_BKV%d",
                     cfg / 1000000, (cfg / 1000) % 1000, cfg % 1000);
        }
    }
    const int nd = cfg / 1000000, mt = (cfg / 1000) % 1000, bkv = cfg % 1000;
    dispatchPrefillV8(nd, mt, bkv, d, stream, Q, Kcache, Vcache, O,
                      B, Hq, G, sq, skv, max_seq, past_len, scale);
    return 0;
}

}  // namespace (fused prefill v5/v7/v8)


extern "C" const char* hip_gqa_flash_prefill_v5_last_tune_info() {
    return g_v5_tune_info;
}

// Perf-diagnostics ablation (bit0 skip softmax, bit1 skip valueGEMM,
// bit2 skip scoreGEMM, bit3 skip global loads). Wrong numerics; latency only.
extern "C" void hip_gqa_flash_prefill_v5_set_ablate(int mask) {
    g_v5_ablate = mask;
}

static int prefillV5Run(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale, const void* head_sink, int num_heads, int smooth_softmax)
{
    if (d != 64 && d != 128) return -1;

    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
    const _Float16* Q_ptr = static_cast<const _Float16*>(Q);
    const _Float16* K_ptr = static_cast<const _Float16*>(Kcache);
    const _Float16* V_ptr = static_cast<const _Float16*>(Vcache);
    _Float16* O_ptr       = static_cast<_Float16*>(O);
    const _Float16* sink_ptr = static_cast<const _Float16*>(head_sink);

    // Neither skv nor the sink belongs in the key. skv only scales the KV loop
    // length, uniformly across candidates, so it cannot reorder them: measured
    // on gpt-oss-20b, M_TILES=2/BKV=32 wins at all 32 chunk skv values from 512
    // to 16384 with an identical candidate ranking. Keying on skv re-tuned once
    // per chunk of a chunked prefill instead of once per process. The sink is
    // one FMA per row in the epilogue (see tunePrefillV5Cfg).
    std::string key = std::to_string(d) + "_" + std::to_string(sq) + "_" +
                      std::to_string(Hq) + "_" + std::to_string(G);
    int cfg;
    {
        std::lock_guard<std::mutex> lk(g_v5_cfg_mutex);
        auto it = g_v5_cfg_cache.find(key);
        if (it != g_v5_cfg_cache.end()) {
            cfg = it->second;
        } else {
            cfg = tunePrefillV5Cfg(d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                                   B, Hq, G, sq, skv, max_seq, past_len, scale);
            g_v5_cfg_cache[key] = cfg;
            snprintf(g_v5_tune_info, sizeof(g_v5_tune_info),
                     "gqa_flash_prefill_v5 cfg=M%d_BKV%d", cfg / 1000, cfg % 1000);
        }
    }
    int m_tiles = cfg / 1000, bkv = cfg % 1000;
    dispatchPrefillV5(m_tiles, bkv, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                      B, Hq, G, sq, skv, max_seq, past_len, scale,
                      sink_ptr, num_heads, smooth_softmax);
    return 0;
}

extern "C" int hip_gqa_flash_prefill_v5_v2(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale)
{
    return prefillV5Run(stream_ptr, Q, Kcache, Vcache, O, B, Hq, G, sq, skv, d,
                        max_seq, past_len, scale, nullptr, Hq, 0);
}


extern "C" const char* hip_gqa_flash_prefill_v7_last_tune_info() {
    return g_v7_tune_info;
}

extern "C" void hip_gqa_flash_prefill_v7_set_ablate(int mask) {
    g_v7_ablate = mask;
}

extern "C" int hip_gqa_flash_prefill_v7_v2(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale)
{
    if (d != 64 && d != 128 && d != 256) return -1;

    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
    const _Float16* Q_ptr = static_cast<const _Float16*>(Q);
    const _Float16* K_ptr = static_cast<const _Float16*>(Kcache);
    const _Float16* V_ptr = static_cast<const _Float16*>(Vcache);
    _Float16* O_ptr       = static_cast<_Float16*>(O);

    std::string key = std::to_string(d) + "_" + std::to_string(sq) + "_" +
                      std::to_string(skv) + "_" + std::to_string(Hq) + "_" +
                      std::to_string(G);
    int cfg;
    {
        std::lock_guard<std::mutex> lk(g_v7_cfg_mutex);
        auto it = g_v7_cfg_cache.find(key);
        if (it != g_v7_cfg_cache.end()) {
            cfg = it->second;
        } else {
            cfg = tunePrefillV7Cfg(d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                                   B, Hq, G, sq, skv, max_seq, past_len, scale);
            g_v7_cfg_cache[key] = cfg;
            snprintf(g_v7_tune_info, sizeof(g_v7_tune_info),
                     "gqa_flash_prefill_v7 cfg=NW%d_BKV%d_MT%d",
                     (cfg / 1000) % 1000, cfg % 1000, cfg / 1000000);
        }
    }
    int mt = cfg / 1000000, nw = (cfg / 1000) % 1000, bkv = cfg % 1000;
    dispatchPrefillV7(nw, bkv, mt, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                      B, Hq, G, sq, skv, max_seq, past_len, scale);
    return 0;
}


extern "C" const char* hip_gqa_flash_prefill_v8_last_tune_info() {
    return g_v8_tune_info;
}

extern "C" void hip_gqa_flash_prefill_v8_set_ablate(int mask) {
    g_v8_ablate = mask;
}


// Unified fused flash-prefill entry. The kernel-selection strategy lives here
// (in the kernel TU) instead of in the runtime dispatcher: v5 wins at d==64
// (warp-private register-resident), v7 at d==128 (M-register-blocked with
// global-streamed K), v8 at d==256 (output-D split + secondary score reduction
// to lift occupancy). The runtime only has to call this one symbol for any
// eligible (fp16, causal, GQA) prefill and gets the right kernel automatically.
// Returns -1 for unsupported head dims so the caller can fall back.
// Wide entry: adds attention sinks (and reserves sliding-window) on top of v2.
// v2 stays the narrow ABI so the standalone harnesses and shim headers that
// redeclare it keep linking unchanged.
//
// Sinks are implemented for d == 64 (the v5 kernel) only. For any other d we
// return -1 rather than silently ignoring the sink, so the runtime keeps routing
// those shapes to the decomposed path, which does implement it.
extern "C" int hip_gqa_flash_prefill_v3(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax)
{
    const bool wants_sink = (head_sink != nullptr) || (smooth_softmax != 0);
    // Sliding window is not implemented in the flash prefill kernels yet.
    if (local_window_size > 0) return -1;
    if (wants_sink && d != 64) return -1;
    if (!wants_sink) {
        return hip_gqa_flash_prefill_v2(stream_ptr, Q, Kcache, Vcache, O, B, Hq,
                                        G, sq, skv, d, max_seq, past_len, scale);
    }
    return prefillV5Run(stream_ptr, Q, Kcache, Vcache, O, B, Hq, G, sq, skv, d,
                        max_seq, past_len, scale, head_sink, num_heads,
                        smooth_softmax);
}

extern "C" int hip_gqa_flash_prefill_v2(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale)
{
    if (d == 64) {
        return hip_gqa_flash_prefill_v5_v2(stream_ptr, Q, Kcache, Vcache, O,
                                        B, Hq, G, sq, skv, d, max_seq, past_len,
                                        scale);
    }
    if (d == 256) {
        // v8 splits the output-D across ND waves (halved Q_reg/O_h -> higher
        // occupancy) with a secondary score reduction; the winning D=256 path.
        // Falls back to v7 if v8 declines (it shouldn't for d=256).
        hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
        int rc = runPrefillV8(stream,
                              static_cast<const _Float16*>(Q),
                              static_cast<const _Float16*>(Kcache),
                              static_cast<const _Float16*>(Vcache),
                              static_cast<_Float16*>(O),
                              B, Hq, G, sq, skv, d, max_seq, past_len, scale);
        if (rc == 0) return 0;
        // fall through to v7
    }
    if (d == 128 || d == 256) {
        // v7 is D-generic (M-register-blocked, K streamed from global).
        return hip_gqa_flash_prefill_v7_v2(stream_ptr, Q, Kcache, Vcache, O,
                                        B, Hq, G, sq, skv, d, max_seq, past_len,
                                        scale);
    }
    return -1;
}

//===----------------------------------------------------------------------===//
// Legacy decode kernels (folded in from the retired gqa_kernel_legacy.hip).
//
// The production decode FAST path is hip_gqa_flash_decode_v2 above. These two
// retired-but-still-needed entries back gqa.cpp::gqa_forward_hipblaslt (the
// decomposed fallback) decode cases:
//   * hip_gqa_fused_decode  : one-block-per-(batch,head_q), d in {64,128,256},
//                             arbitrary HpG -- the general small-/odd-geometry
//                             decode the fallback uses when flash isn't eligible.
//   * hip_gqa_flash_decode  : FA-2 split-K (scalar|WMMA), HPG in {4,8}, with
//                             sliding-window + head-sink/smooth-softmax -- the
//                             fast windowed/sink decode the fallback uses.
// Kernels carry a legacy_ prefix so they coexist with the v2 decode kernels
// above; they reuse this TU's WAVE_SIZE / LOG2E / kWmmaTile / half16 / HALF16 /
// FLASH_DECODE_BKV / flash_decode_partial_offset / GQA_RETURN_LAUNCH_STATUS.
//===----------------------------------------------------------------------===//

// Fused GQA decode (sq == 1, d in {64,128,256}). One block per (batch,head_q),
// D threads; LDS-tiles TILE=8 KV rows to drive memory-level parallelism.
template <int D>
__global__ void __launch_bounds__(D)
legacy_fused_gqa_decode_kernel(
    const __half* __restrict__ Q_roped,
    const __half* __restrict__ Kcache,
    const __half* __restrict__ Vcache,
    __half* __restrict__ O,
    int H, int G, int d, int skv, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k)
{
    constexpr int NUM_WAVES = D / WAVE_SIZE;
    constexpr int TILE = 8;

    const int head_q  = __builtin_amdgcn_readfirstlane(blockIdx.x % H);
    const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.x / H);
    const int head_kv = __builtin_amdgcn_readfirstlane(head_q / (H / G));
    const int tid     = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;

    const int raw_skv = seqlens_k
        ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1) : skv;
    const int clamp_hi = raw_skv < max_seq ? raw_skv : max_seq;
    const int eff_skv = clamp_hi > 0 ? clamp_hi : 0;

    __shared__ float smem[NUM_WAVES];
    __shared__ _Float16 K_tile[TILE * D];
    __shared__ _Float16 V_tile[TILE * D];

    const float scale_log2e = scale * LOG2E;

    const float Q_val = __half2float(Q_roped[(batch * H + head_q) * d + tid]);

    const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
        (unsigned)((batch * G + head_kv) * max_seq)) * d;

    float m = -INFINITY;
    float l = 0.0f;
    float O_acc = 0.0f;

    for (int t_base = 0; t_base < eff_skv; t_base += TILE) {
        const int tile_n = (t_base + TILE <= eff_skv) ? TILE : (eff_skv - t_base);

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            if (i < tile_n) {
                K_tile[i * D + tid] = Kcache[kv_base + (size_t)(t_base + i) * d + tid];
                V_tile[i * D + tid] = Vcache[kv_base + (size_t)(t_base + i) * d + tid];
            }
        }

        for (int t_off = 0; t_off < tile_n; ++t_off) {
            float dot = Q_val * __half2float(K_tile[t_off * D + tid]);

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

            float V_val = __half2float(V_tile[t_off * D + tid]);
            O_acc = correction * O_acc + p * V_val;
            m = m_new;
        }
    }

    O_acc /= fmaxf(l, 1e-6f);
    O[(batch * H + head_q) * d + tid] = __float2half(O_acc);
}

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
  (void)hipGetLastError();  // clear stale error for correct attribution
  switch (d) {
    case 64:
      legacy_fused_gqa_decode_kernel<64><<<grid, 64, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    case 128:
      legacy_fused_gqa_decode_kernel<128><<<grid, 128, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    case 256:
      legacy_fused_gqa_decode_kernel<256><<<grid, 256, 0, stream>>>(
          hQ, hK, hV, hO, H, G, d, skv, max_seq, scale, iSeqlensK);
      break;
    default:
      fprintf(stderr, "hip_gqa_fused_decode: unsupported d=%d "
              "(must be 64, 128, or 256)\n", d);
      return -1;
  }
  GQA_RETURN_LAUNCH_STATUS();
}

// FA-2 split-K decode (scalar). Emits [B,H,K_SPLITS,D+2] partials; handles
// sliding-window via kv_lo clamp. HPG query heads share one KV group's LDS.
template <int D, int K_SPLITS, int HPG>
__global__ void __launch_bounds__(HPG * WAVE_SIZE)
legacy_gqa_flash_decode_kernel(
    const __half* __restrict__ Q_roped,
    const __half* __restrict__ Kcache,
    const __half* __restrict__ Vcache,
    float* __restrict__ partials,
    int H, int G, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size) {
  constexpr int BKV = FLASH_DECODE_BKV;
  constexpr int THREADS = HPG * WAVE_SIZE;
  constexpr int EPT = D / WAVE_SIZE;

  static_assert(D % WAVE_SIZE == 0, "D must be a multiple of WAVE_SIZE");
  static_assert(EPT >= 1 && EPT <= 4, "EPT must be 1..4");

  const int batch     = __builtin_amdgcn_readfirstlane(blockIdx.x / G);
  const int head_kv   = __builtin_amdgcn_readfirstlane(blockIdx.x % G);
  const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int tid       = threadIdx.x;
  const int wave_id   = tid / WAVE_SIZE;
  const int lane_id   = tid % WAVE_SIZE;
  const int head_q    = head_kv * HPG + wave_id;

  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;

  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;

  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split_idx;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  float* partial_dst = partials +
      flash_decode_partial_offset(batch, head_q, split_idx, H, K_SPLITS, D);

  if (kv_start >= kv_end) {
    if (lane_id == 0) {
      partial_dst[0] = -INFINITY;
      partial_dst[1] = 0.0f;
    }
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = 0.0f;
    return;
  }

  __shared__ __half K_lds[BKV * D];
  __shared__ __half V_lds[BKV * D];

  __half Q_reg[EPT];
  {
    const __half* Q_ptr = Q_roped + (size_t)(batch * H + head_q) * D;
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      Q_reg[e] = Q_ptr[lane_id * EPT + e];
  }

  const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
      (unsigned)((batch * G + head_kv) * max_seq)) * (size_t)D;

  const float scale_log2e = scale * LOG2E;

  float m_acc = -INFINITY;
  float l_acc = 0.0f;
  float O_acc[EPT];
  #pragma unroll
  for (int e = 0; e < EPT; ++e) O_acc[e] = 0.0f;

  for (int tile_start = kv_start; tile_start < kv_end; tile_start += BKV) {
    const int tile_n = (tile_start + BKV <= kv_end)
        ? BKV : (kv_end - tile_start);

    for (int i = tid; i < tile_n * D; i += THREADS) {
      const int t = i / D;
      const int dd = i % D;
      const size_t g_off = kv_base + (size_t)(tile_start + t) * D + dd;
      K_lds[t * D + dd] = Kcache[g_off];
      V_lds[t * D + dd] = Vcache[g_off];
    }
    __syncthreads();

    for (int t = 0; t < tile_n; ++t) {
      float dot_partial = 0.0f;
      #pragma unroll
      for (int e = 0; e < EPT; ++e) {
        dot_partial += __half2float(Q_reg[e]) *
                       __half2float(K_lds[t * D + lane_id * EPT + e]);
      }
      dot_partial += __shfl_xor(dot_partial, 16);
      dot_partial += __shfl_xor(dot_partial, 8);
      dot_partial += __shfl_xor(dot_partial, 4);
      dot_partial += __shfl_xor(dot_partial, 2);
      dot_partial += __shfl_xor(dot_partial, 1);

      const float score = dot_partial * scale_log2e;
      const float m_new = fmaxf(m_acc, score);
      const float correction = exp2f(m_acc - m_new);
      const float p = exp2f(score - m_new);
      l_acc = correction * l_acc + p;

      #pragma unroll
      for (int e = 0; e < EPT; ++e) {
        const float V_val = __half2float(V_lds[t * D + lane_id * EPT + e]);
        O_acc[e] = correction * O_acc[e] + p * V_val;
      }
      m_acc = m_new;
    }
    __syncthreads();
  }

  if (lane_id == 0) {
    partial_dst[0] = m_acc;
    partial_dst[1] = l_acc;
  }
  #pragma unroll
  for (int e = 0; e < EPT; ++e) {
    partial_dst[2 + lane_id * EPT + e] = O_acc[e];
  }
}

// FA-2 split-K decode (WMMA). Same [B,H,K_SPLITS,D+2] partials as the scalar
// kernel, so the shared reduce (window/sink) is identical. One wave/block.
template <int D, int K_SPLITS, int HPG, int BKV>
__global__ void __launch_bounds__(32)
legacy_gqa_flash_decode_wmma_kernel(
    const __half* __restrict__ Q_in,
    const __half* __restrict__ K_in,
    const __half* __restrict__ V_in,
    float* __restrict__ partials,
    int H, int G, int max_seq, float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size) {
  constexpr int DSTR    = D + 2;
  constexpr int S_STR   = BKV + 1;
  constexpr int P_STR   = BKV + 2;
  constexpr int Q_TILES = D / kWmmaTile;
  constexpr int O_TILES = D / kWmmaTile;
  constexpr int N_TILES = BKV / kWmmaTile;

  const _Float16* Q = reinterpret_cast<const _Float16*>(Q_in);
  const _Float16* K = reinterpret_cast<const _Float16*>(K_in);
  const _Float16* V = reinterpret_cast<const _Float16*>(V_in);

  const int head_kv = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int split   = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.z);

  const int lane      = threadIdx.x;
  const int wmma_lane = lane % kWmmaTile;
  const int pair      = lane / kWmmaTile;

  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;
  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;
  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  if (kv_start >= kv_end) {
    if (lane < HPG) {
      const int head_q = head_kv * HPG + lane;
      float* pd = partials +
          flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
      pd[0] = -INFINITY;
      pd[1] = 0.0f;
      #pragma unroll
      for (int c = 0; c < D; ++c) pd[2 + c] = 0.0f;
    }
    return;
  }

  extern __shared__ char smem_dw[];
  float*    S_lds = reinterpret_cast<float*>(smem_dw);
  _Float16* K_lds = reinterpret_cast<_Float16*>(smem_dw + 16 * S_STR * sizeof(float));
  _Float16* V_lds = K_lds + BKV * DSTR;
  _Float16* P_lds = V_lds + BKV * DSTR;
  float*    ml    = reinterpret_cast<float*>(P_lds + 16 * P_STR);

  const size_t kv_base = (static_cast<size_t>(batch) * G + head_kv)
                         * static_cast<size_t>(max_seq) * D;

  half16 Q_reg[Q_TILES];
  {
    const bool valid = wmma_lane < HPG;
    const int head = head_kv * HPG + wmma_lane;
    const size_t q_row = (static_cast<size_t>(batch) * H + head) * D;
    #pragma unroll
    for (int tk = 0; tk < Q_TILES; ++tk)
      Q_reg[tk] = valid ? HALF16(Q[q_row + tk * 16]) : half16{};
  }

  if (lane < 16) {
    ml[lane * 3]     = -INFINITY;
    ml[lane * 3 + 1] = 0.0f;
    ml[lane * 3 + 2] = 1.0f;
  }

  float8 O_frag[O_TILES] = {};

  const float scale_log2e = scale * LOG2E;
  const int num_tiles = (kv_end - kv_start + BKV - 1) / BKV;

  constexpr int kPET = (BKV * D) / 32;
  constexpr bool kPrefetch = (kPET <= 32);

  _Float16 k_pf[kPrefetch ? kPET : 1];
  _Float16 v_pf[kPrefetch ? kPET : 1];
  if (kPrefetch) {
    #pragma unroll
    for (int j = 0; j < kPET; ++j) {
      const int i = lane + j * 32;
      const int row = i / D, col = i % D;
      const int kv_pos = kv_start + row;
      const bool ok = kv_pos < kv_end;
      k_pf[j] = ok ? K[kv_base + static_cast<size_t>(kv_pos) * D + col] : (_Float16)0;
      v_pf[j] = ok ? V[kv_base + static_cast<size_t>(kv_pos) * D + col] : (_Float16)0;
    }
  }

  for (int t = 0; t < num_tiles; ++t) {
    const int kv0 = kv_start + t * BKV;

    if (kPrefetch) {
      #pragma unroll
      for (int j = 0; j < kPET; ++j) {
        const int i = lane + j * 32;
        const int row = i / D, col = i % D;
        K_lds[row * DSTR + col] = k_pf[j];
        V_lds[row * DSTR + col] = v_pf[j];
      }
      __syncthreads();
      const int kv0n = kv_start + (t + 1) * BKV;
      if (t + 1 < num_tiles) {
        #pragma unroll
        for (int j = 0; j < kPET; ++j) {
          const int i = lane + j * 32;
          const int row = i / D, col = i % D;
          const int kv_pos = kv0n + row;
          const bool ok = kv_pos < kv_end;
          k_pf[j] = ok ? K[kv_base + static_cast<size_t>(kv_pos) * D + col] : (_Float16)0;
          v_pf[j] = ok ? V[kv_base + static_cast<size_t>(kv_pos) * D + col] : (_Float16)0;
        }
      }
    } else {
      for (int i = lane; i < BKV * D; i += 32) {
        const int row = i / D, col = i % D;
        const int kv_pos = kv0 + row;
        if (kv_pos < kv_end) {
          K_lds[row * DSTR + col] = K[kv_base + static_cast<size_t>(kv_pos) * D + col];
          V_lds[row * DSTR + col] = V[kv_base + static_cast<size_t>(kv_pos) * D + col];
        } else {
          K_lds[row * DSTR + col] = (_Float16)0;
          V_lds[row * DSTR + col] = (_Float16)0;
        }
      }
      __syncthreads();
    }

    #pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      float8 s_frag = {};
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk) {
        half16 kf = HALF16(K_lds[(sj * 16 + wmma_lane) * DSTR + tk * 16]);
        s_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Q_reg[tk], kf, s_frag);
      }
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele) {
        const int r = ele * 2 + pair;
        const int c = sj * 16 + wmma_lane;
        S_lds[r * S_STR + c] = s_frag[ele] * scale_log2e;
      }
    }
    __syncthreads();

    {
      const int row  = lane / 2;
      const int half = lane & 1;
      const int col_start = half * (BKV / 2);

      float local_max = -INFINITY;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const int kv_pos = kv0 + col_start + c;
        float s = (kv_pos < kv_end) ? S_lds[row * S_STR + col_start + c]
                                    : -INFINITY;
        S_lds[row * S_STR + col_start + c] = s;
        local_max = fmaxf(local_max, s);
      }
      local_max = fmaxf(local_max, __shfl_xor(local_max, 1));

      const float m_old = ml[row * 3];
      const float l_old = ml[row * 3 + 1];
      const float m_new = fmaxf(m_old, local_max);
      const float correction = exp2f(m_old - m_new);

      float local_sum = 0.0f;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const float p = exp2f(S_lds[row * S_STR + col_start + c] - m_new);
        local_sum += p;
        P_lds[row * P_STR + col_start + c] = static_cast<_Float16>(p);
      }
      local_sum += __shfl_xor(local_sum, 1);

      if (half == 0) {
        ml[row * 3]     = m_new;
        ml[row * 3 + 1] = correction * l_old + local_sum;
        ml[row * 3 + 2] = correction;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj)
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele)
        O_frag[tj][ele] *= ml[(ele * 2 + pair) * 3 + 2];

    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj) {
      #pragma unroll
      for (int tk = 0; tk < N_TILES; ++tk) {
        half16 p = HALF16(P_lds[wmma_lane * P_STR + tk * 16]);
        half16 v;
        #pragma unroll
        for (int e = 0; e < 16; ++e)
          v[e] = V_lds[(tk * 16 + e) * DSTR + tj * 16 + wmma_lane];
        O_frag[tj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p, v, O_frag[tj]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int tj = 0; tj < O_TILES; ++tj)
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
      const int r = ele * 2 + pair;
      const int c = tj * 16 + wmma_lane;
      if (r < HPG) {
        const int head_q = head_kv * HPG + r;
        float* pd = partials +
            flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
        pd[2 + c] = O_frag[tj][ele];
      }
    }
  if (lane < HPG) {
    const int head_q = head_kv * HPG + lane;
    float* pd = partials +
        flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
    pd[0] = ml[lane * 3];
    pd[1] = ml[lane * 3 + 1];
  }
}

// FA-2 split-K reduction: merge K_SPLITS partials, fold head-sink/smooth into
// the denominator. One block per (batch,head_q), D threads.
template <int D, int K_SPLITS>
__global__ void __launch_bounds__(D)
legacy_gqa_flash_decode_reduce_kernel(
    const float* __restrict__ partials,
    __half* __restrict__ O,
    int H,
    const __half* __restrict__ head_sink,
    int num_heads,
    int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int batch  = __builtin_amdgcn_readfirstlane(linear / H);
  const int head_q = __builtin_amdgcn_readfirstlane(linear % H);
  const int tid    = threadIdx.x;

  float global_m = -INFINITY;
  #pragma unroll
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    global_m = fmaxf(global_m, m_k);
  }

  // All splits empty (eff_skv==0): softmax over sink only -> zero V output.
  if (!isfinite(global_m)) {
    O[(batch * H + head_q) * D + tid] = __float2half(0.0f);
    return;
  }

  float l_sum = 0.0f;
  float O_sum = 0.0f;
  #pragma unroll
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    const float l_k = p[1];
    const float scale_k = exp2f(m_k - global_m);
    l_sum += scale_k * l_k;
    O_sum += scale_k * p[2 + tid];
  }

  // Smooth softmax / sink: add exp2(s*LOG2E - global_m) to the denominator only.
  if (head_sink != nullptr) {
    const int actual_head = head_q % num_heads;
    const float s = __half2float(head_sink[actual_head]);
    l_sum += exp2f(s * (float)LOG2E - global_m);
  } else if (use_smooth_softmax) {
    l_sum += exp2f(-global_m);
  }

  const float inv_l = 1.0f / fmaxf(l_sum, 1e-6f);
  O[(batch * H + head_q) * D + tid] = __float2half(O_sum * inv_l);
}

extern "C" int hip_gqa_flash_decode(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int max_seq, int K_SPLITS,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax) {
  if (G <= 0 || H % G != 0) {
    fprintf(stderr, "hip_gqa_flash_decode: H=%d not divisible by G=%d\n", H, G);
    return -1;
  }
  const int hpg = H / G;
  if (hpg != 4 && hpg != 8) {
    fprintf(stderr,
            "hip_gqa_flash_decode: unsupported HPG=%d (must be 4 or 8; H=%d, G=%d)\n",
            hpg, H, G);
    return -1;
  }
  if (d != 64 && d != 128) {
    fprintf(stderr,
            "hip_gqa_flash_decode: unsupported d=%d (must be 64 or 128)\n", d);
    return -1;
  }
  if (hpg == 8 && d != 64) {
    fprintf(stderr,
            "hip_gqa_flash_decode: HPG=8 currently requires d=64 (got d=%d)\n",
            d);
    return -1;
  }
  if (K_SPLITS != 8) {
    fprintf(stderr,
            "hip_gqa_flash_decode: only K_SPLITS=8 supported (got %d)\n",
            K_SPLITS);
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  const __half* hQ = static_cast<const __half*>(Q);
  const __half* hK = static_cast<const __half*>(Kcache);
  const __half* hV = static_cast<const __half*>(Vcache);
  __half* hO = static_cast<__half*>(O);
  float* hP = static_cast<float*>(partials_workspace);
  const int* iSeq = static_cast<const int*>(seqlens_k);

  bool use_wmma = (d == 64);
  if (const char* e = getenv("HIPDNN_GQA_DECODE_SCALAR")) { if (atoi(e) != 0) use_wmma = false; }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_WMMA"))   { if (atoi(e) != 0) use_wmma = true; }

  if (use_wmma) {
    auto smem_bytes = [](int D_, int bkv) -> size_t {
      const int DSTR = D_ + 2;
      return (size_t)16 * (bkv + 1) * sizeof(float)
           + (size_t)2 * bkv * DSTR * sizeof(_Float16)
           + (size_t)16 * (bkv + 2) * sizeof(_Float16)
           + (size_t)16 * 3 * sizeof(float);
    };
    dim3 grid_wmma((unsigned)G, (unsigned)K_SPLITS, (unsigned)B);
    #define LAUNCH_LEGACY_DW(D_, HPG_, BKV_)                                   \
      legacy_gqa_flash_decode_wmma_kernel<D_, 8, HPG_, BKV_>                   \
          <<<grid_wmma, 32, smem_bytes(D_, BKV_), stream>>>(                   \
              hQ, hK, hV, hP, H, G, max_seq, scale, iSeq, local_window_size)
    if (hpg == 4) {
      if (d == 128) LAUNCH_LEGACY_DW(128, 4, 32); else LAUNCH_LEGACY_DW(64, 4, 16);
    } else { /* hpg == 8, d == 64 */
      LAUNCH_LEGACY_DW(64, 8, 16);
    }
    #undef LAUNCH_LEGACY_DW
  } else {
    dim3 grid_main(B * G, K_SPLITS);
    if (hpg == 4) {
      if (d == 128) {
        legacy_gqa_flash_decode_kernel<128, 8, 4>
            <<<grid_main, 4 * WAVE_SIZE, 0, stream>>>(
                hQ, hK, hV, hP, H, G, max_seq, scale, iSeq, local_window_size);
      } else { /* d == 64 */
        legacy_gqa_flash_decode_kernel<64, 8, 4>
            <<<grid_main, 4 * WAVE_SIZE, 0, stream>>>(
                hQ, hK, hV, hP, H, G, max_seq, scale, iSeq, local_window_size);
      }
    } else { /* hpg == 8, d == 64 */
      legacy_gqa_flash_decode_kernel<64, 8, 8>
          <<<grid_main, 8 * WAVE_SIZE, 0, stream>>>(
              hQ, hK, hV, hP, H, G, max_seq, scale, iSeq, local_window_size);
    }
  }

  dim3 grid_reduce(B * H);
  const __half* hSink = static_cast<const __half*>(head_sink);
  if (d == 128) {
    legacy_gqa_flash_decode_reduce_kernel<128, 8>
        <<<grid_reduce, 128, 0, stream>>>(hP, hO, H, hSink, H,
                                          use_smooth_softmax);
  } else { /* d == 64 */
    legacy_gqa_flash_decode_reduce_kernel<64, 8>
        <<<grid_reduce, 64, 0, stream>>>(hP, hO, H, hSink, H,
                                         use_smooth_softmax);
  }

  return hipGetLastError();
}
