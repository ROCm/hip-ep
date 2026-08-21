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
#include "hip_arch_compat.h"
#include "debug_log.h"

static constexpr int GQA_THREADS = 256;
// Wavefront size: 32 on RDNA (wave32), 64 on CDNA (wave64, e.g. MI350). Keyed
// off the device-pass arch macros via hip_arch_compat.h so the shuffle-based
// reductions and per-wave shared-memory fan-in below span the whole hardware
// wave on both families.
static constexpr int WAVE_SIZE = HIPDNN_WAVE_SIZE;
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

// Widest copy unit for the KV data-movement kernels. Both append and concat are
// pure relocation, so the element type only matters for indexing -- the payload
// can move as opaque 16 B chunks. One fp16 per lane leaves the copy issue-rate
// limited well short of the memory roofline (measured ~99 GB/s of 256 GB/s at
// 12K context); eight per lane is what closes that gap.
struct __align__(16) KvVec16 {
  uint32_t x, y, z, w;
};

// The vectorised path indexes in KvVec16 units, which is only valid when the
// head dim divides evenly and every buffer is 16 B aligned. Element offsets are
// then automatically aligned too: each is (multiple of d) + (multiple of VEC),
// and d % VEC == 0. Callers fall back to the scalar kernel when this fails.
static inline bool kv_is_16b_aligned(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

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

// 16 B-per-lane form of kv_cache_append_kernel. VEC elements of T are moved as
// one KvVec16, so the thread count drops by VEC and each lane issues a single
// dwordx4. Indexing is identical, just in vector units along d.
template <typename T, int VEC>
__global__ void kv_cache_append_vec_kernel(
    const T* __restrict__ src,
    T* __restrict__ cache,
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dv = d / VEC;
  int total_per_batch = sq * G * dv;
  if (idx >= total_per_batch) return;

  int dvi = idx % dv;
  int remaining = idx / dv;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + dvi * VEC;
  int dst_idx =
      ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + dvi * VEC;
  *reinterpret_cast<KvVec16*>(cache + dst_idx) =
      *reinterpret_cast<const KvVec16*>(src + src_idx);
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

// 16 B-per-lane form of kv_cache_concat_kernel. A VEC-wide chunk never straddles
// the past/new boundary (the split is along s, the chunk runs along d), so the
// source branch stays uniform per lane exactly as in the scalar kernel.
template <typename T, int VEC>
__global__ void kv_cache_concat_vec_kernel(
    const T* __restrict__ past,
    const T* __restrict__ current,
    T* __restrict__ present,
    int past_len, int sq, int G, int d,
    int past_seq, int present_seq) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dv = d / VEC;
  int total_seq = past_len + sq;
  int total_per_batch = total_seq * G * dv;
  if (idx >= total_per_batch) return;

  int dvi = idx % dv;
  int remaining = idx / dv;
  int g = remaining % G;
  int s = remaining / G;

  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + dvi * VEC;

  if (s < past_len) {
    int src_idx = ((batch_idx * G + g) * past_seq + s) * d + dvi * VEC;
    *reinterpret_cast<KvVec16*>(present + dst_idx) =
        *reinterpret_cast<const KvVec16*>(past + src_idx);
  } else {
    int cs = s - past_len;
    int src_idx = ((batch_idx * sq + cs) * G + g) * d + dvi * VEC;
    *reinterpret_cast<KvVec16*>(present + dst_idx) =
        *reinterpret_cast<const KvVec16*>(current + src_idx);
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

// Warp-level reduction helpers (wave-size portable: 32 on RDNA, 64 on CDNA)

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor(val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
    val += __shfl_xor(val, offset);
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
  constexpr int VEC = 16 / sizeof(T);
  if ((d % VEC) == 0 && kv_is_16b_aligned(src) && kv_is_16b_aligned(cache)) {
    int total = sq * G * (d / VEC);
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    kv_cache_append_vec_kernel<T, VEC><<<dim3(blocks, batch_size), GQA_THREADS,
                                         0, stream>>>(
        static_cast<const T*>(src), static_cast<T*>(cache),
        sq, G, d, present_seq, past_len, static_cast<const int*>(seqlens_k));
    return;
  }
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
  constexpr int VEC = 16 / sizeof(T);
  if ((d % VEC) == 0 && kv_is_16b_aligned(past) && kv_is_16b_aligned(current) &&
      kv_is_16b_aligned(present)) {
    int total = total_seq * G * (d / VEC);
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    kv_cache_concat_vec_kernel<T, VEC><<<dim3(blocks, batch_size), GQA_THREADS,
                                         0, stream>>>(
        static_cast<const T*>(past), static_cast<const T*>(current),
        static_cast<T*>(present), past_len, sq, G, d, past_seq, present_seq);
    return;
  }
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
  // The block must be a whole multiple of the hardware wave, else the tail of a
  // partial wave is inactive and the __shfl_xor reduction folds in its lanes.
  // The floor is therefore the device wave size: 32 on wave32 (RDNA, unchanged
  // from the original geometry) and 64 on wave64 (CDNA/MI350). num_waves must
  // use the same value so the LDS fan-in buffer matches the kernel's
  // blockDim.x / WAVE_SIZE.
  const int wave = hipdnn_device_wave_size();
  int threads = wave;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  int num_waves = threads / wave;
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
  // Block floor is the device wave size: see launch_softmax_f32_to_out.
  const int wave = hipdnn_device_wave_size();
  int threads = wave;
  while (threads < std::min(rows, 256)) threads *= 2;
  if (threads > 256) threads = 256;
  int num_waves = threads / wave;
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

// 16-byte opaque vector, used only to widen KV cache global loads to a single
// dwordx4 per lane. The element type is irrelevant; the bytes are memcpy'd
// into the typed prefetch registers.
typedef unsigned int bytes16 __attribute__((ext_vector_type(4)));

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
// single path -- hip_gqa_flash_decode (FA-2 split-K, scalar|WMMA selected by
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

// Element type of the KV cache for a given format. Lets the launchers below be
// written once against the format rather than once per element type.
template <KvDtype FMT>
using KvElem = std::conditional_t<FMT == KvDtype::kI8, int8_t, __half>;

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
  // Size the K+V LDS tile for block residency, since this kernel is bandwidth
  // bound and the split-K grid always supplies more blocks than fit. 2*BKV*D*2
  // bytes: D=128 -> BKV=32 (16KB), D=64 -> BKV=64 (16KB). At BKV=64,D=128 the
  // tile would be 32KB -> only 2 resident blocks -> latency-bound (decode
  // stalls ~1.6x).
  //
  // D>=256 uses 8KB (BKV=8) rather than the 16KB the other widths get, which
  // doubles resident blocks per WGP. Measured on gfx1151 with
  // test_gqa_decode --iters 500, vs BKV=16, taking the min of 5-7 runs
  // (the absolute numbers drift a few % with clocks, so only compare within a
  // run):
  //   Qwen3.5-9B   H16 G4 d256 skv3860   0.0440 -> 0.0409 ms  (7-9% faster)
  //   Qwen3.6-35B  H16 G2 d256 skv18646  0.1772 -> 0.1745 ms  (neutral..+3%)
  // Going further the wrong way in either direction is worse: BKV=32 costs 37%
  // on the 9B (halved residency), and BKV=4 costs 6% on the 35B (the tile stops
  // amortizing the per-tile __syncthreads and staging setup).
  constexpr int BKV = (D >= 256) ? 8 : (D == 128) ? 32 : FLASH_DECODE_BKV;
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
  // The __half2-vectorized staging below assumes an even EPT. Flash-decode is a
  // wave32 (RDNA) path only -- wrap_group_query_attention clears fused_supported
  // on wave64 (hipdnn_device_is_wave64()), so this invariant only needs to hold
  // there. On wave64 (CDNA/MI350) D=64 yields EPT=1; the kernel is then dead
  // code that must still compile, so exempt the wave64 device pass.
  static_assert(EPT % 2 == 0 || WAVE_SIZE != 32,
                "EPT must be even for __half2 vectorized loads (wave32)");
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
      // Warp reduce across the wave -> all lanes hold the full dot.
#pragma unroll
      for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
        dot_partial += __shfl_xor(dot_partial, offset);

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

  const size_t kv_base = (static_cast<size_t>(batch) * G + head_kv)
                         * static_cast<size_t>(max_seq) * D;

  // Q fragments: lane wmma_lane holds row wmma_lane (head_kv*HPG + wmma_lane).
  // For INT8 the per-channel K scale is folded in here, exactly as the scalar
  // kernel does: dot = sum_e (Q[e]*Ksc[e]) * K_i8[e]. Q is 16xD per block and
  // read once, whereas scaling K would cost a multiply per staged element on
  // every tile. Fragment element e is D-index tk*16+e, which is the channel the
  // scale is indexed by.
  half16 Q_reg[Q_TILES];
  {
    const bool valid = wmma_lane < HPG;
    const int head = head_kv * HPG + wmma_lane;
    const size_t q_row = (static_cast<size_t>(batch) * H + head) * D;
    if constexpr (KV_I8) {
      const float* ks = k_scale + (size_t)head_kv * D;
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk) {
        half16 q{};
        if (valid) {
          #pragma unroll
          for (int e = 0; e < 16; ++e)
            q[e] = (_Float16)((float)Q[q_row + tk * 16 + e] * ks[tk * 16 + e]);
        }
        Q_reg[tk] = q;
      }
    } else {
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk)
        Q_reg[tk] = valid ? HALF16(Q[q_row + tk * 16]) : half16{};
    }
  }

  if (lane < 16) {
    ml[lane * 3]     = -INFINITY;
    ml[lane * 3 + 1] = 0.0f;
    ml[lane * 3 + 2] = 1.0f;
  }

  float8 O_frag[O_TILES] = {};

  const float scale_log2e = scale * LOG2E;
  const int num_tiles = (kv_end - kv_start + BKV - 1) / BKV;

  // Software pipeline: prefetch the next tile's K/V global loads into registers
  // while the current tile's GEMMs run. For INT8 the prefetch registers stay
  // int8 (KVt) and widen to fp16 only on the LDS write, so the global read
  // stays at 1 byte per element.
  constexpr int kPET = (BKV * D) / 32;          // elements per lane per tile
  // Budget the prefetch by the register footprint it costs, not by element
  // count: K and V together hold 2*kPFBytes per lane, so 128 B caps them at
  // 64 VGPRs. Element count alone would reject D=64/BKV=32, which fits.
  constexpr int kPFBytes = kPET * (int)sizeof(KVt);
  constexpr bool kPrefetch = (kPFBytes <= 128);

  // A KV row is D*sizeof(KVt) bytes; for D=64 fp16 that is exactly one 128 B
  // gfx1151 cache line. Loading one element per lane only reaches 64 B across
  // the wave, so every line is requested in two halves and RGP shows twice the
  // DRAM traffic the kernel actually needs. Loading 16 B per lane makes one
  // instruction cover four whole lines. Safe as long as the cache base is 16 B
  // aligned (hipMalloc guarantees 256 B) since every lane offset here is a
  // multiple of kVecElems.
  constexpr int kVecElems = KV_I8 ? 16 : 8;
  constexpr bool kVecLoad =
      kPrefetch && (D % kVecElems == 0) && (kPET % kVecElems == 0);
  constexpr int kSlotElems = kVecLoad ? kVecElems : 1;
  // Zero when there is no prefetch, so the staging loops below collapse away
  // instead of indexing the placeholder one-element register arrays.
  constexpr int kSlots     = kPrefetch ? (kPET / kSlotElems) : 0;

  KVt k_pf[kPrefetch ? kPET : 1];
  KVt v_pf[kPrefetch ? kPET : 1];

  // First element this lane owns in slot s. kSlotElems divides D, so a slot
  // never straddles two KV rows and the bounds check stays per row.
  auto slot_first = [&](int s) { return (lane + s * 32) * kSlotElems; };

  auto prefetch_tile = [&](int tile_kv0) {
    #pragma unroll
    for (int s = 0; s < kSlots; ++s) {
      const int i = slot_first(s);
      const int row = i / D, col = i % D;
      const int kv_pos = tile_kv0 + row;
      const bool ok = kv_pos < kv_end;
      const size_t off = kv_base + static_cast<size_t>(kv_pos) * D + col;
      if constexpr (kVecLoad) {
        bytes16 kv{}, vv{};
        if (ok) {
          kv = *reinterpret_cast<const bytes16*>(Ksrc + off);
          vv = *reinterpret_cast<const bytes16*>(Vsrc + off);
        }
        __builtin_memcpy(&k_pf[s * kSlotElems], &kv, sizeof(bytes16));
        __builtin_memcpy(&v_pf[s * kSlotElems], &vv, sizeof(bytes16));
      } else {
        k_pf[s] = ok ? Ksrc[off] : (KVt)0;
        v_pf[s] = ok ? Vsrc[off] : (KVt)0;
      }
    }
  };

  if constexpr (kPrefetch) prefetch_tile(kv_start);

  for (int t = 0; t < num_tiles; ++t) {
    const int kv0 = kv_start + t * BKV;

    if constexpr (kPrefetch) {
      // Write the prefetched tile to LDS (int8 widens to fp16 here; its scales
      // are applied outside the KV loop).
      #pragma unroll
      for (int s = 0; s < kSlots; ++s) {
        const int i0 = slot_first(s);
        const int row = i0 / D, col0 = i0 % D;
        #pragma unroll
        for (int e = 0; e < kSlotElems; ++e) {
          const int col = col0 + e;
          const KVt kval = k_pf[s * kSlotElems + e];
          const KVt vval = v_pf[s * kSlotElems + e];
          if constexpr (KV_I8) {
            // Raw int8 magnitudes; both scales are applied outside the KV loop.
            K_lds[row * DSTR + col] = (_Float16)(float)kval;
            V_lds[row * DSTR + col] = (_Float16)(float)vval;
          } else {
            K_lds[row * DSTR + col] = kval;
            V_lds[row * DSTR + col] = vval;
          }
        }
      }
      __syncthreads();
      if (t + 1 < num_tiles) prefetch_tile(kv_start + (t + 1) * BKV);
    } else {
      for (int i = lane; i < BKV * D; i += 32) {
        const int row = i / D, col = i % D;
        const int kv_pos = kv0 + row;
        if (kv_pos < kv_end) {
          const size_t off = kv_base + static_cast<size_t>(kv_pos) * D + col;
          if constexpr (KV_I8) {
            K_lds[row * DSTR + col] = (_Float16)(float)Ksrc[off];
            V_lds[row * DSTR + col] = (_Float16)(float)Vsrc[off];
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
  // INT8 applies the deferred per-channel V scale here. O accumulates in fp32
  // over the KV range for a fixed channel, so one multiply at the end is both
  // cheaper and more accurate than scaling every staged V element.
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
        if constexpr (KV_I8) {
          pd[2 + c] = O_frag[tj][ele] * v_scale[(size_t)head_kv * D + c];
        } else {
          pd[2 + c] = O_frag[tj][ele];
        }
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

// bkv is the WMMA KV tile height and only applies to d==64, where both 16 and
// 32 are instantiated. A taller tile halves the barrier count per KV element
// but wastes more of the last tile and costs more registers, so which one wins
// depends on how many tiles a split actually covers; it is autotuned. d==128
// has only the BKV=32 instantiation and ignores this.
struct FlashDecodeCfg { bool use_wmma; int splits; int bkv = 16; };

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

// Launch one (impl, splits, bkv) configuration: split-K decode + FA-2 reduce.
// Both impls emit identical [B,H,splits,D+2] partials, so the reduce is shared.
//
// Templated on the KV format because the decode kernels already are, and the
// two formats pick their kernel by the same (d, hpg, bkv) rules. Writing those
// ladders once is what keeps a newly supported geometry from being reachable in
// fp16 but not INT8; they used to be two copies and the INT8 one had already
// drifted. kSc/vSc are ignored when FMT is kF16 -- pass nullptr.
template <KvDtype FMT>
static inline void launchFlashDecodeConfig(
    const FlashDecodeCfg& cfg, hipStream_t stream, const __half* hQ,
    const KvElem<FMT>* hK, const KvElem<FMT>* hV,
    const float* kSc, const float* vSc, __half* hO, float* hP,
    int B, int H, int G, int d, int hpg, int max_seq, float scale,
    const int* iSeq, int local_window_size,
    const __half* hSink, int use_smooth_softmax) {
  const int splits = cfg.splits;
  const bool use_wmma = cfg.use_wmma && flash_decode_wmma_supported(d, hpg);
  if (use_wmma) {
    // BKV: D=64 has both 16 and 32 instantiated and cfg.bkv picks between them;
    // D=128 has only 32. Grid (G, splits, B), one wave per block.
    auto smem_bytes = [](int D_, int bkv) -> size_t {
      const int DSTR = D_ + 2;
      return (size_t)16 * (bkv + 1) * sizeof(float)
           + (size_t)2 * bkv * DSTR * sizeof(_Float16)
           + (size_t)16 * (bkv + 2) * sizeof(_Float16)
           + (size_t)16 * 3 * sizeof(float);
    };
    dim3 grid_wmma((unsigned)G, (unsigned)splits, (unsigned)B);
    #define LAUNCH_DW(D_, HPG_, BKV_)                                          \
      gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, FMT>                        \
          <<<grid_wmma, 32, smem_bytes(D_, BKV_), stream>>>(                   \
              hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,            \
              local_window_size, splits)
    if (hpg == 4) {
      if (d == 128) LAUNCH_DW(128, 4, 32);
      else if (cfg.bkv == 32) LAUNCH_DW(64, 4, 32);
      else LAUNCH_DW(64, 4, 16);
    } else { /* hpg == 8, d == 64 */
      if (cfg.bkv == 32) LAUNCH_DW(64, 8, 32); else LAUNCH_DW(64, 8, 16);
    }
    #undef LAUNCH_DW
  } else {
    dim3 grid_main(B * G, splits);
    // Scalar decode: one wave (32 lanes) per query head, HpG waves per block.
    // The kernel body is HpG-generic (each wave owns one query head and reads
    // the KV cache with the max_seq stride), so a single template covers GQA
    // and MHA (HpG==1) alike; we just instantiate every supported HpG.
    #define LAUNCH_SCALAR_D(D_, HPG_)                                          \
      gqa_flash_decode_kernel<D_, HPG_, FMT>                                   \
          <<<grid_main, (HPG_) * WAVE_SIZE, 0, stream>>>(                      \
              hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,            \
              local_window_size, splits)
    #define LAUNCH_SCALAR(HPG_)                                                \
      do {                                                                     \
        if (d == 256)      LAUNCH_SCALAR_D(256, HPG_);                         \
        else if (d == 128) LAUNCH_SCALAR_D(128, HPG_);                         \
        else               LAUNCH_SCALAR_D(64, HPG_);                          \
      } while (0)
    switch (hpg) {
      case 1:  LAUNCH_SCALAR(1);  break;
      case 2:  LAUNCH_SCALAR(2);  break;
      case 3:  LAUNCH_SCALAR(3);  break;
      case 4:  LAUNCH_SCALAR(4);  break;
      case 5:  LAUNCH_SCALAR(5);  break;
      case 8:  LAUNCH_SCALAR(8);  break;
      case 16: LAUNCH_SCALAR(16); break;
      default: break;  // unsupported HpG is rejected in hip_gqa_flash_decode
    }
    #undef LAUNCH_SCALAR
    #undef LAUNCH_SCALAR_D
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
  // skv=8192: splits=2 -> 1.13ms, splits=4 -> 0.59ms, vs splits>=8 ->
  // <=0.36ms). Guard it structurally: never offer a split whose per-block KV
  // scan exceeds kMaxKeysPerSplit.
  //
  // The bound is 1024, not the 256 it started at: at 256 the floor was
  // min_splits=17 at skv=4096 and 33 at skv=8192, which excluded splits=16
  // outright, and 16 measured at least as fast as every wider split there for
  // both impls (the spread across 16/32/48/64 was single-digit percent, i.e.
  // inside this part's run-to-run drift -- what matters is that 16 was not
  // worse, so there is no reason to forbid it). The tight bound existed because
  // a 10-iteration tune could noisily mis-pick a catastrophic small split;
  // timeDecodeCandidate now budgets by duration and keeps the fastest of
  // several bursts, so the tuner can be trusted with the wider set. 1024 still
  // fences off the cliff (it leaves min_splits=8 at skv=8192, where the worst
  // survivor is ~1.4x off the best rather than 4.5x).
  static const int kBase[7] = {2, 4, 8, 16, 32, 48, 64};
  constexpr int kMaxKeysPerSplit = 1024;
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

// Waves the split kernel should aim to launch, derived from the device's actual
// wave-slot capacity rather than an assumed SIMD count -- multiProcessorCount
// reports WGPs on RDNA (a 40-CU 8060S reports 20), so counting SIMDs from it
// would be off by 2x. maxThreadsPerMultiProcessor/WAVE_SIZE is the wave slots
// per unit and needs no such assumption: 2048/32 = 64 slots, i.e. 4 SIMD32 with
// 16 waves each. A quarter of the machine's slots is ~4 waves/SIMD, enough to
// hide DRAM latency without inflating the partials workspace.
static int flashDecodeTargetWaves() {
  static const int target = [] {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess ||
        hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        prop.multiProcessorCount <= 0 || prop.maxThreadsPerMultiProcessor <= 0)
      return 128;
    const int slots =
        prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / WAVE_SIZE);
    return slots >= 4 ? slots / 4 : 1;
  }();
  return target;
}

// Fallback split count (used when autotune is disabled or overridden).
// Decode is bandwidth-bound -- d=64 GQA reads 2*G*skv*d*2 bytes to do
// 4*H*skv*d flops, i.e. 8 flop/byte against a machine balance two orders up --
// so splits buy nothing but memory-level parallelism, and every extra split
// adds partials traffic to the reduce. Take the smallest candidate that fills
// the device, or the widest one when the KV range is too short to fill it at
// all (a 128-key sliding window cannot, and should not try).
//
// This has to be impl-aware: the scalar kernel launches HpG waves per block and
// WMMA exactly one, so a fixed count under-fills the WMMA path by HpG-x. The
// old constant 8 is G*8 = 64 waves for gpt-oss-20b, which covers all 32 SIMDs
// of an 8-WGP part twice over but cannot even cover the 80 SIMDs of a 20-WGP
// 8060S once -- a fifth of them would get no work at all.
//
// Candidates come from flashDecodeCandidateSplits so the pick inherits its
// min_splits floor; occupancy alone would happily choose splits=1 for the
// scalar kernel at skv=8192 (already 64 waves), which is the 4.5x serial-scan
// cliff that floor exists to fence off.
static int flashDecodeDefaultSplits(bool use_wmma, int B, int H, int G, int skv,
                                    int cap) {
  int split_set[7];
  int n = 0;
  flashDecodeCandidateSplits(skv, cap, split_set, n);
  const int waves_per_split = use_wmma ? (B * G) : (B * H);
  const int target = flashDecodeTargetWaves();
  for (int i = 0; i < n; ++i)
    if (waves_per_split * split_set[i] >= target) return split_set[i];
  return split_set[n - 1];  // ascending; nothing reaches the target
}

// A taller WMMA tile pays for itself by halving the barriers per KV element,
// but it only amortises once a split spans many tiles, and it costs registers
// and a longer ragged tail. Measured on gfx1151, BKV=32 wins by 4-6% once a
// split covers a few hundred KV positions and loses by a similar margin below
// that. Keeping the short-range candidate set exactly as it was also stops the
// extra candidates from perturbing tuning decisions where they cannot win.
static inline bool flashDecodeTallTileUseful(int bkv, int kv_range,
                                             int splits) {
  if (bkv <= 16) return true;
  return kv_range / (splits > 0 ? splits : 1) >= 512;
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

// Time one candidate, shared by the fp16 and INT8 tuners.
//
// A fixed iteration count is wrong at both ends of this kernel's range. Decode
// spans ~3us (a 128-key sliding window) to ~300us (8k context): 10 iterations
// of the 3us case is 30us of GPU work, small enough that launch jitter swamps
// it -- repeated 10-iteration bursts on an IDENTICAL config measured a 1.4-2.3x
// spread, enough for the tuner to hand a 2.4x-slower config to every sliding
// layer for the process lifetime. Meanwhile 10 iterations of the 300us case
// costs 3ms per candidate for statistics it never needed.
//
// So budget by duration and run several bursts. The reduction over bursts was
// originally the MINIMUM (fastest burst == sample least polluted by system
// interference). That is right for SLOW outliers -- contention only ever makes
// a burst slower -- but it is catastrophic for FAST ones: on the 8060S the
// hipEvent timer intermittently reports a sub-physical elapsed time (measured a
// scalar/8 gpt-oss full@512 burst at 0.0006 ms, ~18x faster than the kernel can
// physically run), and `min` latches onto exactly that glitch, caching a config
// that loses to the baseline for the whole process lifetime -- the very failure
// mode the duration budget above was added to prevent. The MEDIAN of the bursts
// rejects an outlier in EITHER direction, so a lone timer glitch (fast) or a
// lone scheduling hiccup (slow) can no longer decide the ranking; a physical
// floor (a real burst cannot beat the warm probe by more than 2x) drops
// glitches before they can even enter the median. kBursts is raised to 5 so the
// median survives up to two bad samples; the ~1ms total-work budget is
// unchanged (it is divided across the bursts).
template <typename LaunchFn>
static float timeDecodeCandidate(hipStream_t stream, LaunchFn&& launch) {
  constexpr float kBudgetMs = 1.0f;  // GPU work per candidate, across bursts
  constexpr int kBursts = 5;
  constexpr int kProbeIters = 3;
  constexpr int kMaxIters = 4096;

  hipEvent_t ev0, ev1;
  (void)hipEventCreate(&ev0);
  (void)hipEventCreate(&ev1);

  // Warm up first: the initial launch of each kernel pays lazy module load.
  launch();
  launch();

  (void)hipEventRecord(ev0, stream);
  for (int i = 0; i < kProbeIters; ++i) launch();
  (void)hipEventRecord(ev1, stream);
  (void)hipEventSynchronize(ev1);
  float probe_ms = 0.0f;
  (void)hipEventElapsedTime(&probe_ms, ev0, ev1);
  const float per_iter = probe_ms / kProbeIters;

  int iters = 1;
  if (per_iter > 0.0f) {
    const float want = kBudgetMs / (kBursts * per_iter);
    iters = want < 1.0f ? 1 : (want > kMaxIters ? kMaxIters : (int)want);
  }

  // A real burst averages many warm launches; it cannot beat the (already warm)
  // probe estimate by more than this factor. Anything faster is a timer glitch.
  const float glitch_floor = (per_iter > 0.0f) ? per_iter * 0.5f : 0.0f;

  float samples[kBursts];
  int n = 0;
  for (int b = 0; b < kBursts; ++b) {
    (void)hipEventRecord(ev0, stream);
    for (int i = 0; i < iters; ++i) launch();
    (void)hipEventRecord(ev1, stream);
    (void)hipEventSynchronize(ev1);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev0, ev1);
    ms /= iters;
    if (ms > glitch_floor) samples[n++] = ms;  // drop sub-physical readings
  }

  (void)hipEventDestroy(ev0);
  (void)hipEventDestroy(ev1);

  if (n == 0) return per_iter;  // every burst glitched -> fall back to the probe
  std::sort(samples, samples + n);
  return samples[n / 2];  // median of the surviving bursts
}

// Benchmark every (impl, splits, bkv) candidate on the actual device buffers
// and pick the fastest. Leaves the output buffers holding the *last*
// candidate's result; the caller relaunches the winner afterwards to produce
// the correct output.
//
// fp16 and INT8 share this: the candidate set and the selection rule are the
// same, only the launcher differs, so `launch` is a callable taking the config.
// Keeping one copy is the point. The two used to be separate near-identical
// functions, and a tuning change applied to one but not the other would make
// that format silently pick a worse configuration while every test still
// passed, since the choice affects speed and not results. `launch` is a lambda
// and inlines, so this costs nothing at runtime.
template <typename LaunchFn>
static FlashDecodeCfg tuneFlashDecodeConfig(
    hipStream_t stream, int d, int hpg, int skv, int max_splits,
    const char* tag, LaunchFn&& launch) {
  int split_set[7]; int n_splits = 0;
  flashDecodeCandidateSplits(skv, max_splits, split_set, n_splits);
  const bool impls[2] = {false, true};  // {scalar, WMMA}; geometry is identical

  const bool wmma_ok = flash_decode_wmma_supported(d, hpg);
  FlashDecodeCfg best{wmma_ok && (d == 64), split_set[0]};
  float best_ms = 1e30f;

  // Only d==64 WMMA has a choice of tile height; everything else has one.
  const int bkv_set[2] = {16, 32};
  const int n_bkv = (wmma_ok && d == 64) ? 2 : 1;

  for (int ii = 0; ii < 2; ++ii) {
    if (impls[ii] && !wmma_ok) continue;  // skip WMMA where it isn't templated
    const int bkv_count = impls[ii] ? n_bkv : 1;
    for (int bi = 0; bi < bkv_count; ++bi) {
      for (int si = 0; si < n_splits; ++si) {
        if (!flashDecodeTallTileUseful(bkv_set[bi], skv, split_set[si]))
          continue;
        const FlashDecodeCfg cfg{impls[ii], split_set[si], bkv_set[bi]};
        const float ms = timeDecodeCandidate(stream, [&] { launch(cfg); });
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   %s cand impl=%-6s splits=%2d bkv=%2d "
            ": %.4f ms%s\n",
            tag, cfg.use_wmma ? "wmma" : "scalar", cfg.splits, cfg.bkv, ms,
            ms < best_ms ? "  <-- best so far" : "");
        if (ms < best_ms) { best_ms = ms; best = cfg; }
      }
    }
  }
  return best;
}

// INT8 decode autotune state + helpers (defined below). Forward-declared here so
// the unified hip_gqa_flash_decode can dispatch the int8 path when scales are
// supplied. Launching and tuning are both shared with fp16:
// launchFlashDecodeConfig<KvDtype::kI8> and tuneFlashDecodeConfig above. Only
// the config cache is per format, because the two time differently on the same
// geometry.
static std::unordered_map<uint64_t, FlashDecodeCfg> g_flash_decode_i8_cache;
static std::mutex g_flash_decode_i8_mutex;

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
static int runGqaFlashDecode(
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
    const void* v_scale,
    int configured_use_wmma,
    int configured_splits,
    int configured_bkv,
    bool allow_autotune) {
  // HPG = heads-per-group (Q heads sharing one KV head). The kernels are
  // instantiated for HPG=4 (Llama-3.x family at d=64 and d=128), HPG=5
  // (Qwen2.5-14B's 40:8 at d=128) and HPG=8 (gpt-oss-20b at d=64). HPG=8 with
  // d=128 is rejected to keep template bloat bounded; no current model needs it.
  if (G <= 0 || H % G != 0) {
    fprintf(stderr, "hip_gqa_flash_decode: H=%d not divisible by G=%d\n", H, G);
    return -1;
  }
  const int hpg = H / G;
  // Scalar decode is templated for these HpG (GQA + MHA); WMMA is layered on
  // top where flash_decode_wmma_supported() says so. Anything else is rejected.
  if (hpg != 1 && hpg != 2 && hpg != 3 && hpg != 4 && hpg != 5 && hpg != 8 &&
      hpg != 16) {
    fprintf(stderr,
            "hip_gqa_flash_decode: unsupported HPG=%d (must be "
            "1/2/3/4/5/8/16; H=%d, G=%d)\n",
            hpg, H, G);
    return -1;
  }
  if (d != 64 && d != 128 && d != 256) {
    fprintf(stderr,
            "hip_gqa_flash_decode: unsupported d=%d (must be 64/128/256)\n",
            d);
    return -1;
  }
  // Cache format is chosen by kv_dtype (extensible: add a case for INT4/FP8).
  if (kv_dtype != HIP_KV_DTYPE_FP16 && kv_dtype != HIP_KV_DTYPE_INT8) {
    fprintf(stderr, "hip_gqa_flash_decode: unsupported kv_dtype=%d\n",
            kv_dtype);
    return -1;
  }
  const bool kv_i8 = (kv_dtype == HIP_KV_DTYPE_INT8);
  if (kv_i8 && (k_scale == nullptr || v_scale == nullptr)) {
    fprintf(stderr,
            "hip_gqa_flash_decode: INT8 KV requires k_scale and v_scale\n");
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

  // --- Config selection: explicit LUT config > env override > online tune ---
  // Heuristic default favours WMMA at d==64, but only where it is templated;
  // every other geometry (incl. MHA) starts from the scalar kernel. The split
  // count is sized from the device rather than fixed, and keys off tune_len so
  // sliding-window layers do not over-split their (much shorter) range.
  const bool default_wmma = flash_decode_wmma_supported(d, hpg) && (d == 64);
  FlashDecodeCfg cfg{
      default_wmma,
      flashDecodeDefaultSplits(default_wmma, B, H, G, tune_len, cap)};

  bool forced = !allow_autotune;
  if (!allow_autotune) {
    if (configured_use_wmma < 0 || configured_splits < 1 ||
        configured_bkv < 1) {
      fprintf(stderr,
              "hip_gqa_flash_decode_configured: missing explicit config\n");
      return -1;
    }
    if (configured_splits > cap || (configured_bkv != 16 &&
                                    configured_bkv != 32) ||
        (configured_use_wmma == 0 && configured_bkv != 16) ||
        (configured_use_wmma != 0 &&
         !flash_decode_wmma_supported(d, hpg))) {
      fprintf(stderr,
              "hip_gqa_flash_decode_configured: invalid config "
              "(wmma=%d splits=%d bkv=%d cap=%d H=%d G=%d d=%d)\n",
              configured_use_wmma, configured_splits, configured_bkv, cap, H,
              G, d);
      return -1;
    }
    cfg = FlashDecodeCfg{configured_use_wmma != 0, configured_splits,
                         configured_bkv};
  }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_SCALAR")) { if (atoi(e) != 0) { cfg.use_wmma = false; forced = true; } }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_WMMA"))   { if (atoi(e) != 0) { cfg.use_wmma = true;  forced = true; } }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_SPLITS")) {
    int v = atoi(e); if (v >= 1) { cfg.splits = v > cap ? cap : v; forced = true; }
  }
  if (const char* e = getenv("HIPDNN_GQA_DECODE_BKV")) {
    int v = atoi(e); if (v == 16 || v == 32) { cfg.bkv = v; forced = true; }
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
        cfg = tuneFlashDecodeConfig(
            stream, d, hpg, tune_len, cap, "flash_decode_i8",
            [&](const FlashDecodeCfg& c) {
              launchFlashDecodeConfig<KvDtype::kI8>(
                  c, stream, hQ, hK, hV, kSc, vSc, hO, hP, B, H, G, d, hpg,
                  max_seq, scale, iSeq, local_window_size, hSink,
                  use_smooth_softmax);
            });
        g_flash_decode_i8_cache[key] = cfg;
      }
    }
    launchFlashDecodeConfig<KvDtype::kI8>(
        cfg, stream, hQ, hK, hV, kSc, vSc, hO, hP, B, H, G, d, hpg, max_seq,
        scale, iSeq, local_window_size, hSink, use_smooth_softmax);
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
      cfg = tuneFlashDecodeConfig(
          stream, d, hpg, tune_len, cap, "flash_decode",
          [&](const FlashDecodeCfg& c) {
            launchFlashDecodeConfig<KvDtype::kF16>(
                c, stream, hQ, hK, hV, nullptr, nullptr, hO, hP, B, H, G, d,
                hpg, max_seq, scale, iSeq, local_window_size, hSink,
                use_smooth_softmax);
          });
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
  launchFlashDecodeConfig<KvDtype::kF16>(
      cfg, stream, hQ, hK, hV, nullptr, nullptr, hO, hP, B, H, G, d, hpg,
      max_seq, scale, iSeq, local_window_size, hSink, use_smooth_softmax);

  return hipGetLastError();
}

// Keep the established public entry as the online-tuning path. The lookup-only
// sibling below takes an explicit config from gqa_autotune.cpp.
extern "C" int hip_gqa_flash_decode(
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
  return runGqaFlashDecode(
      stream_ptr, Q, Kcache, Vcache, O, partials_workspace, B, H, G, d, skv,
      max_seq, max_splits, scale, seqlens_k, local_window_size, head_sink,
      use_smooth_softmax, kv_dtype, k_scale, v_scale,
      /*configured_use_wmma=*/-1, /*configured_splits=*/0,
      /*configured_bkv=*/-1,
      /*allow_autotune=*/true);
}

extern "C" int hip_gqa_flash_decode_configured(
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
    const void* v_scale,
    int use_wmma,
    int splits,
    int bkv) {
  return runGqaFlashDecode(
      stream_ptr, Q, Kcache, Vcache, O, partials_workspace, B, H, G, d, skv,
      max_seq, max_splits, scale, seqlens_k, local_window_size, head_sink,
      use_smooth_softmax, kv_dtype, k_scale, v_scale, use_wmma, splits, bkv,
      /*allow_autotune=*/false);
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
// HAS_WINDOW is a template parameter rather than a runtime test because this
// kernel runs at the VGPR ceiling: carrying the window's per-row state live in
// the full-attention instantiation pushed M_TILES=2 from 120 to 180 bytes/lane
// of scratch (38 -> 68 spills) and cost that path 21% at deep KV.
template <int M_TILES, int BKV, int D, bool HAS_WINDOW>
__global__ void __launch_bounds__(32)
gqa_flash_prefill_v5_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate,
    const _Float16* __restrict__ head_sink,  // [num_heads] or null
    int num_heads, int smooth_softmax,
    int local_window_size)                   // <= 0 for full attention
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

    // With a window, every row in this Q tile has a lower bound at least as high
    // as the first row's, so starting from the first row's bound covers the whole
    // tile and the per-row mask trims the rest. This is what makes the window
    // cheaper rather than merely masked: the number of tiles walked stops growing
    // with skv once the window is filled.
    int t_start = 0;
    if constexpr (HAS_WINDOW) {
        const int kv_lo_tile = past_len + q_start - local_window_size + 1;
        t_start = (kv_lo_tile > 0) ? (kv_lo_tile / BKV) : 0;
    }

    for (int t = t_start; t < num_tiles; ++t) {
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
                // causal mask, sliding-window lower bound, and per-row partial
                // max over this lane's columns
                const int abs_q = past_len + q_pos;
                int kv_lo = 0;
                if constexpr (HAS_WINDOW) kv_lo = abs_q - local_window_size + 1;
                float rmax = -INFINITY;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const int kv_pos = kv0 + sj * 16 + wmma_lane;
                    bool masked = (kv_pos >= skv || kv_pos > abs_q);
                    if constexpr (HAS_WINDOW) masked = masked || (kv_pos < kv_lo);
                    if (masked) S_reg[mi][sj][e] = -INFINITY;
                    rmax = fmaxf(rmax, S_reg[mi][sj][e]);
                }
                rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                const float m_old = m_reg[mi][e];
                const float m_new = fmaxf(m_old, rmax);
                // A window can mask a whole tile for a row while that row has
                // still seen nothing, leaving m_old and m_new both at -inf.
                // exp2f(-inf - -inf) is NaN, so short-circuit: contribute
                // nothing and leave the running state untouched. Causal-only
                // never reaches this, since tile 0 always holds key 0, so the
                // guard folds away in the full-attention instantiation.
                bool empty = false;
                if constexpr (HAS_WINDOW) empty = (m_new == -INFINITY);
                const float corr  = empty ? 1.0f : exp2f(m_old - m_new);
                corr_reg[mi][e]   = corr;

                float rsum = 0.0f;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const float p =
                        empty ? 0.0f : exp2f(S_reg[mi][sj][e] - m_new);
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
    const _Float16* head_sink, int num_heads, int smooth_softmax,
    int local_window_size)
{
    const int rows = m_tiles * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV5SmemBytes(m_tiles, bkv, d);
    const int ablate = g_v5_ablate;

    #define LAUNCH_V5(MT_, BKV_, D_, HW_)                                       \
        gqa_flash_prefill_v5_kernel<MT_, BKV_, D_, HW_>                         \
            <<<grid, 32, smem, stream>>>(                                       \
                Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,     \
                scale, ablate, head_sink, num_heads, smooth_softmax,            \
                local_window_size)

    // Only d == 64 gets windowed instantiations: hip_gqa_flash_prefill
    // rejects a window at any other head dim, so d == 128 with a window would
    // be dead code.
    if (d == 64) {
        if (local_window_size > 0) {
            if (m_tiles == 2 && bkv == 64)  LAUNCH_V5(2, 64, 64, true);
            else if (m_tiles == 2)          LAUNCH_V5(2, 32, 64, true);
            else if (bkv == 64)             LAUNCH_V5(1, 64, 64, true);
            else                            LAUNCH_V5(1, 32, 64, true);
        } else {
            if (m_tiles == 2 && bkv == 64)  LAUNCH_V5(2, 64, 64, false);
            else if (m_tiles == 2)          LAUNCH_V5(2, 32, 64, false);
            else if (bkv == 64)             LAUNCH_V5(1, 64, 64, false);
            else                            LAUNCH_V5(1, 32, 64, false);
        }
    } else {
        if (m_tiles == 2 && bkv == 64)      LAUNCH_V5(2, 64, 128, false);
        else if (m_tiles == 2)              LAUNCH_V5(2, 32, 128, false);
        else if (bkv == 64)                 LAUNCH_V5(1, 64, 128, false);
        else                                LAUNCH_V5(1, 32, 128, false);
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
    int past_len, float scale, int local_window_size)
{
    PrefillV5Cfg cands[8];
    const int nc = prefillV5Candidates(d, cands);
    constexpr int kWarmup = 30, kRounds = 4, kIters = 30;

    // Tuning launches carry no sink: it is a single fused-multiply-add per row in
    // the epilogue, so it cannot shift which (M_TILES, BKV) wins, and leaving it
    // out keeps tuned configs comparable with the non-sink case. The window is
    // passed through, because it changes how many KV tiles each block walks and
    // therefore genuinely can change the winner.
    auto launch_cfg = [&](const PrefillV5Cfg& c) {
        dispatchPrefillV5(c.m_tiles, c.bkv, d, stream, Q, Kcache, Vcache, O,
                          B, Hq, G, sq, skv, max_seq, past_len, scale,
                          nullptr, Hq, 0, local_window_size);
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
    // A WMMA accumulator holds 8 rows per lane (row = e*2 + pair), and the
    // softmax over those rows is row-independent, so it splits ND ways instead of
    // being repeated ND times or dumped on one wave while the rest wait.
    constexpr int EPW       = 8 / ND;          // accumulator rows per wave
    static_assert(EPW * ND == 8, "ND must divide 8 for the softmax row split");
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
    // Per-row softmax correction, published by the one wave that owns the
    // softmax and read by all of them; reused after the KV loop to hand out the
    // final l. ROWS floats, so 64 B at MT=1.
    float*    Cr_lds  = reinterpret_cast<float*>(
        P_lds + (size_t)ROWS * P_STR);

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

    // Accumulate O in fp32 registers. The WMMA accumulator is already fp32, so
    // the previous half8_t store meant unpacking to float, accumulating, and
    // repacking on every KV tile -- 2 conversions per element per tile, plus the
    // same round-trip again in the rescale, for no register saving that mattered
    // at the winning config. Holding fp32 also drops the per-tile rounding of a
    // running sum.
    float8_t O_acc[MT][DPG] = {};
    // Running (m, l) only for the rows this wave owns in the softmax split.
    float   m_reg[MT][EPW];
    float   l_reg[MT][EPW];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int ei = 0; ei < EPW; ++ei) { m_reg[mt][ei] = -INFINITY; l_reg[mt][ei] = 0.0f; }

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
        // abl bit 32 drops all three per-tile barriers. The result is wrong, but
        // it prices the barriers before any restructuring is done to remove them:
        // this kernel runs 3 barriers per KV tile and ~583 tiles per block at
        // BKV=16, so "reduce the barrier count" is only worth doing if that shows
        // up as time.
        if (!(abl & 32)) __syncthreads();   // b_a: partials + V staging visible

        // ---- softmax, rows split ND ways across the waves ----
        // Every wave used to sum the ND partials and run a bit-identical softmax,
        // with only group 0's P store skipped for the others. Ablating this block
        // (ablate bit 1, which also drops the cross-wave partial sum below, so the
        // figure covers that read-back too) cost 136 ms of 348 ms at
        // sq=18646, the largest single item in the kernel, so repeating it ND times
        // was the main waste -- and it is why raising ND used to make this kernel
        // slower rather than faster despite the extra occupancy.
        //
        // Handing the whole softmax to group 0 instead only moves the problem: the
        // other waves then idle at b_b while group 0's copy sits on the critical
        // path. The rows are independent, so wave g takes accumulator rows
        // [g*EPW, (g+1)*EPW) and the critical path shrinks with ND instead of
        // growing. Each wave publishes the per-row correction and, after the loop,
        // its share of l; both are read back by every wave, since a wave writes
        // all 16 rows of the D tiles it owns.
        //
        // corr and l are uniform across wmma_lane (both come out of a 16-lane
        // shuffle reduction), so the 16 lanes sharing a row write the same value
        // to Cr_lds -- a same-value race, not a data race.
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                #pragma unroll
                for (int ei = 0; ei < EPW; ++ei) {
                    const int e     = group * EPW + ei;
                    const int rl    = e * 2 + pair;
                    const int q_pos = q_start + mt * 16 + rl;

                    // Sum the ND partial scores for this row only.
                    float s_row[S_TILES_J];
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int col = sj * 16 + wmma_lane;
                        float acc = 0.0f;
                        #pragma unroll
                        for (int g = 0; g < ND; ++g)
                            acc += Sp_lds[((size_t)g * ROWS + mt * 16 + rl) * SP_STR + col];
                        s_row[sj] = acc * scale * kLog2e;
                    }

                    float rmax = -INFINITY;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int kv_pos = kv0 + sj * 16 + wmma_lane;
                        if (kv_pos >= skv || kv_pos > past_len + q_pos)
                            s_row[sj] = -INFINITY;
                        rmax = fmaxf(rmax, s_row[sj]);
                    }
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                    const float m_old = m_reg[mt][ei];
                    const float m_new = fmaxf(m_old, rmax);
                    const float corr  = exp2f(m_old - m_new);

                    float rsum = 0.0f;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const float p = exp2f(s_row[sj] - m_new);
                        rsum += p;
                        P_lds[(mt * 16 + rl) * P_STR + sj * 16 + wmma_lane] =
                            static_cast<_Float16>(p);
                    }
                    rsum += __shfl_xor(rsum, 1);
                    rsum += __shfl_xor(rsum, 2);
                    rsum += __shfl_xor(rsum, 4);
                    rsum += __shfl_xor(rsum, 8);

                    m_reg[mt][ei] = m_new;
                    l_reg[mt][ei] = corr * l_reg[mt][ei] + rsum;
                    Cr_lds[mt * 16 + rl] = corr;
                }
            }
        }
        if (!(abl & 32)) __syncthreads();   // b_b: P_lds + Cr_lds visible

        // ---- every wave rescales its own O tiles by the published correction ----
        // Skipping this when corr == 1.0 (i.e. no row's running max grew, which is
        // most tiles past the first few) was tried and measured 0.3%, inside run
        // noise: these fp32 multiplies hide behind the tile's memory traffic, so
        // the ballot and the branch bought nothing and are not here.
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float corr_e[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    corr_e[e] = Cr_lds[mt * 16 + e * 2 + pair];
                #pragma unroll
                for (int tj = 0; tj < DPG; ++tj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        O_acc[mt][tj][e] *= corr_e[e];
            }
        }

        // ---- value GEMM: this wave's owned D tiles only ----
        if (!(abl & 2))
        for (int tj = 0; tj < DPG; ++tj) {
            const int d_tile = dt_base + tj;
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
                    O_acc[mt][tj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        p_frag, v_frag, O_acc[mt][tj]);
                }
            }
        }
        if (!(abl & 32)) __syncthreads();   // b_c: WAR on V_lds/Sp_lds/P_lds
    }

    // ---- pool the per-wave l so every wave can normalise all 16 rows ----
    // The softmax split gives each wave l for only EPW of the 8 accumulator rows,
    // but a wave writes all 16 rows of the D tiles it owns. b_c above already
    // separated this from the last tile's reads of Cr_lds, so one barrier does.
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int ei = 0; ei < EPW; ++ei)
            Cr_lds[mt * 16 + (group * EPW + ei) * 2 + pair] = l_reg[mt][ei];
    __syncthreads();
    float l_fin[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            l_fin[mt][e] = Cr_lds[mt * 16 + e * 2 + pair];

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
                    const float l_val   = fmaxf(l_fin[mt][e], 1e-6f);
                    const float out_val = O_acc[mt][tj][e] / l_val;
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
    const size_t cr_lds = (size_t)rows * sizeof(float);
    return v_lds + sp_lds + p_lds + cr_lds;
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
    else if (nd == 2 && mt == 1 && bkv == 16) LAUNCH_V8(2, 1, 16);
    else if (nd == 4 && mt == 1 && bkv == 16) LAUNCH_V8(4, 1, 16);
    else if (nd == 8 && mt == 1 && bkv == 16) LAUNCH_V8(8, 1, 16);
    else if (nd == 8 && mt == 1 && bkv == 32) LAUNCH_V8(8, 1, 32);
    else if (nd == 2 && mt == 2 && bkv == 16) LAUNCH_V8(2, 2, 16);
    else if (nd == 4 && mt == 2 && bkv == 16) LAUNCH_V8(4, 2, 16);
    else if (nd == 1 && mt == 1 && bkv == 16) LAUNCH_V8(1, 1, 16);
    else if (nd == 1 && mt == 2 && bkv == 16) LAUNCH_V8(1, 2, 16);
    else                                      LAUNCH_V8(2, 1, 32);
    #undef LAUNCH_V8
}

struct PrefillV8Cfg { int nd; int mt; int bkv; };

// The configs the tuner sweeps, kept to those that do not spill and can plausibly
// win. Each candidate costs ~40 s of first-call sweep at sq=18646, and -O3 register
// allocation (read with -Rpass-analysis=kernel-resource-usage) disqualifies most
// of the original list outright:
//
//   cfg      vgpr  spill   occ    ms @ sq=18646
//   2,1,16    239      0  37.5%   204   <- long-prompt winner
//   4,1,16    135      0  62.5%   219
//   4,1,32    175      0  50.0%   249
//   8,1,16     82      0 100.0%   276
//   2,1,32    256     92  31.2%    -    (spills)
//   2,2,32    256      -  31.2%    -    (spills)
//   2,1,64    256      -  31.2%    -    (spills; V_lds alone is 33 KB)
//   2,2,64    256      -  31.2%    -    (spills)
//   1,1,16    256      -  31.2%   410   (spills)
//   2,2,16    256      -  31.2%   423   (spills)
//   1,2,16    256      -  31.2%   638   (spills)
//
// Occupancy does not order this table: 8,1,16 is fully occupant and still 35%
// slower than 2,1,16 at 37.5%, because raising ND adds cross-wave
// score-reduction traffic that costs more than the extra residency buys. That is
// the opposite of what the kernel's "lift occupancy" design note assumes, so do
// not re-add candidates on an occupancy argument alone. Anything that spills is
// hopeless regardless, which removes BKV=64 and all of MT=2.
static inline int prefillV8Candidates(int d, PrefillV8Cfg* out) {
    static const PrefillV8Cfg kTuned[] = {
        {2, 1, 16}, {4, 1, 16}, {4, 1, 32}, {8, 1, 16}};
    int n = 0;
    for (const PrefillV8Cfg& c : kTuned)
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
    PrefillV8Cfg cands[16];
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
    float best_ms[16];
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
    float scale, const void* head_sink, int num_heads, int smooth_softmax,
    int local_window_size)
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
    //
    // The window does belong in the key: it caps how many KV tiles a block walks,
    // which is a different work profile, not a scaled one.
    std::string key = std::to_string(d) + "_" + std::to_string(sq) + "_" +
                      std::to_string(Hq) + "_" + std::to_string(G) + "_w" +
                      std::to_string(local_window_size > 0 ? local_window_size : 0);
    int cfg;
    {
        std::lock_guard<std::mutex> lk(g_v5_cfg_mutex);
        auto it = g_v5_cfg_cache.find(key);
        if (it != g_v5_cfg_cache.end()) {
            cfg = it->second;
        } else {
            cfg = tunePrefillV5Cfg(d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                                   B, Hq, G, sq, skv, max_seq, past_len, scale,
                                   local_window_size);
            g_v5_cfg_cache[key] = cfg;
            snprintf(g_v5_tune_info, sizeof(g_v5_tune_info),
                     "gqa_flash_prefill_v5 cfg=M%d_BKV%d", cfg / 1000, cfg % 1000);
        }
    }
    int m_tiles = cfg / 1000, bkv = cfg % 1000;
    dispatchPrefillV5(m_tiles, bkv, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                      B, Hq, G, sq, skv, max_seq, past_len, scale,
                      sink_ptr, num_heads, smooth_softmax, local_window_size);
    return 0;
}

extern "C" int hip_gqa_flash_prefill_v5_v2(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale)
{
    return prefillV5Run(stream_ptr, Q, Kcache, Vcache, O, B, Hq, G, sq, skv, d,
                        max_seq, past_len, scale, nullptr, Hq, 0, 0);
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


// Wide fused flash-prefill entry: adds attention sinks and the sliding window
// on top of v2. v2 stays the narrow ABI so the standalone harnesses and shim
// headers that redeclare it keep linking unchanged.
//
// Both features are implemented for d == 64 (the v5 kernel) only. For any other
// d we return -1 rather than silently ignoring them, so the runtime keeps
// routing those shapes to the decomposed path, which does implement both. A
// request carrying neither is forwarded to v2 and keeps v2's head-dim coverage
// and its v5/v7/v8 selection; d == 64 lands on v5 either way, so a sink or a
// window never changes which kernel runs.
extern "C" int hip_gqa_flash_prefill(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax)
{
    const bool wants_sink   = (head_sink != nullptr) || (smooth_softmax != 0);
    const bool wants_window = local_window_size > 0;
    if ((wants_sink || wants_window) && d != 64) return -1;
    if (!wants_sink && !wants_window) {
        return hip_gqa_flash_prefill_v2(stream_ptr, Q, Kcache, Vcache, O, B, Hq,
                                        G, sq, skv, d, max_seq, past_len, scale);
    }
    return prefillV5Run(stream_ptr, Q, Kcache, Vcache, O, B, Hq, G, sq, skv, d,
                        max_seq, past_len, scale, head_sink, num_heads,
                        smooth_softmax, local_window_size);
}

// Unified fused flash-prefill entry. The kernel-selection strategy lives here
// (in the kernel TU) instead of in the runtime dispatcher: v5 wins at d==64
// (warp-private register-resident), v7 at d==128 (M-register-blocked with
// global-streamed K), v8 at d==256 (output-D split + secondary score reduction
// to lift occupancy). The runtime only has to call this one symbol for any
// eligible (fp16, causal, GQA) prefill and gets the right kernel automatically.
// Returns -1 for unsupported head dims so the caller can fall back.
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

// Lookup-only prefill entry. Unlike v2/v3, every launch parameter comes from a
// runtime-validated offline LUT or deterministic heuristic, so this path never
// enters tunePrefillV5Cfg/tunePrefillV7Cfg/tunePrefillV8Cfg.
extern "C" int hip_gqa_flash_prefill_v3_configured(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax,
    int m_tiles, int bkv, int nw, int mt, int nd)
{
    const bool wants_sink = head_sink != nullptr || smooth_softmax != 0;
    const bool wants_window = local_window_size > 0;
    if ((wants_sink || wants_window) && d != 64) return -1;

    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
    const _Float16* Q_ptr = static_cast<const _Float16*>(Q);
    const _Float16* K_ptr = static_cast<const _Float16*>(Kcache);
    const _Float16* V_ptr = static_cast<const _Float16*>(Vcache);
    _Float16* O_ptr = static_cast<_Float16*>(O);

    if (d == 64) {
        PrefillV5Cfg candidates[8];
        const int count = prefillV5Candidates(d, candidates);
        bool valid = false;
        for (int i = 0; i < count; ++i)
            if (candidates[i].m_tiles == m_tiles && candidates[i].bkv == bkv)
                valid = true;
        if (!valid) return -1;
        dispatchPrefillV5(m_tiles, bkv, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                          B, Hq, G, sq, skv, max_seq, past_len, scale,
                          static_cast<const _Float16*>(head_sink), num_heads,
                          smooth_softmax, local_window_size);
        return 0;
    }
    if (d == 128) {
        PrefillV7Cfg candidates[12];
        const int count = prefillV7Candidates(d, candidates);
        bool valid = false;
        for (int i = 0; i < count; ++i)
            if (candidates[i].nw == nw && candidates[i].bkv == bkv &&
                candidates[i].mt == mt)
                valid = true;
        if (!valid) return -1;
        dispatchPrefillV7(nw, bkv, mt, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                          B, Hq, G, sq, skv, max_seq, past_len, scale);
        return 0;
    }
    if (d == 256) {
        PrefillV8Cfg candidates[8];
        const int count = prefillV8Candidates(d, candidates);
        bool valid = false;
        for (int i = 0; i < count; ++i)
            if (candidates[i].nd == nd && candidates[i].bkv == bkv &&
                candidates[i].mt == mt)
                valid = true;
        if (!valid) return -1;
        dispatchPrefillV8(nd, mt, bkv, d, stream, Q_ptr, K_ptr, V_ptr, O_ptr,
                          B, Hq, G, sq, skv, max_seq, past_len, scale);
        return 0;
    }
    return -1;
}

//===----------------------------------------------------------------------===//
// Legacy decode kernel (folded in from the retired gqa_kernel_legacy.hip).
//
// The production decode path is hip_gqa_flash_decode above; it is the SINGLE
// decode kernel the runtime dispatches for every fp16 causal GQA/MHA decode --
// sliding window and head sink folded in, split count autotuned per shape.
// This one retired entry backs gqa.cpp::gqa_forward_hipblaslt (the decomposed
// fallback) for the only decode case v2 does not template:
//   * hip_gqa_fused_decode : one-block-per-(batch,head_q), d in {64,128,256},
//                            arbitrary HpG -- the general odd-geometry decode
//                            the fallback uses. It carries neither window nor
//                            sink; those cases route to the decomposed pipeline.
// The kernel carries a legacy_ prefix so it coexists with the v2 decode kernels
// above; it reuses this TU's WAVE_SIZE / LOG2E / GQA_RETURN_LAUNCH_STATUS.
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

#pragma unroll
            for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
                dot += __shfl_xor(dot, offset);

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
