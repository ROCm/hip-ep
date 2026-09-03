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
#include <vector>

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"
#include "debug_log.h"

#include "rtc/hiprtc_module.h"

// Generated per arch from hip/autotune/gqa/lut/<arch>.json; see
// hip/autotune/gqa/scripts/gen_rtc_names.py.
#include "gqa_rtc_tuned_names.h"

static constexpr int GQA_THREADS = 256;

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

// The __global__ definitions live in a header so the AOT build and hipRTC
// compile the same text. It also carries what both halves share (WAVE_SIZE,
// KvDtype), so this include has to precede the host dispatch below. Keeping
// the kernels outside the anonymous namespace below gives them the same
// linkage on both paths.
#include "hip/rtc/gqa_device.h"

// The same header, embedded as bytes at build time with hip_arch_compat.h
// prepended: the device half uses WAVE_SIZE and the WMMA macros without
// including it.
extern "C" const unsigned char kGqaDeviceSource[];
extern "C" const size_t kGqaDeviceSource_size;

// nullptr whenever RTC is off or the module failed to build, which is the
// signal for HIPDNN_RTC_LAUNCH to use the AOT kernel. Defined at the bottom of
// the file, once every family has contributed its name expressions.
static hipFunction_t gqaRtcFunction(const char* name_expression);

// A name expression is stringized from the tokens at the launch site, so
// KERNEL<T> would register the letter T rather than the type it stands for.
// Branching on the element type puts a spelled-out instantiation in its place.
#define GQA_LAUNCH_BY_ELEM(TPARAM, GRID, BLOCK, SMEM, ARGS, KERNEL)            \
  do {                                                                         \
    if constexpr (std::is_same_v<TPARAM, __half>) {                            \
      HIPDNN_RTC_LAUNCH(gqaRtcFunction, GRID, BLOCK, SMEM, stream, ARGS,       \
                        KERNEL<__half>);                                       \
    } else {                                                                   \
      HIPDNN_RTC_LAUNCH(gqaRtcFunction, GRID, BLOCK, SMEM, stream, ARGS,       \
                        KERNEL<float>);                                        \
    }                                                                          \
  } while (0)

// As above, with VEC (16 / sizeof(T)) spelled out too.
#define GQA_LAUNCH_BY_ELEM_VEC(GRID, BLOCK, ARGS, KERNEL)                      \
  do {                                                                         \
    if constexpr (std::is_same_v<T, __half>) {                                 \
      HIPDNN_RTC_LAUNCH(gqaRtcFunction, GRID, BLOCK, 0, stream, ARGS,          \
                        KERNEL<__half, 8>);                                    \
    } else {                                                                   \
      HIPDNN_RTC_LAUNCH(gqaRtcFunction, GRID, BLOCK, 0, stream, ARGS,          \
                        KERNEL<float, 4>);                                     \
    }                                                                          \
  } while (0)

#define GQA_ELEM_KERNEL_LIST(X)                                                \
  X(rope_kernel) X(transpose_mid_dims_kernel) X(kv_cache_append_kernel)        \
  X(kv_cache_concat_kernel) X(expand_kv_kernel) X(split_qkv_kernel)            \
  X(causal_mask_kernel_impl) X(add_attention_bias_kernel)                      \
  X(softmax_f32_to_out_kernel)

#define GQA_ELEM_VEC_KERNEL_LIST(X)                                            \
  X(kv_cache_append_vec_kernel) X(kv_cache_concat_vec_kernel)

// The vectorised path indexes in KvVec16 units, which is only valid when the
// head dim divides evenly and every buffer is 16 B aligned. Element offsets are
// then automatically aligned too: each is (multiple of d) + (multiple of VEC),
// and d % VEC == 0. Callers fall back to the scalar kernel when this fails.
static inline bool kv_is_16b_aligned(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
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
  GQA_LAUNCH_BY_ELEM(
      T, dim3(blocks, batch_size), GQA_THREADS, 0,
      (static_cast<const T*>(input), static_cast<T*>(output),
       static_cast<const T*>(cos_cache), static_cast<const T*>(sin_cache),
       seq_len, num_heads, head_dim, half_rot, past_len,
       static_cast<const int*>(seqlens_k)),
      rope_kernel);
}

template <typename T>
static void launch_transpose_mid_dims(
    hipStream_t stream, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D) {
  int per_batch = dim1 * dim2 * D;
  int blocks = (per_batch + GQA_THREADS - 1) / GQA_THREADS;
  GQA_LAUNCH_BY_ELEM(
      T, dim3(blocks, batch_size), GQA_THREADS, 0,
      (static_cast<const T*>(src), static_cast<T*>(dst), dim1, dim2, D),
      transpose_mid_dims_kernel);
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
    GQA_LAUNCH_BY_ELEM_VEC(
        dim3(blocks, batch_size), GQA_THREADS,
        (static_cast<const T*>(src), static_cast<T*>(cache), sq, G, d,
         present_seq, past_len, static_cast<const int*>(seqlens_k)),
        kv_cache_append_vec_kernel);
    return;
  }
  int total = sq * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  GQA_LAUNCH_BY_ELEM(
      T, dim3(blocks, batch_size), GQA_THREADS, 0,
      (static_cast<const T*>(src), static_cast<T*>(cache), sq, G, d,
       present_seq, past_len, static_cast<const int*>(seqlens_k)),
      kv_cache_append_kernel);
}

template <typename T>
static void launch_kv_cache_concat(
    hipStream_t stream, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int s_lo) {
  int total_seq = past_len + sq;
  constexpr int VEC = 16 / sizeof(T);
  if ((d % VEC) == 0 && kv_is_16b_aligned(past) && kv_is_16b_aligned(current) &&
      kv_is_16b_aligned(present)) {
    int total = (total_seq - s_lo) * G * (d / VEC);
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    GQA_LAUNCH_BY_ELEM_VEC(
        dim3(blocks, batch_size), GQA_THREADS,
        (static_cast<const T*>(past), static_cast<const T*>(current),
         static_cast<T*>(present), past_len, sq, G, d, past_seq, present_seq,
         s_lo),
        kv_cache_concat_vec_kernel);
    return;
  }
  int total = (total_seq - s_lo) * G * d;
  int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
  GQA_LAUNCH_BY_ELEM(
      T, dim3(blocks, batch_size), GQA_THREADS, 0,
      (static_cast<const T*>(past), static_cast<const T*>(current),
       static_cast<T*>(present), past_len, sq, G, d, past_seq, present_seq,
       s_lo),
      kv_cache_concat_kernel);
}

template <typename T>
static void launch_expand_kv(
    hipStream_t stream, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems) {
  int blocksY = (copy_elems + GQA_THREADS - 1) / GQA_THREADS;
  GQA_LAUNCH_BY_ELEM(
      T, dim3(total_heads, blocksY), GQA_THREADS, 0,
      (static_cast<const T*>(src), static_cast<T*>(dst), heads_per_group,
       src_stride, dst_stride, copy_elems),
      expand_kv_kernel);
}

template <typename T>
static void launch_split_qkv(
    hipStream_t stream, const void* packed, void* Q, void* K, void* V,
    int BS, int D_total, int Q_dim, int K_dim) {
  int total_elems = BS * D_total;
  int blocks = (total_elems + GQA_THREADS - 1) / GQA_THREADS;
  GQA_LAUNCH_BY_ELEM(
      T, blocks, GQA_THREADS, 0,
      (static_cast<const T*>(packed), static_cast<T*>(Q), static_cast<T*>(K),
       static_cast<T*>(V), BS, D_total, Q_dim, K_dim),
      split_qkv_kernel);
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
    HIPDNN_RTC_LAUNCH(
        gqaRtcFunction, dim3(blocks, batch_size), GQA_THREADS, 0, stream,
        (static_cast<const _Float16*>(src), static_cast<int8_t*>(cache),
         static_cast<const float*>(scale), sq, G, d, present_seq, past_len,
         static_cast<const int*>(seqlens_k)),
        kv_cache_append_quant_i8_kernel);
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
//
// s_lo skips the past prefix below it. It is clamped to [0, past_len] here, once
// for all three kernels: past_len is the hard ceiling because [past_len,
// total_seq) is the new tokens, which are the one thing this call must never
// drop. A caller asking for more than that is asking for silent data loss, so
// the boundary refuses rather than obeys.
extern "C" int hip_gqa_kv_cache_concat(
    void* stream_ptr, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int element_size_bytes,
    int kv_dtype, const void* scale, int s_lo) {
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  if (s_lo < 0) s_lo = 0;
  if (s_lo > past_len) s_lo = past_len;
  if (kv_dtype == HIP_KV_DTYPE_INT8) {
    if (scale == nullptr) {
      fprintf(stderr, "hip_gqa_kv_cache_concat: INT8 requires scale\n");
      return -1;
    }
    int total_seq = past_len + sq;
    int total = (total_seq - s_lo) * G * d;
    int blocks = (total + GQA_THREADS - 1) / GQA_THREADS;
    (void)hipGetLastError();
    HIPDNN_RTC_LAUNCH(
        gqaRtcFunction, dim3(blocks, batch_size), GQA_THREADS, 0, stream,
        (static_cast<const int8_t*>(past),
         static_cast<const _Float16*>(current), static_cast<int8_t*>(present),
         static_cast<const float*>(scale), past_len, sq, G, d, past_seq,
         present_seq, s_lo),
        kv_cache_concat_quant_i8_kernel);
    GQA_RETURN_LAUNCH_STATUS();
  }
  if (kv_dtype != HIP_KV_DTYPE_FP16) {
    fprintf(stderr, "hip_gqa_kv_cache_concat: unsupported kv_dtype=%d\n",
            kv_dtype);
    return -1;
  }
  GQA_DISPATCH_BY_ELEM_SIZE(element_size_bytes, launch_kv_cache_concat, stream,
                            past, current, present, batch_size, past_len, sq, G,
                            d, past_seq, present_seq, s_lo);
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
  HIPDNN_RTC_LAUNCH(
      gqaRtcFunction, dim3(blocks, batch_size), GQA_THREADS, 0, stream,
      (static_cast<const int8_t*>(src), static_cast<_Float16*>(dst),
       static_cast<const float*>(scale), total_seq, G, d, src_seq, dst_seq),
      dequant_kv_i8_to_fp16_kernel);
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
  HIPDNN_RTC_LAUNCH(gqaRtcFunction, dim3(blocksX, total_heads), GQA_THREADS, 0,
                    stream,
                    (static_cast<__half*>(S), skv, sq, batch_stride, past_len,
                     local_window_size),
                    causal_mask_kernel_impl<__half>);
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
  HIPDNN_RTC_LAUNCH(gqaRtcFunction, dim3(blocksX, total_heads), GQA_THREADS, 0,
                    stream,
                    (static_cast<float*>(S), skv, sq, batch_stride, past_len,
                     local_window_size),
                    causal_mask_kernel_impl<float>);
  GQA_RETURN_LAUNCH_STATUS();
}

extern "C" int hip_gqa_add_attention_bias_f32(
    void* stream_ptr, void* scores, const void* bias, int total_heads,
    int num_heads, int bias_batch, int bias_heads, int sq, int score_cols,
    int score_batch_stride, int bias_element_size_bytes, int bias_sq,
    int bias_row_offset, int bias_total_seq, int bias_col_offset) {
  if (!scores || !bias)
    return -1;
  hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
  if (bias_sq <= 0)
    bias_sq = sq;
  if (bias_total_seq <= 0)
    bias_total_seq = score_cols;
  int n = sq * score_cols;
  int blocksX = (n + GQA_THREADS - 1) / GQA_THREADS;
  (void)hipGetLastError();
  if (bias_element_size_bytes == 2) {
    HIPDNN_RTC_LAUNCH(gqaRtcFunction, dim3(blocksX, total_heads), GQA_THREADS,
                      0, stream,
                      (static_cast<float*>(scores),
                       static_cast<const __half*>(bias), total_heads, num_heads,
                       bias_batch, bias_heads, sq, score_cols,
                       score_batch_stride, bias_sq, bias_row_offset,
                       bias_total_seq, bias_col_offset),
                      add_attention_bias_kernel<__half>);
  } else if (bias_element_size_bytes == 4) {
    HIPDNN_RTC_LAUNCH(gqaRtcFunction, dim3(blocksX, total_heads), GQA_THREADS,
                      0, stream,
                      (static_cast<float*>(scores),
                       static_cast<const float*>(bias), total_heads, num_heads,
                       bias_batch, bias_heads, sq, score_cols,
                       score_batch_stride, bias_sq, bias_row_offset,
                       bias_total_seq, bias_col_offset),
                      add_attention_bias_kernel<float>);
  } else {
    fprintf(stderr,
            "hip_gqa_add_attention_bias_f32: unsupported bias elem size %d\n",
            bias_element_size_bytes);
    return -1;
  }
  GQA_RETURN_LAUNCH_STATUS();
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
  GQA_LAUNCH_BY_ELEM(
      TOut, total_head_queries, threads, num_waves * sizeof(float),
      (static_cast<const float*>(input_f32), static_cast<TOut*>(output), rows,
       cols, input_batch_stride, output_batch_stride,
       static_cast<const __half*>(head_sink), num_heads, use_smooth_softmax),
      softmax_f32_to_out_kernel);
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
  HIPDNN_RTC_LAUNCH(gqaRtcFunction, total_head_queries, threads,
                    num_waves * sizeof(float), stream,
                    (static_cast<__half*>(data), rows, cols, batch_stride,
                     static_cast<const __half*>(head_sink), num_heads,
                     use_smooth_softmax),
                    softmax_inplace_kernel);
  GQA_RETURN_LAUNCH_STATUS();
}

// [removed] fused_gqa_decode_kernel / hip_gqa_fused_decode: the legacy
// one-block-per-(batch,head_q) decode kernel has been retired. Decode is now a
// single path -- hip_gqa_flash_decode (FA-2 split-K, scalar|WMMA selected by
// geometry) -- which serves GQA and MHA (HpG==1) alike. See gqa.cpp decode
// dispatch and the SCALAR/WMMA kernels above.

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

// Mirror the decode dispatch ladders below; GQA_DECODE_HPG_LIST drives the
// scalar dispatch switch itself.
#define GQA_DECODE_WMMA_LIST(X)                                                \
  X(128, 4, 32) X(64, 4, 32) X(64, 4, 16) X(64, 8, 32) X(64, 8, 16)
#define GQA_DECODE_HPG_LIST(X) X(1) X(2) X(3) X(4) X(5) X(8) X(16)
#define GQA_DECODE_D_LIST(X) X(256) X(128) X(64)

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
  // Diagnostic: print the config actually launched (works in both lookup and
  // online/autotune modes) so a run can be diffed against the other mode.
  static const bool s_log_cfg = getenv("HIPDNN_EP_GQA_LOG_CONFIG") != nullptr;
  if (s_log_cfg)
    fprintf(stderr,
            "[gqa-cfg] decode B=%d H=%d G=%d d=%d hpg=%d maxseq=%d -> "
            "impl=%s splits=%d bkv=%d\n",
            B, H, G, d, hpg, max_seq, use_wmma ? "wmma" : "scalar", splits,
            cfg.bkv);
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
    // FMT is this function's template parameter, so stringizing it would
    // register the token "FMT"; branching puts the enumerator literal in the
    // name expression instead.
    #define LAUNCH_DW(D_, HPG_, BKV_)                                          \
      do {                                                                     \
        if constexpr (FMT == KvDtype::kF16) {                                  \
          HIPDNN_RTC_LAUNCH(                                                   \
              gqaRtcFunction, grid_wmma, 32, smem_bytes(D_, BKV_), stream,     \
              (hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,           \
               local_window_size, splits),                                     \
              gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kF16>);    \
        } else {                                                               \
          HIPDNN_RTC_LAUNCH(                                                   \
              gqaRtcFunction, grid_wmma, 32, smem_bytes(D_, BKV_), stream,     \
              (hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,           \
               local_window_size, splits),                                     \
              gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kI8>);     \
        }                                                                      \
      } while (0)
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
      do {                                                                     \
        if constexpr (FMT == KvDtype::kF16) {                                  \
          HIPDNN_RTC_LAUNCH(                                                   \
              gqaRtcFunction, grid_main, (HPG_) * WAVE_SIZE, 0, stream,        \
              (hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,           \
               local_window_size, splits),                                     \
              gqa_flash_decode_kernel<D_, HPG_, KvDtype::kF16>);               \
        } else {                                                               \
          HIPDNN_RTC_LAUNCH(                                                   \
              gqaRtcFunction, grid_main, (HPG_) * WAVE_SIZE, 0, stream,        \
              (hQ, hK, hV, kSc, vSc, hP, H, G, max_seq, scale, iSeq,           \
               local_window_size, splits),                                     \
              gqa_flash_decode_kernel<D_, HPG_, KvDtype::kI8>);                \
        }                                                                      \
      } while (0)
    #define LAUNCH_SCALAR(HPG_)                                                \
      do {                                                                     \
        if (d == 256)      LAUNCH_SCALAR_D(256, HPG_);                         \
        else if (d == 128) LAUNCH_SCALAR_D(128, HPG_);                         \
        else               LAUNCH_SCALAR_D(64, HPG_);                          \
      } while (0)
    switch (hpg) {
#define X(HPG_) case HPG_: LAUNCH_SCALAR(HPG_); break;
      GQA_DECODE_HPG_LIST(X)
#undef X
      default: break;  // unsupported HpG is rejected in hip_gqa_flash_decode
    }
    #undef LAUNCH_SCALAR
    #undef LAUNCH_SCALAR_D
  }

  // Reduce: one block per query head, block dim D. seqlens_k / sliding-window /
  // head_sink / smooth_softmax are all folded in here (not the partials).
  dim3 grid_reduce(B * H);
  #define LAUNCH_REDUCE(D_)                                                    \
    HIPDNN_RTC_LAUNCH(gqaRtcFunction, grid_reduce, D_, 0, stream,              \
                      (hP, hO, H, hSink, H, use_smooth_softmax, splits),       \
                      gqa_flash_decode_reduce_kernel<D_>)
  if (d == 256) {
    LAUNCH_REDUCE(256);
  } else if (d == 128) {
    LAUNCH_REDUCE(128);
  } else {
    LAUNCH_REDUCE(64);
  }
  #undef LAUNCH_REDUCE
}

// Not autotuned: every split-K decode config launches the same reduce, so it
// stays registered even when the LUT narrows the decode kernels themselves.
static void appendGqaDecodeReduceNames(std::vector<std::string>& names) {
#define X(D_)                                                                  \
  names.push_back(HIPDNN_RTC_STR(gqa_flash_decode_reduce_kernel<D_>));
  GQA_DECODE_D_LIST(X)
#undef X
}

#if !HIPDNN_GQA_RTC_TUNED_SUBSET
static void appendGqaDecodeNames(std::vector<std::string>& names) {
#define X(D_, HPG_, BKV_)                                                      \
  names.push_back(HIPDNN_RTC_STR(                                              \
      gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kF16>));           \
  names.push_back(HIPDNN_RTC_STR(                                              \
      gqa_flash_decode_wmma_kernel<D_, HPG_, BKV_, KvDtype::kI8>));
  GQA_DECODE_WMMA_LIST(X)
#undef X
#define X(HPG_)                                                                \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<256, HPG_, KvDtype::kF16>));      \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<128, HPG_, KvDtype::kF16>));      \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<64, HPG_, KvDtype::kF16>));       \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<256, HPG_, KvDtype::kI8>));       \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<128, HPG_, KvDtype::kI8>));       \
  names.push_back(                                                             \
      HIPDNN_RTC_STR(gqa_flash_decode_kernel<64, HPG_, KvDtype::kI8>));
  GQA_DECODE_HPG_LIST(X)
#undef X
}
#endif  // !HIPDNN_GQA_RTC_TUNED_SUBSET

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
    static const bool s_log_cfg = getenv("HIPDNN_EP_GQA_LOG_CONFIG") != nullptr;
    if (s_log_cfg)
        fprintf(stderr,
                "[gqa-cfg] prefill v5 B=%d H=%d G=%d d=%d sq=%d skv=%d -> "
                "m_tiles=%d bkv=%d\n",
                B, Hq, G, d, sq, skv, m_tiles, bkv);
    const int rows = m_tiles * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV5SmemBytes(m_tiles, bkv, d);
    const int ablate = g_v5_ablate;

    #define LAUNCH_V5(MT_, BKV_, D_, HW_)                                       \
        HIPDNN_RTC_LAUNCH(gqaRtcFunction, grid, 32, smem, stream,               \
            (Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,        \
             scale, ablate, head_sink, num_heads, smooth_softmax,               \
             local_window_size),                                                \
            gqa_flash_prefill_v5_kernel<MT_, BKV_, D_, HW_>)

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

// Mirrors the (M_TILES, BKV, D, HAS_WINDOW) ladder above.
#define GQA_PREFILL_V5_LIST(X)                                                \
    X(2, 64, 64, true)   X(2, 32, 64, true)                                   \
    X(1, 64, 64, true)   X(1, 32, 64, true)                                   \
    X(2, 64, 64, false)  X(2, 32, 64, false)                                  \
    X(1, 64, 64, false)  X(1, 32, 64, false)                                  \
    X(2, 64, 128, false) X(2, 32, 128, false)                                 \
    X(1, 64, 128, false) X(1, 32, 128, false)

#if !HIPDNN_GQA_RTC_TUNED_SUBSET
static void appendGqaPrefillV5Names(std::vector<std::string>& names) {
#define X(MT_, BKV_, D_, HW_)                                                 \
    names.push_back(                                                          \
        HIPDNN_RTC_STR(gqa_flash_prefill_v5_kernel<MT_, BKV_, D_, HW_>));
    GQA_PREFILL_V5_LIST(X)
#undef X
}
#endif  // !HIPDNN_GQA_RTC_TUNED_SUBSET

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
    static const bool s_log_cfg = getenv("HIPDNN_EP_GQA_LOG_CONFIG") != nullptr;
    if (s_log_cfg)
        fprintf(stderr,
                "[gqa-cfg] prefill v7 B=%d H=%d G=%d d=%d sq=%d skv=%d -> "
                "nw=%d bkv=%d mt=%d\n",
                B, Hq, G, d, sq, skv, nw, bkv, mt);
    const int rows = nw * mt * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV7SmemBytes(nw, bkv, d, mt);
    const int ablate = g_v7_ablate;

    #define LAUNCH_V7(NW_, BKV_, D_, MT_)                                       \
        HIPDNN_RTC_LAUNCH(gqaRtcFunction, grid, NW_ * 32, smem, stream,         \
            (Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,        \
             scale, ablate),                                                    \
            gqa_flash_prefill_v7_kernel<NW_, BKV_, D_, MT_>)
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

// Mirrors the (NW, BKV) ladder of DISPATCH_V7_D across its D and MT arguments.
#define GQA_PREFILL_V7_LIST(X)                                                \
    X(4, 64, 64, 1)  X(4, 32, 64, 1)  X(2, 64, 64, 1)                         \
    X(2, 32, 64, 1)  X(1, 64, 64, 1)  X(1, 32, 64, 1)                         \
    X(4, 64, 64, 2)  X(4, 32, 64, 2)  X(2, 64, 64, 2)                         \
    X(2, 32, 64, 2)  X(1, 64, 64, 2)  X(1, 32, 64, 2)                         \
    X(4, 64, 128, 1) X(4, 32, 128, 1) X(2, 64, 128, 1)                        \
    X(2, 32, 128, 1) X(1, 64, 128, 1) X(1, 32, 128, 1)                        \
    X(4, 64, 128, 2) X(4, 32, 128, 2) X(2, 64, 128, 2)                        \
    X(2, 32, 128, 2) X(1, 64, 128, 2) X(1, 32, 128, 2)                        \
    X(4, 64, 256, 1) X(4, 32, 256, 1) X(2, 64, 256, 1)                        \
    X(2, 32, 256, 1) X(1, 64, 256, 1) X(1, 32, 256, 1)                        \
    X(4, 64, 256, 2) X(4, 32, 256, 2) X(2, 64, 256, 2)                        \
    X(2, 32, 256, 2) X(1, 64, 256, 2) X(1, 32, 256, 2)

#if !HIPDNN_GQA_RTC_TUNED_SUBSET
static void appendGqaPrefillV7Names(std::vector<std::string>& names) {
#define X(NW_, BKV_, D_, MT_)                                                 \
    names.push_back(                                                          \
        HIPDNN_RTC_STR(gqa_flash_prefill_v7_kernel<NW_, BKV_, D_, MT_>));
    GQA_PREFILL_V7_LIST(X)
#undef X
}
#endif  // !HIPDNN_GQA_RTC_TUNED_SUBSET

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
    static const bool s_log_cfg = getenv("HIPDNN_EP_GQA_LOG_CONFIG") != nullptr;
    if (s_log_cfg)
        fprintf(stderr,
                "[gqa-cfg] prefill v8 B=%d H=%d G=%d d=%d sq=%d skv=%d -> "
                "nd=%d mt=%d bkv=%d\n",
                B, Hq, G, d, sq, skv, nd, mt, bkv);
    const int rows = mt * 16;
    const int num_q_tiles = (sq + rows - 1) / rows;
    dim3 grid(num_q_tiles * B, Hq);
    const size_t smem = prefillV8SmemBytes(nd, mt, bkv, d);
    const int ablate = g_v8_ablate;

    #define LAUNCH_V8(ND_, MT_, BKV_)                                           \
        HIPDNN_RTC_LAUNCH(gqaRtcFunction, grid, ND_ * 32, smem, stream,         \
            (Q, Kcache, Vcache, O, B, Hq, G, sq, skv, max_seq, past_len,        \
             scale, ablate),                                                    \
            gqa_flash_prefill_v8_kernel<ND_, MT_, BKV_, 256>)

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

// Mirrors the (ND, MT, BKV) ladder above; D is always 256 there.
#define GQA_PREFILL_V8_LIST(X)                                                \
    X(2, 1, 32) X(2, 2, 32) X(2, 1, 64) X(2, 2, 64)                           \
    X(4, 1, 32) X(4, 2, 32) X(2, 1, 16) X(4, 1, 16)                           \
    X(8, 1, 16) X(8, 1, 32) X(2, 2, 16) X(4, 2, 16)                           \
    X(1, 1, 16) X(1, 2, 16)

#if !HIPDNN_GQA_RTC_TUNED_SUBSET
static void appendGqaPrefillV8Names(std::vector<std::string>& names) {
#define X(ND_, MT_, BKV_)                                                     \
    names.push_back(                                                          \
        HIPDNN_RTC_STR(gqa_flash_prefill_v8_kernel<ND_, MT_, BKV_, 256>));
    GQA_PREFILL_V8_LIST(X)
#undef X
}
#endif  // !HIPDNN_GQA_RTC_TUNED_SUBSET

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
        {2, 1, 16}, {4, 1, 16}, {4, 1, 32}, {8, 1, 16},
        // dispatchPrefillV8's default launch and what the offline LUT selects
        // for D256; keep it here so the configured entry accepts that row.
        {2, 1, 32}};
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
  #define LAUNCH_LEGACY(D_)                                                    \
    HIPDNN_RTC_LAUNCH(gqaRtcFunction, grid, D_, 0, stream,                     \
                      (hQ, hK, hV, hO, H, G, d, skv, max_seq, scale,           \
                       iSeqlensK),                                             \
                      legacy_fused_gqa_decode_kernel<D_>)
  switch (d) {
    case 64:
      LAUNCH_LEGACY(64);
      break;
    case 128:
      LAUNCH_LEGACY(128);
      break;
    case 256:
      LAUNCH_LEGACY(256);
      break;
    default:
      fprintf(stderr, "hip_gqa_fused_decode: unsupported d=%d "
              "(must be 64, 128, or 256)\n", d);
      return -1;
  }
  #undef LAUNCH_LEGACY
  GQA_RETURN_LAUNCH_STATUS();
}

// ---------------------------------------------------------------------------
// hipRTC module
//
// Every name expression is registered before the single compile: hipRTC
// resolves them while parsing.
// ---------------------------------------------------------------------------

static void appendGqaElemNames(std::vector<std::string>& names) {
#define X(K)                                                                   \
  names.push_back(HIPDNN_RTC_STR(K<__half>));                                  \
  names.push_back(HIPDNN_RTC_STR(K<float>));
  GQA_ELEM_KERNEL_LIST(X)
#undef X
#define X(K)                                                                   \
  names.push_back(HIPDNN_RTC_STR(K<__half, 8>));                               \
  names.push_back(HIPDNN_RTC_STR(K<float, 4>));
  GQA_ELEM_VEC_KERNEL_LIST(X)
#undef X
}

// The kernels launched outside a list-driven family: one instance each, so the
// name expression is spelled here rather than generated.
static void appendGqaMiscNames(std::vector<std::string>& names) {
  names.push_back(HIPDNN_RTC_STR(kv_cache_append_quant_i8_kernel));
  names.push_back(HIPDNN_RTC_STR(kv_cache_concat_quant_i8_kernel));
  names.push_back(HIPDNN_RTC_STR(dequant_kv_i8_to_fp16_kernel));
  names.push_back(HIPDNN_RTC_STR(softmax_inplace_kernel));
#define X(D_)                                                                  \
  names.push_back(HIPDNN_RTC_STR(legacy_fused_gqa_decode_kernel<D_>));
  GQA_DECODE_D_LIST(X)
#undef X
}

// The autotuned families register only what the offline LUT picks; a config
// left out resolves to nullptr and HIPDNN_RTC_LAUNCH takes its AOT kernel.
static void appendGqaTunedNames(std::vector<std::string>& names) {
#if HIPDNN_GQA_RTC_TUNED_SUBSET
#define X(NAME_) names.push_back(NAME_);
  HIPDNN_GQA_RTC_TUNED_NAME_LIST(X)
#undef X
#else
  appendGqaDecodeNames(names);
  appendGqaPrefillV5Names(names);
  appendGqaPrefillV7Names(names);
  appendGqaPrefillV8Names(names);
#endif
}

static hipdnn_ep::rtc::Module& gqaRtcModule() {
  static hipdnn_ep::rtc::Module module(
      "gqa", reinterpret_cast<const char*>(kGqaDeviceSource),
      kGqaDeviceSource_size, [] {
        std::vector<std::string> names;
        appendGqaElemNames(names);
        appendGqaMiscNames(names);
        appendGqaDecodeReduceNames(names);
        appendGqaTunedNames(names);
        return names;
      }());
  return module;
}

static hipFunction_t gqaRtcFunction(const char* name_expression) {
  if (!hipdnn_ep::rtc::enabled()) {
    return nullptr;
  }
  return gqaRtcModule().getFunction(name_expression);
}
