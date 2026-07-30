/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GQA runtime wrapper (self-contained: optimized fused fast path + legacy
// decomposed hipBLASLt fallback).
//
// The generated IR calls `wrap_group_query_attention` (39-arg ABI, unchanged so
// the HipToLLVM lowering keeps resolving). Path selection:
//
//   * Common fp16 causal GQA (head_dim in {64,128}, templated decode geometry,
//     no sliding window) -> optimized fused custom kernels:
//       prefill (sq > 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_prefill_v3
//       decode  (sq == 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_decode_v2
//     Attention sinks / smooth softmax are applied here too: by flash decode
//     for any supported geometry, and by flash prefill at head_dim == 64 (the
//     v5 kernel). Prefill at head_dim 128/256 with a sink still goes decomposed.
//
//   * Everything else the fused kernels do not implement (fp32, no_causal /
//     bidirectional, sliding window, a sink on 128/256-wide prefill, additive
//     attention bias, other head_dim, untemplated decode geometry) -> the
//     feature-complete legacy decomposed hipBLASLt pipeline
//     gqa_forward_hipblaslt below. This is a verbatim port of the proven
//     gqa_back.cpp strategy (the read-only backup stays out of the build); its
//     fast decode kernels (hip_gqa_fused_decode / hip_gqa_flash_decode) are
//     folded into gqa_kernel.hip so the sliding-window / sink decode case keeps
//     the legacy kernel's performance.
//
//   * The additive attention bias (onnx.Attention attn_mask) IS supported, but
//     only by the decomposed path (Step 8b adds it; a causal op then masks the
//     triangle in Step 9), so its presence forces the decomposed path.
//   * Symmetric per-channel INT8 KV cache (k/v_quant_type=PER_CHANNEL,
//     kv_cache_bit_width=8, static fp32 k/v_scale [G,d]) IS supported on the
//     fused path: new tokens are quantize-appended into the int8 cache; decode
//     reads int8 directly (bandwidth win), prefill dequantizes the cache
//     to fp16 once and reuses the tuned fp16 prefill (compute-bound ->
//     ~parity).
//   * Inputs NEITHER path supports (other KV quantization, position ids,
//     qk_output) are rejected up front.
//===----------------------------------------------------------------------===//

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// Legacy fast-path decode kernels (folded into gqa_kernel.hip as legacy_*
// device kernels) back the decomposed hipBLASLt fallback below. Their entries
// hip_gqa_fused_decode / hip_gqa_flash_decode are declared (and HIP_KERNEL_API
// exported) in hip_custom_kernels.h, so the EP resolves them out of
// custom_kernels_<arch> at JIT link / native import.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dispatch helpers (shared by the fused and decomposed paths)
//===----------------------------------------------------------------------===//

// Env-var gate to cache seqlens_k_val across the GQA layers in a single
// forward pass. Default ON. Caching skips the per-layer
// hipMemcpyAsync(D2H) + hipStreamSynchronize on both paths after the first
// GQA call -- a 32-layer Llama decode then issues one D2H instead of 32,
// eliminating ~30-45 ms/token of pipeline stalls on Strix Halo. Set
// HIPDNN_EP_GQA_CACHE_SEQLENS=0 to disable (escape hatch for running against
// an older per-model bitcode without the begin_compute export, or for A/B
// measurement).
//
// Correctness depends on the EP-side MlirCustomOp::Compute() invoking
// hipdnn_ep_runtime_begin_compute(state) at the start of each forward pass to
// invalidate the cache. Older per-model bitcode without that symbol exported
// is detected at session creation and produces a LOG(WARNING) directing the
// user to set HIPDNN_EP_GQA_CACHE_SEQLENS=0 (otherwise the cache would survive
// across forward passes and return stale total_seq values).
static bool gqa_cache_seqlens_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    // Default on; explicit "0" disables.
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Sentinel returned by read_seqlens_k_for_dispatch when the pre-dispatch read
// is not applicable (multi-batch or missing seqlens_k_ptr) or failed (D2H copy
// / stream sync error). Outside the valid range of real seqlens_k values (-1 is
// ORT's prefill sentinel; 0..max_seq are real). Callers must treat this as "no
// pre-read available" and fall back to the legacy per-call D2H readback site
// they already implement.
static constexpr int32_t kSeqlensKNotRead = -2;

// Read seqlens_k_val from device (or the per-Compute() cache when
// HIPDNN_EP_GQA_CACHE_SEQLENS=1) once per call before the fused/decomposed
// dispatch decision. Two purposes:
//   1. Give the smart-dispatch heuristic access to total_seq for the decode
//      case (sq == 1) so it can compare against gqa_fused_decode_max_t().
//   2. Populate the per-Compute() cache for B == 1 so subsequent GQA layers
//      within the same forward pass reuse the value with zero D2H.
//
// Applies to B == 1 regardless of sq (both prefill and decode share the same
// seqlens_k pointer and benefit from caching). On B != 1 we return
// kSeqlensKNotRead because per-batch validation in the multi-batch path
// requires reading every entry; the legacy readback site there handles it.
//
// Behaviour:
//   - cache enabled and hit:  zero D2H, return cached value.
//   - cache enabled and miss: one D2H + sync, populate cache, return.
//   - cache disabled:         one D2H + sync, return (no cache write).
//   - B != 1, !seqlens_k_ptr, or D2H/sync failure: return kSeqlensKNotRead.
//
// The returned int32_t is the raw device value: -1 is ORT's prefill sentinel
// (callers map it to total_seq=sq, past_len=0); 0..max_seq is the live
// (total_seq - 1).
static int32_t read_seqlens_k_for_dispatch(hipStream_t stream,
                                           const void *seqlens_k_ptr, int64_t B,
                                           RuntimeState *state) {
  if (!seqlens_k_ptr || B != 1)
    return kSeqlensKNotRead;

  if (gqa_cache_seqlens_enabled() && state && state->seqlens_k_cached_valid &&
      state->seqlens_k_cached_ptr == seqlens_k_ptr)
    return state->seqlens_k_cached_val;

  int32_t seqlens_k_val = 0;
  if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                     hipMemcpyDeviceToHost, stream) != hipSuccess)
    return kSeqlensKNotRead;
  if (hipStreamSynchronize(stream) != hipSuccess)
    return kSeqlensKNotRead;

  if (gqa_cache_seqlens_enabled() && state) {
    state->seqlens_k_cached_val = seqlens_k_val;
    state->seqlens_k_cached_ptr = seqlens_k_ptr;
    state->seqlens_k_cached_valid = true;
  }
  return seqlens_k_val;
}

// FA-2 split-K decode workspace capacity, in splits (matches gqa_kernel.hip).
static constexpr int kFlashDecodeMaxSplits = 64;

// Geometry gate for the flash_decode kernel template instantiations. The scalar
// decode kernel is templated for HpG in {1,2,3,4,5,8,16} (so it covers MHA and
// the common GQA ratios, incl. Qwen2.5-14B's 40:8=5) at d in {64,128,256}
// (d=256 covers Qwen3-family 16:4 heads); WMMA is layered on top inside the
// kernel where it helps (d<=128 only). Anything outside this set has no decode
// kernel.
static inline bool flash_decode_geometry_ok(int64_t H, int64_t G, int64_t d) {
  if (G <= 0)
    return false;
  if (d != 64 && d != 128 && d != 256)
    return false;
  int64_t hpg = H / G;
  if (hpg * G != H)
    return false;
  return hpg == 1 || hpg == 2 || hpg == 3 || hpg == 4 || hpg == 5 || hpg == 8 ||
         hpg == 16;
}

// Logical storage format of the KV cache the fused path must read/write. This
// is the single vocabulary the rest of gqa.cpp branches on; adding a new
// quantized cache (e.g. INT4 per-channel, FP8) means adding an entry here, a
// branch in classify_kv_cache(), a case in kv_dtype_abi() and the matching
// kernel specialization downstream -- NOT a new boolean threaded through every
// signature, and not scattering `k_quant_type == N` checks through wrap_*.
enum class KvCacheFormat {
  Fp16,           // unquantized (default path)
  Int8PerChannel, // symmetric per-channel int8, static fp32 scales [G,d]
};

// Map the runtime KvCacheFormat onto the kernel ABI dtype code (hip_kv_dtype_t
// in hip_custom_kernels.h). Single choke point: one case per format, so the
// kernel entries stay dtype-driven (switch) rather than
// boolean/pointer-sniffing.
static inline int kv_dtype_abi(KvCacheFormat f) {
  switch (f) {
  case KvCacheFormat::Int8PerChannel:
    return HIP_KV_DTYPE_INT8;
  case KvCacheFormat::Fp16:
  default:
    return HIP_KV_DTYPE_FP16;
  }
}

//===----------------------------------------------------------------------===//
// KV cache update: concat past+new, append new tokens in place, or (no_causal)
// the bidirectional copy/append branch used by the decomposed pipeline.
//===----------------------------------------------------------------------===//
// past_buf_seq is the buffer dim of past_key (may exceed past_len for
// pre-allocated caches). seqlens_k_ptr: when non-null on the append path the
// kernel reads past_len from device memory (zero D2H). The fused path calls
// this with the default no_causal=false / skv=-1; the decomposed pipeline
// passes them for the Whisper no_causal cases. Returns 0 on success.
//
// kv_format: for any quantized format (Int8PerChannel today, INT4/FP8 later)
// the (causal) concat/append quantizes the incoming fp16 K/V with the static
// per-channel k_scale/v_scale into the quantized present cache instead of a
// plain copy. Only the causal concat/append tail honours it (the no_causal
// Whisper branches are fp16/fp32-only), so the format decision lives here
// rather than in a separate helper or at each call site.
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int past_buf_seq, int present_seq,
                           const void *seqlens_k_ptr, int elem_sz,
                           bool no_causal = false, int skv = -1,
                           KvCacheFormat kv_format = KvCacheFormat::Fp16,
                           const void *k_scale = nullptr,
                           const void *v_scale = nullptr) {
  // no_causal (Whisper encoder / cross-attn): bidirectional, no past KV.
  // The KV to attend over is the FULL `new_key`/`new_value` (Skv tokens), not
  // `sq` newly-appended tokens. Two sub-cases distinguished by sq vs Skv:
  //   * Cross-attn (sq != Skv): `key`/`value` arrive as rank-4 BNSD
  //     [B, G, Skv, d] -- already in the present_key layout. A straight
  //     device-to-device copy of all Skv tokens populates present_*.
  //   * Encoder self-attn (sq == Skv): `key`/`value` are BSHD [B, Skv, G, d];
  //     fall through to the append kernel below with past_len forced to 0 and
  //     seqlens_k=nullptr so it transposes all Skv tokens to offset 0 WITHOUT
  //     the +1 PAST-token convention.
  if (no_causal && skv >= 0 && skv != sq) {
    // Cross-attn: key/value arrive rank-4 BNSD [B,G,skv,d]; straight D2D copy.
    // elem_sz is 2 (fp16) or 4 (fp32); the decomposed pipeline supports both.
    size_t bytes = static_cast<size_t>(B) * G * static_cast<size_t>(skv) * d *
                   static_cast<size_t>(elem_sz);
    if (hipMemcpyAsync(present_key, new_key, bytes, hipMemcpyDeviceToDevice,
                       stream) != hipSuccess)
      return -1;
    if (hipMemcpyAsync(present_value, new_value, bytes, hipMemcpyDeviceToDevice,
                       stream) != hipSuccess)
      return -1;
    return 0;
  }
  if (no_causal) {
    // Encoder self-attn: append all Skv (== sq) tokens at offset 0, bypassing
    // the seqlens_k +1 convention (pass nullptr so the kernel uses past_len=0).
    // no_causal is fp16/fp32 only (decomposed pipeline), never quantized.
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, /*past_len=*/0,
                                /*seqlens_k_ptr=*/nullptr, elem_sz,
                                HIP_KV_DTYPE_FP16, /*scale=*/nullptr) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, /*past_len=*/0,
                                /*seqlens_k_ptr=*/nullptr, elem_sz,
                                HIP_KV_DTYPE_FP16, /*scale=*/nullptr) != 0)
      return -1;
    return 0;
  }
  // Quantized cache: pass the per-channel scale so the kernel quantizes on
  // write; fp16/fp32: pass nullptr for a plain copy. The append/concat entries
  // dispatch on kv_dtype, so no separate per-format code path is needed here.
  const int kv_dtype = kv_dtype_abi(kv_format);
  const bool quantized = (kv_format != KvCacheFormat::Fp16);
  const void *k_sc = quantized ? k_scale : nullptr;
  const void *v_sc = quantized ? v_scale : nullptr;
  if (past_key && past_len > 0 && past_key != present_key) {
    // Separate-buffer concat: needs host-side past_len for stride computation.
    if (hip_gqa_kv_cache_concat(stream, past_key, new_key, present_key, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz, kv_dtype, k_sc) != 0)
      return -1;
    if (hip_gqa_kv_cache_concat(stream, past_value, new_value, present_value, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz, kv_dtype, v_sc) != 0)
      return -1;
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr.
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr, elem_sz,
                                kv_dtype, k_sc) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr, elem_sz,
                                kv_dtype, v_sc) != 0)
      return -1;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Fused-only forward: fp16, causal, GQA. No hipBLASLt, no decomposed fallback.
//===----------------------------------------------------------------------===//
static int gqa_forward_fused(
    RuntimeState *state, hipStream_t stream,
    const void *query,    // BSHD [B, sq, H, d] or packed [B, sq, (H+2G)*d]
    const void *key,      // BSHD [B, sq, G, d] or null (packed QKV)
    const void *value,    // BSHD [B, sq, G, d] or null (packed QKV)
    const void *past_key, // BNSD [B, G, past_buf_seq, d] or null
    const void *past_value, const void *seqlens_k_ptr, const void *cos_cache,
    const void *sin_cache, void *output, void *present_key, void *present_value,
    int64_t B, int64_t sq, int64_t skv, int64_t past_buf_seq, int64_t H,
    int64_t G, int64_t d, float scale, int64_t do_rotary,
    const void *k_scale = nullptr, const void *v_scale = nullptr,
    KvCacheFormat kv_format = KvCacheFormat::Fp16,
    const void *head_sink = nullptr, bool use_smooth_softmax = false) {

  // Any non-fp16 cache reads/writes quantized bytes on the concat/append/decode
  // path; the specific scheme is carried by kv_format (extensible to INT4/FP8).
  const bool kv_quantized = (kv_format != KvCacheFormat::Fp16);
  const int64_t present_seq = skv; // present_key buffer stride (may be max_seq)
  const size_t elem_sz = 2;        // fp16 only on the fused path
  const bool need_rope = do_rotary && cos_cache && sin_cache;
  const bool packed_qkv = (!key && !value);

  // B==1 pre-read (cached per Compute) feeds host past_len where needed.
  const int32_t seqlens_k_pre =
      read_seqlens_k_for_dispatch(stream, seqlens_k_ptr, B, state);

  const size_t Q_full_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
  const size_t K_full_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;

  //===------------------------------------------------------------------===//
  // Decode (sq == 1)
  //===------------------------------------------------------------------===//
  if (sq == 1) {
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // past_len only needed host-side for the concat branch (separate buffers);
    // in-place caches let the kernels read it from device.
    int64_t past_len = 0;
    const bool need_host_past_len =
        seqlens_k_ptr && past_key && past_key != present_key;
    if (need_host_past_len) {
      int32_t seqlens_k_val = 0;
      if (seqlens_k_pre != kSeqlensKNotRead) {
        seqlens_k_val = seqlens_k_pre;
      } else {
        if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                           hipMemcpyDeviceToHost, stream) != hipSuccess)
          return -1;
        if (hipStreamSynchronize(stream) != hipSuccess)
          return -1;
      }
      if (seqlens_k_val < 0) {
        past_len = 0; // ORT prefill sentinel
      } else {
        int64_t total_seq = static_cast<int64_t>(seqlens_k_val) + 1;
        int64_t past_len_check = total_seq - sq;
        if (total_seq < 1 || past_len_check < 0 || total_seq > present_seq ||
            past_len_check > past_buf_seq) {
          fprintf(stderr,
                  "gqa_forward_fused (decode): invalid seqlens_k[0]+1=%lld "
                  "(sq=%lld, past_len=%lld, present_seq=%lld, "
                  "past_buf_seq=%lld)\n",
                  (long long)total_seq, (long long)sq,
                  (long long)past_len_check, (long long)present_seq,
                  (long long)past_buf_seq);
          return -1;
        }
        past_len = past_len_check;
      }
    } else if (!seqlens_k_ptr) {
      past_len = skv - sq;
    }
    if (past_len < 0)
      past_len = 0;

    // Decode has a SINGLE path: hip_gqa_flash_decode_v2. It selects WMMA (D64/
    // HpG>=8) vs scalar internally and serves GQA and MHA (HpG==1) alike, by
    // GEOMETRY only -- never by KV depth. There is no legacy fused fallback;
    // geometries the kernel cannot template are rejected here.
    if (!flash_decode_geometry_ok(H, G, d)) {
      fprintf(stderr,
              "gqa_forward_fused (decode): unsupported geometry H=%lld G=%lld "
              "d=%lld (HpG must be 1/2/3/4/5/8/16 and d 64/128/256)\n",
              (long long)H, (long long)G, (long long)d);
      return -1;
    }

    // One combined workspace request: [split? | rope-temp? | flash-partials].
    const size_t split_bytes =
        packed_qkv ? (Q_full_bytes + K_full_bytes + K_full_bytes) : 0;
    const size_t rope_temp_bytes =
        need_rope ? (Q_full_bytes + K_full_bytes) : 0;
    const size_t flash_partials_bytes = static_cast<size_t>(B) * H *
                                        kFlashDecodeMaxSplits * (d + 2) *
                                        sizeof(float);
    const size_t total_ws_bytes =
        split_bytes + rope_temp_bytes + flash_partials_bytes;
    if (total_ws_bytes > 0 &&
        hipdnn_ep_state_ensure_workspace(state, total_ws_bytes) != 0)
      return -1;

    const size_t off_split = 0;
    const size_t off_rope = off_split + split_bytes;
    const size_t off_partials = off_rope + rope_temp_bytes;

    if (packed_qkv) {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *d_Qsplit = ws + off_split;
      void *d_Ksplit = ws + off_split + Q_full_bytes;
      void *d_Vsplit = ws + off_split + Q_full_bytes + K_full_bytes;
      if (hip_gqa_split_qkv(
              stream, query, d_Qsplit, d_Ksplit, d_Vsplit, static_cast<int>(B),
              static_cast<int>(sq), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(elem_sz)) != 0)
        return -1;
      qSrc = d_Qsplit;
      kSrc = d_Ksplit;
      vSrc = d_Vsplit;
    }

    if (need_rope) {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *d_Qroped = ws + off_rope;
      void *d_Kroped = ws + off_rope + Q_full_bytes;
      int half_rot = static_cast<int>(d / 2);
      if (hip_gqa_rope(stream, qSrc, d_Qroped, cos_cache, sin_cache,
                       static_cast<int>(B), static_cast<int>(sq),
                       static_cast<int>(H), static_cast<int>(d), half_rot,
                       static_cast<int>(past_len), seqlens_k_ptr,
                       static_cast<int>(elem_sz)) != 0)
        return -1;
      if (hip_gqa_rope(stream, kSrc, d_Kroped, cos_cache, sin_cache,
                       static_cast<int>(B), static_cast<int>(sq),
                       static_cast<int>(G), static_cast<int>(d), half_rot,
                       static_cast<int>(past_len), seqlens_k_ptr,
                       static_cast<int>(elem_sz)) != 0)
        return -1;
      qSrc = d_Qroped;
      kSrc = d_Kroped; // V is never RoPE'd.
    }

    // Append the new token to the KV cache. Quantized cache:
    // quantize-and-append with the static per-channel scale; fp16 cache: plain
    // append. update_kv_cache dispatches on kv_format internally.
    if (update_kv_cache(
            stream, past_key, past_value, kSrc, vSrc, present_key,
            present_value, static_cast<int>(B), static_cast<int>(past_len),
            static_cast<int>(sq), static_cast<int>(G), static_cast<int>(d),
            static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
            seqlens_k_ptr, static_cast<int>(elem_sz), /*no_causal=*/false,
            /*skv=*/-1, kv_format, k_scale, v_scale) != 0)
      return -1;

    {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *partials = ws + off_partials;
      // Decode is bandwidth-bound: a quantized cache is read directly (e.g.
      // int8 halves the DRAM traffic). One entry serves every format --
      // kv_dtype selects the code path inside the kernel; scales feed dequant.
      int drc = hip_gqa_flash_decode_v2(
          stream, qSrc, present_key, present_value, output, partials,
          static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
          static_cast<int>(d), static_cast<int>(skv),
          static_cast<int>(present_seq), kFlashDecodeMaxSplits, scale,
          seqlens_k_ptr, /*local_window_size=*/0, head_sink,
          use_smooth_softmax ? 1 : 0, kv_dtype_abi(kv_format),
          kv_quantized ? k_scale : nullptr, kv_quantized ? v_scale : nullptr);
      if (drc != 0)
        return -1;
      RUNTIME_DEBUG_LOG(
          "[REAL] flash GQA decode (%s): B=%lld skv=%lld H=%lld G=%lld "
          "d=%lld max_splits=%d\n",
          kv_quantized ? "quant" : "fp16", (long long)B, (long long)skv,
          (long long)H, (long long)G, (long long)d, kFlashDecodeMaxSplits);
    }
    return 0;
  }

  //===------------------------------------------------------------------===//
  // Prefill (sq > 1)
  //===------------------------------------------------------------------===//
  int64_t total_seq = skv;
  int64_t past_len = skv - sq;
  if (seqlens_k_ptr) {
    int32_t seqlens_k_val = 0;
    if (seqlens_k_pre != kSeqlensKNotRead) {
      seqlens_k_val = seqlens_k_pre;
    } else if (B > 1) {
      std::vector<int32_t> seqlens_k_host(B);
      if (hipMemcpyAsync(seqlens_k_host.data(), seqlens_k_ptr,
                         B * sizeof(int32_t), hipMemcpyDeviceToHost,
                         stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
      seqlens_k_val = seqlens_k_host[0];
      for (int64_t b = 1; b < B; ++b) {
        if (seqlens_k_host[b] != seqlens_k_val) {
          fprintf(stderr,
                  "gqa_forward_fused: per-batch seqlens_k not supported "
                  "(batch %lld has %d, batch 0 has %d)\n",
                  (long long)b, seqlens_k_host[b], seqlens_k_val);
          return -1;
        }
      }
    } else {
      if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                         hipMemcpyDeviceToHost, stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
    }
    if (seqlens_k_val < 0) {
      total_seq = sq; // ORT prefill sentinel
      past_len = 0;
    } else {
      total_seq = static_cast<int64_t>(seqlens_k_val) + 1;
      past_len = total_seq - sq;
      if (total_seq < 1 || past_len < 0 || total_seq > present_seq ||
          past_len > past_buf_seq) {
        fprintf(
            stderr,
            "gqa_forward_fused (prefill): invalid seqlens_k[0]+1=%lld "
            "(sq=%lld, past_len=%lld, present_seq=%lld, past_buf_seq=%lld)\n",
            (long long)total_seq, (long long)sq, (long long)past_len,
            (long long)present_seq, (long long)past_buf_seq);
        return -1;
      }
    }
  }
  if (past_len < 0)
    past_len = 0;

  const void *qSrc = query;
  const void *kSrc = key;
  const void *vSrc = value;

  // Workspace: [split? | rope-temp? | i8-dequant K/V?]. The fp16 flash_prefill
  // reads the BNSD present cache + BSHD roped Q directly, so it needs no extra
  // scratch. For the INT8 cache we additionally dequantize the [0,total_seq)
  // cache range ONCE into an fp16 BNSD scratch (K and V) and run the tuned fp16
  // prefill on it -- prefill is compute-bound, so dequantizing per WMMA
  // fragment (re-done for every query tile) would be pure overhead; a single
  // dequant pass amortizes to ~parity with fp16.
  const size_t split_bytes =
      packed_qkv ? (Q_full_bytes + K_full_bytes + K_full_bytes) : 0;
  const size_t rope_temp_bytes = need_rope ? (Q_full_bytes + K_full_bytes) : 0;
  // 2 bytes/elem (fp16) x 2 (K and V).
  const size_t deq_kv_bytes =
      kv_quantized ? static_cast<size_t>(B) * G * total_seq * d * 2 * 2 : 0;
  const size_t total_ws_bytes = split_bytes + rope_temp_bytes + deq_kv_bytes;
  if (total_ws_bytes > 0 &&
      hipdnn_ep_state_ensure_workspace(state, total_ws_bytes) != 0)
    return -1;
  const size_t off_split = 0;
  const size_t off_rope = off_split + split_bytes;
  const size_t off_deq = off_rope + rope_temp_bytes;

  if (packed_qkv) {
    char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    void *d_Qsplit = ws + off_split;
    void *d_Ksplit = ws + off_split + Q_full_bytes;
    void *d_Vsplit = ws + off_split + Q_full_bytes + K_full_bytes;
    if (hip_gqa_split_qkv(stream, query, d_Qsplit, d_Ksplit, d_Vsplit,
                          static_cast<int>(B), static_cast<int>(sq),
                          static_cast<int>(H), static_cast<int>(G),
                          static_cast<int>(d), static_cast<int>(elem_sz)) != 0)
      return -1;
    qSrc = d_Qsplit;
    kSrc = d_Ksplit;
    vSrc = d_Vsplit;
  }

  if (need_rope) {
    char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    void *d_Qroped = ws + off_rope;
    void *d_Kroped = ws + off_rope + Q_full_bytes;
    int half_rot = static_cast<int>(d / 2);
    if (hip_gqa_rope(stream, qSrc, d_Qroped, cos_cache, sin_cache,
                     static_cast<int>(B), static_cast<int>(sq),
                     static_cast<int>(H), static_cast<int>(d), half_rot,
                     static_cast<int>(past_len), nullptr,
                     static_cast<int>(elem_sz)) != 0)
      return -1;
    if (hip_gqa_rope(stream, kSrc, d_Kroped, cos_cache, sin_cache,
                     static_cast<int>(B), static_cast<int>(sq),
                     static_cast<int>(G), static_cast<int>(d), half_rot,
                     static_cast<int>(past_len), nullptr,
                     static_cast<int>(elem_sz)) != 0)
      return -1;
    qSrc = d_Qroped;
    kSrc = d_Kroped; // V is never RoPE'd.
  }

  if (present_key && present_value) {
    // Quantized cache: quantize-and-append/concat; fp16: plain. update_kv_cache
    // dispatches on kv_format internally.
    if (update_kv_cache(
            stream, past_key, past_value, kSrc, vSrc, present_key,
            present_value, static_cast<int>(B), static_cast<int>(past_len),
            static_cast<int>(sq), static_cast<int>(G), static_cast<int>(d),
            static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
            seqlens_k_ptr, static_cast<int>(elem_sz), /*no_causal=*/false,
            /*skv=*/-1, kv_format, k_scale, v_scale) != 0)
      return -1;
  }

  // Choose the KV the prefill attends over. fp16 cache: read present directly.
  // Quantized cache: dequantize [0,total_seq) into a compact fp16 BNSD scratch
  // ONCE and feed the tuned fp16 prefill (max_seq = total_seq). This attends
  // over the exact rounded values decode will later read, so prefill/decode
  // stay numerically consistent, and keeps prefill at ~fp16 speed (no
  // per-fragment dequant). (INT8 today; a new quantized format adds its dequant
  // here.)
  const void *kAttn = present_key;
  const void *vAttn = present_value;
  int attn_max_seq = static_cast<int>(present_seq);
  if (kv_quantized) {
    char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    void *d_Kf16 = ws + off_deq;
    void *d_Vf16 =
        ws + off_deq + static_cast<size_t>(B) * G * total_seq * d * 2;
    if (hip_gqa_dequant_kv_i8_to_fp16(
            stream, present_key, d_Kf16, k_scale, static_cast<int>(B),
            static_cast<int>(total_seq), static_cast<int>(G),
            static_cast<int>(d), static_cast<int>(present_seq),
            static_cast<int>(total_seq)) != 0)
      return -1;
    if (hip_gqa_dequant_kv_i8_to_fp16(
            stream, present_value, d_Vf16, v_scale, static_cast<int>(B),
            static_cast<int>(total_seq), static_cast<int>(G),
            static_cast<int>(d), static_cast<int>(present_seq),
            static_cast<int>(total_seq)) != 0)
      return -1;
    kAttn = d_Kf16;
    vAttn = d_Vf16;
    attn_max_seq = static_cast<int>(total_seq);
  }

  // Single unified entry; v5 (d==64) / v7 (d==128) selection lives in the
  // kernel TU (gqa_kernel.hip). v3 carries the sink; it returns -1 rather than
  // dropping it when the kernel for this d cannot apply it, and the caller then
  // falls back to the decomposed path.
  int fp_rc = hip_gqa_flash_prefill_v3(
      stream, qSrc, kAttn, vAttn, output, static_cast<int>(B),
      static_cast<int>(H), static_cast<int>(G), static_cast<int>(sq),
      static_cast<int>(total_seq), static_cast<int>(d), attn_max_seq,
      static_cast<int>(past_len), scale, /*local_window_size=*/-1, head_sink,
      static_cast<int>(H), use_smooth_softmax ? 1 : 0);
  RUNTIME_DEBUG_LOG(
      "[REAL] GQA fused prefill (%s d=%lld -> v%d): B=%lld sq=%lld "
      "total_seq=%lld H=%lld G=%lld past_len=%lld sink=%d smooth=%d rc=%d\n",
      kv_quantized ? "quant" : "fp16", (long long)d, (d == 64 ? 5 : 7),
      (long long)B, (long long)sq, (long long)total_seq, (long long)H,
      (long long)G, (long long)past_len,
      static_cast<int>(head_sink != nullptr),
      static_cast<int>(use_smooth_softmax), fp_rc);
  return fp_rc != 0 ? -1 : 0;
}

//===----------------------------------------------------------------------===//
// Legacy decomposed hipBLASLt pipeline (verbatim port of gqa_back.cpp's
// strategy). Reached from wrap_group_query_attention for every case the
// optimized fused path does not implement. The read-only backup gqa_back.cpp
// stays out of the build; this is the production copy.
//===----------------------------------------------------------------------===//

// Env-var gate for the group-batched "no-expand" hipBLASLt GQA pipeline.
// When HIPDNN_EP_GQA_NO_EXPAND=1 (default) the Score and Value GEMMs read
// K and V directly from the BNSD [B, G, skv, d] present cache using
// strided-batched mode with batch = B*G and per-operand batch strides:
//   A (K or V):  stride = present_seq * d       (one BNSD group matrix)
//   B (Q or S):  stride = HPG * sq * d          (score)
//                         HPG * sq * total_seq  (value)
//   C (S or O):  stride = HPG * sq * total_seq  (score)
//                         HPG * sq * d          (value)
// This eliminates the explicit expand_kv_k / expand_kv_v kernels and the
// B*H*total_seq*d fp16 scratch buffers they wrote into.
//
// At sq == 1 (decode) BSHD [B, 1, H, d] and BNSD [B, H, 1, d] share the
// same memory, so Q and O are also read / written in place (no
// Q-transpose / O-transpose kernels). At sq > 1 (prefill) the Q-transpose
// and O-transpose kernels still run because the two layouts diverge.
//
// Output is S / O bit-identical (modulo fp16 rounding in a different GEMM
// tile schedule) to the expand + transpose path. Set
// HIPDNN_EP_GQA_NO_EXPAND=0 to fall back fully for A/B testing.
static bool gqa_no_expand_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_NO_EXPAND");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Env-var gate for enabling the no-expand path on prefill (sq > 1). Default
// off: today only decode (sq == 1) takes the no-expand fast path, matching the
// verified pre-step-2 behaviour. Set HIPDNN_EP_GQA_NO_EXPAND_PREFILL=1 to opt
// prefill into the same group-batched pipeline -- same strides as decode, but
// with Q/O transpose kernels kept in place because BSHD and BNSD diverge at
// sq > 1.
//
// Keeping this behind a separate flag lets us A/B just the new prefill
// behaviour without touching decode. Once verified across model families
// (Mistral / Llama / GPT-OSS / ...), this can be folded into the main
// HIPDNN_EP_GQA_NO_EXPAND flag.
static bool gqa_no_expand_prefill_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_NO_EXPAND_PREFILL");
    return v && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Env-var gate to force decode through the decomposed hipBLASLt pipeline
// instead of the fused custom kernel hip_gqa_fused_decode. Default off
// (fused path is preferred). Set HIPDNN_EP_GQA_DISABLE_FUSED_DECODE=1 to
// A/B against decomposed at sq==1 -- useful for measuring whether the custom
// fused kernel is actually faster than hipBLASLt's auto-tuned GEMMs at decode
// shapes.
static bool gqa_fused_decode_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_DISABLE_FUSED_DECODE");
    return v && std::strcmp(v, "0") != 0;
  }();
  return disabled;
}

// Env-var gate for the FA-2 split-K flash_decode path (legacy decode kernel).
// HIPDNN_EP_GQA_FLASH_DECODE=0 disables it (falls back to
// hip_gqa_fused_decode). flash_decode wins at high KV depth where the existing
// one-block-per-head kernel is bandwidth-bound; below the threshold
// (gqa_flash_decode_min_skv) its 2-kernel overhead may not pay back, so we keep
// the existing fused_decode for short sequences.
static bool gqa_flash_decode_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FLASH_DECODE");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Smart-dispatch threshold for the legacy GQA decode (sq == 1). When total_seq
// exceeds this value, dispatch routes through the decomposed hipBLASLt pipeline
// instead of the fused custom kernel hip_gqa_fused_decode. The fused kernel
// uses a serial-over-time scheme with cross-wave reductions on the critical
// path of every iteration, so it loses to the GEMM-based decomposed path on
// long sequences (measured ~12x slower at total_seq~=2048 on Strix Halo). When
// flash_decode is eligible we keep the fused branch active even at long
// total_seq -- flash_decode is exactly what this threshold was working around.
//
// Default 256 is a starter value pending a full threshold sweep. Set
// HIPDNN_EP_GQA_FUSED_DECODE_MAX_T=N to override (or a very large value like
// 999999 to effectively disable smart-dispatch and preserve the always-fused-
// when-eligible behaviour for A/B testing).
static int gqa_fused_decode_max_t() {
  static const int max_t = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FUSED_DECODE_MAX_T");
    if (!v || !*v)
      return 256;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed <= 0)
      return 256;
    return static_cast<int>(parsed);
  }();
  return max_t;
}

// Depth threshold for legacy flash_decode eligibility. flash_decode wins at
// high KV depth; below this (default skv >= 256) the existing one-block-per-
// head fused_decode amortizes its single-kernel cost better than flash_decode's
// (split + reduce) launches. Set HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV=N to
// override.
static int gqa_flash_decode_min_skv() {
  static const int threshold = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV");
    if (!v || !*v)
      return 256;
    int n = std::atoi(v);
    return n > 0 ? n : 256;
  }();
  return threshold;
}

// FA-2 split-K geometry (must match the legacy hip_gqa_flash_decode launcher).
static constexpr int kFlashDecodeKSplits = 8;

// Geometry gate for the legacy flash_decode kernel. The launcher has template
// instantiations for:
//   - HPG=4, d in {64, 128}  (Llama-3.x family)
//   - HPG=8, d == 64         (gpt-oss-20b)
// Any other (HPG, d) combination must fall back to fused_decode / decomposed.
// Distinct from flash_decode_geometry_ok above, which gates the optimized _v2
// decode kernel (HpG in {1,2,3,4,5,8,16}, d in {64,128,256}).
static inline bool legacy_flash_decode_geometry_ok(int64_t H, int64_t G,
                                                   int64_t d) {
  if (G <= 0)
    return false;
  int64_t hpg = H / G;
  if (hpg * G != H)
    return false;
  if (hpg == 4 && (d == 64 || d == 128))
    return true;
  if (hpg == 8 && d == 64)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// hipBLASLt layout helper
//===----------------------------------------------------------------------===//
static hipblasStatus_t setLayoutBatch(hipblasLtMatrixLayout_t layout,
                                      int32_t batchCount, int64_t stride) {
  hipblasStatus_t status;
  status = hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
      sizeof(batchCount));
  if (status != HIPBLAS_STATUS_SUCCESS)
    return status;
  status = hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride));
  return status;
}

//===----------------------------------------------------------------------===//
// GQA GEMM descriptor cache
//===----------------------------------------------------------------------===//
//
// hipBLASLt descriptors and heuristic-selected algorithms are created once per
// unique (m, n, k, batch, transA, ...) shape and reused for the process
// lifetime. This avoids repeated descriptor creation and heuristic queries on
// every GQA inference call in the decomposed (prefill) path.
struct GqaGemmKey {
  int64_t m, n, k, batch;
  // true = Score GEMM (K^T * Q); false = Value GEMM (V * S)
  bool transA;
  // true = HIP_R_32F C/D layouts (Score GEMM always accumulates fp32)
  bool outputFp32;
  // true = fp32 A/B inputs (Whisper no_causal); false = fp16
  bool inputFp32;
  // Explicit per-operand batch strides (elements); 0 = default dense stride
  // (m*k for A, n*k for B, n*m for C/D). Non-zero enables the no-expand layout
  // where K/V are shared across HPG heads (stride = present_seq*d over G
  // groups) while Q/O advance by HPG entries per group.
  int64_t strideA;
  int64_t strideB;
  int64_t strideC;
  bool operator==(const GqaGemmKey &o) const {
    return m == o.m && n == o.n && k == o.k && batch == o.batch &&
           transA == o.transA && outputFp32 == o.outputFp32 &&
           inputFp32 == o.inputFp32 && strideA == o.strideA &&
           strideB == o.strideB && strideC == o.strideC;
  }
};

struct GqaGemmKeyHash {
  size_t operator()(const GqaGemmKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.m);
    hash_combine_val(h, k.n);
    hash_combine_val(h, k.k);
    hash_combine_val(h, k.batch);
    hash_combine_val(h, k.transA);
    hash_combine_val(h, k.outputFp32);
    hash_combine_val(h, k.inputFp32);
    hash_combine_val(h, k.strideA);
    hash_combine_val(h, k.strideB);
    hash_combine_val(h, k.strideC);
    return h;
  }
};

/// Cached hipBLASLt state for a single GEMM shape.
/// Ownership: descriptors are created in queryOrCreateGemmState() and live for
/// the process lifetime (destroyed together when the owning op-state slot is
/// torn down, in GqaGemmCache's destructor).
struct GqaGemmCacheEntry {
  hipblasLtMatmulDesc_t desc;                     // matmul operation descriptor
  hipblasLtMatrixLayout_t layA, layB, layC, layD; // matrix layouts
  hipblasLtMatmulAlgo_t algo; // heuristic-selected algorithm
  size_t workspace_size;      // workspace bytes required by algo
};

struct GqaGemmCache {
  std::unordered_map<GqaGemmKey, GqaGemmCacheEntry, GqaGemmKeyHash> entries;
  // Destroys every cached hipBLASLt descriptor/layout entry. Defined
  // out-of-line below. Runs when the owning op-state slot is torn down
  // (GqaState's deleter).
  ~GqaGemmCache();
};

// Per-instance GQA op-state (see op-state-slots-design.md): owns this
// instance's per-GEMM-shape hipBLASLt descriptor/algorithm cache. Replaces the
// former shared RuntimeState::gqa_gemm_cache, so concurrent sessions (and
// distinct GQA layers) no longer share one descriptor map.
struct GqaState : OpStateT<GqaState> {
  GqaGemmCache cache;
};

// Resolve this GQA instance's descriptor cache from its op-state slot. Returns
// nullptr when the slot is unconstructed (init failure) -- callers propagate
// the error rather than lazily allocating, since the slot is built at session
// init.
static GqaGemmCache *get_gemm_cache(RuntimeState *state, int op_state_slot) {
  GqaState *gs = GqaState::get_op_state(state, op_state_slot);
  return gs ? &gs->cache : nullptr;
}

static const GqaGemmCacheEntry *queryOrCreateGemmState(RuntimeState *state,
                                                       hipblasLtHandle_t handle,
                                                       const GqaGemmKey &key,
                                                       int op_state_slot) {
  assert(handle && "queryOrCreateGemmState: null handle");
  auto *cache = get_gemm_cache(state, op_state_slot);
  if (!cache) {
    fprintf(stderr, "queryOrCreateGemmState: no GqaState at slot %d\n",
            op_state_slot);
    return nullptr;
  }
  auto it = cache->entries.find(key);
  if (it != cache->entries.end())
    return &it->second;

  int64_t m = key.m, n = key.n, k = key.k;
  int32_t batch = static_cast<int32_t>(key.batch);

  GqaGemmCacheEntry entry = {};

  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasStatus_t st;

#define GQA_CACHE_CHECK(call)                                                  \
  do {                                                                         \
    st = (call);                                                               \
    if (st != HIPBLAS_STATUS_SUCCESS)                                          \
      goto cache_fail;                                                         \
  } while (0)

  GQA_CACHE_CHECK(
      hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  {
    hipblasOperation_t opA = key.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opN = HIPBLAS_OP_N;
    GQA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    GQA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  }

  {
    int64_t strideA = key.strideA != 0 ? key.strideA : m * k;
    int64_t strideB = key.strideB != 0 ? key.strideB : n * k;
    int64_t strideC = key.strideC != 0 ? key.strideC : n * m;

    // Input operand element type: HIP_R_16F (fp16 GQA) or HIP_R_32F (fp32
    // GQA, e.g. Whisper no_causal). Compute is HIPBLAS_COMPUTE_32F either way.
    hipDataType inType = key.inputFp32 ? HIP_R_32F : HIP_R_16F;
    int64_t a_rows = key.transA ? k : m;
    int64_t a_cols = key.transA ? m : k;
    GQA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layA, inType, a_rows,
                                                a_cols, a_rows));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layA, batch, strideA));

    GQA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layB, inType, k, n, k));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layB, batch, strideB));

    hipDataType outType = key.outputFp32 ? HIP_R_32F : HIP_R_16F;
    GQA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layC, outType, m, n, m));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layC, batch, strideC));
    GQA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layD, outType, m, n, m));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layD, batch, strideC));
  }

  GQA_CACHE_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  {
    const size_t max_ws = kMaxWorkspaceBytes;
    GQA_CACHE_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
        sizeof(max_ws)));
  }

  {
    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    GQA_CACHE_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, entry.desc, entry.layA, entry.layB, entry.layC, entry.layD,
        pref, 1, &heur, &returned));
    hipblasLtMatmulPreferenceDestroy(pref);
    pref = nullptr;

    if (returned == 0) {
      fprintf(stderr,
              "GQA: no algorithm found for GEMM m=%lld n=%lld k=%lld "
              "batch=%lld\n",
              (long long)m, (long long)n, (long long)k, (long long)key.batch);
      goto cache_fail;
    }

    entry.algo = heur.algo;
    entry.workspace_size = heur.workspaceSize;
  }

#undef GQA_CACHE_CHECK
  goto cache_done;

cache_fail:
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (entry.layD)
    hipblasLtMatrixLayoutDestroy(entry.layD);
  if (entry.layC)
    hipblasLtMatrixLayoutDestroy(entry.layC);
  if (entry.layB)
    hipblasLtMatrixLayoutDestroy(entry.layB);
  if (entry.layA)
    hipblasLtMatrixLayoutDestroy(entry.layA);
  if (entry.desc)
    hipblasLtMatmulDescDestroy(entry.desc);
  return nullptr;

cache_done:
  auto [ins, _] = cache->entries.emplace(key, entry);
  return &ins->second;
}

//===----------------------------------------------------------------------===//
// 12-step hipBLASLt GQA pipeline (Step 0 + Steps 1-11; fp16 + fp32)
//===----------------------------------------------------------------------===//
static int gqa_forward_hipblaslt(
    RuntimeState *state, hipStream_t stream, hipblasLtHandle_t ltHandle,
    const void *query, const void *key, const void *value, const void *past_key,
    const void *past_value, const void *seqlens_k_ptr, const void *cos_cache,
    const void *sin_cache, void *head_sink, bool use_smooth_softmax,
    // onnx.Attention external additive mask [B,H,S,T] (broadcastable on the
    // batch / head dims); null for plain GQA.
    const void *attention_bias, int64_t attn_bias_batch,
    int64_t attn_bias_num_heads, void *output, void *present_key,
    void *present_value, int64_t B, int64_t sq, int64_t skv,
    int64_t past_buf_seq, int64_t H, int64_t G, int64_t d, float scale,
    int64_t do_rotary, int64_t local_window_size, bool no_causal,
    int64_t element_size_bytes, int op_state_slot) {

  // Whisper bidirectional no-past path only when no_causal AND no external
  // attention_bias. ONNX Attention with is_causal=0 still carries past KV and
  // uses the standard decode seqlens_k convention.
  const bool bidirectional_no_past = no_causal && (attention_bias == nullptr);

  int64_t HPG = H / G;
  int64_t present_seq = skv;
  size_t elem_sz = static_cast<size_t>(element_size_bytes);
  bool gemm_fp32 = (elem_sz == 4);
  bool need_rope = do_rotary && cos_cache && sin_cache;

  // Pre-dispatch read of seqlens_k (or per-Compute() cache lookup for B==1
  // when HIPDNN_EP_GQA_CACHE_SEQLENS=1). Applies to both prefill (sq>1) and
  // decode (sq==1) for B==1 -- both share the same seqlens_k pointer and
  // benefit from caching. The result is reused by the fused-path
  // need_host_past_len block (eliminating its inline D2H) and the
  // decomposed-path readback site (consumed unconditionally instead of issuing
  // its own D2H). On B>1 this returns kSeqlensKNotRead and both downstream
  // paths fall back to their legacy per-call reads. Stored in seqlens_k_pre;
  // total_seq_pre is the derived total_seq (-1 means unknown / not applicable).
  int32_t seqlens_k_pre =
      read_seqlens_k_for_dispatch(stream, seqlens_k_ptr, B, state);
  int64_t total_seq_pre = -1;
  if (bidirectional_no_past) {
    // bidirectional_no_past (Whisper encoder / cross-attn): seqlens_k = skv
    // means "all skv keys valid", there is no past. total_seq is exactly skv --
    // do NOT apply the +1 decode convention (would over-count and trip the
    // smart-dispatch size check / fused validation). See the matching exemption
    // at the decomposed-path total_seq derivation below.
    total_seq_pre = skv;
  } else if (seqlens_k_pre != kSeqlensKNotRead) {
    // -1 is ORT's prefill sentinel: total_seq=sq, past_len=0. Real values are
    // 0..max_seq; total_seq = seqlens_k_val + 1.
    total_seq_pre =
        (seqlens_k_pre < 0) ? sq : static_cast<int64_t>(seqlens_k_pre) + 1;
  }

  bool fused_d = (d == 64 || d == 128 || d == 256);

  // Smart dispatch: the legacy fused decode kernel (hip_gqa_fused_decode)
  // serializes over the time dimension (cross-wave reduction tree on the
  // critical path of every iteration). For total_seq above
  // gqa_fused_decode_max_t() the GEMM-based decomposed path wins (~12x at
  // total_seq=2048 on Strix Halo). The newer flash_decode kernel
  // (hip_gqa_flash_decode) fixes that scaling via FA-2 split-K, so when it is
  // eligible we keep the fused branch active even at long total_seq --
  // flash_decode is exactly what the smart-dispatch threshold was working
  // around. When we can't read total_seq (B>1, no seqlens_k, or D2H failure)
  // default to permitting fused -- preserves bit-for-bit behaviour on workloads
  // that pass the predicate today.
  bool flash_decode_eligible = gqa_flash_decode_enabled() &&
                               legacy_flash_decode_geometry_ok(H, G, d) &&
                               skv >= gqa_flash_decode_min_skv();
  bool size_ok_for_fused =
      (total_seq_pre < 0) ||
      (total_seq_pre <= static_cast<int64_t>(gqa_fused_decode_max_t())) ||
      flash_decode_eligible;

  // ONNX uses local_window_size=-1 for "no sliding window"; <= 0 means
  // disabled. The original hip_gqa_fused_decode kernel does NOT support sliding
  // window, but hip_gqa_flash_decode does (it clamps kv_lo to
  // max(0, eff_skv - window) when local_window_size > 0). So we admit the fused
  // branch with sliding window only when flash_decode is eligible -- the
  // (use_flash_decode) check inside the branch then routes us correctly. This
  // is what unlocks the gpt-oss-20b sliding-attention layers (12 of 24) at long
  // context: they were previously rejected here and fell through to the
  // decomposed path, which reads the full skv KV cache instead of just the
  // 128-element window.
  bool sliding_ok_for_fused = (local_window_size <= 0) || flash_decode_eligible;
  // head_sink / smooth_softmax: legacy hip_gqa_fused_decode does not support
  // these, but hip_gqa_flash_decode now folds the sink term into the reduce
  // kernel's denominator. Admit them only when flash_decode is the eligible
  // dispatch -- the (use_flash_decode) check inside the branch then routes
  // correctly. This is what unlocks gpt-oss-20b decode (all 24 GQA layers pass
  // head_sink, which previously forced fall-through to the decomposed hipBLASLt
  // path that scales linearly with skv).
  bool sink_ok_for_fused =
      (!head_sink && !use_smooth_softmax) || flash_decode_eligible;
  // Packed-QKV inputs (gpt-oss-20b style: query is the [B,sq,(H+2G)*d]
  // qkv_proj output, key and value are null) are supported by the fused branch
  // by routing through hip_gqa_split_qkv into workspace before rope and
  // KV-append. Only flash_decode is exercised by these models in practice
  // (HPG=8, d=64), but split is correct for the legacy fused_decode branch too.
  bool fused_packed_qkv = (!key && !value);
  bool kv_inputs_ok = (key && value) || fused_packed_qkv;
  // The fused (hip_gqa_fused_decode) and flash (hip_gqa_flash_decode) decode
  // kernels are __half-only (their Q/K/V/O pointers are `const __half*`). They
  // are correct for the fp16 causal decode of Llama / gpt-oss (elem_size==2).
  // The fp32 GQA path (Whisper decoder self-attn, elem_size==4) must NOT reach
  // them: feeding fp32 buffers to a __half kernel reinterprets the bytes and
  // produces garbage. The decomposed hipBLASLt pipeline below IS fp32-capable,
  // so route fp32 decode there. This is the decode-side analogue of the
  // no_causal exemption (prefill sq>1 fp32 already used decomposed).
  bool fused_fp16 = (element_size_bytes == 2);
  // no_causal (Whisper encoder / cross-attn) always takes the decomposed
  // hipBLASLt path. The fused/flash decode kernels are decode-only (sq==1) and
  // read seqlens_k with the +1 PAST-token convention, plus assume the KV cache
  // is appended in BSHD->BNSD layout from `sq` new tokens. Neither holds for
  // bidirectional no-past attention where `key` is the full Skv-length KV
  // (cross-attn ships it as rank-4 BNSD with Skv != sq). Routing no_causal to
  // the decomposed path keeps a single correct code path for these models.
  bool fused_predicate =
      (!gqa_fused_decode_disabled() && !no_causal && !attention_bias &&
       fused_fp16 && fused_d && sq == 1 && kv_inputs_ok && present_key &&
       present_value && sliding_ok_for_fused && sink_ok_for_fused &&
       size_ok_for_fused);

  //===--------------------------------------------------------------------===//
  // Fused / flash GQA decode path (sq == 1, d in {64,128,256}, KV cache on).
  //
  // Collapses Steps 3 and 6-11 of the decomposed pipeline into a single kernel
  // that reads Q in BSHD and KV from the BNSD cache, producing O in BSHD.
  // Steps 0 (split), 1-2 (RoPE) and 4-5 (KV cache update) still run as separate
  // kernels below before the fused dispatch.
  //
  // When seqlens_k is provided, the device pointer is passed directly to each
  // kernel so they can read the actual sequence length on-device, eliminating
  // the D2H copy + hipStreamSynchronize stall entirely for the decode hot path.
  //
  // All prefill (sq > 1) goes through the decomposed hipBLASLt path below where
  // auto-tuned GEMMs outperform fixed WMMA tiling and all ORT GQA features
  // (sliding window, smooth softmax, head sink) are supported.
  //===--------------------------------------------------------------------===//
  if (fused_predicate) {
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // For fused decode, kernels read seqlens_k from device memory directly.
    // past_len is only needed on host for the concat branch (separate buffers);
    // for in-place caches (past_key == present_key) it is unused on host.
    int64_t past_len = 0;
    bool need_host_past_len =
        seqlens_k_ptr && past_key && past_key != present_key;
    if (need_host_past_len) {
      // Reuse the value the pre-dispatch helper already read above. Fall back
      // to a per-call D2H + sync only when the pre-read was not applicable
      // (multi-batch, or copy/sync failure). For the asym Llama decode hot path
      // (B==1, sq==1) the pre-read is always applicable, so this branch becomes
      // pure host arithmetic.
      int32_t seqlens_k_val = 0;
      if (seqlens_k_pre != kSeqlensKNotRead) {
        seqlens_k_val = seqlens_k_pre;
      } else {
        if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                           hipMemcpyDeviceToHost, stream) != hipSuccess) {
          return -1;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
          return -1;
        }
      }

      // ORT prefill sentinel: when there is no past KV yet, the producer
      // initialises seqlens_k[b] to -1 (so seqlens_k[b]+1 == 0). Treat that as
      // a fresh prefill (past_len=0) instead of rejecting it as invalid.
      if (seqlens_k_val < 0) {
        past_len = 0;
      } else {
        int64_t total_seq = static_cast<int64_t>(seqlens_k_val) + 1;
        int64_t past_len_check = total_seq - sq;
        if (total_seq < 1 || past_len_check < 0 || total_seq > present_seq ||
            past_len_check > past_buf_seq) {
          fprintf(stderr,
                  "gqa_forward_hipblaslt (fused decode): invalid "
                  "seqlens_k[0]+1=%lld (sq=%lld, past_len=%lld, "
                  "present_seq=%lld, past_buf_seq=%lld)\n",
                  (long long)total_seq, (long long)sq,
                  (long long)past_len_check, (long long)present_seq,
                  (long long)past_buf_seq);
          return -1;
        }
        past_len = past_len_check;
      }
    } else if (!seqlens_k_ptr) {
      past_len = skv - sq;
    }
    if (past_len < 0)
      past_len = 0;

    // Flash-decode path is taken when:
    //   - depth threshold met (default skv >= 256)
    //   - geometry matches a kernel template instantiation
    //     (HPG=4 with d in {64,128}, or HPG=8 with d=64 for gpt-oss-20b)
    //   - not disabled via env var
    // Below threshold the existing one-block-per-head fused_decode is faster
    // because its single-kernel cost amortizes better than flash_decode's
    // (split + reduce) launches and per-call workspace setup.
    const bool use_flash_decode = gqa_flash_decode_enabled() &&
                                  legacy_flash_decode_geometry_ok(H, G, d) &&
                                  skv >= gqa_flash_decode_min_skv();

    // Sum split + rope-temp + flash-partials in a single ensure_workspace call.
    // ensure_workspace does NOT preserve data on grow (free + malloc), so one
    // combined request avoids clobbering earlier writes; the offsets below
    // match the call order so each step's input region stays live while
    // consumed. Region order: split (Q/K/V), then rope-temp (Q/K), then
    // flash-partials -- each present only when its feature is active.
    const size_t Q_full_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
    const size_t K_full_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;
    const size_t split_bytes =
        fused_packed_qkv ? (Q_full_bytes + K_full_bytes + K_full_bytes) : 0;
    const size_t rope_temp_bytes =
        need_rope ? (Q_full_bytes + K_full_bytes) : 0;
    const size_t flash_partials_bytes =
        use_flash_decode ? static_cast<size_t>(B) * H * kFlashDecodeKSplits *
                               (d + 2) * sizeof(float)
                         : 0;
    const size_t total_ws_bytes =
        split_bytes + rope_temp_bytes + flash_partials_bytes;

    if (total_ws_bytes > 0) {
      if (hipdnn_ep_state_ensure_workspace(state, total_ws_bytes) != 0)
        return -1;
    }

    const size_t off_split = 0;
    const size_t off_rope = off_split + split_bytes;
    const size_t off_partials = off_rope + rope_temp_bytes;

    // ---- Step 0: Split packed QKV (if needed) ----
    if (fused_packed_qkv) {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *d_Qsplit = ws + off_split;
      void *d_Ksplit = ws + off_split + Q_full_bytes;
      void *d_Vsplit = ws + off_split + Q_full_bytes + K_full_bytes;
      if (hip_gqa_split_qkv(
              stream, query, d_Qsplit, d_Ksplit, d_Vsplit, static_cast<int>(B),
              static_cast<int>(sq), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(elem_sz)) != 0)
        return -1;
      qSrc = d_Qsplit;
      kSrc = d_Ksplit;
      vSrc = d_Vsplit;
    }

    // ---- Steps 1-2: RoPE (optional) ----
    if (need_rope) {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *d_Qroped = ws + off_rope;
      void *d_Kroped = ws + off_rope + Q_full_bytes;

      int half_rot = static_cast<int>(d / 2);
      if (hip_gqa_rope(stream, qSrc, d_Qroped, cos_cache, sin_cache,
                       static_cast<int>(B), static_cast<int>(sq),
                       static_cast<int>(H), static_cast<int>(d), half_rot,
                       static_cast<int>(past_len), seqlens_k_ptr,
                       static_cast<int>(elem_sz)) != 0)
        return -1;
      if (hip_gqa_rope(stream, kSrc, d_Kroped, cos_cache, sin_cache,
                       static_cast<int>(B), static_cast<int>(sq),
                       static_cast<int>(G), static_cast<int>(d), half_rot,
                       static_cast<int>(past_len), seqlens_k_ptr,
                       static_cast<int>(elem_sz)) != 0)
        return -1;

      qSrc = d_Qroped;
      kSrc = d_Kroped;
      // vSrc is intentionally NOT updated: V is never RoPE'd.
    }

    // ---- Steps 4-5: KV cache update ----
    if (update_kv_cache(
            stream, past_key, past_value, kSrc, vSrc, present_key,
            present_value, static_cast<int>(B), static_cast<int>(past_len),
            static_cast<int>(sq), static_cast<int>(G), static_cast<int>(d),
            static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
            seqlens_k_ptr, static_cast<int>(elem_sz)) != 0)
      return -1;

    if (use_flash_decode) {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *partials = ws + off_partials;
      if (hip_gqa_flash_decode(
              stream, qSrc, present_key, present_value, output, partials,
              static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(present_seq),
              kFlashDecodeKSplits, scale, seqlens_k_ptr,
              static_cast<int>(local_window_size), head_sink,
              static_cast<int>(use_smooth_softmax)) != 0)
        return -1;
      RUNTIME_DEBUG_LOG(
          "[REAL] flash GQA decode (legacy): B=%lld sq=%lld skv=%lld H=%lld "
          "G=%lld d=%lld K_SPLITS=%d window=%lld sink=%d smooth=%d\n",
          (long long)B, (long long)sq, (long long)skv, (long long)H,
          (long long)G, (long long)d, kFlashDecodeKSplits,
          (long long)local_window_size, static_cast<int>(head_sink != nullptr),
          static_cast<int>(use_smooth_softmax));
    } else {
      // The original hip_gqa_fused_decode kernel does not implement sliding
      // window or head_sink/smooth_softmax. The predicate above only admits
      // those features when flash_decode is eligible, so this branch should
      // never see them -- assert defensively rather than silently producing
      // wrong results.
      if (local_window_size > 0) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: BUG -- fused_decode (non-flash) cannot "
                "handle local_window_size=%lld\n",
                (long long)local_window_size);
        return -1;
      }
      if (head_sink != nullptr || use_smooth_softmax) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: BUG -- fused_decode (non-flash) cannot "
                "handle head_sink=%p smooth=%d\n",
                head_sink, static_cast<int>(use_smooth_softmax));
        return -1;
      }
      // skv is passed as a fallback; kernel reads seqlens_k[b]+1 when
      // available.
      if (hip_gqa_fused_decode(
              stream, qSrc, present_key, present_value, output,
              static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(skv),
              static_cast<int>(present_seq), scale, seqlens_k_ptr) != 0)
        return -1;
      RUNTIME_DEBUG_LOG("[REAL] fused GQA decode (legacy): B=%lld sq=%lld "
                        "skv=%lld H=%lld G=%lld d=%lld\n",
                        (long long)B, (long long)sq, (long long)skv,
                        (long long)H, (long long)G, (long long)d);
    }
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Decomposed hipBLASLt pipeline (all prefill sq > 1, unsupported d, or
  // features requiring sliding window / smooth softmax / head sink / fp32)
  //===--------------------------------------------------------------------===//

  // D2H readback of seqlens_k is required here because hipBLASLt descriptor
  // creation and workspace sizing are host-side APIs that need total_seq. For
  // B == 1 the value was already read (and cached when
  // HIPDNN_EP_GQA_CACHE_SEQLENS=1) by the pre-dispatch helper above; we just
  // consume seqlens_k_pre. The B > 1 branch keeps the legacy per-call read
  // because per-batch validation requires reading every entry and we have no
  // validated multi-batch decode workload yet.
  int64_t total_seq = skv;
  int64_t past_len = skv - sq;
  // no_causal (Whisper encoder self-attn + decoder cross-attn) is bidirectional
  // with NO past KV: the converters emit a compile-time seqlens_k = skv meaning
  // "all skv keys are valid". The ORT decode convention below (seqlens_k[b] =
  // PAST tokens => total_seq = seqlens_k+1) does NOT apply here -- it would
  // give total_seq = skv+1 > present_seq = skv -> rc=-1 -> zeroed output, and
  // past_len = total_seq - sq is invalid when sq != skv (cross-attn has sq=1,
  // skv=1500 => bogus past_len=1499). Gated on bidirectional_no_past (not raw
  // no_causal): an onnx.Attention with an external mask still carries past KV,
  // so it keeps the standard seqlens_k path below. For the no-past case
  // total_seq = skv (== present_seq) and past_len = 0; skip the readback.
  if (bidirectional_no_past) {
    total_seq = skv;
    past_len = 0;
  } else if (seqlens_k_ptr) {
    int32_t seqlens_k_val = 0;

    if (seqlens_k_pre != kSeqlensKNotRead) {
      seqlens_k_val = seqlens_k_pre;
    } else if (B > 1) {
      std::vector<int32_t> seqlens_k_host(B);
      if (hipMemcpyAsync(seqlens_k_host.data(), seqlens_k_ptr,
                         B * sizeof(int32_t), hipMemcpyDeviceToHost,
                         stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
      seqlens_k_val = seqlens_k_host[0];
      for (int64_t b = 1; b < B; ++b) {
        if (seqlens_k_host[b] != seqlens_k_val) {
          fprintf(stderr,
                  "gqa_forward_hipblaslt: per-batch seqlens_k not yet "
                  "supported (batch %lld has %d, batch 0 has %d)\n",
                  (long long)b, seqlens_k_host[b], seqlens_k_val);
          return -1;
        }
      }
    } else {
      // Defensive fallback for B == 1 when the pre-dispatch helper bailed out
      // (D2H or sync failure). Rare path; not cached because the same failure
      // mode would have prevented the helper from caching too.
      if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                         hipMemcpyDeviceToHost, stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
    }

    // ORT prefill sentinel: when there is no past KV yet, the producer
    // initialises seqlens_k[b] to -1. Treat that as a fresh prefill
    // (past_len=0, total_seq=sq) instead of rejecting it as invalid.
    if (seqlens_k_val < 0) {
      total_seq = sq;
      past_len = 0;
    } else {
      total_seq = static_cast<int64_t>(seqlens_k_val) + 1;
      past_len = total_seq - sq;
      if (total_seq < 1 || past_len < 0 || total_seq > present_seq ||
          past_len > past_buf_seq) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: invalid seqlens_k[0]+1=%lld "
                "(sq=%lld, past_len=%lld, present_seq=%lld, "
                "past_buf_seq=%lld)\n",
                (long long)total_seq, (long long)sq, (long long)past_len,
                (long long)present_seq, (long long)past_buf_seq);
        return -1;
      }
    }
  }
  if (past_len < 0)
    past_len = 0;

  bool packed_qkv = (key == nullptr && value == nullptr);

  //===--------------------------------------------------------------------===//
  // Unified hipBLASLt GQA pipeline (shared by expand and no-expand paths).
  //
  // Two orthogonal knobs control the layout choices for the two GEMMs:
  //
  //   use_no_expand : when true, Score reads K and Value reads V directly from
  //                   the BNSD cache (present_key / present_value) and both
  //                   GEMMs use strided-batched mode with batch = B*G plus
  //                   per-operand strides to broadcast each KV group across its
  //                   HPG queries. When false (original path) expand_kv_*
  //                   duplicates the G groups into H heads in d_Kexp / d_Vexp
  //                   and both GEMMs run with dense batch = B*H. The no-expand
  //                   flavour skips two kernels and two B*H*total_seq*d fp16
  //                   scratch buffers.
  //
  //   need_transpose: true when sq > 1. The BSHD (Q, output) and BNSD
  //                   (GEMM-native) layouts only coincide when sq == 1, so for
  //                   prefill we still need a Q-transpose before Score and an
  //                   O-transpose after Value. For decode (sq == 1) both
  //                   transposes are pure pointer reinterpretations, skipped.
  //
  // Prefill (sq > 1) is additionally guarded by HIPDNN_EP_GQA_NO_EXPAND_PREFILL
  // (default off) so the new behaviour can be verified in isolation.
  //
  // Pointer plumbing:
  //   Score A (K):  use_no_expand ? present_key  : d_Kexp
  //   Score B (Q):  need_transpose ? d_Qtrans    : qSrc  (BSHD == BNSD @sq=1)
  //   Value A (V):  use_no_expand ? present_value: d_Vexp
  //   Value C (O):  need_transpose ? d_O         : output
  //===--------------------------------------------------------------------===//
  bool use_no_expand = gqa_no_expand_enabled() && present_key &&
                       present_value &&
                       (sq == 1 || gqa_no_expand_prefill_enabled());
  bool need_transpose = (sq > 1);

  // GEMM descriptor keys. The no-expand flavour uses explicit per-operand
  // strides (non-zero stride fields); the expand flavour leaves them zero so
  // queryOrCreateGemmState falls back to the dense packed-batch defaults.
  GqaGemmKey scoreKey, valueKey;
  if (use_no_expand) {
    // Score: C[total_seq, HPG*sq] = K^T[d,total_seq] * Q[d, HPG*sq] per (b, g)
    // pair. strideA steps over the buffer page (present_seq*d) even though only
    // the first total_seq tokens are read, keeping the descriptor stable across
    // token steps.
    scoreKey = {/*m=*/total_seq,
                /*n=*/HPG * sq,
                /*k=*/d,
                /*batch=*/B * G,
                /*transA=*/true,
                /*outputFp32=*/true,
                /*inputFp32=*/gemm_fp32,
                /*strideA=*/present_seq * d,
                /*strideB=*/HPG * sq * d,
                /*strideC=*/HPG * sq * total_seq};
    // Value: C[d, HPG*sq] = V[d, total_seq] * S[total_seq, HPG*sq] per (b, g)
    // pair, writing into BNSD [B, G, HPG, sq, d] which at sq==1 coincides with
    // BSHD [B, 1, H, d].
    valueKey = {/*m=*/d,
                /*n=*/HPG * sq,
                /*k=*/total_seq,
                /*batch=*/B * G,
                /*transA=*/false,
                /*outputFp32=*/gemm_fp32,
                /*inputFp32=*/gemm_fp32,
                /*strideA=*/present_seq * d,
                /*strideB=*/HPG * sq * total_seq,
                /*strideC=*/HPG * sq * d};
  } else {
    scoreKey = {total_seq,           sq,        d, B * H, true,
                /*outputFp32=*/true, gemm_fp32,
                /*strideA=*/0,
                /*strideB=*/0,
                /*strideC=*/0};
    valueKey = {d,
                sq,
                total_seq,
                B * H,
                false,
                /*outputFp32=*/gemm_fp32,
                gemm_fp32,
                /*strideA=*/0,
                /*strideB=*/0,
                /*strideC=*/0};
  }

  const GqaGemmCacheEntry *scoreState =
      queryOrCreateGemmState(state, ltHandle, scoreKey, op_state_slot);
  if (!scoreState)
    return -1;
  const GqaGemmCacheEntry *valueState =
      queryOrCreateGemmState(state, ltHandle, valueKey, op_state_slot);
  if (!valueState)
    return -1;

  // ---- Workspace layout ----
  // All temp buffers are packed contiguously into the shared workspace,
  // followed by the GEMM workspace region. This eliminates per-call
  // hipMalloc/hipFree -- after the first inference the workspace is already
  // large enough and reuse is zero-cost.
  //
  // Region order: Qtrans?, Kexp?, Vexp?, S_f32, S_fp16, O?, Qroped?, Kroped?,
  // Qsplit?, Ksplit?, Vsplit?, then the GEMM workspace. Optional (?) regions
  // are omitted when their feature is inactive.
  //
  // Qtrans / O are only allocated when need_transpose is true.
  // Kexp / Vexp are only allocated when use_no_expand is false.
  // S_f32 and S_fp16 are always allocated (softmax is on every path).
  size_t Qtrans_bytes =
      need_transpose ? static_cast<size_t>(B) * H * sq * d * elem_sz : 0;
  size_t Kexp_bytes =
      use_no_expand ? 0 : static_cast<size_t>(B) * H * total_seq * d * elem_sz;
  size_t Vexp_bytes = Kexp_bytes;
  size_t S_f32_bytes =
      static_cast<size_t>(B) * H * sq * total_seq * sizeof(float);
  size_t S_fp16_bytes = static_cast<size_t>(B) * H * sq * total_seq * elem_sz;
  size_t O_bytes =
      need_transpose ? static_cast<size_t>(B) * H * sq * d * elem_sz : 0;

  size_t off_Qtrans = 0;
  size_t off_Kexp = off_Qtrans + Qtrans_bytes;
  size_t off_Vexp = off_Kexp + Kexp_bytes;
  size_t off_S_f32 = off_Vexp + Vexp_bytes;
  size_t off_S_fp16 = off_S_f32 + S_f32_bytes;
  size_t off_O = off_S_fp16 + S_fp16_bytes;
  size_t temp_end = off_O + O_bytes;

  // Optional RoPE buffers: allocated only when do_rotary is enabled.
  size_t off_Qroped = 0, off_Kroped = 0;
  if (need_rope) {
    size_t Q_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
    size_t K_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;
    off_Qroped = temp_end;
    off_Kroped = off_Qroped + Q_bytes;
    temp_end = off_Kroped + K_bytes;
  }

  // Optional packed-QKV split buffers: allocated only when key/value are null
  // (GPT-OSS style packed input). Placed AFTER the RoPE buffers so that split
  // outputs remain live while RoPE reads from them and writes to the RoPE
  // region (no overlap).
  size_t off_Qsplit = 0, off_Ksplit = 0, off_Vsplit = 0;
  if (packed_qkv) {
    size_t Q_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
    size_t K_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;
    off_Qsplit = temp_end;
    off_Ksplit = off_Qsplit + Q_bytes;
    off_Vsplit = off_Ksplit + K_bytes;
    temp_end = off_Vsplit + K_bytes;
  }

  int result = 0;

  // Single workspace allocation: temp buffers + GEMM workspace.
  {
    size_t gemm_ws =
        std::max(scoreState->workspace_size, valueState->workspace_size);
    size_t total_needed = temp_end + gemm_ws;
    HIP_CHECK(hipdnn_ep_state_ensure_workspace(state, total_needed));
  }

  {
    char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    size_t ws_total = hipdnn_ep_state_get_workspace_size(state);

    void *d_Qtrans = need_transpose ? (ws + off_Qtrans) : nullptr;
    void *d_Kexp = use_no_expand ? nullptr : (ws + off_Kexp);
    void *d_Vexp = use_no_expand ? nullptr : (ws + off_Vexp);
    void *d_S_f32 = ws + off_S_f32;
    void *d_S_fp16 = ws + off_S_fp16;
    void *d_O = need_transpose ? (ws + off_O) : nullptr;

    void *gemm_ws_ptr = ws + temp_end;
    size_t gemm_ws_bytes = ws_total - temp_end;

    // Mutable source pointers: initially the raw inputs, redirected to
    // workspace buffers as pipeline steps (split, RoPE) produce intermediate
    // results. Downstream steps always read through these so they pick up the
    // latest transformed data.
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // ---- Step 0: Split packed QKV (if needed) ----
    // key/value null => query is packed [B, sq, (H+2G)*d]; split into Q/K/V
    // workspace buffers and redirect qSrc/kSrc/vSrc to them.
    if (packed_qkv) {
      void *d_Qsplit = ws + off_Qsplit;
      void *d_Ksplit = ws + off_Ksplit;
      void *d_Vsplit = ws + off_Vsplit;
      HIP_CHECK(hip_gqa_split_qkv(
          stream, query, d_Qsplit, d_Ksplit, d_Vsplit, static_cast<int>(B),
          static_cast<int>(sq), static_cast<int>(H), static_cast<int>(G),
          static_cast<int>(d), static_cast<int>(elem_sz)));
      qSrc = d_Qsplit;
      kSrc = d_Ksplit;
      vSrc = d_Vsplit;
    }

    // ---- Steps 1-2: RoPE (optional) ----
    // Reads qSrc/kSrc so packed-QKV split output feeds RoPE; V is never RoPE'd.
    if (need_rope) {
      int half_rot = static_cast<int>(d / 2);
      void *d_Qroped = ws + off_Qroped;
      void *d_Kroped = ws + off_Kroped;

      HIP_CHECK(hip_gqa_rope(stream, qSrc, d_Qroped, cos_cache, sin_cache,
                             static_cast<int>(B), static_cast<int>(sq),
                             static_cast<int>(H), static_cast<int>(d), half_rot,
                             static_cast<int>(past_len), nullptr,
                             static_cast<int>(elem_sz)));
      HIP_CHECK(hip_gqa_rope(stream, kSrc, d_Kroped, cos_cache, sin_cache,
                             static_cast<int>(B), static_cast<int>(sq),
                             static_cast<int>(G), static_cast<int>(d), half_rot,
                             static_cast<int>(past_len), nullptr,
                             static_cast<int>(elem_sz)));

      qSrc = d_Qroped;
      kSrc = d_Kroped;
    }

    // ---- Step 3: Q Transpose BSHD [B,sq,H,d] -> BNSD [B,H,sq,d] ----
    // Skipped at sq == 1: BSHD and BNSD share memory, so the Score GEMM reads
    // qSrc directly.
    if (need_transpose) {
      HIP_CHECK(hip_gqa_transpose_mid_dims(
          stream, qSrc, d_Qtrans, static_cast<int>(B), static_cast<int>(sq),
          static_cast<int>(H), static_cast<int>(d), static_cast<int>(elem_sz)));
    }

    // ---- Steps 4-5: KV Cache Update (concat/append into BNSD present) ----
    // no-expand hands seqlens_k_ptr to the append kernel (on-device past_len,
    // no D2H); the expand path already read total_seq host-side so passes null.
    if (present_key && present_value) {
      // bidirectional_no_past (Whisper encoder / cross-attn): never hand
      // seqlens_k to the append kernel (it would apply the +1 convention).
      // ONNX Attention with is_causal=0 + external mask keeps the standard
      // decode KV path (bidirectional_no_past=false).
      HIP_CHECK(update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          static_cast<int>(B), static_cast<int>(past_len), static_cast<int>(sq),
          static_cast<int>(G), static_cast<int>(d),
          static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
          (use_no_expand && !bidirectional_no_past) ? seqlens_k_ptr : nullptr,
          static_cast<int>(elem_sz), bidirectional_no_past,
          static_cast<int>(skv)));
    }

    // ---- Steps 6-7: KV Expand [B*G,present_seq,d] -> [B*H,total_seq,d] ----
    // Skipped in no-expand mode: the Score/Value GEMMs read K/V directly from
    // the BNSD cache via per-operand batch strides instead.
    if (!use_no_expand) {
      const void *kCache = present_key ? present_key : key;
      const void *vCache = present_value ? present_value : value;
      int kvSrcStride = static_cast<int>(present_seq * d);
      int kvDstStride = static_cast<int>(total_seq * d);
      int expandCopy = static_cast<int>(total_seq * d);

      HIP_CHECK(
          hip_gqa_expand_kv(stream, kCache, d_Kexp, static_cast<int>(B * H),
                            static_cast<int>(HPG), kvSrcStride, kvDstStride,
                            expandCopy, static_cast<int>(elem_sz)));
      HIP_CHECK(
          hip_gqa_expand_kv(stream, vCache, d_Vexp, static_cast<int>(B * H),
                            static_cast<int>(HPG), kvSrcStride, kvDstStride,
                            expandCopy, static_cast<int>(elem_sz)));
    }

    // ---- Step 8: Score GEMM (fp16/fp32 in, fp32 out) ----
    // A = K: no-expand reads present_key directly, expand reads d_Kexp.
    // B = Q: need_transpose reads d_Qtrans (BNSD), else qSrc (BSHD==BNSD@sq=1).
    const void *scoreA = use_no_expand ? present_key : d_Kexp;
    const void *scoreB = need_transpose ? d_Qtrans : qSrc;
    float scoreAlpha = scale;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t sAlgo = scoreState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, scoreState->desc, &scoreAlpha, scoreA, scoreState->layA,
        scoreB, scoreState->layB, &beta, d_S_f32, scoreState->layC, d_S_f32,
        scoreState->layD, &sAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 8b: Add external attention bias (onnx.Attention attn_mask) ----
    int scoreF32BatchStride = static_cast<int>(sq * total_seq);
    if (attention_bias) {
      HIP_CHECK(hip_gqa_add_attention_bias_f32(
          stream, d_S_f32, const_cast<void *>(attention_bias),
          static_cast<int>(B * H), static_cast<int>(H),
          static_cast<int>(attn_bias_batch),
          static_cast<int>(attn_bias_num_heads), static_cast<int>(sq),
          static_cast<int>(total_seq), scoreF32BatchStride,
          static_cast<int>(elem_sz)));
    }

    // ---- Step 9: Causal Mask (fp32) + Softmax (fp32 -> fp16/fp32) ----
    // S is treated as [B*H, sq, total_seq] (head stride sq*total_seq) by both
    // GEMM flavours. softmax dtype follows gemm_fp32.
    //
    // The built-in causal triangle is applied whenever !no_causal,
    // INDEPENDENTLY of attention_bias. This mirrors the ONNX Attention
    // reference and the ORT GQA op: the additive mask (Step 8b, e.g.
    // onnx.Attention attn_mask or a GQA/ALiBi bias) is ADDED first, then, if
    // the op is causal, the upper triangle is masked out. For a mask that
    // already encodes causal (the common HF export) this is idempotent (-inf
    // stays -inf); for a padding-only mask + is_causal it supplies the missing
    // triangle. The bidirectional paths (no_causal, e.g. Whisper
    // encoder/cross-attn or onnx.Attention is_causal=0) set no_causal=true and
    // thus skip this, letting the mask carry all masking. Note this Step is a
    // no-op at sq==1 (single-query decode has no future tokens), gated by (sq >
    // 1 || local_window_size > 0).
    int scoreFp16BatchStride = static_cast<int>(sq * total_seq);
    if ((sq > 1 || local_window_size > 0) && !no_causal) {
      HIP_CHECK(hip_gqa_causal_mask_f32(
          stream, d_S_f32, static_cast<int>(B * H), static_cast<int>(total_seq),
          static_cast<int>(sq), scoreF32BatchStride, static_cast<int>(past_len),
          static_cast<int>(local_window_size)));
    }
    // fp16 GQA: softmax writes fp16 probabilities for the fp16 Value GEMM.
    // fp32 GQA (Whisper no_causal): softmax writes fp32 probabilities for the
    // fp32 Value GEMM. d_S_fp16 is the probabilities buffer either way (sized
    // by elem_sz above), so the name is fp16-specific but holds fp32 when
    // gemm_fp32.
    if (gemm_fp32) {
      HIP_CHECK(hip_gqa_softmax_f32_to_f32(
          stream, d_S_f32, d_S_fp16, static_cast<int>(B * H * sq),
          static_cast<int>(total_seq), static_cast<int>(sq),
          scoreF32BatchStride, scoreFp16BatchStride, head_sink,
          static_cast<int>(H), static_cast<int>(use_smooth_softmax)));
    } else {
      HIP_CHECK(hip_gqa_softmax_f32_to_f16(
          stream, d_S_f32, d_S_fp16, static_cast<int>(B * H * sq),
          static_cast<int>(total_seq), static_cast<int>(sq),
          scoreF32BatchStride, scoreFp16BatchStride, head_sink,
          static_cast<int>(H), static_cast<int>(use_smooth_softmax)));
    }

    // ---- Step 10: Value GEMM (fp16/fp32 in, fp16/fp32 out) ----
    // A = V: no-expand reads present_value directly, expand reads d_Vexp.
    // C = O: need_transpose writes d_O (BNSD, transposed below); at sq==1 the
    //        GEMM writes straight to output (BSHD==BNSD).
    const void *valueA = use_no_expand ? present_value : d_Vexp;
    void *valueC = need_transpose ? d_O : output;
    float valAlpha = 1.0f;
    hipblasLtMatmulAlgo_t vAlgo = valueState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueState->desc, &valAlpha, valueA, valueState->layA,
        d_S_fp16, valueState->layB, &beta, valueC, valueState->layC, valueC,
        valueState->layD, &vAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 11: O Transpose BNSD [B,H,sq,d] -> BSHD [B,sq,H,d] ----
    // Skipped at sq == 1: the Value GEMM already wrote into output.
    if (need_transpose) {
      HIP_CHECK(hip_gqa_transpose_mid_dims(
          stream, d_O, output, static_cast<int>(B), static_cast<int>(H),
          static_cast<int>(sq), static_cast<int>(d),
          static_cast<int>(elem_sz)));
    }

    RUNTIME_DEBUG_LOG(
        "[REAL] GQA hipBLASLt: B=%lld sq=%lld total_seq=%lld H=%lld G=%lld "
        "d=%lld no_expand=%d transpose=%d\n",
        (long long)B, (long long)sq, (long long)total_seq, (long long)H,
        (long long)G, (long long)d, static_cast<int>(use_no_expand),
        static_cast<int>(need_transpose));
  }

cleanup:
  return result;
}

//===----------------------------------------------------------------------===//
// Op-state slot. Owns the per-instance hipBLASLt GEMM descriptor cache that the
// decomposed pipeline (gqa_forward_hipblaslt) needs. The fused fast path holds
// no per-instance state and ignores op_state_slot.
//===----------------------------------------------------------------------===//
GqaGemmCache::~GqaGemmCache() {
  for (auto &[k, e] : entries) {
    if (e.layD)
      hipblasLtMatrixLayoutDestroy(e.layD);
    if (e.layC)
      hipblasLtMatrixLayoutDestroy(e.layC);
    if (e.layB)
      hipblasLtMatrixLayoutDestroy(e.layB);
    if (e.layA)
      hipblasLtMatrixLayoutDestroy(e.layA);
    if (e.desc)
      hipblasLtMatmulDescDestroy(e.desc);
  }
}

extern "C" int8_t hipdnn_ep_op_state_construct_gqa(RuntimeState *state,
                                                   int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, GqaState::create().release());
  return 0;
}

//===----------------------------------------------------------------------===//
// KV-cache quantization scheme handling.
//
// The quant-type integers below MUST match GqaLowering.cpp's quantTypeToEnum():
// the lowering converts the hip.gqa string attribute ("NONE"/"PER_TENSOR"/
// "PER_CHANNEL") into these values before the runtime ABI sees them. Keep the
// two in sync.
//===----------------------------------------------------------------------===//
enum GqaQuantType : int64_t {
  GQA_QUANT_NONE = 0,
  GQA_QUANT_PER_TENSOR = 1,
  GQA_QUANT_PER_CHANNEL = 2,
};

// Map the ONNX GQA quantization attributes onto a supported KvCacheFormat.
// Returns false (and logs) for any combination not (yet) implemented, so the
// caller can reject rather than silently mis-reading the cache bytes.
static bool classify_kv_cache(int64_t k_quant_type, int64_t v_quant_type,
                              int64_t kv_cache_bit_width, const void *k_scale,
                              const void *v_scale, KvCacheFormat *out) {
  const bool any_quant =
      (k_quant_type != GQA_QUANT_NONE || v_quant_type != GQA_QUANT_NONE ||
       k_scale != nullptr || v_scale != nullptr);
  if (!any_quant) {
    *out = KvCacheFormat::Fp16;
    return true;
  }

  // K and V share one typed cache buffer, so they must use the same scheme.
  if (k_quant_type != v_quant_type) {
    fprintf(stderr,
            "wrap_group_query_attention: mixed KV quantization not supported "
            "(k_quant_type=%lld v_quant_type=%lld)\n",
            (long long)k_quant_type, (long long)v_quant_type);
    return false;
  }

  // Symmetric per-channel int8: static fp32 scales [G,d], no zero point.
  if (k_quant_type == GQA_QUANT_PER_CHANNEL && kv_cache_bit_width == 8 &&
      k_scale != nullptr && v_scale != nullptr) {
    *out = KvCacheFormat::Int8PerChannel;
    return true;
  }

  // Future schemes (int4 per-channel, fp8, per-tensor, ...) would add branches
  // above. Anything not matched is rejected.
  fprintf(stderr,
          "wrap_group_query_attention: unsupported KV quantization "
          "(k_quant_type=%lld v_quant_type=%lld bit_width=%lld "
          "k_scale=%d v_scale=%d); only symmetric per-channel int8 "
          "(quant_type=PER_CHANNEL=2, bit_width=8) is supported\n",
          (long long)k_quant_type, (long long)v_quant_type,
          (long long)kv_cache_bit_width, k_scale != nullptr,
          v_scale != nullptr);
  return false;
}

//===----------------------------------------------------------------------===//
// Public wrapper called by generated IR. ABI MUST stay identical to the
// HipToLLVM lowering (kWrapGQA = "wrap_group_query_attention", 41 params).
//===----------------------------------------------------------------------===//
int wrap_group_query_attention(
    RuntimeState *state, int op_state_slot,
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (13)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width, int32_t no_causal,
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes,
    int64_t attn_bias_batch, int64_t attn_bias_num_heads) {
  OP_PROFILE(
      "gqa",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "b=%lld,sq=%lld,skv=%lld,h=%lld,d=%lld",
                 (long long)batch_size, (long long)seq_len_q,
                 (long long)seq_len_kv, (long long)num_heads,
                 (long long)head_dim);
        return std::string(b);
      },
      state);

  if (!state) {
    fprintf(stderr, "wrap_group_query_attention: null state\n");
    return -1;
  }
  if (!query || !output) {
    fprintf(stderr, "wrap_group_query_attention: null required argument\n");
    return -1;
  }
  if (kv_num_heads <= 0 || num_heads % kv_num_heads != 0) {
    fprintf(stderr,
            "wrap_group_query_attention: num_heads (%lld) must be divisible "
            "by kv_num_heads (%lld)\n",
            (long long)num_heads, (long long)kv_num_heads);
    return -1;
  }

  //===------------------------------------------------------------------===//
  // Features NEITHER the optimized fused path NOR the legacy decomposed
  // fallback implement -> reject up front. wrap_gqa_legacy ignores these
  // inputs entirely, so routing to it would silently drop them and produce
  // wrong results. present_key/present_value are required by both paths (the
  // legacy decomposed pipeline reads/writes them as the BNSD KV cache).
  //===------------------------------------------------------------------===//
  if (position_ids != nullptr) {
    fprintf(stderr, "wrap_group_query_attention: position_ids not supported\n");
    return -1;
  }
  // attention_bias (onnx.Attention external mask) is intentionally NOT rejected
  // here: the legacy decomposed pipeline applies it (Step 8b in
  // gqa_forward_hipblaslt), and fused_supported below excludes it so masked
  // attention is routed to that path rather than the lean fused kernels.
  // Resolve the KV-cache storage format from the ONNX quant attributes. The
  // classifier is the single place that decides which quantized caches are
  // supported; unsupported combinations are rejected here rather than silently
  // mis-read downstream.
  KvCacheFormat kv_format = KvCacheFormat::Fp16;
  if (!classify_kv_cache(k_quant_type, v_quant_type, kv_cache_bit_width,
                         k_scale, v_scale, &kv_format))
    return -1;
  const bool kv_quantized = (kv_format != KvCacheFormat::Fp16);
  if (output_qk != nullptr || qk_output != 0) {
    fprintf(stderr, "wrap_group_query_attention: qk_output not supported\n");
    return -1;
  }
  if (!present_key || !present_value) {
    fprintf(stderr, "wrap_group_query_attention: GQA requires "
                    "present_key/present_value KV cache\n");
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_group_query_attention: null stream\n");
    return -1;
  }

  // ORT uses scale == 0.0 as sentinel for "auto-compute 1/sqrt(head_size)"
  // (gqa_attention_base.h: scale_ == 0.0f ? 1/sqrt(head_size) : scale_).
  if (scale == 0.0f && head_dim > 0)
    scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  (void)total_seq_len;      // runtime derives total_seq from seqlens_k
  (void)rotary_interleaved; // interleaved layout handled inside hip_gqa_rope
  (void)softcap;            // softcap not applied on either path
  (void)kv_cache_bit_width; // validated above for the int8 KV path

  //===------------------------------------------------------------------===//
  // Path selection. The optimized fused/flash kernels are fp16 causal GQA
  // with head_dim in {64,128} and a templated decode geometry (HpG in
  // {1,2,3,4,8,16}), and now also cover attention sinks (decode always,
  // prefill at head_dim == 64). Anything they do not implement -- fp32,
  // no_causal / bidirectional, sliding window, a sink on 128/256-wide prefill,
  // other head_dim, or an untemplated decode geometry -- is handled by the
  // legacy decomposed hipBLASLt pipeline (gqa_forward_hipblaslt above), which is
  // the verbatim-ported gqa_back.cpp strategy: feature-complete, and keeps the
  // legacy fast decode kernel for the sliding-window case so that path is not
  // slower than the original.
  //===------------------------------------------------------------------===//
  const bool is_decode = (seq_len_q == 1);
  const bool decode_geometry_ok =
      !is_decode || flash_decode_geometry_ok(num_heads, kv_num_heads, head_dim);
  // head_dim gate: the fused WMMA prefill (v7) and the scalar flash-decode
  // kernels both cover d in {64,128,256} now (d=256 for Qwen3-family 16:4). For
  // decode, flash_decode_geometry_ok already validates d; for prefill we clamp
  // to the templated set here.
  const bool head_dim_ok =
      is_decode ? true : (head_dim == 64 || head_dim == 128 || head_dim == 256);
  // attention_bias (onnx.Attention external mask) is only applied by the legacy
  // decomposed pipeline (Step 8b); the lean fused path would silently drop it,
  // so exclude it here to route masked attention to gqa_forward_hipblaslt
  // below.
  //
  // Smooth softmax is active when a sink is supplied OR the attribute is
  // explicitly 1, matching ORT (gqa_attention_base.h:
  // use_smooth_softmax_ || head_sink != nullptr).
  const bool has_smooth_softmax = (head_sink != nullptr || smooth_softmax == 1);
  // Sinks on the fused path: the flash decode kernel applies them for any
  // supported geometry, and flash prefill applies them at head_dim == 64 (the
  // v5 kernel). Prefill at head_dim 128/256 has no sink support yet, so those
  // shapes must keep routing to the decomposed pipeline -- admitting them here
  // would make hip_gqa_flash_prefill_v3 return -1 and fail the op outright
  // instead of falling back.
  const bool sink_ok = !has_smooth_softmax || is_decode || head_dim == 64;
  const bool fused_supported = element_size_bytes == 2 && no_causal == 0 &&
                               local_window_size <= 0 && sink_ok &&
                               head_dim_ok && decode_geometry_ok &&
                               attention_bias == nullptr;

  // A quantized KV cache is implemented ONLY on the fused path (quant decode +
  // fp16 prefill-over-dequant), for head_dim in {64,128}. The legacy decomposed
  // pipeline reads the cache as fp16 and would misinterpret quantized bytes, so
  // we must reject rather than silently fall through to it.
  if (kv_quantized &&
      (!fused_supported || (head_dim != 64 && head_dim != 128))) {
    fprintf(stderr,
            "wrap_group_query_attention: quantized KV cache requires the fused "
            "path (fp16, causal, no window/bias, head_dim 64 or 128); got "
            "fused_supported=%d head_dim=%lld\n",
            static_cast<int>(fused_supported), (long long)head_dim);
    return -1;
  }

  if (!fused_supported) {
    hipblasLtHandle_t ltHandle = static_cast<hipblasLtHandle_t>(
        hipdnn_ep_state_get_hipblas_handle(state));
    if (!ltHandle) {
      fprintf(stderr, "wrap_group_query_attention: null hipblas handle\n");
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_query_attention: routing to legacy decomposed "
        "pipeline "
        "(elem=%lld no_causal=%d window=%lld sink=%d smooth=%lld d=%lld "
        "sq=%lld geom_ok=%d)\n",
        (long long)element_size_bytes, static_cast<int>(no_causal),
        (long long)local_window_size, static_cast<int>(head_sink != nullptr),
        (long long)smooth_softmax, (long long)head_dim, (long long)seq_len_q,
        static_cast<int>(decode_geometry_ok));
    int lrc = gqa_forward_hipblaslt(
        state, stream, ltHandle, query, key, value, past_key, past_value,
        seqlens_k, cos_cache, sin_cache, head_sink, has_smooth_softmax,
        attention_bias, attn_bias_batch, attn_bias_num_heads, output,
        present_key, present_value, batch_size, seq_len_q, seq_len_kv,
        past_buf_seq, num_heads, kv_num_heads, head_dim, scale, do_rotary,
        local_window_size, no_causal != 0, element_size_bytes, op_state_slot);
    if (lrc != 0)
      fprintf(stderr,
              "wrap_group_query_attention: legacy decomposed pipeline failed "
              "(rc=%d)\n",
              lrc);
    return lrc;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_group_query_attention (slim/fused): batch=%lld seq_q=%lld "
      "seq_kv=%lld H=%lld G=%lld d=%lld do_rotary=%lld packed_qkv=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (long long)do_rotary,
      static_cast<int>(key == nullptr && value == nullptr));

  int rc = gqa_forward_fused(
      state, stream, query, key, value, past_key, past_value, seqlens_k,
      cos_cache, sin_cache, output, present_key, present_value, batch_size,
      seq_len_q, seq_len_kv, past_buf_seq, num_heads, kv_num_heads, head_dim,
      scale, do_rotary, k_scale, v_scale, kv_format, head_sink,
      has_smooth_softmax);
  if (rc != 0)
    fprintf(stderr,
            "wrap_group_query_attention: gqa_forward_fused failed "
            "(rc=%d)\n",
            rc);
  return rc;
}
