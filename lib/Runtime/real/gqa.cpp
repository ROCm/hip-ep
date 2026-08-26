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
//   * Common fp16 causal GQA (head_dim in {64,128,256}, templated decode
//     geometry) -> optimized fused custom kernels:
//       prefill (sq > 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_prefill
//       decode  (sq == 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_decode
//     Decode additionally covers the sliding window and the head sink /
//     smooth softmax: hip_gqa_flash_decode clamps kv_lo to
//     max(0, total_seq - window) in the split kernel and folds the sink into
//     the reduce denominator, and it autotunes (impl, split-count) per shape --
//     keying sliding layers on the window rather than the full context. Prefill
//     applies sinks and sliding windows through hip_gqa_flash_prefill for
//     head_dim == 64; unsupported prefill geometries route to decomposed.
//
//   * Everything else the fused kernels do not implement (fp32, no_causal /
//     bidirectional, additive attention bias, other head_dim, untemplated
//     decode geometry) -> the feature-complete legacy decomposed hipBLASLt
//     pipeline gqa_forward_hipblaslt below. This is a verbatim port of the
//     proven gqa_back.cpp strategy (the read-only backup stays out of the
//     build); it keeps hip_gqa_fused_decode for decode geometries the v2
//     kernels do not template.
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
#include "hip_arch_compat.h"
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
// Legacy fast-path decode kernel (folded into gqa_kernel.hip as a legacy_*
// device kernel) backs the decomposed hipBLASLt fallback below. Its entry
// hip_gqa_fused_decode is declared (and HIP_KERNEL_API exported) in
// hip_custom_kernels.h, so the EP resolves it out of custom_kernels_<arch> at
// JIT link / native import.
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
//
// copy_lo bounds the separate-buffer concat from below: past positions
// [0, copy_lo) are not copied into present and are left whatever the allocator
// held. It exists for the one caller that can prove it will not read them back
// -- a sliding-window layer at decode passes the same lower bound it scores
// from -- and must be 0 for everyone else. It is required rather than defaulted
// so that a new call site has to say which it is. The append branches ignore
// it: they only ever write the new tokens, and an in-place cache has no prefix
// to copy in the first place.
//
// Which means the narrowing is now the fallback rather than the common path:
// a model configured with past_present_share_buffer aliases present onto past
// and appends, so nothing here runs. It stays because that configuration is a
// property of the model directory, not of this repo, and a caller that ships
// without it still gets a separate buffer and still wants a narrowed copy.
//
// ring says the present buffer is a right-sized sliding-window cache holding
// only the newest present_seq positions, so the append wraps and drops anything
// older. It is meaningless on the concat branch -- a separate present is always
// allocated to the full length -- and so is required to be 0 there.
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int past_buf_seq, int present_seq,
                           const void *seqlens_k_ptr, int elem_sz, int copy_lo,
                           int ring, bool no_causal = false, int skv = -1,
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
                                HIP_KV_DTYPE_FP16, /*scale=*/nullptr,
                                /*ring=*/0) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, /*past_len=*/0,
                                /*seqlens_k_ptr=*/nullptr, elem_sz,
                                HIP_KV_DTYPE_FP16, /*scale=*/nullptr,
                                /*ring=*/0) != 0)
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
    if (ring) {
      fprintf(stderr,
              "update_kv_cache: ring append requested with separate past and "
              "present buffers (past_buf_seq=%d present_seq=%d)\n",
              past_buf_seq, present_seq);
      return -1;
    }
    if (hip_gqa_kv_cache_concat(stream, past_key, new_key, present_key, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz, kv_dtype, k_sc, copy_lo) != 0)
      return -1;
    if (hip_gqa_kv_cache_concat(stream, past_value, new_value, present_value, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz, kv_dtype, v_sc, copy_lo) != 0)
      return -1;
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr.
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr, elem_sz,
                                kv_dtype, k_sc, ring) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr, elem_sz,
                                kv_dtype, v_sc, ring) != 0)
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
    const void *head_sink = nullptr, bool use_smooth_softmax = false,
    int local_window_size = -1,
    // Additive attention mask, applied by flash decode only (sq == 1). Prefill
    // has no bias support on this path and is gated out by the caller.
    const void *attention_bias = nullptr, int64_t attn_bias_batch = 1,
    int64_t attn_bias_num_heads = 1) {

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

    // Decode has a SINGLE path: hip_gqa_flash_decode. It selects WMMA (D64/
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
    //
    // copy_lo stays 0 here even though this path knows its window. Narrowing
    // the concat needs a separate growing present buffer, and this path never
    // gets one: seq_len_kv reaches the runtime as the present_key memref's
    // sequence dim, which for a GroupQueryAttention export with a growing cache
    // arrives one short of the declared present, and the seqlens_k check above
    // then rejects the call before it can reach the concat. When the cache is
    // instead in-place there is no prefix to copy and update_kv_cache appends.
    // So a bound here would be unreachable, and therefore untestable, code.
    if (update_kv_cache(
            stream, past_key, past_value, kSrc, vSrc, present_key,
            present_value, static_cast<int>(B), static_cast<int>(past_len),
            static_cast<int>(sq), static_cast<int>(G), static_cast<int>(d),
            static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
            seqlens_k_ptr, static_cast<int>(elem_sz), /*copy_lo=*/0,
            /*ring=*/0, /*no_causal=*/false, /*skv=*/-1, kv_format, k_scale,
            v_scale) != 0)
      return -1;

    {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *partials = ws + off_partials;
      // Decode is bandwidth-bound: a quantized cache is read directly (e.g.
      // int8 halves the DRAM traffic). One entry serves every format --
      // kv_dtype selects the code path inside the kernel; scales feed dequant.
      //
      // Decode structurally clamps kv_lo to the requested local window. This
      // keeps sliding layers on the fused autotuned path rather than treating
      // the window as a decomposed score-mask operation.
      //
      // The mask is dense [bias_batch, bias_heads, 1, total_seq], so its kv
      // stride is the VALID total length, not the present buffer stride --
      // those differ whenever past and present share a max_length buffer. The
      // decomposed path strides it by total_seq for the same reason (Step 8b).
      // total_seq is only host-known here via the B==1 pre-read, which is
      // exactly the shape the caller admits a masked decode for.
      int64_t bias_kv_stride = 0;
      if (attention_bias) {
        if (seqlens_k_pre == kSeqlensKNotRead || seqlens_k_pre < 0) {
          fprintf(stderr,
                  "gqa_forward_fused (decode): attention_bias needs a host-side "
                  "total_seq, but seqlens_k could not be pre-read\n");
          return -1;
        }
        bias_kv_stride = static_cast<int64_t>(seqlens_k_pre) + 1;
        // The caller's predicate excludes a ring cache structurally (it cannot
        // read total_seq that early). Re-check it here, where total_seq IS
        // known, because the two disagreeing is silent: the kernel would index
        // the mask by ring slot instead of absolute position and return a
        // plausible-looking wrong answer rather than an error.
        if (bias_kv_stride > present_seq) {
          fprintf(stderr,
                  "gqa_forward_fused (decode): attention_bias with a ring cache "
                  "(total_seq=%lld > present_seq=%lld) indexes the mask by ring "
                  "slot, not position; this call should have been routed to the "
                  "decomposed pipeline\n",
                  (long long)bias_kv_stride, (long long)present_seq);
          return -1;
        }
      }
      int drc = hip_gqa_flash_decode(
          stream, qSrc, present_key, present_value, output, partials,
          static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
          static_cast<int>(d), static_cast<int>(skv),
          static_cast<int>(present_seq), kFlashDecodeMaxSplits, scale,
          seqlens_k_ptr, static_cast<int>(local_window_size), head_sink,
          static_cast<int>(use_smooth_softmax), kv_dtype_abi(kv_format),
          kv_quantized ? k_scale : nullptr, kv_quantized ? v_scale : nullptr,
          attention_bias, static_cast<int>(attn_bias_batch),
          static_cast<int>(attn_bias_num_heads),
          static_cast<int>(bias_kv_stride));
      if (drc != 0)
        return -1;
      RUNTIME_DEBUG_LOG(
          "[REAL] flash GQA decode (%s): B=%lld skv=%lld H=%lld G=%lld "
          "d=%lld max_splits=%d window=%lld sink=%d smooth=%d\n",
          kv_quantized ? "quant" : "fp16", (long long)B, (long long)skv,
          (long long)H, (long long)G, (long long)d, kFlashDecodeMaxSplits,
          (long long)local_window_size, static_cast<int>(head_sink != nullptr),
          static_cast<int>(use_smooth_softmax));
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
            seqlens_k_ptr, static_cast<int>(elem_sz), /*copy_lo=*/0,
            /*ring=*/0, /*no_causal=*/false, /*skv=*/-1, kv_format, k_scale,
            v_scale) != 0)
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
  // kernel TU (gqa_kernel.hip). v3 carries the sink and the window; it returns
  // -1 rather than dropping either when the kernel for this d cannot apply it,
  // and the caller then falls back to the decomposed path.
  int fp_rc = hip_gqa_flash_prefill(
      stream, qSrc, kAttn, vAttn, output, static_cast<int>(B),
      static_cast<int>(H), static_cast<int>(G), static_cast<int>(sq),
      static_cast<int>(total_seq), static_cast<int>(d), attn_max_seq,
      static_cast<int>(past_len), scale, local_window_size, head_sink,
      static_cast<int>(H), use_smooth_softmax ? 1 : 0);
  // window is logged because it selects the HAS_WINDOW instantiation, so a
  // dispatch that looks identical here can be two different kernels.
  RUNTIME_DEBUG_LOG(
      "[REAL] GQA fused prefill (%s d=%lld -> v%d): B=%lld sq=%lld "
      "total_seq=%lld H=%lld G=%lld past_len=%lld sink=%d smooth=%d window=%d "
      "rc=%d\n",
      kv_quantized ? "quant" : "fp16", (long long)d, (d == 64 ? 5 : 7),
      (long long)B, (long long)sq, (long long)total_seq, (long long)H,
      (long long)G, (long long)past_len, static_cast<int>(head_sink != nullptr),
      static_cast<int>(use_smooth_softmax), local_window_size, fp_rc);
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

// Byte budget for the pair of score matrices the decomposed prefill holds. Once
// the pair would exceed this, the prefill is tiled over query rows instead (see
// the chunking comment in gqa_forward_hipblaslt). Set
// HIPDNN_EP_GQA_SCORE_BUDGET_MB=0 to restore the untiled single-shot behaviour
// for A/B testing.
//
// The default is deliberately far above any shape that already runs well: at
// 1 GiB nothing tiles below roughly 3.5K tokens on a 16-head model, so short
// and medium sequences keep their existing kernel launch counts and descriptor
// cache entries exactly, and only the lengths whose score matrices are already
// tens of gigabytes change behaviour.
static size_t gqa_score_budget_bytes() {
  static const size_t budget = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_SCORE_BUDGET_MB");
    long long mb = 1024;
    if (v && *v) {
      char *end = nullptr;
      long long parsed = std::strtoll(v, &end, 10);
      if (end != v && parsed >= 0)
        mb = parsed;
    }
    return static_cast<size_t>(mb) * 1024u * 1024u;
  }();
  return budget;
}

// Query rows per chunk are rounded down to a multiple of this so the Score and
// Value GEMMs see a tile-friendly n. Only applied when it does not cost an
// extra chunk, since a ragged n on one chunk is cheaper than a whole extra
// pass.
static constexpr int64_t kScoreChunkAlign = 128;

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

// Env-var gate to keep a MASKED decode on the decomposed pipeline, i.e. to undo
// just the additive-bias admission in fused_supported below. Set
// HIPDNN_EP_GQA_DISABLE_FUSED_DECODE_BIAS=1 to A/B the two implementations of
// the same masking from one build. That comparison is the only way to check the
// bias is really applied: a dropped or misindexed mask shows up as a changed
// answer only where the mask actually removes keys, e.g. Gemma-4's 1024-wide
// sliding-window layers past 1024 tokens of context.
static bool gqa_fused_decode_bias_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_DISABLE_FUSED_DECODE_BIAS");
    return v && std::strcmp(v, "0") != 0;
  }();
  return disabled;
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
  // total_seq=2048 on Strix Halo), so route long sequences there. When we can't
  // read total_seq (B>1, no seqlens_k, or D2H failure) default to permitting
  // fused -- preserves behaviour on workloads that pass the predicate today.
  bool size_ok_for_fused =
      (total_seq_pre < 0) ||
      (total_seq_pre <= static_cast<int64_t>(gqa_fused_decode_max_t()));

  // hip_gqa_fused_decode implements neither the sliding window nor the head
  // sink / smooth softmax, so a windowed or sink decode cannot use it -- gate
  // it out here and let the decomposed pipeline below own those cases.
  //
  // NOTE: fp16 causal decode is the domain of hip_gqa_flash_decode, which
  // wrap_group_query_attention keeps on the fused path (it implements window +
  // sink and autotunes its split count). This function only sees the decodes v2
  // rejects for other reasons (fp32, no_causal, attention_bias, or a geometry
  // v2 does not template); the fused_decode branch below serves just the last
  // of those, and only when there is no window / sink.
  bool sliding_ok_for_fused = (local_window_size <= 0);
  bool sink_ok_for_fused = (!head_sink && !use_smooth_softmax);
  // Packed-QKV inputs (gpt-oss-20b style: query is the [B,sq,(H+2G)*d]
  // qkv_proj output, key and value are null) are supported by the fused branch
  // by routing through hip_gqa_split_qkv into workspace before rope and
  // KV-append.
  bool fused_packed_qkv = (!key && !value);
  bool kv_inputs_ok = (key && value) || fused_packed_qkv;
  // hip_gqa_fused_decode is __half-only (its Q/K/V/O pointers are
  // `const __half*`). It is correct for the fp16 causal decode of Llama /
  // gpt-oss (elem_size==2).
  // The fp32 GQA path (Whisper decoder self-attn, elem_size==4) must NOT reach
  // them: feeding fp32 buffers to a __half kernel reinterprets the bytes and
  // produces garbage. The decomposed hipBLASLt pipeline below IS fp32-capable,
  // so route fp32 decode there. This is the decode-side analogue of the
  // no_causal exemption (prefill sq>1 fp32 already used decomposed).
  bool fused_fp16 = (element_size_bytes == 2);
  // no_causal (Whisper encoder / cross-attn) always takes the decomposed
  // hipBLASLt path. hip_gqa_fused_decode is decode-only (sq==1) and reads
  // seqlens_k with the +1 PAST-token convention, plus assumes the KV cache
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
  // Fused GQA decode fallback (sq == 1, d in {64,128,256}, KV cache on).
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

    // Sum split + rope-temp in a single ensure_workspace call. ensure_workspace
    // does NOT preserve data on grow (free + malloc), so one combined request
    // avoids clobbering earlier writes; the offsets below match the call order
    // so each step's input region stays live while consumed. Region order:
    // split (Q/K/V), then rope-temp (Q/K) -- each present only when its feature
    // is active. hip_gqa_fused_decode needs no scratch of its own.
    const size_t Q_full_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
    const size_t K_full_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;
    const size_t split_bytes =
        fused_packed_qkv ? (Q_full_bytes + K_full_bytes + K_full_bytes) : 0;
    const size_t rope_temp_bytes =
        need_rope ? (Q_full_bytes + K_full_bytes) : 0;
    const size_t total_ws_bytes = split_bytes + rope_temp_bytes;

    if (total_ws_bytes > 0) {
      if (hipdnn_ep_state_ensure_workspace(state, total_ws_bytes) != 0)
        return -1;
    }

    const size_t off_split = 0;
    const size_t off_rope = off_split + split_bytes;

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
            seqlens_k_ptr, static_cast<int>(elem_sz), /*copy_lo=*/0,
            /*ring=*/0) != 0)
      return -1;

    // hip_gqa_flash_decode owns every geometry it templates; this fallback
    // only runs for the odd geometries it does not, and fused_predicate already
    // gated out window / sink (sliding_ok_for_fused / sink_ok_for_fused above).
    // Keep the defensive asserts so a future predicate change cannot silently
    // feed an unsupported feature to the __half-only kernel.
    if (local_window_size > 0) {
      fprintf(stderr,
              "gqa_forward_hipblaslt: BUG -- hip_gqa_fused_decode cannot "
              "handle local_window_size=%lld\n",
              (long long)local_window_size);
      return -1;
    }
    if (head_sink != nullptr || use_smooth_softmax) {
      fprintf(stderr,
              "gqa_forward_hipblaslt: BUG -- hip_gqa_fused_decode cannot "
              "handle head_sink=%p smooth=%d\n",
              head_sink, static_cast<int>(use_smooth_softmax));
      return -1;
    }
    // skv is passed as a fallback; kernel reads seqlens_k[b]+1 when available.
    if (hip_gqa_fused_decode(
            stream, qSrc, present_key, present_value, output,
            static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
            static_cast<int>(d), static_cast<int>(skv),
            static_cast<int>(present_seq), scale, seqlens_k_ptr) != 0)
      return -1;
    RUNTIME_DEBUG_LOG("[REAL] fused GQA decode (legacy fallback): B=%lld "
                      "sq=%lld skv=%lld H=%lld G=%lld d=%lld\n",
                      (long long)B, (long long)sq, (long long)skv, (long long)H,
                      (long long)G, (long long)d);
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
      //
      // Both failures are reported rather than returned bare: this is the one
      // place the decomposed path can fail before it has done any work, and a
      // silent -1 here surfaces only as a zero-filled attention output, which
      // looks like a kernel bug rather than a seqlens_k that could not be read.
      hipError_t cp =
          hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                         hipMemcpyDeviceToHost, stream);
      if (cp != hipSuccess) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: could not read seqlens_k from %p: %s\n",
                seqlens_k_ptr, hipGetErrorString(cp));
        return -1;
      }
      hipError_t sy = hipStreamSynchronize(stream);
      if (sy != hipSuccess) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: sync after seqlens_k read failed: %s\n",
                hipGetErrorString(sy));
        return -1;
      }
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
      // A cache right-sized to a sliding window is legitimately shorter than
      // the context, so present_seq stops being an upper bound on total_seq for
      // those layers. It still bounds a layer without a window, where a short
      // buffer means the graph and the runtime disagree about the cache. The
      // ring block further down re-checks the capacity against the window
      // before letting anything wrap.
      const bool short_ok =
          (local_window_size > 0 && present_seq >= local_window_size);
      if (total_seq < 1 || past_len < 0 ||
          ((total_seq > present_seq || past_len > past_buf_seq) && !short_ok)) {
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

  RUNTIME_DEBUG_LOG(
      "[REAL] decomposed GQA lengths: sq=%lld total_seq=%lld past_len=%lld "
      "present_seq=%lld past_buf_seq=%lld aliased=%d\n",
      (long long)sq, (long long)total_seq, (long long)past_len,
      (long long)present_seq, (long long)past_buf_seq,
      static_cast<int>(past_key == present_key));

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

  //===--------------------------------------------------------------------===//
  // Sliding-window narrowing of the decode key range
  //
  // A windowed layer can only attend to the last `window` key positions, but
  // this path scored all of them and then had hip_gqa_causal_mask_f32 write
  // -inf over everything older. On Gemma-4 at 16K that is 16x the KV traffic
  // the layer needs: 25 of its 30 layers have a 1024-token window, and their
  // measured gqa time scaled 7.3x from 2K to 16K as if the window did not
  // exist.
  //
  // Dropping the out-of-window keys is exact rather than an approximation. The
  // entries being dropped are -INFINITY: the softmax takes a max then
  // exp2f(x - max), so they never win the max and add exactly 0 to the
  // denominator, and the Value GEMM runs alpha=1/beta=0 so they contribute
  // exactly 0 to the output. What is left is a difference in GEMM tile
  // reassociation, the same standard as use_no_expand.
  //
  // The KV cache is BNSD [B, G, present_seq, d] with d fastest, so a run of key
  // positions is contiguous at + kv_lo * d and the batch stride stays
  // present_seq * d whatever the offset. The A pointer is a hipblasLtMatmul
  // call argument rather than part of the descriptor, so the offset costs
  // nothing.
  //
  // Restrictions:
  //  - Decode only (sq == 1). Windowed prefill wants a banded score matrix, not
  //    a single narrowed range, and that is a separate effort.
  //  - use_no_expand only. The expand flavour materialises Kexp/Vexp over the
  //    full range, so narrowing would have to narrow the expand copies too;
  //    that path is not the default at decode.
  //
  // A positive local_window_size is a promise that the window is ENFORCED
  // somewhere, and narrowing is exact only because of that. Both of the ways it
  // can reach us keep that promise, by different mechanisms:
  //
  //  - !no_causal: the ONNX attribute, forwarded from
  //    com.microsoft.GroupQueryAttention. hip_gqa_causal_mask_f32 below applies
  //    the window itself, so narrowing is provably the same arithmetic.
  //
  //  - no_causal: recovered from the additive mask by the converter's
  //    AttentionWindowFold, which matched `And(q >= k, q - k < W)` in the
  //    mask's own keep-condition. The mask is then the enforcer -- the entries
  //    being dropped are exactly the ones it already set to its large-negative
  //    value. This is the shape that needs it: onnx.Attention gained a window
  //    attribute only in opset 25, so a windowed export below that arrives with
  //    is_causal=0 and the window in the mask (Gemma-4's 25 local layers are
  //    exactly this).
  //
  // So no_causal is deliberately NOT a gate here. It was one while the window
  // could only arrive as an attribute, because a bidirectional op's window was
  // being ignored rather than applied; now that a window can also come from the
  // mask, gating on no_causal would reject the only case that needs narrowing.
  // Nothing else changes as a result: no producer emits a window together with
  // no_causal except the recovery above (GqaConversion sets no_causal=false
  // explicitly, and the Whisper bidirectional builders set no window).

  //===--------------------------------------------------------------------===//
  // Right-sized sliding-window cache (ring)
  //
  // OGA can allocate a sliding-window layer's cache to the window rather than
  // max_length, which on Gemma-4 is 25 of 30 layers and 3.0 GB. The buffer then
  // holds only the newest present_seq positions and the append wraps, so
  // present_seq < total_seq stops being a malformed call and becomes the steady
  // state once the context passes the window.
  //
  // Capacity is required to equal the window exactly, and that is a correctness
  // condition rather than a convenience. A ring of exactly `window` cells holds
  // precisely the positions the current query may attend to, so the causal and
  // window masks have nothing to reject and can be skipped -- which matters
  // because the mask kernel derives a column's position from its index and a
  // ring presents them rotated. Give it one cell more and the oldest slot is
  // outside the window with no way to say so.
  //===--------------------------------------------------------------------===//
  const bool ring_cache = (present_key != nullptr && present_seq < total_seq);
  if (ring_cache) {
    if (local_window_size <= 0 || present_seq != local_window_size) {
      fprintf(
          stderr,
          "gqa_forward_hipblaslt: present KV buffer (%lld) is shorter than "
          "the context (%lld), which is only supported for a sliding-window "
          "layer whose cache is right-sized to exactly the window, but "
          "local_window_size=%lld\n",
          (long long)present_seq, (long long)total_seq,
          (long long)local_window_size);
      return -1;
    }
    if (past_key != present_key) {
      fprintf(stderr,
              "gqa_forward_hipblaslt: a right-sized sliding-window cache needs "
              "past and present aliased (past_present_share_buffer), but they "
              "differ\n");
      return -1;
    }
  }

  // The two halves of ring mode, which want opposite things. A prefill scores
  // against the whole context, so it reads a full-length linear copy and only
  // the cache write wraps. A decode reads the ring itself, whose keys arrive
  // rotated.
  const bool ring_prefill = ring_cache && sq > 1;
  const bool ring_read = ring_cache && sq == 1;

  // Number of key positions actually scored, and the first one. kv_span ==
  // total_seq and kv_lo == 0 whenever narrowing is inactive, which keeps every
  // downstream expression below identical to what it was.
  int64_t kv_span = total_seq;
  int64_t kv_lo = 0;
  if (ring_cache && sq == 1) {
    // The whole ring is in-window by construction, so #795's narrowing
    // collapses: read every slot from 0 and let the ring rotation, not an
    // offset, place them. kv_lo must stay 0 -- a ring has no contiguous "first
    // scored position" to point at.
    kv_span = present_seq;
    kv_lo = 0;
  } else if (sq == 1 && use_no_expand && local_window_size > 0 &&
             local_window_size < total_seq) {
    kv_span = local_window_size;
    kv_lo = total_seq - kv_span;
  }

  //===--------------------------------------------------------------------===//
  // Query-row chunking of the score matrix
  //
  // This path materialises the whole [B*H, sq, total_seq] score matrix twice --
  // once in fp32 for the bias add, the causal mask and the softmax reduction,
  // and once in the GEMM input dtype as the probabilities the Value GEMM
  // consumes. Both are quadratic in sequence length, and together they dominate
  // every other allocation the runtime makes: at 16K on a 16-head model they
  // are 96 bytes per S^2, or 23.6 GiB, which is 97.9% of the workspace.
  //
  // The softmax here reduces along total_seq, so each query row is independent
  // of every other. A block of rows can therefore be scored, biased, masked,
  // softmaxed and multiplied by V to completion before the next block starts,
  // and the result is identical -- this is a tiling of the same arithmetic, not
  // an approximation, and it needs no running maximum or rescaling because
  // every row sees its full key range within one chunk.
  //
  // Only the two score buffers shrink. Q, K, V and O are linear in sq and stay
  // whole, so Q is read and O is written through a per-chunk offset while their
  // batch strides continue to describe the full sq.
  //
  // Chunking is skipped entirely unless the score pair exceeds the budget, in
  // which case sq_chunk == sq, the loop below runs once, the GEMM keys keep
  // their dense default strides and the behaviour is exactly what it was.
  //
  // The no-expand flavour is excluded: its Q and O operands are laid out
  // [B, G, HPG, sq, d] with n = HPG*sq, so a range of query rows is not a
  // contiguous range of n and the same offset trick does not apply. That path
  // is off by default for prefill.
  //===--------------------------------------------------------------------===//
  int64_t sq_chunk = sq;
  if (sq > 1 && !use_no_expand) {
    const size_t score_row_bytes =
        static_cast<size_t>(B) * H * total_seq * (sizeof(float) + elem_sz);
    const size_t budget = gqa_score_budget_bytes();
    if (budget > 0 && score_row_bytes > 0 &&
        static_cast<size_t>(sq) * score_row_bytes > budget) {
      int64_t rows = static_cast<int64_t>(budget / score_row_bytes);
      // A single row can already exceed the budget on a long enough context.
      // Correctness wins: one row per chunk is the floor.
      if (rows < 1)
        rows = 1;
      if (rows >= 2 * kScoreChunkAlign)
        rows -= rows % kScoreChunkAlign;
      if (rows > sq)
        rows = sq;
      sq_chunk = rows;
    }
  }
  const bool chunked = (sq_chunk < sq);
  const int64_t sq_tail = chunked ? (sq % sq_chunk) : 0;

  // GEMM descriptor keys. The no-expand flavour uses explicit per-operand
  // strides (non-zero stride fields); the expand flavour leaves them zero so
  // queryOrCreateGemmState falls back to the dense packed-batch defaults.
  //
  // When chunking, the expand flavour must state Q's and O's strides
  // explicitly: n is the chunk length but those two operands still advance by
  // the full sq between heads. The score matrices are freshly packed per chunk,
  // so their strides stay at the dense default.
  auto makeKeys = [&](int64_t n_rows, GqaGemmKey *score, GqaGemmKey *value) {
    *score = {total_seq,
              n_rows,
              d,
              B * H,
              true,
              /*outputFp32=*/true,
              gemm_fp32,
              /*strideA=*/0,
              /*strideB=*/chunked ? sq * d : 0,
              /*strideC=*/0};
    *value = {d,
              n_rows,
              total_seq,
              B * H,
              false,
              /*outputFp32=*/gemm_fp32,
              gemm_fp32,
              /*strideA=*/0,
              /*strideB=*/0,
              /*strideC=*/chunked ? sq * d : 0};
  };

  // strideA is the buffer page the K/V operand steps over per (b, g). It is the
  // present cache's own capacity, except on a ring prefill, which reads the
  // full-length workspace copy instead.
  const int64_t kv_page_seq = ring_prefill ? total_seq : present_seq;

  GqaGemmKey scoreKey, valueKey;
  if (use_no_expand) {
    // Score: C[kv_span, HPG*sq] = K^T[d,kv_span] * Q[d, HPG*sq] per (b, g)
    // pair. strideA steps over the buffer page (present_seq*d) even though only
    // kv_span tokens are read, keeping the descriptor stable across token
    // steps. Under window narrowing kv_span also stops changing per token once
    // the context passes the window, so the descriptor cache stops missing on
    // every token and the per-token hipblasLtMatmulAlgoGetHeuristic calls
    // collapse to one entry per shape.
    scoreKey = {/*m=*/kv_span,
                /*n=*/HPG * sq,
                /*k=*/d,
                /*batch=*/B * G,
                /*transA=*/true,
                /*outputFp32=*/true,
                /*inputFp32=*/gemm_fp32,
                /*strideA=*/kv_page_seq * d,
                /*strideB=*/HPG * sq * d,
                /*strideC=*/HPG * sq * kv_span};
    // Value: C[d, HPG*sq] = V[d, kv_span] * S[kv_span, HPG*sq] per (b, g)
    // pair, writing into BNSD [B, G, HPG, sq, d] which at sq==1 coincides with
    // BSHD [B, 1, H, d].
    valueKey = {/*m=*/d,
                /*n=*/HPG * sq,
                /*k=*/kv_span,
                /*batch=*/B * G,
                /*transA=*/false,
                /*outputFp32=*/gemm_fp32,
                /*inputFp32=*/gemm_fp32,
                /*strideA=*/kv_page_seq * d,
                /*strideB=*/HPG * sq * kv_span,
                /*strideC=*/HPG * sq * d};
  } else {
    makeKeys(sq_chunk, &scoreKey, &valueKey);
  }

  const GqaGemmCacheEntry *scoreState =
      queryOrCreateGemmState(state, ltHandle, scoreKey, op_state_slot);
  if (!scoreState)
    return -1;
  const GqaGemmCacheEntry *valueState =
      queryOrCreateGemmState(state, ltHandle, valueKey, op_state_slot);
  if (!valueState)
    return -1;

  // A ragged final chunk is a different n and so a different descriptor. Both
  // shapes are resolved before sizing the workspace because the GEMM workspace
  // region has to cover whichever algorithm asks for more.
  const GqaGemmCacheEntry *scoreTailState = nullptr;
  const GqaGemmCacheEntry *valueTailState = nullptr;
  if (sq_tail > 0) {
    GqaGemmKey scoreTailKey, valueTailKey;
    makeKeys(sq_tail, &scoreTailKey, &valueTailKey);
    scoreTailState =
        queryOrCreateGemmState(state, ltHandle, scoreTailKey, op_state_slot);
    if (!scoreTailState)
      return -1;
    valueTailState =
        queryOrCreateGemmState(state, ltHandle, valueTailKey, op_state_slot);
    if (!valueTailState)
      return -1;
  }

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
  // S_f32 and S_fp16 are always allocated (softmax is on every path), and are
  // sized to one query chunk rather than the whole query range -- sq_chunk ==
  // sq whenever chunking is inactive, so this is the full matrix in that case.
  size_t Qtrans_bytes =
      need_transpose ? static_cast<size_t>(B) * H * sq * d * elem_sz : 0;
  size_t Kexp_bytes =
      use_no_expand ? 0 : static_cast<size_t>(B) * H * total_seq * d * elem_sz;
  size_t Vexp_bytes = Kexp_bytes;
  // Both score buffers span kv_span keys, not total_seq: they hold exactly what
  // the Score GEMM writes. These offsets chain into off_S_fp16, off_O, temp_end
  // and the GEMM workspace, so kv_span has to appear in both or the region
  // boundaries disagree with the GEMM extents and it is a silent heap
  // overwrite rather than a crash.
  size_t S_f32_bytes =
      static_cast<size_t>(B) * H * sq_chunk * kv_span * sizeof(float);
  size_t S_fp16_bytes =
      static_cast<size_t>(B) * H * sq_chunk * kv_span * elem_sz;
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

  // Full-length BNSD K/V for a ring prefill. A prefill scores every query
  // against every earlier key, so it needs all total_seq keys laid out
  // contiguously -- but the ring only has `window` cells, and the incoming
  // kSrc/vSrc are BSHD, which is not the layout the expand step and the
  // no-expand GEMM strides read. So the prefill gets a transient BNSD copy of
  // the full range here, and the persistent cache separately receives just the
  // tail the next decode will read. This is workspace, sized once and reused by
  // every layer, not 25 more caches: it is the same order as Kexp/Vexp
  // alongside it, and it disappears when prefill does.
  size_t off_Kfull = 0, off_Vfull = 0;
  if (ring_prefill) {
    size_t KV_bytes = static_cast<size_t>(B) * G * total_seq * d * elem_sz;
    off_Kfull = temp_end;
    off_Vfull = off_Kfull + KV_bytes;
    temp_end = off_Vfull + KV_bytes;
  }

  int result = 0;

  // Single workspace allocation: temp buffers + GEMM workspace.
  {
    size_t gemm_ws =
        std::max(scoreState->workspace_size, valueState->workspace_size);
    if (scoreTailState)
      gemm_ws = std::max(gemm_ws, scoreTailState->workspace_size);
    if (valueTailState)
      gemm_ws = std::max(gemm_ws, valueTailState->workspace_size);
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
      //
      // copy_lo is kv_lo, deliberately the identical value and not a separately
      // derived one: the bytes this skips writing are then exactly the bytes
      // the Score and Value GEMMs below skip reading, so the narrowing cannot
      // be half-applied. That equality is the whole safety argument, and it is
      // why this needs no gate of its own -- kv_lo is already 0 for prefill,
      // for the expand flavour and whenever no window applies.
      //
      // It holds across steps as well as within one: this step writes
      // [kv_lo, total_seq) and the next step reads from kv_lo + 1 (the window
      // slides by exactly the one token it appends), so every step's read range
      // is a subset of the previous step's written range.
      //
      // Only the separate-buffer concat can skip anything. When the cache is
      // in-place (past_key == present_key) the prefix is already there and
      // update_kv_cache appends, ignoring this.
      HIP_CHECK(update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          static_cast<int>(B), static_cast<int>(past_len), static_cast<int>(sq),
          static_cast<int>(G), static_cast<int>(d),
          static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
          (use_no_expand && !bidirectional_no_past) ? seqlens_k_ptr : nullptr,
          static_cast<int>(elem_sz), static_cast<int>(kv_lo),
          ring_cache ? 1 : 0, bidirectional_no_past, static_cast<int>(skv)));

      // Ring prefill: the append above stored the tail the next decode reads.
      // The attention below still needs the whole range, so lay it out BNSD in
      // workspace. past_len is 0 for a prefill, so this is a plain linear
      // append into a buffer that is exactly total_seq long.
      if (ring_prefill) {
        HIP_CHECK(update_kv_cache(
            stream, /*past_key=*/nullptr, /*past_value=*/nullptr, kSrc, vSrc,
            ws + off_Kfull, ws + off_Vfull, static_cast<int>(B),
            /*past_len=*/0, static_cast<int>(sq), static_cast<int>(G),
            static_cast<int>(d), /*past_buf_seq=*/0,
            static_cast<int>(total_seq), /*seqlens_k_ptr=*/nullptr,
            static_cast<int>(elem_sz), /*copy_lo=*/0, /*ring=*/0,
            /*no_causal=*/false, /*skv=*/-1));
      }
    }

    // ---- Steps 6-7: KV Expand [B*G,present_seq,d] -> [B*H,total_seq,d] ----
    // Skipped in no-expand mode: the Score/Value GEMMs read K/V directly from
    // the BNSD cache via per-operand batch strides instead.
    if (!use_no_expand) {
      // A ring prefill expands out of the full-length workspace copy, whose
      // batch stride is total_seq rather than the ring's capacity.
      const void *kCache =
          ring_prefill ? (ws + off_Kfull) : (present_key ? present_key : key);
      const void *vCache = ring_prefill
                               ? (ws + off_Vfull)
                               : (present_value ? present_value : value);
      int kvSrcStride =
          static_cast<int>((ring_prefill ? total_seq : present_seq) * d);
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

    // ---- Steps 8-10, per query chunk ----
    // One iteration when chunking is inactive, in which case q0 is 0, c is sq
    // and every pointer and stride below reduces to what it was.
    //
    // K and V are advanced to the first key position in range. The cache is
    // BNSD with d fastest, so this selects a contiguous span and leaves the
    // per-(b,g) batch stride at present_seq*d. kv_lo is 0 unless the sliding
    // window narrowed the range.
    // A ring prefill reads the full-length workspace copy instead of the cache,
    // for the same reason the expand step above does: the ring holds only the
    // window, and a prefill scores against everything before it.
    const size_t kv_byte_off = static_cast<size_t>(kv_lo) * d * elem_sz;
    const char *noExpandK = ring_prefill
                                ? (ws + off_Kfull)
                                : static_cast<const char *>(present_key);
    const char *noExpandV = ring_prefill
                                ? (ws + off_Vfull)
                                : static_cast<const char *>(present_value);
    const void *scoreA = (use_no_expand ? noExpandK + kv_byte_off
                                        : static_cast<const char *>(d_Kexp));
    const void *scoreBBase = need_transpose ? d_Qtrans : qSrc;
    const void *valueA = (use_no_expand ? noExpandV + kv_byte_off
                                        : static_cast<const char *>(d_Vexp));
    void *valueCBase = need_transpose ? d_O : output;
    float scoreAlpha = scale;
    float beta = 0.0f;
    float valAlpha = 1.0f;

    for (int64_t q0 = 0; q0 < sq; q0 += sq_chunk) {
      const int64_t c = std::min<int64_t>(sq_chunk, sq - q0);
      const bool is_tail = (c != sq_chunk);
      const GqaGemmCacheEntry *sSt = is_tail ? scoreTailState : scoreState;
      const GqaGemmCacheEntry *vSt = is_tail ? valueTailState : valueState;

      // Q and O keep their full-sq batch strides (stated in the GEMM key), so
      // selecting a chunk is a plain offset into the query axis of each.
      const void *scoreB = static_cast<const char *>(scoreBBase) +
                           static_cast<size_t>(q0) * d * elem_sz;
      void *valueC = static_cast<char *>(valueCBase) +
                     static_cast<size_t>(q0) * d * elem_sz;
      // The score buffers are re-packed per chunk, so their per-head stride is
      // the chunk's own row count over the key range actually scored.
      const int scoreBatchStride = static_cast<int>(c * kv_span);

      // ---- Step 8: Score GEMM (fp16/fp32 in, fp32 out) ----
      // A = K: no-expand reads present_key directly, expand reads d_Kexp.
      // B = Q: need_transpose reads d_Qtrans (BNSD), else qSrc
      // (BSHD==BNSD@sq=1).
      hipblasLtMatmulAlgo_t sAlgo = sSt->algo;
      HIPBLAS_CHECK(hipblasLtMatmul(
          ltHandle, sSt->desc, &scoreAlpha, scoreA, sSt->layA, scoreB,
          sSt->layB, &beta, d_S_f32, sSt->layC, d_S_f32, sSt->layD, &sAlgo,
          gemm_ws_ptr, gemm_ws_bytes, stream));

      // ---- Step 8b: Add external attention bias (onnx.Attention attn_mask) --
      // The bias is indexed over the full query and key ranges, so the chunk's
      // row offset and the window's key offset are passed through rather than
      // folded into the pointer: with bias_batch or bias_heads > 1 the plane
      // stride still spans all sq rows and all total_seq columns.
      if (attention_bias) {
        HIP_CHECK(hip_gqa_add_attention_bias_f32(
            stream, d_S_f32, const_cast<void *>(attention_bias),
            static_cast<int>(B * H), static_cast<int>(H),
            static_cast<int>(attn_bias_batch),
            static_cast<int>(attn_bias_num_heads), static_cast<int>(c),
            static_cast<int>(kv_span), scoreBatchStride,
            static_cast<int>(elem_sz), static_cast<int>(sq),
            static_cast<int>(q0), static_cast<int>(total_seq),
            static_cast<int>(kv_lo),
            static_cast<int>(ring_read ? total_seq - present_seq : 0),
            static_cast<int>(ring_read ? present_seq : 0)));
      }

      // ---- Step 9: Causal Mask (fp32) + Softmax (fp32 -> fp16/fp32) ----
      // S is treated as [B*H, c, kv_span] (head stride c*kv_span) by both
      // GEMM flavours. softmax dtype follows gemm_fp32.
      //
      // The built-in causal triangle is applied whenever !no_causal,
      // INDEPENDENTLY of attention_bias. This mirrors the ONNX Attention
      // reference and the ORT GQA op: the additive mask (Step 8b, e.g.
      // onnx.Attention attn_mask or a GQA/ALiBi bias) is ADDED first, then, if
      // the op is causal, the upper triangle is masked out. For a mask that
      // already encodes causal (the common HF export) this is idempotent (-inf
      // stays -inf); for a padding-only mask + is_causal it supplies the
      // missing triangle. The bidirectional paths (no_causal, e.g. Whisper
      // encoder/cross-attn or onnx.Attention is_causal=0) set no_causal=true
      // and thus skip this, letting the mask carry all masking. Note this Step
      // is a no-op at sq==1 (single-query decode has no future tokens), gated
      // by (sq > 1 || local_window_size > 0).
      //
      // The mask kernel derives each row's absolute query position as
      // past_len + row, so advancing past_len by the chunk's start puts the
      // triangle and the sliding window in the right place for the chunk.
      // Under window narrowing the score columns start at kv_lo rather than 0,
      // and since both of the kernel's predicates reference the query position
      // only as (past_len_arg + row), shifting past_len down by kv_lo is
      // exactly equivalent to shifting the column index up by it. No kernel
      // change needed. (At sq == 1 the narrowed range is precisely the window,
      // so the kernel then writes nothing -- correct, just redundant.)
      // A ring read is exempt, and has to be: the kernel derives a column's
      // absolute position from its index, which a rotated ring breaks. It is
      // also unnecessary, and for the same reason the capacity is pinned to the
      // window -- a ring of exactly `window` cells holds exactly the positions
      // the single query may attend to, so both the triangle and the window
      // admit every column and the kernel would write nothing.
      if ((sq > 1 || local_window_size > 0) && !no_causal && !ring_read) {
        HIP_CHECK(hip_gqa_causal_mask_f32(
            stream, d_S_f32, static_cast<int>(B * H), static_cast<int>(kv_span),
            static_cast<int>(c), scoreBatchStride,
            static_cast<int>(past_len + q0 - kv_lo),
            static_cast<int>(local_window_size)));
      }
      // fp16 GQA: softmax writes fp16 probabilities for the fp16 Value GEMM.
      // fp32 GQA (Whisper no_causal): softmax writes fp32 probabilities for the
      // fp32 Value GEMM. d_S_fp16 is the probabilities buffer either way (sized
      // by elem_sz above), so the name is fp16-specific but holds fp32 when
      // gemm_fp32.
      if (gemm_fp32) {
        HIP_CHECK(hip_gqa_softmax_f32_to_f32(
            stream, d_S_f32, d_S_fp16, static_cast<int>(B * H * c),
            static_cast<int>(kv_span), static_cast<int>(c), scoreBatchStride,
            scoreBatchStride, head_sink, static_cast<int>(H),
            static_cast<int>(use_smooth_softmax)));
      } else {
        HIP_CHECK(hip_gqa_softmax_f32_to_f16(
            stream, d_S_f32, d_S_fp16, static_cast<int>(B * H * c),
            static_cast<int>(kv_span), static_cast<int>(c), scoreBatchStride,
            scoreBatchStride, head_sink, static_cast<int>(H),
            static_cast<int>(use_smooth_softmax)));
      }

      // ---- Step 10: Value GEMM (fp16/fp32 in, fp16/fp32 out) ----
      // A = V: no-expand reads present_value directly, expand reads d_Vexp.
      // C = O: need_transpose writes d_O (BNSD, transposed below); at sq==1 the
      //        GEMM writes straight to output (BSHD==BNSD).
      hipblasLtMatmulAlgo_t vAlgo = vSt->algo;
      HIPBLAS_CHECK(hipblasLtMatmul(
          ltHandle, vSt->desc, &valAlpha, valueA, vSt->layA, d_S_fp16,
          vSt->layB, &beta, valueC, vSt->layC, valueC, vSt->layD, &vAlgo,
          gemm_ws_ptr, gemm_ws_bytes, stream));
    }

    // ---- Step 11: O Transpose BNSD [B,H,sq,d] -> BSHD [B,sq,H,d] ----
    // Skipped at sq == 1: the Value GEMM already wrote into output.
    if (need_transpose) {
      HIP_CHECK(hip_gqa_transpose_mid_dims(
          stream, d_O, output, static_cast<int>(B), static_cast<int>(H),
          static_cast<int>(sq), static_cast<int>(d),
          static_cast<int>(elem_sz)));
    }

    // kv_inplace and the three sequence lengths are here because they decide
    // whether any of this narrowing does anything: the copy can only be skipped
    // on the separate-buffer branch, which is exactly kv_inplace=0, and that in
    // turn is visible only as present_seq differing from past_buf_seq. A model
    // exporting its cache in place takes the append branch and the copy_lo
    // below is inert, which is otherwise indistinguishable from the narrowing
    // failing to engage.
    RUNTIME_DEBUG_LOG(
        "[REAL] GQA hipBLASLt: B=%lld sq=%lld sq_chunk=%lld total_seq=%lld "
        "kv_lo=%lld kv_span=%lld H=%lld G=%lld d=%lld no_expand=%d "
        "transpose=%d kv_inplace=%d past_len=%lld past_buf_seq=%lld "
        "present_seq=%lld\n",
        (long long)B, (long long)sq, (long long)sq_chunk, (long long)total_seq,
        (long long)kv_lo, (long long)kv_span, (long long)H, (long long)G,
        (long long)d, static_cast<int>(use_no_expand),
        static_cast<int>(need_transpose),
        static_cast<int>(past_key != nullptr && past_key == present_key),
        (long long)past_len, (long long)past_buf_seq, (long long)present_seq);
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
  // Path selection. The optimized fused/flash kernels are fp16 causal GQA with
  // head_dim in {64,128,256} and a templated decode geometry (HpG in
  // {1,2,3,4,5,8,16}). Decode supports sink/window for every templated
  // geometry; prefill v3 supports them at head_dim == 64. Everything else uses
  // the feature-complete decomposed hipBLASLt fallback.
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
  // Sink/window are first-class decode features. Prefill v3 implements both at
  // D64; D128/D256 prefill must fall back rather than silently drop them.
  const bool sink_ok = !has_smooth_softmax || is_decode || head_dim == 64;
  // Decode handles the window itself (kv_lo clamp), so it is allowed for any
  // supported geometry; prefill only implements it at D64. This subsumes the
  // narrower prefill-only form, so nothing prefill accepted before is widened.
  const bool window_ok = is_decode || local_window_size <= 0 || head_dim == 64;
  // The whole fused path (gqa_forward_fused, including the sink and windowed
  // prefill kernels above) is built on the RDNA-only WMMA intrinsics, which
  // trap on CDNA (wave64, e.g. MI350). Route wave64 to the decomposed hipBLASLt
  // pipeline below (MFMA GEMMs + wave-portable scalar kernels), which is
  // feature-complete and correct on both wave sizes. RDNA is unaffected.
  // A masked decode CAN take the fused path: flash decode folds the additive
  // mask into its online softmax. Three conditions make that sound, and all
  // three are decode-only, which is why prefill stays excluded:
  //   - sq == 1, so the mask is a single row and the kernel's dense
  //     [bias_batch, bias_heads, 1, total_seq] view is exact;
  //   - batch 1, so total_seq (the mask's kv stride) is host-known from the
  //     seqlens_k pre-read;
  //   - fp16 KV, since the int8 launcher does not thread the mask.
  // Flash prefill has no bias support at all, so a masked prefill must keep
  // going decomposed or the mask would be silently dropped.
  //
  // A ring cache is the fourth condition, and it is not a limitation of the
  // mask so much as of indexing. flash decode reads the present buffer slot by
  // slot and indexes the mask by that slot, which is only the key's absolute
  // position while the buffer is linear. A cache right-sized to a sliding
  // window holds position p at p % capacity, so once the context passes the
  // window the slots arrive rotated and every mask entry lands on the wrong
  // key. The decomposed path already knows this -- it passes a ring base and
  // capacity into hip_gqa_add_attention_bias_f32 and skips its causal/window
  // kernel entirely under ring_read, for exactly this reason -- and the fused
  // path has no equivalent yet.
  //
  // The test is structural rather than a comparison against total_seq, which is
  // not host-known this early: the ring block in gqa_forward_hipblaslt admits a
  // short buffer ONLY when it is right-sized to the window exactly, so a buffer
  // longer than the window can never rotate. That makes this conservative by
  // one phase -- a right-sized layer is excluded even before the context
  // exceeds its window, where slot and position still agree -- which costs a
  // few early tokens on the decomposed path and needs no device read.
  const bool ring_cache_possible =
      local_window_size > 0 && seq_len_kv <= local_window_size;
  const bool bias_ok =
      attention_bias == nullptr ||
      (is_decode && batch_size == 1 && !kv_quantized && !ring_cache_possible &&
       !gqa_fused_decode_bias_disabled());
  // no_causal on a masked decode is likewise safe: is_causal=0 exports (Gemma-4)
  // carry causal, padding AND any sliding window inside the additive mask, and
  // the built-in causal triangle is a documented no-op at sq == 1 anyway (see
  // Step 9), so admitting these costs no masking. Unmasked no_causal is
  // bidirectional attention and still belongs on the decomposed path.
  const bool causal_ok =
      no_causal == 0 || (is_decode && attention_bias != nullptr);
  const bool fused_supported =
      element_size_bytes == 2 && causal_ok && window_ok && sink_ok &&
      head_dim_ok && decode_geometry_ok && bias_ok &&
      !hipdnn_device_is_wave64();

  // The quantized-cache kernels live exclusively on the fused path, which is
  // disabled above on wave64 because it is built on the RDNA-only WMMA
  // intrinsics. A quantized cache therefore has no implementation at all on
  // CDNA. Reject it here, naming the architecture: the generic check below
  // would only report fused_supported=0, pointing a reader at the feature gates
  // (causal / window / sink / bias) instead of the real reason.
  if (kv_quantized && hipdnn_device_is_wave64()) {
    fprintf(
        stderr,
        "wrap_group_query_attention: quantized KV cache is not supported on "
        "wave64 devices (CDNA, e.g. MI350) -- its kernels are on the WMMA "
        "fused path, which is RDNA-only. Use an fp16 KV cache.\n");
    return -1;
  }

  // A quantized KV cache is implemented ONLY on the fused path (quant decode +
  // fp16 prefill-over-dequant), for head_dim in {64,128}. The legacy decomposed
  // pipeline reads the cache as fp16 and would misinterpret quantized bytes, so
  // we must reject rather than silently fall through to it.
  if (kv_quantized &&
      (!fused_supported || (head_dim != 64 && head_dim != 128))) {
    fprintf(stderr,
            "wrap_group_query_attention: quantized KV cache requires the fused "
            "path (fp16, causal, no attention bias, any window/sink only where "
            "the fused kernels implement it, head_dim 64 or 128); got "
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
      has_smooth_softmax, static_cast<int>(local_window_size), attention_bias,
      attn_bias_batch, attn_bias_num_heads);
  if (rc != 0)
    fprintf(stderr,
            "wrap_group_query_attention: gqa_forward_fused failed "
            "(rc=%d)\n",
            rc);
  return rc;
}
