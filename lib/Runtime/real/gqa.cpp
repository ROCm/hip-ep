/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GQA runtime wrapper (self-contained: optimized fused fast path + legacy
// decomposed hipBLASLt fallback).
//
// The generated IR calls `wrap_gqa_flash` (39-arg ABI, unchanged so the
// HipToLLVM lowering keeps resolving). Path selection:
//
//   * Common fp16 causal GQA (head_dim in {64,128}, templated decode geometry,
//     no sliding window / sink / smooth) -> optimized fused custom kernels:
//       prefill (sq > 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_prefill_v2
//       decode  (sq == 1): [split] -> [rope] -> kv-cache update ->
//                         hip_gqa_flash_decode_v2
//
//   * Everything else the fused kernels do not implement (fp32, no_causal /
//     bidirectional, sliding window, head sink / smooth softmax, other
//     head_dim, untemplated decode geometry) -> the feature-complete legacy
//     decomposed hipBLASLt pipeline gqa_forward_hipblaslt below. This is a
//     verbatim port of the proven gqa_back.cpp strategy (the read-only backup
//     stays out of the build); its fast decode kernels (hip_gqa_fused_decode /
//     hip_gqa_flash_decode) live in the isolated gqa_kernel_legacy.hip so the
//     sliding-window / sink decode case keeps the legacy kernel's performance.
//
//   * Inputs NEITHER path supports (KV-cache quantization, attention bias,
//     position ids, qk_output) are rejected up front.
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
// Legacy fast-path decode kernels (defined in gqa_kernel_legacy.hip, an
// isolated copy of the backup kernels). The current hip_custom_kernels.h no
// longer declares these (the production fused path uses the _v2 kernels), so
// declare them here for the decomposed-fallback fast decode path.
//===----------------------------------------------------------------------===//
extern "C" int hip_gqa_fused_decode(void *stream, const void *Q,
                                    const void *Kcache, const void *Vcache,
                                    void *O, int B, int H, int G, int d,
                                    int skv, int max_seq, float scale,
                                    const void *seqlens_k);

extern "C" int
hip_gqa_flash_decode(void *stream, const void *Q, const void *Kcache,
                     const void *Vcache, void *O, void *partials_workspace,
                     int B, int H, int G, int d, int max_seq, int K_SPLITS,
                     float scale, const void *seqlens_k, int local_window_size,
                     const void *head_sink, int use_smooth_softmax);

//===----------------------------------------------------------------------===//
// Dispatch helpers (shared by the fused and decomposed paths)
//===----------------------------------------------------------------------===//

// Per-Compute() seqlens_k D2H cache. Default on; explicit "0" disables.
static bool gqa_cache_seqlens_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Sentinel: pre-dispatch seqlens_k read not applicable / failed.
static constexpr int32_t kSeqlensKNotRead = -2;

// Read seqlens_k[0] once before dispatch (B==1 only), reusing the per-Compute()
// cache when enabled so subsequent GQA layers in the same forward pass cost
// zero D2H. Returns the raw device value (-1 is ORT's prefill sentinel) or
// kSeqlensKNotRead when not applicable.
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
// decode kernel is templated for HpG in {1,2,3,4,8,16} (so it covers MHA and
// the common GQA ratios) at d in {64,128}; WMMA is layered on top inside the
// kernel where it helps. Anything outside this set has no decode kernel.
static inline bool flash_decode_geometry_ok(int64_t H, int64_t G, int64_t d) {
  if (G <= 0)
    return false;
  if (d != 64 && d != 128)
    return false;
  int64_t hpg = H / G;
  if (hpg * G != H)
    return false;
  return hpg == 1 || hpg == 2 || hpg == 3 || hpg == 4 || hpg == 8 || hpg == 16;
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
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int past_buf_seq, int present_seq,
                           const void *seqlens_k_ptr, int elem_sz,
                           bool no_causal = false, int skv = -1) {
  // no_causal (Whisper encoder / cross-attn): bidirectional, no past KV.
  // The KV to attend over is the FULL `new_key`/`new_value` (Skv tokens), not
  // `sq` newly-appended tokens.
  if (no_causal && skv >= 0 && skv != sq) {
    // Cross-attn: key/value arrive rank-4 BNSD [B,G,skv,d]; straight D2D copy.
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
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, /*past_len=*/0,
                                /*seqlens_k_ptr=*/nullptr, elem_sz) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, /*past_len=*/0,
                                /*seqlens_k_ptr=*/nullptr, elem_sz) != 0)
      return -1;
    return 0;
  }
  if (past_key && past_len > 0 && past_key != present_key) {
    // Separate-buffer concat: needs host-side past_len for stride computation.
    if (hip_gqa_kv_cache_concat(stream, past_key, new_key, present_key, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz) != 0)
      return -1;
    if (hip_gqa_kv_cache_concat(stream, past_value, new_value, present_value, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz) != 0)
      return -1;
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr.
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr,
                                elem_sz) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, past_len, seqlens_k_ptr,
                                elem_sz) != 0)
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
    int64_t G, int64_t d, float scale, int64_t do_rotary) {

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
              "d=%lld (HpG must be 1/2/3/4/8/16 and d 64 or 128)\n",
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

    if (update_kv_cache(
            stream, past_key, past_value, kSrc, vSrc, present_key,
            present_value, static_cast<int>(B), static_cast<int>(past_len),
            static_cast<int>(sq), static_cast<int>(G), static_cast<int>(d),
            static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
            seqlens_k_ptr, static_cast<int>(elem_sz)) != 0)
      return -1;

    {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *partials = ws + off_partials;
      if (hip_gqa_flash_decode_v2(
              stream, qSrc, present_key, present_value, output, partials,
              static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(skv),
              static_cast<int>(present_seq), kFlashDecodeMaxSplits, scale,
              seqlens_k_ptr,
              /*local_window_size=*/0, /*head_sink=*/nullptr,
              /*smooth_softmax=*/0) != 0)
        return -1;
      RUNTIME_DEBUG_LOG(
          "[REAL] flash GQA decode: B=%lld skv=%lld H=%lld G=%lld "
          "d=%lld max_splits=%d\n",
          (long long)B, (long long)skv, (long long)H, (long long)G,
          (long long)d, kFlashDecodeMaxSplits);
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

  // Workspace: [split? | rope-temp?]. flash_prefill reads the BNSD present
  // cache + BSHD roped Q directly, so it needs no extra scratch.
  const size_t split_bytes =
      packed_qkv ? (Q_full_bytes + K_full_bytes + K_full_bytes) : 0;
  const size_t rope_temp_bytes = need_rope ? (Q_full_bytes + K_full_bytes) : 0;
  const size_t total_ws_bytes = split_bytes + rope_temp_bytes;
  if (total_ws_bytes > 0 &&
      hipdnn_ep_state_ensure_workspace(state, total_ws_bytes) != 0)
    return -1;
  const size_t off_split = 0;
  const size_t off_rope = off_split + split_bytes;

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

  if (present_key && present_value &&
      update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          static_cast<int>(B), static_cast<int>(past_len), static_cast<int>(sq),
          static_cast<int>(G), static_cast<int>(d),
          static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
          seqlens_k_ptr, static_cast<int>(elem_sz)) != 0)
    return -1;

  // Single unified entry; v5 (d==64) / v7 (d==128) selection lives in the
  // kernel TU (gqa_kernel.hip).
  int fp_rc = hip_gqa_flash_prefill_v2(
      stream, qSrc, present_key, present_value, output, static_cast<int>(B),
      static_cast<int>(H), static_cast<int>(G), static_cast<int>(sq),
      static_cast<int>(total_seq), static_cast<int>(d),
      static_cast<int>(present_seq), static_cast<int>(past_len), scale);
  RUNTIME_DEBUG_LOG("[REAL] GQA fused prefill (d=%lld -> v%d): B=%lld sq=%lld "
                    "total_seq=%lld H=%lld G=%lld past_len=%lld rc=%d\n",
                    (long long)d, (d == 64 ? 5 : 7), (long long)B,
                    (long long)sq, (long long)total_seq, (long long)H,
                    (long long)G, (long long)past_len, fp_rc);
  return fp_rc != 0 ? -1 : 0;
}

//===----------------------------------------------------------------------===//
// Legacy decomposed hipBLASLt pipeline (verbatim port of gqa_back.cpp's
// strategy). Reached from wrap_gqa_flash for every case the optimized fused
// path does not implement. The read-only backup gqa_back.cpp stays out of the
// build; this is the production copy.
//===----------------------------------------------------------------------===//

// Env-var gate for the group-batched "no-expand" hipBLASLt GQA pipeline.
// Default ON: the Score/Value GEMMs read K/V directly from the BNSD present
// cache via strided-batched mode, eliminating the expand_kv kernels/scratch.
static bool gqa_no_expand_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_NO_EXPAND");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Env-var gate for enabling the no-expand path on prefill (sq > 1). Default
// off: today only decode (sq == 1) takes the no-expand fast path.
static bool gqa_no_expand_prefill_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_NO_EXPAND_PREFILL");
    return v && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Env-var gate to force decode through the decomposed hipBLASLt pipeline
// instead of the fused custom kernel hip_gqa_fused_decode. Default off.
static bool gqa_fused_decode_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_DISABLE_FUSED_DECODE");
    return v && std::strcmp(v, "0") != 0;
  }();
  return disabled;
}

// Env-var gate for the FA-2 split-K flash_decode path. Default on; "0" falls
// back to hip_gqa_fused_decode.
static bool gqa_flash_decode_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FLASH_DECODE");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Smart-dispatch threshold for legacy fused decode (sq == 1). When total_seq
// exceeds this and flash_decode is not eligible, route through the decomposed
// hipBLASLt pipeline (the serial-over-time fused kernel loses on long seqs).
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

// Depth threshold for flash_decode eligibility (default skv >= 256).
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

// Geometry gate for the legacy flash_decode kernel: HPG=4 (d in {64,128}) or
// HPG=8 (d == 64). Distinct from flash_decode_geometry_ok above, which gates
// the optimized _v2 decode kernel (HpG in {1,2,3,4,8,16}).
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
// GQA GEMM descriptor cache (per-instance, keyed by GEMM shape)
//===----------------------------------------------------------------------===//
struct GqaGemmKey {
  int64_t m, n, k, batch;
  bool transA;
  bool outputFp32;
  bool inputFp32;
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

struct GqaGemmCacheEntry {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t layA, layB, layC, layD;
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
};

struct GqaGemmCache {
  std::unordered_map<GqaGemmKey, GqaGemmCacheEntry, GqaGemmKeyHash> entries;
  ~GqaGemmCache();
};

// Per-instance GQA op-state: owns this instance's per-GEMM-shape hipBLASLt
// descriptor/algorithm cache (see op-state-slots-design.md).
struct GqaState : OpStateT<GqaState> {
  GqaGemmCache cache;
};

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
    void *output, void *present_key, void *present_value, int64_t B, int64_t sq,
    int64_t skv, int64_t past_buf_seq, int64_t H, int64_t G, int64_t d,
    float scale, int64_t do_rotary, int64_t local_window_size, bool no_causal,
    int64_t element_size_bytes, int op_state_slot) {

  int64_t HPG = H / G;
  int64_t present_seq = skv;
  size_t elem_sz = static_cast<size_t>(element_size_bytes);
  bool gemm_fp32 = (elem_sz == 4);
  bool need_rope = do_rotary && cos_cache && sin_cache;

  int32_t seqlens_k_pre =
      read_seqlens_k_for_dispatch(stream, seqlens_k_ptr, B, state);
  int64_t total_seq_pre = -1;
  if (no_causal) {
    total_seq_pre = skv;
  } else if (seqlens_k_pre != kSeqlensKNotRead) {
    total_seq_pre =
        (seqlens_k_pre < 0) ? sq : static_cast<int64_t>(seqlens_k_pre) + 1;
  }

  bool fused_d = (d == 64 || d == 128 || d == 256);

  bool flash_decode_eligible = gqa_flash_decode_enabled() &&
                               legacy_flash_decode_geometry_ok(H, G, d) &&
                               skv >= gqa_flash_decode_min_skv();
  bool size_ok_for_fused =
      (total_seq_pre < 0) ||
      (total_seq_pre <= static_cast<int64_t>(gqa_fused_decode_max_t())) ||
      flash_decode_eligible;

  bool sliding_ok_for_fused = (local_window_size <= 0) || flash_decode_eligible;
  bool sink_ok_for_fused =
      (!head_sink && !use_smooth_softmax) || flash_decode_eligible;
  bool fused_packed_qkv = (!key && !value);
  bool kv_inputs_ok = (key && value) || fused_packed_qkv;
  bool fused_fp16 = (element_size_bytes == 2);
  bool fused_predicate =
      (!gqa_fused_decode_disabled() && !no_causal && fused_fp16 && fused_d &&
       sq == 1 && kv_inputs_ok && present_key && present_value &&
       sliding_ok_for_fused && sink_ok_for_fused && size_ok_for_fused);

  if (fused_predicate) {
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    int64_t past_len = 0;
    bool need_host_past_len =
        seqlens_k_ptr && past_key && past_key != present_key;
    if (need_host_past_len) {
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

    const bool use_flash_decode = gqa_flash_decode_enabled() &&
                                  legacy_flash_decode_geometry_ok(H, G, d) &&
                                  skv >= gqa_flash_decode_min_skv();

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
    }

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
  int64_t total_seq = skv;
  int64_t past_len = skv - sq;
  if (no_causal) {
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
      if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                         hipMemcpyDeviceToHost, stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
    }

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

  bool use_no_expand = gqa_no_expand_enabled() && present_key &&
                       present_value &&
                       (sq == 1 || gqa_no_expand_prefill_enabled());
  bool need_transpose = (sq > 1);

  GqaGemmKey scoreKey, valueKey;
  if (use_no_expand) {
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

  size_t off_Qroped = 0, off_Kroped = 0;
  if (need_rope) {
    size_t Q_bytes = static_cast<size_t>(B) * sq * H * d * elem_sz;
    size_t K_bytes = static_cast<size_t>(B) * sq * G * d * elem_sz;
    off_Qroped = temp_end;
    off_Kroped = off_Qroped + Q_bytes;
    temp_end = off_Kroped + K_bytes;
  }

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

    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

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

    if (need_transpose) {
      HIP_CHECK(hip_gqa_transpose_mid_dims(
          stream, qSrc, d_Qtrans, static_cast<int>(B), static_cast<int>(sq),
          static_cast<int>(H), static_cast<int>(d), static_cast<int>(elem_sz)));
    }

    if (present_key && present_value) {
      HIP_CHECK(update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          static_cast<int>(B), static_cast<int>(past_len), static_cast<int>(sq),
          static_cast<int>(G), static_cast<int>(d),
          static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
          (use_no_expand && !no_causal) ? seqlens_k_ptr : nullptr,
          static_cast<int>(elem_sz), no_causal, static_cast<int>(skv)));
    }

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

    const void *scoreA = use_no_expand ? present_key : d_Kexp;
    const void *scoreB = need_transpose ? d_Qtrans : qSrc;
    float scoreAlpha = scale;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t sAlgo = scoreState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, scoreState->desc, &scoreAlpha, scoreA, scoreState->layA,
        scoreB, scoreState->layB, &beta, d_S_f32, scoreState->layC, d_S_f32,
        scoreState->layD, &sAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    int scoreF32BatchStride = static_cast<int>(sq * total_seq);
    int scoreFp16BatchStride = static_cast<int>(sq * total_seq);
    if ((sq > 1 || local_window_size > 0) && !no_causal) {
      HIP_CHECK(hip_gqa_causal_mask_f32(
          stream, d_S_f32, static_cast<int>(B * H), static_cast<int>(total_seq),
          static_cast<int>(sq), scoreF32BatchStride, static_cast<int>(past_len),
          static_cast<int>(local_window_size)));
    }
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

    const void *valueA = use_no_expand ? present_value : d_Vexp;
    void *valueC = need_transpose ? d_O : output;
    float valAlpha = 1.0f;
    hipblasLtMatmulAlgo_t vAlgo = valueState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueState->desc, &valAlpha, valueA, valueState->layA,
        d_S_fp16, valueState->layB, &beta, valueC, valueState->layC, valueC,
        valueState->layD, &vAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

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
// Public wrapper called by generated IR. ABI MUST stay identical to the
// HipToLLVM lowering (kWrapGQA = "wrap_gqa_flash", 39 params).
//===----------------------------------------------------------------------===//
int wrap_gqa_flash(
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
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes) {
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
    fprintf(stderr, "wrap_gqa_flash: null state\n");
    return -1;
  }
  if (!query || !output) {
    fprintf(stderr, "wrap_gqa_flash: null required argument\n");
    return -1;
  }
  if (kv_num_heads <= 0 || num_heads % kv_num_heads != 0) {
    fprintf(stderr,
            "wrap_gqa_flash: num_heads (%lld) must be divisible "
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
    fprintf(stderr, "wrap_gqa_flash: position_ids not supported\n");
    return -1;
  }
  if (attention_bias != nullptr) {
    fprintf(stderr, "wrap_gqa_flash: attention_bias not supported\n");
    return -1;
  }
  if (k_scale != nullptr || v_scale != nullptr || k_quant_type != 0 ||
      v_quant_type != 0) {
    fprintf(stderr, "wrap_gqa_flash: KV cache quantization not "
                    "supported\n");
    return -1;
  }
  if (output_qk != nullptr || qk_output != 0) {
    fprintf(stderr, "wrap_gqa_flash: qk_output not supported\n");
    return -1;
  }
  if (!present_key || !present_value) {
    fprintf(stderr, "wrap_gqa_flash: GQA requires "
                    "present_key/present_value KV cache\n");
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_gqa_flash: null stream\n");
    return -1;
  }

  // ORT scale==0.0 sentinel -> auto 1/sqrt(head_dim).
  if (scale == 0.0f && head_dim > 0)
    scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  (void)total_seq_len;      // runtime derives total_seq from seqlens_k
  (void)rotary_interleaved; // interleaved layout handled inside hip_gqa_rope
  (void)softcap;            // softcap not applied on either path
  (void)kv_cache_bit_width; // KV quant rejected above

  //===------------------------------------------------------------------===//
  // Path selection. The optimized fused/flash kernels are fp16 causal GQA
  // with head_dim in {64,128} and a templated decode geometry (HpG in
  // {1,2,3,4,8,16}). Anything they do not implement -- fp32, no_causal /
  // bidirectional, sliding window, head sink / smooth softmax, other
  // head_dim, or an untemplated decode geometry -- is handled by the legacy
  // decomposed hipBLASLt pipeline (gqa_forward_hipblaslt above), which is the
  // verbatim-ported gqa_back.cpp strategy: feature-complete, and keeps the
  // legacy fast decode kernel for the sliding-window / sink decode case so
  // that path is not slower than the original. The common fp16 causal case
  // still takes the fast fused path here.
  //===------------------------------------------------------------------===//
  const bool decode_geometry_ok =
      (seq_len_q != 1) ||
      flash_decode_geometry_ok(num_heads, kv_num_heads, head_dim);
  const bool fused_supported =
      element_size_bytes == 2 && no_causal == 0 && local_window_size <= 0 &&
      head_sink == nullptr && smooth_softmax != 1 &&
      (head_dim == 64 || head_dim == 128) && decode_geometry_ok;

  if (!fused_supported) {
    hipblasLtHandle_t ltHandle = static_cast<hipblasLtHandle_t>(
        hipdnn_ep_state_get_hipblas_handle(state));
    if (!ltHandle) {
      fprintf(stderr, "wrap_gqa_flash: null hipblas handle\n");
      return -1;
    }
    const bool has_smooth_softmax =
        (head_sink != nullptr || smooth_softmax == 1);
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_gqa_flash: routing to legacy decomposed pipeline "
        "(elem=%lld no_causal=%d window=%lld sink=%d smooth=%lld d=%lld "
        "sq=%lld geom_ok=%d)\n",
        (long long)element_size_bytes, static_cast<int>(no_causal),
        (long long)local_window_size, static_cast<int>(head_sink != nullptr),
        (long long)smooth_softmax, (long long)head_dim, (long long)seq_len_q,
        static_cast<int>(decode_geometry_ok));
    int lrc = gqa_forward_hipblaslt(
        state, stream, ltHandle, query, key, value, past_key, past_value,
        seqlens_k, cos_cache, sin_cache, head_sink, has_smooth_softmax, output,
        present_key, present_value, batch_size, seq_len_q, seq_len_kv,
        past_buf_seq, num_heads, kv_num_heads, head_dim, scale, do_rotary,
        local_window_size, no_causal != 0, element_size_bytes, op_state_slot);
    if (lrc != 0)
      fprintf(stderr,
              "wrap_gqa_flash: legacy decomposed pipeline failed (rc=%d)\n",
              lrc);
    return lrc;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gqa_flash (slim/fused): batch=%lld seq_q=%lld "
      "seq_kv=%lld H=%lld G=%lld d=%lld do_rotary=%lld packed_qkv=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (long long)do_rotary,
      static_cast<int>(key == nullptr && value == nullptr));

  int rc = gqa_forward_fused(state, stream, query, key, value, past_key,
                             past_value, seqlens_k, cos_cache, sin_cache,
                             output, present_key, present_value, batch_size,
                             seq_len_q, seq_len_kv, past_buf_seq, num_heads,
                             kv_num_heads, head_dim, scale, do_rotary);
  if (rc != 0)
    fprintf(stderr,
            "wrap_gqa_flash: gqa_forward_fused failed "
            "(rc=%d)\n",
            rc);
  return rc;
}
