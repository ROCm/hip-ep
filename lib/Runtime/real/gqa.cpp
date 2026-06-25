/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Slim GQA runtime wrapper.
//
// This is the matmul_nbits-style thin entry: the generated IR calls
// `wrap_gqa_flash` (same 39-arg ABI as before, unchanged so the
// HipToLLVM lowering keeps resolving), and we route straight into the
// optimized fused custom kernels with the minimum prep:
//
//   prefill (sq > 1): [split] -> [rope] -> kv-cache update -> hip_gqa_flash_prefill_v2
//   decode  (sq == 1): [split] -> [rope] -> kv-cache update -> hip_gqa_flash_decode_v2
//
// The previous decomposed hipBLASLt pipeline, the per-instance GEMM descriptor
// cache, and all env-var dispatch gates are gone. Features the fused kernels do
// not implement (fp32, bidirectional/no_causal, sliding window, head sink /
// smooth softmax, KV-cache quantization, attention bias, position ids,
// qk_output) are rejected up front -- "not supported for now".
//===----------------------------------------------------------------------===//

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

//===----------------------------------------------------------------------===//
// Dispatch helpers (fused path only)
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
// decode kernel is templated for HpG in {1,2,3,4,8,16} (so it covers MHA and the
// common GQA ratios) at d in {64,128}; WMMA is layered on top inside the kernel
// where it helps. Anything outside this set has no decode kernel.
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
// KV cache update (causal): concat past+new, or append new tokens in place.
//===----------------------------------------------------------------------===//
// past_buf_seq is the buffer dim of past_key (may exceed past_len for
// pre-allocated caches). seqlens_k_ptr: when non-null on the append path the
// kernel reads past_len from device memory (zero D2H). Returns 0 on success.
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int past_buf_seq, int present_seq,
                           const void *seqlens_k_ptr, int elem_sz) {
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
                  (long long)total_seq, (long long)sq, (long long)past_len_check,
                  (long long)present_seq, (long long)past_buf_seq);
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
    const size_t rope_temp_bytes = need_rope ? (Q_full_bytes + K_full_bytes) : 0;
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

    if (update_kv_cache(stream, past_key, past_value, kSrc, vSrc, present_key,
                        present_value, static_cast<int>(B),
                        static_cast<int>(past_len), static_cast<int>(sq),
                        static_cast<int>(G), static_cast<int>(d),
                        static_cast<int>(past_buf_seq),
                        static_cast<int>(present_seq), seqlens_k_ptr,
                        static_cast<int>(elem_sz)) != 0)
      return -1;

    {
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *partials = ws + off_partials;
      if (hip_gqa_flash_decode_v2(stream, qSrc, present_key, present_value, output,
                               partials, static_cast<int>(B),
                               static_cast<int>(H), static_cast<int>(G),
                               static_cast<int>(d), static_cast<int>(skv),
                               static_cast<int>(present_seq),
                               kFlashDecodeMaxSplits, scale, seqlens_k_ptr,
                               /*local_window_size=*/0, /*head_sink=*/nullptr,
                               /*smooth_softmax=*/0) != 0)
        return -1;
      RUNTIME_DEBUG_LOG("[REAL] flash GQA decode: B=%lld skv=%lld H=%lld G=%lld "
                        "d=%lld max_splits=%d\n",
                        (long long)B, (long long)skv, (long long)H,
                        (long long)G, (long long)d, kFlashDecodeMaxSplits);
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
        fprintf(stderr,
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
      update_kv_cache(stream, past_key, past_value, kSrc, vSrc, present_key,
                      present_value, static_cast<int>(B),
                      static_cast<int>(past_len), static_cast<int>(sq),
                      static_cast<int>(G), static_cast<int>(d),
                      static_cast<int>(past_buf_seq),
                      static_cast<int>(present_seq), seqlens_k_ptr,
                      static_cast<int>(elem_sz)) != 0)
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
                    (long long)d, (d == 64 ? 5 : 7), (long long)B, (long long)sq,
                    (long long)total_seq, (long long)H, (long long)G,
                    (long long)past_len, fp_rc);
  return fp_rc != 0 ? -1 : 0;
}

//===----------------------------------------------------------------------===//
// Op-state slot: kept as an empty slot so the compiler-emitted
// inference_init -> hipdnn_ep_op_state_construct_gqa chain still resolves.
// The slim path holds no per-instance state (the hipBLASLt GEMM cache is gone).
//===----------------------------------------------------------------------===//
struct GqaState : OpStateT<GqaState> {};

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
  // Fused-path constraints. Everything below is "not supported for now" --
  // the slim path is fp16 causal GQA only and routes straight to the fused
  // kernels. (Was: decomposed hipBLASLt fallback handled these.)
  //===------------------------------------------------------------------===//
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "wrap_gqa_flash: fused path is fp16-only "
            "(element_size=2), got %lld -- not supported\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (no_causal != 0) {
    fprintf(stderr, "wrap_gqa_flash: no_causal/bidirectional "
                    "attention not supported on the fused path\n");
    return -1;
  }
  if (local_window_size > 0) {
    fprintf(stderr, "wrap_gqa_flash: sliding window "
                    "(local_window_size=%lld) not supported on the fused "
                    "path\n",
            (long long)local_window_size);
    return -1;
  }
  if (head_sink != nullptr || smooth_softmax == 1) {
    fprintf(stderr, "wrap_gqa_flash: head_sink / smooth_softmax "
                    "not supported on the fused path\n");
    return -1;
  }
  if (head_dim != 64 && head_dim != 128) {
    fprintf(stderr,
            "wrap_gqa_flash: fused path supports head_dim in "
            "{64,128}, got %lld -- not supported\n",
            (long long)head_dim);
    return -1;
  }
  if (!present_key || !present_value) {
    fprintf(stderr, "wrap_gqa_flash: fused path requires "
                    "present_key/present_value KV cache\n");
    return -1;
  }
  if (position_ids != nullptr) {
    fprintf(stderr,
            "wrap_gqa_flash: position_ids not supported\n");
    return -1;
  }
  if (attention_bias != nullptr) {
    fprintf(stderr,
            "wrap_gqa_flash: attention_bias not supported\n");
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

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_gqa_flash: null stream\n");
    return -1;
  }

  // ORT scale==0.0 sentinel -> auto 1/sqrt(head_dim).
  if (scale == 0.0f && head_dim > 0)
    scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  (void)total_seq_len;       // runtime derives total_seq from seqlens_k
  (void)rotary_interleaved;  // interleaved layout handled inside hip_gqa_rope
  (void)softcap;             // not applied on the fused path
  (void)kv_cache_bit_width;  // KV quant rejected above
  (void)op_state_slot;       // slim path is stateless

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gqa_flash (slim/fused): batch=%lld seq_q=%lld "
      "seq_kv=%lld H=%lld G=%lld d=%lld do_rotary=%lld packed_qkv=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (long long)do_rotary, static_cast<int>(key == nullptr && value == nullptr));

  int rc = gqa_forward_fused(state, stream, query, key, value, past_key,
                             past_value, seqlens_k, cos_cache, sin_cache, output,
                             present_key, present_value, batch_size, seq_len_q,
                             seq_len_kv, past_buf_seq, num_heads, kv_num_heads,
                             head_dim, scale, do_rotary);
  if (rc != 0)
    fprintf(stderr, "wrap_gqa_flash: gqa_forward_fused failed "
                    "(rc=%d)\n",
            rc);
  return rc;
}
