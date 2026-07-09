/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

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

// Env-var gate for enabling the no-expand path on prefill (sq > 1).
// Default off: today only decode (sq == 1) takes the no-expand fast path,
// matching the verified pre-step-2 behaviour. Set
// HIPDNN_EP_GQA_NO_EXPAND_PREFILL=1 to opt prefill into the same
// group-batched pipeline -- same strides as decode, but with Q/O transpose
// kernels kept in place because BSHD and BNSD diverge at sq > 1.
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
// A/B against decomposed at sq==1 -- useful for measuring whether the
// custom fused kernel is actually faster than hipBLASLt's auto-tuned GEMMs
// at decode shapes.
static bool gqa_fused_decode_disabled() {
  static const bool disabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_DISABLE_FUSED_DECODE");
    return v && std::strcmp(v, "0") != 0;
  }();
  return disabled;
}

// Env-var gate to cache seqlens_k_val across the GQA layers in a single
// forward pass. Default ON. Caching skips the per-layer
// hipMemcpyAsync(D2H) + hipStreamSynchronize on the decomposed path after
// the first GQA call -- a 32-layer Llama decode then issues one D2H
// instead of 32, eliminating ~30-45 ms/token of pipeline stalls on Strix
// Halo. Set HIPDNN_EP_GQA_CACHE_SEQLENS=0 to disable (escape hatch for
// running against an older per-model bitcode without the begin_compute
// export, or for A/B measurement).
//
// Correctness depends on the EP-side MlirCustomOp::Compute() invoking
// hipdnn_ep_runtime_begin_compute(state) at the start of each forward
// pass to invalidate the cache. Older per-model bitcode without that
// symbol exported is detected at session creation and produces a
// LOG(WARNING) directing the user to set HIPDNN_EP_GQA_CACHE_SEQLENS=0
// (otherwise the cache would survive across forward passes and return
// stale total_seq values).
static bool gqa_cache_seqlens_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
    // Default on; explicit "0" disables.
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Env-var gates for the FA-2 split-K flash_decode path (Phase 1).
// HIPDNN_EP_GQA_FLASH_DECODE=0 disables it (falls back to
// hip_gqa_fused_decode). HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV overrides the depth
// threshold (default 256). flash_decode wins at high KV depth where the
// existing one-block-per-head kernel is bandwidth-bound; below the threshold
// its 2-kernel overhead may not pay back, so we keep the existing fused_decode
// for short sequences.
static bool gqa_flash_decode_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FLASH_DECODE");
    return !v || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// Smart-dispatch threshold for GQA decode (sq == 1). When total_seq exceeds
// this value, dispatch routes through the decomposed hipBLASLt pipeline
// instead of the fused custom kernel hip_gqa_fused_decode. The fused kernel
// uses a serial-over-time scheme with cross-wave reductions on the critical
// path of every iteration, so it loses to the GEMM-based decomposed path on
// long sequences (measured ~12x slower at total_seq~=2048 on Strix Halo).
//
// Default 256 is a starter value pending a full threshold sweep. Set
// HIPDNN_EP_GQA_FUSED_DECODE_MAX_T=N to override (or set a very large value
// like 999999 to effectively disable smart-dispatch and preserve the
// pre-smart-dispatch always-fused-when-eligible behaviour for A/B testing).
static int gqa_fused_decode_max_t() {
  static const int max_t = [] {
    const char *v = std::getenv("HIPDNN_EP_GQA_FUSED_DECODE_MAX_T");
    if (!v || !*v) {
      return 256;
    }
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed <= 0) {
      return 256;
    }
    return static_cast<int>(parsed);
  }();
  return max_t;
}

// Sentinel returned by read_seqlens_k_for_dispatch when the pre-dispatch
// read is not applicable (multi-batch or missing seqlens_k_ptr) or failed
// (D2H copy / stream sync error). Outside the valid range of real
// seqlens_k values (-1 is ORT's prefill sentinel; 0..max_seq are real).
// Callers must treat this as "no pre-read available" and fall back to the
// legacy per-call D2H readback site they already implement.
static constexpr int32_t kSeqlensKNotRead = -2;

// Read seqlens_k_val from device (or the per-Compute() cache when
// HIPDNN_EP_GQA_CACHE_SEQLENS=1) once per call before the fused/decomposed
// dispatch decision. Two purposes:
//   1. Give the smart-dispatch heuristic access to total_seq for the
//      decode case (sq == 1) so it can compare against
//      gqa_fused_decode_max_t().
//   2. Populate the per-Compute() cache for B == 1 so subsequent GQA
//      layers within the same forward pass reuse the value with zero D2H.
//
// Applies to B == 1 regardless of sq (both prefill and decode share the
// same seqlens_k pointer and benefit from caching). On B != 1 we return
// kSeqlensKNotRead because per-batch validation in the multi-batch path
// requires reading every entry; the legacy readback site there handles it.
//
// Behaviour:
//   - cache enabled and hit:    zero D2H, return cached value.
//   - cache enabled and miss:   one D2H + sync, populate cache, return.
//   - cache disabled:           one D2H + sync, return (no cache write).
//   - B != 1, !seqlens_k_ptr, or D2H/sync failure:
//                               return kSeqlensKNotRead.
//
// The returned int32_t is the raw device value: -1 is ORT's prefill
// sentinel (callers map it to total_seq=sq, past_len=0); 0..max_seq is
// the live (total_seq - 1).
static int32_t read_seqlens_k_for_dispatch(hipStream_t stream,
                                           const void *seqlens_k_ptr, int64_t B,
                                           RuntimeState *state) {
  if (!seqlens_k_ptr || B != 1) {
    return kSeqlensKNotRead;
  }

  if (gqa_cache_seqlens_enabled() && state && state->mm &&
      state->mm->seqlens_k_cache_valid(seqlens_k_ptr)) {
    return state->mm->seqlens_k_cached_val();
  }

  int32_t seqlens_k_val = 0;
  if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                     hipMemcpyDeviceToHost, stream) != hipSuccess) {
    return kSeqlensKNotRead;
  }
  if (hipStreamSynchronize(stream) != hipSuccess) {
    return kSeqlensKNotRead;
  }

  if (gqa_cache_seqlens_enabled() && state && state->mm) {
    state->mm->seqlens_k_cache_set(seqlens_k_ptr, seqlens_k_val);
  }

  return seqlens_k_val;
}

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

// FA-2 split-K geometry (must match hip_gqa_flash_decode launcher).
static constexpr int kFlashDecodeKSplits = 8;

// Geometry gate for the flash_decode kernel. The launcher has template
// instantiations for:
//   - HPG=4, D in {64, 128}  (Llama-3.x family)
//   - HPG=8, D == 64         (gpt-oss-20b)
// Any other (HPG, D) combination must fall back to fused_decode /
// hipBLASLt-decomposed.
static inline bool flash_decode_geometry_ok(int64_t H, int64_t G, int64_t d) {
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
// unique (m, n, k, batch, transA) shape and reused for the process lifetime.
// This avoids repeated descriptor creation and heuristic queries on every
// GQA inference call in the decomposed (prefill) path.

struct GqaGemmKey {
  int64_t m, n, k, batch;
  bool transA; // true for Score GEMM (K^T * Q), false for Value GEMM (V * S)
  bool outputFp32; // true to use HIP_R_32F for C/D layouts (Score GEMM)
  // true to use HIP_R_32F for the A/B input layouts (fp32 GQA, e.g. Whisper
  // no_causal). When false the inputs are HIP_R_16F (fp16 GQA -- Llama /
  // gpt-oss, byte-identical to the pre-fp32 behaviour).
  bool inputFp32;
  // Optional explicit per-operand batch strides (in elements). A value of 0
  // means "use the default dense stride" (m*k for A, n*k for B, n*m for C/D).
  // Non-zero values override the default and enable batch layouts that differ
  // from the packed-batched layout -- used by the no-expand decode path where
  // K/V are shared across HPG heads (stride = present_seq*d across G groups)
  // while Q/O advance by HPG entries per group (stride = HPG*d or HPG*skv).
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
/// Ownership: descriptors are created in queryOrCreateGemmState() and live
/// for the process lifetime (never destroyed individually).
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
  // (GqaState's deletor).
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
// nullptr when the slot is unconstructed (init failure) — callers propagate the
// error rather than lazily allocating, since the slot is built at session init.
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

// Shared KV cache update: concat past+new or append new tokens.
// past_buf_seq is the buffer dimension of past_key (may be larger than past_len
// when the cache is pre-allocated at max_length).
// seqlens_k_ptr: optional device pointer. When non-null and the append path is
// taken, the kernel reads past_len from device memory (zero D2H copy).
// When the concat path is needed (separate buffers), past_len must be valid.
// Returns 0 on success, non-zero on failure.
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int past_buf_seq, int present_seq,
                           const void *seqlens_k_ptr, int elem_sz,
                           bool no_causal = false, int skv = -1) {
  // no_causal (Whisper encoder / cross-attn): bidirectional, no past KV.
  // The KV to attend over is the FULL `new_key`/`new_value` (Skv tokens), not
  // `sq` newly-appended tokens. Two sub-cases distinguished by sq vs Skv:
  //   * Cross-attn (sq != Skv): `key`/`value` arrive as rank-4 BNSH
  //     [B, G, Skv, d] -- already in the present_key layout. A straight
  //     device-to-device copy of all Skv tokens populates present_*.
  //   * Encoder self-attn (sq == Skv): `key`/`value` are BSHD [B, Skv, G, d];
  //     fall through to the append kernel below with past_len forced to 0 and
  //     seqlens_k=nullptr so it transposes all Skv tokens to offset 0 WITHOUT
  //     the +1 PAST-token convention.
  if (no_causal && skv >= 0 && skv != sq) {
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
    // Separate-buffer concat: needs host-side past_len for stride computation
    if (hip_gqa_kv_cache_concat(stream, past_key, new_key, present_key, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz) != 0)
      return -1;
    if (hip_gqa_kv_cache_concat(stream, past_value, new_value, present_value, B,
                                past_len, sq, G, d, past_buf_seq, present_seq,
                                elem_sz) != 0)
      return -1;
  } else {
    // In-place append: kernel can read past_len from device via seqlens_k_ptr
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
// 12-step hipBLASLt GQA pipeline (Step 0 + Steps 1-11, FP16 only)
//===----------------------------------------------------------------------===//

static int gqa_forward_hipblaslt(
    RuntimeState *state, hipStream_t stream, hipblasLtHandle_t ltHandle,
    const void *query,    // BSHD [B, sq, H, d] or packed [B, sq, (H+2G)*d]
    const void *key,      // BSHD [B, sq, G, d] or null (packed QKV)
    const void *value,    // BSHD [B, sq, G, d] or null (packed QKV)
    const void *past_key, // BNSD [B, G, past_buf_seq, d] or null
    const void *past_value,
    const void *seqlens_k_ptr, // GPU pointer to seqlens_k [B] int32
    const void *cos_cache, const void *sin_cache,
    void *head_sink,         // [H] smooth softmax factor or null
    bool use_smooth_softmax, // true when smooth softmax is enabled (ORT:
                             // head_sink || smooth_softmax attr)
    void *output,            // BSHD [B, sq, H, d]
    void *present_key,       // BNSD [B, G, present_buf_seq, d]
    void *present_value, int64_t B, int64_t sq, int64_t skv,
    int64_t past_buf_seq, int64_t H, int64_t G, int64_t d, float scale,
    int64_t do_rotary, int64_t local_window_size, bool no_causal,
    int64_t element_size_bytes, int op_state_slot) {

  int64_t HPG = H / G;
  // present_seq is the buffer dimension of present_key (may be max_length
  // for pre-allocated caches, larger than the actual valid token count).
  int64_t present_seq = skv;
  // 2 = fp16 (Llama / gpt-oss), 4 = fp32 (Whisper no_causal). Drives scratch
  // buffer sizing, the GEMM layout element type, the softmax output dtype, and
  // the data-movement kernel dispatch below.
  size_t elem_sz = static_cast<size_t>(element_size_bytes);
  // hipBLASLt matrix layout element type for the fp16-in/fp16-out GEMMs
  // (Value GEMM operands K/V/P/O). The Score GEMM accumulates to fp32 either
  // way; only its input operands (K, Q) use this type. Compute stays
  // HIPBLAS_COMPUTE_32F regardless.
  bool gemm_fp32 = (elem_sz == 4);

  bool need_rope = do_rotary && cos_cache && sin_cache;

  //===--------------------------------------------------------------------===//
  // Fused GQA decode path (sq == 1, d in {64, 128, 256}, KV cache enabled)
  //
  // Replaces steps 3, 6-11 of the decomposed pipeline with a single kernel
  // that reads Q in BSHD and KV from the BNSD cache, producing O in BSHD.
  // Steps 1-2 (RoPE) and 4-5 (KV cache update) still run as separate
  // kernels before the fused dispatch.
  //
  // When seqlens_k is provided, the device pointer is passed directly to
  // each kernel so they can read the actual sequence length on-device.
  // This eliminates the D2H copy + hipStreamSynchronize stall entirely
  // for the decode hot path.
  //
  // All prefill (sq > 1) goes through the decomposed hipBLASLt path where
  // auto-tuned GEMMs outperform fixed WMMA tiling and all ORT GQA features
  // (sliding window, smooth softmax, head sink) are supported.
  //===--------------------------------------------------------------------===//
  // Pre-dispatch read of seqlens_k (or per-Compute() cache lookup for B==1
  // when HIPDNN_EP_GQA_CACHE_SEQLENS=1). Applies to both prefill (sq>1) and
  // decode (sq==1) for B==1 -- both share the same seqlens_k pointer and
  // benefit from caching. The result is reused by the fused-path
  // need_host_past_len block (eliminating its inline D2H) and the
  // decomposed-path readback site (consumed unconditionally instead of
  // issuing its own D2H). On B>1 this returns kSeqlensKNotRead and both
  // downstream paths fall back to their legacy per-call reads. Stored in
  // seqlens_k_pre; total_seq_pre is the derived total_seq (-1 means
  // unknown / not applicable).
  int32_t seqlens_k_pre =
      read_seqlens_k_for_dispatch(stream, seqlens_k_ptr, B, state);
  int64_t total_seq_pre = -1;
  if (no_causal) {
    // no_causal (Whisper encoder / cross-attn): seqlens_k = skv means "all skv
    // keys valid", there is no past. total_seq is exactly skv -- do NOT apply
    // the +1 decode convention (would over-count and trip the smart-dispatch
    // size check / fused validation). See the matching exemption at the
    // decomposed-path total_seq derivation below.
    total_seq_pre = skv;
  } else if (seqlens_k_pre != kSeqlensKNotRead) {
    // -1 is ORT's prefill sentinel: total_seq=sq, past_len=0. Real values
    // are 0..max_seq; total_seq = seqlens_k_val + 1.
    total_seq_pre =
        (seqlens_k_pre < 0) ? sq : static_cast<int64_t>(seqlens_k_pre) + 1;
  }

  bool fused_d = (d == 64 || d == 128 || d == 256);

  // Smart dispatch: the *legacy* fused decode kernel (hip_gqa_fused_decode)
  // serializes over the time dimension (cross-wave reduction tree on the
  // critical path of every iteration). For total_seq above
  // gqa_fused_decode_max_t() the GEMM-based decomposed path wins (~12x at
  // total_seq=2048 on Strix Halo). The newer flash_decode kernel
  // (hip_gqa_flash_decode) fixes that scaling via FA-2 split-K, so when it
  // is eligible we keep the fused branch active even at long total_seq --
  // flash_decode is exactly what the smart-dispatch threshold was working
  // around. When we can't read total_seq (B>1, no seqlens_k, or D2H failure)
  // default to permitting fused -- preserves bit-for-bit behaviour on
  // workloads that pass the predicate today.
  bool flash_decode_eligible = gqa_flash_decode_enabled() &&
                               flash_decode_geometry_ok(H, G, d) &&
                               skv >= gqa_flash_decode_min_skv();
  bool size_ok_for_fused =
      (total_seq_pre < 0) ||
      (total_seq_pre <= static_cast<int64_t>(gqa_fused_decode_max_t())) ||
      flash_decode_eligible;

  // ONNX uses local_window_size=-1 for "no sliding window"; <= 0 means
  // disabled. The original hip_gqa_fused_decode kernel does NOT support
  // sliding window, but hip_gqa_flash_decode does (it clamps kv_lo to
  // max(0, eff_skv - window) when local_window_size > 0). So we admit the
  // fused branch with sliding window only when flash_decode is eligible --
  // the (use_flash_decode) check inside the branch then routes us correctly.
  // This is what unlocks the gpt-oss-20b sliding-attention layers (12 of 24)
  // at long context: they were previously rejected here and fell through to
  // the decomposed path, which reads the full skv KV cache instead of just
  // the 128-element window.
  bool sliding_ok_for_fused = (local_window_size <= 0) || flash_decode_eligible;
  // head_sink / smooth_softmax: legacy hip_gqa_fused_decode does not support
  // these, but hip_gqa_flash_decode now folds the sink term into the reduce
  // kernel's denominator. Admit them only when flash_decode is the eligible
  // dispatch -- the (use_flash_decode) check inside the branch then routes
  // correctly. This is what unlocks gpt-oss-20b decode (all 24 GQA layers
  // pass head_sink, which previously forced fall-through to the decomposed
  // hipBLASLt path that scales linearly with skv).
  bool sink_ok_for_fused =
      (!head_sink && !use_smooth_softmax) || flash_decode_eligible;
  // Packed-QKV inputs (gpt-oss-20b style: query is the [B,sq,(H+2G)*d]
  // qkv_proj output, key and value are null) are supported by the fused
  // branch by routing through hip_gqa_split_qkv into workspace before rope
  // and KV-append. Only flash_decode is exercised by these models in
  // practice (HPG=8, d=64), but split is correct for the legacy fused_decode
  // branch too.
  bool fused_packed_qkv = (!key && !value);
  bool kv_inputs_ok = (key && value) || fused_packed_qkv;
  // The fused (hip_gqa_fused_decode) and flash (hip_gqa_flash_decode) decode
  // kernels are __half-only (their Q/K/V/O pointers are `const __half*`). They
  // are correct for the fp16 causal decode of Llama / gpt-oss (elem_size==2).
  // The fp32 GQA path (Whisper decoder self-attn, elem_size==4) must NOT reach
  // them: feeding fp32 buffers to a __half kernel reinterprets the bytes and
  // produces garbage (observed: O == -4.28e37 sentinel leak → token 0 spam in
  // greedy decode). The decomposed hipBLASLt pipeline below IS fp32-capable, so
  // route fp32 decode there. This is the decode-side analogue of the no_causal
  // exemption (the prefill sq>1 fp32 path already used decomposed). GATE-1
  // (Llama / gpt-oss) is fp16 so `fused_fp16` stays true for them — byte
  // identical.
  bool fused_fp16 = (element_size_bytes == 2);
  // no_causal (Whisper encoder / cross-attn) always takes the decomposed
  // hipBLASLt path. The fused/flash decode kernels are decode-only (sq==1) and
  // read seqlens_k with the +1 PAST-token convention, plus assume the KV cache
  // is appended in BSHD->BNSD layout from `sq` new tokens. Neither holds for
  // bidirectional no-past attention where `key` is the full Skv-length KV
  // (cross-attn ships it as rank-4 BNSH with Skv != sq). Routing no_causal to
  // the decomposed path keeps a single correct code path for these models.
  bool fused_predicate =
      (!gqa_fused_decode_disabled() && !no_causal && fused_fp16 && fused_d &&
       sq == 1 && kv_inputs_ok && present_key && present_value &&
       sliding_ok_for_fused && sink_ok_for_fused && size_ok_for_fused);

  if (fused_predicate) {
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // For fused decode, kernels read seqlens_k from device memory directly.
    // past_len is only needed on host for the concat branch (separate buffers).
    // For in-place caches (past_key == present_key), past_len is unused on
    // host.
    int64_t past_len = 0;
    bool need_host_past_len =
        seqlens_k_ptr && past_key && past_key != present_key;
    if (need_host_past_len) {
      // Reuse the value the pre-dispatch helper already read above. Fall
      // back to a per-call D2H + sync only when the pre-read was not
      // applicable (multi-batch, or copy/sync failure). For the asym
      // Llama decode hot path (B==1, sq==1) the pre-read is always
      // applicable, so this branch becomes pure host arithmetic.
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
      // initialises seqlens_k[b] to -1 (so seqlens_k[b]+1 == 0). Treat that
      // as a fresh prefill (past_len=0) instead of rejecting it as invalid.
      // IR fixture:
      // test/lit/Conversion/onnx-to-hip/test_gqa_prefill_sentinel.mlir
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
                                  flash_decode_geometry_ok(H, G, d) &&
                                  skv >= gqa_flash_decode_min_skv();

    // Sum split + rope-temp + flash-partials in a single ensure_workspace
    // call. ensure_workspace does NOT preserve data on grow (free + malloc),
    // so any data written earlier into the workspace would be lost if a
    // later request triggered a regrowth. One combined request avoids that
    // hazard, and the offsets below match the call order so each step's
    // input region stays live while it is consumed.
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

    // Layout: [Qsplit?, Ksplit?, Vsplit? | Qroped?, Kroped? | flash_partials?]
    const size_t off_split = 0;
    const size_t off_rope = off_split + split_bytes;
    const size_t off_partials = off_rope + rope_temp_bytes;

    // ---- Step 0: Split packed QKV (if needed) ----
    // For the fused / flash decode branch the split outputs become the new
    // qSrc / kSrc / vSrc and persist in workspace until rope (Q,K) and
    // update_kv_cache (V) consume them. After update_kv_cache returns V is
    // committed to present_value, so the split V slot can be safely reused
    // by downstream callers — but here we never overwrite it again because
    // flash_partials is placed strictly after rope_temp, which is placed
    // strictly after split.
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
      // vSrc is intentionally NOT updated: V is never RoPE'd.
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
          "[REAL] flash GQA decode: B=%lld sq=%lld skv=%lld H=%lld G=%lld "
          "d=%lld K_SPLITS=%d window=%lld sink=%d smooth=%d zero_d2h=%d\n",
          (long long)B, (long long)sq, (long long)skv, (long long)H,
          (long long)G, (long long)d, kFlashDecodeKSplits,
          (long long)local_window_size, static_cast<int>(head_sink != nullptr),
          static_cast<int>(use_smooth_softmax),
          static_cast<int>(seqlens_k_ptr != nullptr && !need_host_past_len));
    } else {
      // The original hip_gqa_fused_decode kernel does not implement sliding
      // window or head_sink/smooth_softmax. The predicate above only admits
      // those features when flash_decode is eligible, so this branch should
      // never see them -- assert defensively rather than silently producing
      // wrong results.
      if (local_window_size > 0) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: BUG -- fused_decode (non-flash) cannot "
                "handle local_window_size=%lld; predicate should have routed "
                "to flash_decode or the decomposed path.\n",
                (long long)local_window_size);
        return -1;
      }
      if (head_sink != nullptr || use_smooth_softmax) {
        fprintf(stderr,
                "gqa_forward_hipblaslt: BUG -- fused_decode (non-flash) cannot "
                "handle head_sink=%p smooth=%d; predicate should have routed "
                "to flash_decode or the decomposed path.\n",
                head_sink, static_cast<int>(use_smooth_softmax));
        return -1;
      }
      // skv is passed as a fallback; kernel reads seqlens_k[b]+1 when available
      if (hip_gqa_fused_decode(
              stream, qSrc, present_key, present_value, output,
              static_cast<int>(B), static_cast<int>(H), static_cast<int>(G),
              static_cast<int>(d), static_cast<int>(skv),
              static_cast<int>(present_seq), scale, seqlens_k_ptr) != 0)
        return -1;
      RUNTIME_DEBUG_LOG(
          "[REAL] fused GQA decode: B=%lld sq=%lld skv=%lld H=%lld G=%lld "
          "d=%lld zero_d2h=%d\n",
          (long long)B, (long long)sq, (long long)skv, (long long)H,
          (long long)G, (long long)d,
          static_cast<int>(seqlens_k_ptr != nullptr && !need_host_past_len));
    }
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Decomposed hipBLASLt pipeline (all prefill sq > 1, unsupported d, or
  // features requiring sliding window / smooth softmax / head sink)
  //===--------------------------------------------------------------------===//

  // D2H readback of seqlens_k is required here because hipBLASLt descriptor
  // creation and workspace sizing are host-side APIs that need total_seq.
  // For B == 1 the value was already read (and cached when
  // HIPDNN_EP_GQA_CACHE_SEQLENS=1) by the pre-dispatch helper above; we
  // just consume seqlens_k_pre. The B > 1 branch keeps the legacy per-call
  // read because per-batch validation requires reading every entry and we
  // have no validated multi-batch decode workload yet.
  int64_t total_seq = skv;
  int64_t past_len = skv - sq;
  // no_causal (Whisper encoder self-attn + decoder cross-attn) is bidirectional
  // with NO past KV: the converters emit a compile-time seqlens_k = skv meaning
  // "all skv keys are valid". The ORT decode convention below (seqlens_k[b] =
  // PAST tokens, total = past + current => total_seq = seqlens_k+1) does NOT
  // apply here. Applying it would give total_seq = skv+1 > present_seq = skv ->
  // rc=-1 -> zeroed output. Also, past_len = total_seq - sq is invalid when
  // sq != skv (cross-attn has sq=1, skv=1500 => bogus past_len=1499). For
  // no_causal there is no past, so total_seq = skv (== present_seq) and
  // past_len = 0; skip the seqlens_k readback entirely.
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
      // Defensive fallback for B == 1 when the pre-dispatch helper bailed
      // out (D2H or sync failure). Rare path; not cached because the same
      // failure mode would have prevented the helper from caching too.
      if (hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                         hipMemcpyDeviceToHost, stream) != hipSuccess)
        return -1;
      if (hipStreamSynchronize(stream) != hipSuccess)
        return -1;
    }

    // ORT prefill sentinel: when there is no past KV yet, the producer
    // initialises seqlens_k[b] to -1. Treat that as a fresh prefill
    // (past_len=0, total_seq=sq) instead of rejecting it as invalid.
    // IR fixture:
    // test/lit/Conversion/onnx-to-hip/test_gqa_prefill_sentinel.mlir
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
  // Unified hipBLASLt GQA pipeline (shared by expand and no-expand paths)
  //
  // Two orthogonal knobs control the layout choices for the two GEMMs:
  //
  //   use_no_expand : when true, Score reads K and Value reads V directly
  //                   from the BNSD cache (present_key / present_value) and
  //                   both GEMMs use strided-batched mode with batch = B*G
  //                   plus per-operand strides to broadcast each KV group
  //                   across its HPG queries. When false (original path)
  //                   expand_kv_* duplicates the G groups into H heads in
  //                   d_Kexp / d_Vexp and both GEMMs run with dense batch
  //                   = B*H. The no-expand flavour skips two kernels and
  //                   two B*H*total_seq*d fp16 scratch buffers.
  //
  //   need_transpose: true when sq > 1. The BSHD (Q, output) and BNSD
  //                   (GEMM-native) layouts only coincide when sq == 1,
  //                   so for prefill we still need a Q-transpose before
  //                   Score and an O-transpose after Value. For decode
  //                   (sq == 1) both transposes are pure pointer
  //                   reinterpretations and the kernels are skipped.
  //
  // Gate rationale: the no-expand path is correctness-orthogonal to most
  // GQA features -- packed-QKV split runs in Step 0 and redirects qSrc /
  // kSrc / vSrc; RoPE runs in Steps 1-2 and redirects them again; the
  // softmax / causal-mask kernels see S in the same BNSD [B, H, sq,
  // total_seq] layout regardless of which GEMM flavour produced it; and
  // the strided-batched GEMM addressing is correct for any B and any sq
  // (I walked the stride math for both dimensions separately before
  // dropping their gates). The only hard correctness requirement is that
  // present_key / present_value be valid BNSD caches the GEMMs can read
  // directly -- everything else is just pipeline plumbing that works
  // identically across the two flavours.
  //
  // Prefill (sq > 1) is additionally guarded by
  // HIPDNN_EP_GQA_NO_EXPAND_PREFILL (default off) so the new behaviour can
  // be verified in isolation -- with the prefill flag off the path is
  // byte-identical to the pre-step-2 decode-only gate. Once prefill is
  // verified across model families this extra gate can be removed.
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
  // strides (non-zero stride fields); the expand flavour leaves them zero
  // so queryOrCreateGemmState falls back to the dense packed-batch defaults.
  GqaGemmKey scoreKey, valueKey;
  if (use_no_expand) {
    // Score: C[total_seq, HPG*sq] = K^T[d,total_seq] * Q[d, HPG*sq] per
    // (b, g) pair. strideA steps over the buffer page (present_seq*d) even
    // though only the first total_seq tokens are read, which lets the
    // descriptor stay stable across token steps.
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
    // Value: C[d, HPG*sq] = V[d, total_seq] * S[total_seq, HPG*sq] per
    // (b, g) pair, writing into BNSD [B, G, HPG, sq, d] which at sq==1
    // coincides with BSHD [B, 1, H, d].
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
  // hipMalloc/hipFree -- after the first inference the workspace is
  // already large enough and reuse is zero-cost.
  //
  // Layout: [Qtrans? | Kexp? | Vexp? | S_f32 | S_fp16 | O? | Qroped? |
  // Kroped? | Qsplit? | Ksplit? | Vsplit? | GEMM ws]
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

  // Optional packed-QKV split buffers: allocated only when key/value are
  // null (GPT-OSS style packed input).  These must be placed AFTER the RoPE
  // buffers in the layout so that split outputs remain live while RoPE reads
  // from them and writes to the RoPE region (no overlap).
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

  // Single workspace allocation: temp buffers + GEMM workspace
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

    // Mutable source pointers: initially point to the raw inputs, but get
    // redirected to workspace buffers as pipeline steps (split, RoPE) produce
    // intermediate results.  Downstream steps always read through these so
    // they automatically pick up the latest transformed data.
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // ---- Step 0: Split packed QKV (if needed) ----
    // When key/value are null, query is a packed [B, S, (H+2G)*d] tensor.
    // Split into separate Q [B*S, H*d], K [B*S, G*d], V [B*S, G*d].
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
    // Uses qSrc/kSrc (not raw query/key) so that packed-QKV split buffers
    // are correctly fed into RoPE when both features are active.
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
      // vSrc is intentionally NOT updated: V is never RoPE'd.
    }

    // ---- Step 3: Q Transpose BSHD [B,S,H,d] -> BNSD [B,H,S,d] ----
    // Skipped for sq == 1 because BSHD [B, 1, H, d] and BNSD [B, H, 1, d]
    // are bit-identical in memory -- the Score GEMM can read qSrc directly.
    if (need_transpose) {
      HIP_CHECK(hip_gqa_transpose_mid_dims(
          stream, qSrc, d_Qtrans, static_cast<int>(B), static_cast<int>(sq),
          static_cast<int>(H), static_cast<int>(d), static_cast<int>(elem_sz)));
    }

    // ---- Steps 4-5: KV Cache Update ----
    // The concat/append kernels write the valid range [0, total_seq) in full;
    // positions [total_seq, present_seq) are never read downstream (expand_kv
    // is called with copy_elems = total_seq * d), so the unused tail is left
    // untouched to avoid redundant memory bandwidth during prefill.
    //
    // For the no-expand path we hand seqlens_k_ptr to the append kernel so
    // it can resolve past_len on-device, avoiding a D2H stall on every
    // decode step. The expand path has already consumed total_seq host-side
    // above, so it passes nullptr to preserve the pre-refactor behaviour.
    if (present_key && present_value) {
      // For no_causal, never hand seqlens_k to the append kernel (it would
      // apply the +1 PAST-token convention); update_kv_cache routes no_causal
      // through its bidirectional copy/append branch using skv directly.
      HIP_CHECK(update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          static_cast<int>(B), static_cast<int>(past_len), static_cast<int>(sq),
          static_cast<int>(G), static_cast<int>(d),
          static_cast<int>(past_buf_seq), static_cast<int>(present_seq),
          (use_no_expand && !no_causal) ? seqlens_k_ptr : nullptr,
          static_cast<int>(elem_sz), no_causal, static_cast<int>(skv)));
    }

    // ---- Steps 6-7: KV Expand [B*G, present_seq, d] -> [B*H, total_seq, d]
    // Skipped in no-expand mode: the Score / Value GEMMs read K/V directly
    // from the BNSD cache via per-operand batch strides instead.
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

    // ---- Step 8: Score GEMM (fp16 in, fp32 out) ----
    // A = K : no-expand reads the BNSD cache (present_key) directly;
    //         the expand path reads the already-duplicated d_Kexp.
    // B = Q : need_transpose reads d_Qtrans (BNSD); at sq==1 qSrc points
    //         straight at the BSHD input which shares memory with BNSD.
    const void *scoreA = use_no_expand ? present_key : d_Kexp;
    const void *scoreB = need_transpose ? d_Qtrans : qSrc;
    float scoreAlpha = scale;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t sAlgo = scoreState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, scoreState->desc, &scoreAlpha, scoreA, scoreState->layA,
        scoreB, scoreState->layB, &beta, d_S_f32, scoreState->layC, d_S_f32,
        scoreState->layD, &sAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 9: Causal Mask (fp32) + Softmax (fp32 -> fp16) ----
    // Both kernels treat S as [B*H, sq, total_seq] with head_stride
    // sq*total_seq, which is the layout produced by both GEMM flavours
    // (the no-expand strided-batched output lands in BNSD order too).
    int scoreF32BatchStride = static_cast<int>(sq * total_seq);
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
    // by elem_sz above), so the name is fp16-specific but the buffer holds
    // fp32 when gemm_fp32.
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

    // ---- Step 10: Value GEMM (fp16 in, fp16 out) ----
    // A = V : no-expand reads present_value directly; expand reads d_Vexp.
    // C = O : need_transpose writes d_O (BNSD), later transposed to output;
    //         at sq==1 the GEMM writes directly to the caller's output
    //         buffer because BSHD and BNSD coincide.
    const void *valueA = use_no_expand ? present_value : d_Vexp;
    void *valueC = need_transpose ? d_O : output;
    float valAlpha = 1.0f;
    hipblasLtMatmulAlgo_t vAlgo = valueState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueState->desc, &valAlpha, valueA, valueState->layA,
        d_S_fp16, valueState->layB, &beta, valueC, valueState->layC, valueC,
        valueState->layD, &vAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 11: O Transpose BNSD [B,H,S,d] -> BSHD [B,S,H,d] ----
    // Skipped at sq == 1 -- the Value GEMM already wrote into `output`.
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
// Public wrapper called by generated IR
//===----------------------------------------------------------------------===//

int wrap_group_query_attention(
    RuntimeState *state,
    // Per-instance op-state slot (GqaState: this layer's GEMM descriptor cache)
    int op_state_slot,
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
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Whisper bidirectional-attention flag: when non-zero, the causal mask
    // step in gqa_forward_hipblaslt is skipped. Default 0 preserves Llama /
    // gpt-oss behaviour.
    int32_t no_causal,
    // Shape values (6): past_buf_seq is the buffer dimension of past_key
    // (may differ from actual past token count for pre-allocated caches)
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
    fprintf(stderr, "wrap_group_query_attention: null state\n");
    return -1;
  }
  if (!query || !output) {
    fprintf(stderr, "wrap_group_query_attention: null required argument\n");
    return -1;
  }
  if (num_heads % kv_num_heads != 0) {
    fprintf(stderr,
            "wrap_group_query_attention: num_heads (%lld) must be "
            "divisible by kv_num_heads (%lld)\n",
            (long long)num_heads, (long long)kv_num_heads);
    return -1;
  }
  // The decomposed hipBLASLt GQA pipeline supports fp16 (elem_size=2) and
  // fp32 (elem_size=4): GEMM matrix layouts are created with the runtime data
  // type (gemm_fp32), the data-movement custom kernels (rope / append / concat
  // / transpose / expand / split) dispatch on element_size_bytes, and softmax
  // writes fp16 or fp32 probabilities to match. The fp32 path is used by
  // Whisper's no_causal attention (encoder self-attn + decoder cross-attn),
  // which is forced down the decomposed path. The fused / flash decode kernels
  // (Llama / gpt-oss) remain FP16-only; no_causal never reaches them. BF16 is
  // not yet supported.
  if (element_size_bytes != 2 && element_size_bytes != 4) {
    fprintf(stderr,
            "wrap_group_query_attention: hipBLASLt pipeline requires "
            "FP16 (elem_size=2) or FP32 (elem_size=4), got %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  hipblasLtHandle_t ltHandle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  if (!stream || !ltHandle) {
    fprintf(stderr, "wrap_group_query_attention: null stream or hipblas "
                    "handle\n");
    return -1;
  }

  // Reject features not yet implemented
  if (position_ids != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: position_ids not yet implemented\n");
    return -1;
  }
  if (attention_bias != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: attention_bias not yet implemented\n");
    return -1;
  }
  if (k_scale != nullptr || v_scale != nullptr) {
    fprintf(stderr, "wrap_group_query_attention: KV cache quantization not yet "
                    "implemented\n");
    return -1;
  }
  if (output_qk != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: output_qk not yet implemented\n");
    return -1;
  }
  if (qk_output != 0) {
    fprintf(stderr,
            "wrap_group_query_attention: qk_output not yet implemented\n");
    return -1;
  }
  if (k_quant_type != 0 || v_quant_type != 0) {
    fprintf(
        stderr,
        "wrap_group_query_attention: quantization types not yet implemented\n");
    return -1;
  }
  if (kv_cache_bit_width != 8) {
    fprintf(stderr,
            "wrap_group_query_attention: non-8bit cache not yet implemented\n");
    return -1;
  }

  // ORT uses scale == 0.0 as sentinel for "auto-compute 1/√head_size"
  // (gqa_attention_base.h line 185: scale_ == 0.0f ? 1/sqrt(head_size) :
  // scale_).
  if (scale == 0.0f && head_dim > 0) {
    scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  }

  // Smooth softmax: activated when head_sink is provided OR smooth_softmax
  // attribute is explicitly 1, matching ORT behaviour (gqa_attention_base.h
  // line 354: use_smooth_softmax_ || head_sink != nullptr).
  // Compare == 1 following ORT's GetAttrOrDefault("smooth_softmax", 0) == 1
  // pattern (default 0 is set in HipOps.td and OnnxToHip.cpp).
  bool has_smooth_softmax = (head_sink != nullptr || smooth_softmax == 1);

  bool is_packed_qkv = (key == nullptr && value == nullptr);
  bool has_rope =
      (do_rotary == 1 && cos_cache != nullptr && sin_cache != nullptr);
  bool has_local_window = (local_window_size > 0);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_group_query_attention:\n"
      "  shapes: batch=%lld, seq_q=%lld, seq_kv=%lld, num_heads=%lld, "
      "kv_heads=%lld, head_dim=%lld\n"
      "  attrs:  scale=%f, do_rotary=%lld, rotary_interleaved=%lld, "
      "softcap=%f, local_window_size=%lld, smooth_softmax=%lld\n"
      "  inputs: query=%p, key=%p, value=%p, past_key=%p, past_value=%p\n"
      "          cos_cache=%p, sin_cache=%p, head_sink=%p\n"
      "  outputs: output=%p, present_key=%p, present_value=%p\n"
      "  mode:   packed_qkv=%d, rope=%d, local_window=%d, smooth_softmax=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (double)scale, (long long)do_rotary, (long long)rotary_interleaved,
      (double)softcap, (long long)local_window_size, (long long)smooth_softmax,
      query, key, value, past_key, past_value, cos_cache, sin_cache, head_sink,
      output, present_key, present_value, static_cast<int>(is_packed_qkv),
      static_cast<int>(has_rope), static_cast<int>(has_local_window),
      static_cast<int>(has_smooth_softmax));

  int rc = gqa_forward_hipblaslt(
      state, stream, ltHandle, query, key, value, past_key, past_value,
      seqlens_k, cos_cache, sin_cache, head_sink, has_smooth_softmax, output,
      present_key, present_value, batch_size, seq_len_q, seq_len_kv,
      past_buf_seq, num_heads, kv_num_heads, head_dim, scale, do_rotary,
      local_window_size, no_causal != 0, element_size_bytes, op_state_slot);

  if (rc != 0) {
    fprintf(stderr, "wrap_group_query_attention: gqa_forward failed (rc=%d)\n",
            rc);
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_query_attention: completed successfully\n");
  }

  return rc;
}

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

// Construct this GQA instance's op-state slot (see op-state-slots-design.md).
// No compile-time params: the descriptor cache fills lazily per GEMM shape. The
// construct fn stores the state into op_states[slot] itself.
extern "C" int8_t hipdnn_ep_op_state_construct_gqa(RuntimeState *state,
                                                   int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, GqaState::create().release());
  return 0;
}
