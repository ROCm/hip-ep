/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"
#include "zp_unpack_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

struct TokenEntry {
  int32_t token_id;
  int32_t slot;
};

int wrap_qmoe(RuntimeState *state, const void *input, const void *router_probs,
              const void *router_weights, const void *fc1_weights,
              const void *fc1_scales, const void *fc1_bias,
              const void *fc2_weights, const void *fc2_scales,
              const void *fc2_bias, const void *fc3_weights,
              const void *fc3_scales, const void *fc3_bias,
              const void *fc1_zero_points, const void *fc2_zero_points,
              const void *fc3_zero_points, void *output, int64_t num_tokens,
              int64_t hidden_size, int64_t inter_size, int64_t num_experts,
              int64_t k, int64_t expert_weight_bits, int64_t block_size,
              int64_t swiglu_fusion, int64_t activation_type,
              float activation_alpha, float activation_beta, float swiglu_limit,
              int64_t normalize_routing_weights, int64_t elem_size) {
  OP_PROFILE(
      "qmoe",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lld,e=%lld", (long long)num_tokens,
                 (long long)hidden_size, (long long)inter_size,
                 (long long)num_experts);
        return std::string(b);
      },
      state);
  if (router_weights) {
    fprintf(stderr, "wrap_qmoe: router_weights is not supported yet\n");
    return -1;
  }
  if (!state || !input || !router_probs || !output) {
    fprintf(stderr, "wrap_qmoe: null argument\n");
    return -1;
  }

  if (swiglu_fusion != 1) {
    fprintf(stderr, "wrap_qmoe: only swiglu_fusion=1 supported, got %lld\n",
            (long long)swiglu_fusion);
    return -1;
  }

  if (fc3_weights || fc3_scales || fc3_bias || fc3_zero_points) {
    fprintf(stderr, "wrap_qmoe: fc3 (unfused SwiGLU) not supported, "
                    "use swiglu_fusion=1\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe(tokens=%lld, hidden=%lld, inter=%lld, "
                    "experts=%lld, k=%lld, bits=%lld, block=%lld, elem=%lld)\n",
                    (long long)num_tokens, (long long)hidden_size,
                    (long long)inter_size, (long long)num_experts, (long long)k,
                    (long long)expert_weight_bits, (long long)block_size,
                    (long long)elem_size);

  // Guard against pathological metadata: block_size==0 would otherwise crash
  // with STATUS_INTEGER_DIVIDE_BY_ZERO inside the k_blocks computations below
  // (and produces invalid quant layouts even at >0 if not a multiple of 2).
  if (block_size <= 0 || (block_size & 1) != 0) {
    fprintf(stderr,
            "wrap_qmoe: invalid block_size=%lld (must be a positive even "
            "value matching the weights' quant block layout)\n",
            (long long)block_size);
    return -1;
  }
  if (hidden_size <= 0 || inter_size <= 0 || num_experts <= 0 || k <= 0 ||
      num_tokens <= 0 || elem_size <= 0) {
    fprintf(stderr,
            "wrap_qmoe: invalid sizes (tokens=%lld hidden=%lld inter=%lld "
            "experts=%lld k=%lld elem=%lld)\n",
            (long long)num_tokens, (long long)hidden_size,
            (long long)inter_size, (long long)num_experts, (long long)k,
            (long long)elem_size);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_qmoe: null stream\n");
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int result = 0;

  int64_t fusion_inter = 2 * inter_size;
  int64_t k_blocks_fc1 = (hidden_size + block_size - 1) / block_size;
  int64_t blob_size_fc1 = block_size / 2;
  int64_t k_blocks_fc2 = (inter_size + block_size - 1) / block_size;
  int64_t blob_size_fc2 = block_size / 2;

  // The grouped-vs-host-loop prefill decision has two parts: a shape/scale test
  // knowable up front (grouped_shape_ok && grouped_scale_ok) and a fill test
  // that needs the post-bucket per-expert row distribution (fill_ok). Only the
  // grouped path uses pf_act/pf_accum/active_eids, so we reserve those buffers
  // ONLY when grouped could actually run (the pre-bucket predicate below).
  // Below the token threshold (short prompts, VLM image prefills) and for
  // non-WMMA shapes the grouped-only scratch is skipped entirely, so the pool
  // high-water matches the host-loop/main footprint and -- since the pool never
  // shrinks -- is not carried into the decode phase. This is realloc-safe: the
  // predicate is computed before bucket_tokens writes into the pool, so the
  // size is final before the single ensure() below (growing after bucketing
  // would realloc and drop the bucketed data). If the shape/scale test passes
  // but fill_ok later fails, pf_* stay reserved-but-unused (rare: tokens >=
  // threshold yet experts don't fill the WMMA tile). Decode buffers stay gated
  // on is_decode.
  const bool is_decode = (num_tokens == 1);
  const bool is_prefill = !is_decode;
  // Pre-bucket portion of the grouped-WMMA prefill gate (see the post-bucket
  // gate near the dispatch for fill_ok). Kept as the single source of truth for
  // both the scratch reservation here and the dispatch decision below.
  const bool grouped_shape_ok =
      elem_size == 2 && expert_weight_bits == 4 && block_size > 0 &&
      (hidden_size % 64 == 0) && (inter_size % 64 == 0) &&
      (num_experts >= hipdnn_ep_qmoe_grouped_min_experts());
  const bool grouped_scale_ok =
      num_tokens >= hipdnn_ep_qmoe_grouped_min_tokens();
  const bool grouped_possible =
      is_prefill && grouped_shape_ok && grouped_scale_ok;

  // Per-state grow-on-demand scratch in place of 8 hipMalloc/8 hipFree per
  // call. Sub-buffers are 64-byte aligned (matches GPU pool alignment, gives
  // each sub-buffer its own cache line). The buffer grows when num_tokens /
  // sizes exceed the cached capacity, never shrinks; freed in state cleanup.
  // Buffers a path doesn't touch are sized 0 (their offset collapses onto the
  // next buffer; the inactive-path pointers below are simply never
  // dereferenced).
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  // Routing outputs (topk_routing) are used by every path.
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  // Host-loop fallback per-expert FFN buffers (sized to num_tokens, reused
  // across experts). Reserved for every prefill because the host-loop path may
  // be selected post-bucket; only the host-loop path dereferences them.
  size_t sz_gather_buf =
      is_prefill ? align_up_64(num_tokens * hidden_size * elem_size) : 0;
  size_t sz_fc1_buf =
      is_prefill ? align_up_64(num_tokens * fusion_inter * elem_size) : 0;
  // Fused decode (num_tokens == 1) reuses act_buf and fc2_buf as the [k,
  // inter] activation slots and [k, hidden] per-expert output slots needed
  // by hip_qmoe_decode_fused (gather/scatter happen inline inside the
  // kernel, indexed by expert_indices). For num_tokens > 1 the host-loop
  // path uses [num_tokens, ...] sizing. Take the max so the per-state
  // scratch is never under-sized regardless of which of those two paths runs.
  // Always reserved: decode needs them, and the host-loop prefill path (which
  // may be selected post-bucket) needs them too. The grouped-WMMA prefill path
  // uses none of these (it has its own pf_*).
  int64_t act_slots = std::max<int64_t>(num_tokens, k);
  size_t sz_act_buf = align_up_64(act_slots * inter_size * elem_size);
  size_t sz_fc2_buf = align_up_64(act_slots * hidden_size * elem_size);
  // bucket_tokens outputs (Phase 2): per-expert counts + exclusive prefix sum
  // offsets, plus tokens/weights re-grouped on-device into per-expert
  // contiguous slices. Used by both prefill paths (grouped + host-loop), not
  // by decode.
  size_t sz_expert_counts =
      is_decode ? 0 : align_up_64(num_experts * sizeof(int32_t));
  size_t sz_expert_offsets =
      is_decode ? 0 : align_up_64((num_experts + 1) * sizeof(int32_t));
  size_t sz_sorted_token_ids =
      is_decode ? 0 : align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_sorted_weights =
      is_decode ? 0 : align_up_64(num_tokens * k * elem_size);
  // dp4a decode scratch (fused decode only, env-gated): int8-quantized
  // activations + per-group fp32 scales for the fc1 input ([hidden]) and the
  // fc2 slot activations ([k, inter]). Decode path only.
  // k_blocks_fc1 == n_blk_in, k_blocks_fc2 == n_blk_mid.
  size_t sz_a_qb_in = is_decode ? align_up_64(hidden_size * sizeof(int8_t)) : 0;
  size_t sz_a_scale_in =
      is_decode ? align_up_64(k_blocks_fc1 * sizeof(float)) : 0;
  size_t sz_a_qb_mid =
      is_decode ? align_up_64(k * inter_size * sizeof(int8_t)) : 0;
  size_t sz_a_scale_mid =
      is_decode ? align_up_64(k * k_blocks_fc2 * sizeof(float)) : 0;

  size_t off_expert_indices = 0;
  size_t off_expert_weights = off_expert_indices + sz_expert_indices;
  size_t off_gather_buf = off_expert_weights + sz_expert_weights;
  size_t off_fc1_buf = off_gather_buf + sz_gather_buf;
  size_t off_act_buf = off_fc1_buf + sz_fc1_buf;
  size_t off_fc2_buf = off_act_buf + sz_act_buf;
  size_t off_expert_counts = off_fc2_buf + sz_fc2_buf;
  size_t off_expert_offsets = off_expert_counts + sz_expert_counts;
  size_t off_sorted_token_ids = off_expert_offsets + sz_expert_offsets;
  size_t off_sorted_weights = off_sorted_token_ids + sz_sorted_token_ids;
  size_t off_a_qb_in = off_sorted_weights + sz_sorted_weights;
  size_t off_a_scale_in = off_a_qb_in + sz_a_qb_in;
  size_t off_a_qb_mid = off_a_scale_in + sz_a_scale_in;
  size_t off_a_scale_mid = off_a_qb_mid + sz_a_qb_mid;
  // Grouped-WMMA prefill scratch (num_tokens > 1 default path). P =
  // num_tokens*k pairs. FC1 fuses gather + SwiGLU into the WMMA kernel and
  // writes act_out directly, and FC2 scatter-adds into an fp32 accumulator, so
  // the path only needs: [P, inter] SwiGLU activations (pf_act), the fp32
  // [num_tokens, hidden] accumulator (pf_accum), and the active-expert id list
  // (active_eids). The old a_all [P, hidden] and fc1_out [P, 2*inter]
  // intermediates are gone.
  int64_t pf_pairs = num_tokens * k;
  size_t sz_pf_act =
      grouped_possible ? align_up_64(pf_pairs * inter_size * elem_size) : 0;
  size_t sz_pf_accum =
      grouped_possible ? align_up_64(num_tokens * hidden_size * sizeof(float))
                       : 0;
  // FC1 gathers its rows straight from `input` and applies SwiGLU in-kernel,
  // writing act_out (pf_act) directly, so neither the [P, hidden] a_all nor the
  // [P, 2*inter] fc1_out buffer is materialized any more.
  size_t sz_active_eids =
      grouped_possible ? align_up_64(num_experts * sizeof(int32_t)) : 0;
  size_t off_pf_act = off_a_scale_mid + sz_a_scale_mid;
  size_t off_pf_accum = off_pf_act + sz_pf_act;
  size_t off_active_eids = off_pf_accum + sz_pf_accum;
  size_t total_scratch = off_active_eids + sz_active_eids;

  if (hipdnn_ep_state_ensure_qmoe_scratch(state, total_scratch) != 0) {
    fprintf(stderr, "wrap_qmoe: ensure_qmoe_scratch(%zu) failed\n",
            total_scratch);
    return -1;
  }
  char *scratch_base =
      static_cast<char *>(hipdnn_ep_state_get_qmoe_scratch(state));
  void *d_expert_indices = scratch_base + off_expert_indices;
  void *d_expert_weights = scratch_base + off_expert_weights;
  void *d_gather_buf = scratch_base + off_gather_buf;
  void *d_fc1_buf = scratch_base + off_fc1_buf;
  void *d_act_buf = scratch_base + off_act_buf;
  void *d_fc2_buf = scratch_base + off_fc2_buf;
  int32_t *d_expert_counts =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_counts);
  int32_t *d_expert_offsets =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_offsets);
  int32_t *d_sorted_token_ids =
      reinterpret_cast<int32_t *>(scratch_base + off_sorted_token_ids);
  char *d_sorted_weights = scratch_base + off_sorted_weights;
  void *d_a_qb_in = scratch_base + off_a_qb_in;
  void *d_a_scale_in = scratch_base + off_a_scale_in;
  void *d_a_qb_mid = scratch_base + off_a_qb_mid;
  void *d_a_scale_mid = scratch_base + off_a_scale_mid;
  void *d_pf_act = scratch_base + off_pf_act;
  void *d_pf_accum = scratch_base + off_pf_accum;
  int32_t *d_active_eids =
      reinterpret_cast<int32_t *>(scratch_base + off_active_eids);

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: topk_routing(tokens=%lld, experts=%lld, "
                    "k=%lld, normalize=%lld)\n",
                    (long long)num_tokens, (long long)num_experts, (long long)k,
                    (long long)normalize_routing_weights);
  HIP_CHECK(hip_qmoe_topk_routing(stream, router_probs, d_expert_indices,
                                  d_expert_weights, num_tokens, num_experts, k,
                                  normalize_routing_weights, elem_size));

  // Fused decode fast path: single-token MoE collapses to three back-to-back
  // kernel launches (FC1+SwiGLU, FC2, weighted reduce) with zero D2H,
  // hipStreamSynchronize, or host-side bucketing. Replaces the multi-pass
  // bucket -> sync -> per-active-expert (gather, fc1, swiglu, fc2,
  // scatter_add) sequence below. d_act_buf is reused as the [k, inter]
  // activation slots, d_fc2_buf as the [k, hidden] per-expert output slots
  // (gather/scatter happen inline via expert_indices).
  if (num_tokens == 1) {
    // W4A8 dp4a decode variant (env-gated). Requires fp16 and 32-aligned
    // block_size / hidden / inter (all true for the MoE targets: block_size
    // 32, hidden/inter multiples of 32). Quantizes the shared input + the k
    // slot activations to int8 once, then runs the fc1/fc2 GEMVs via sudot4.
    // Falls through to the fp fused path otherwise.
    const bool dp4a_ok = hipdnn_ep_matmul_dp4a_enabled() && elem_size == 2 &&
                         block_size > 0 && (block_size % 32 == 0) &&
                         (hidden_size % 32 == 0) && (inter_size % 32 == 0);
    if (dp4a_ok) {
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: fused decode dp4a path (k=%lld)\n",
                        (long long)k);
      HIP_CHECK(hip_qmoe_decode_fused_dp4a(
          stream, input, d_expert_indices, d_expert_weights, fc1_weights,
          fc1_scales, fc1_zero_points, fc1_bias, fc2_weights, fc2_scales,
          fc2_zero_points, fc2_bias, d_fc2_buf, d_act_buf, output, d_a_qb_in,
          d_a_scale_in, d_a_qb_mid, d_a_scale_mid, hidden_size, inter_size, k,
          block_size, activation_alpha, activation_beta, swiglu_limit,
          elem_size));
      return 0;
    }
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: fused decode path (k=%lld)\n",
                      (long long)k);
    HIP_CHECK(hip_qmoe_decode_fused(
        stream, input, d_expert_indices, d_expert_weights, fc1_weights,
        fc1_scales, fc1_zero_points, fc1_bias, fc2_weights, fc2_scales,
        fc2_zero_points, fc2_bias, d_fc2_buf, d_act_buf, output, hidden_size,
        inter_size, k, block_size, activation_alpha, activation_beta,
        swiglu_limit, elem_size));
    return 0;
  }

  {
    // Phase 2: GPU-side bucketing eliminates the per-expert host build +
    // 2 H2D round-trips per active expert per layer. Old flow was:
    //   D2H expert_indices + expert_weights -> sync -> host bucket loop ->
    //   per active expert: build h_ids/h_wts_e -> 2x H2D -> launch chain.
    // New flow:
    //   hip_qmoe_bucket_tokens (counts + prefix-sum + scatter on device) ->
    //   D2H of just num_experts int32 counts -> sync -> compute host offsets
    //   -> per active expert: gather/matmul/scatter from d_sorted_*[offset]
    //   directly. No per-expert H2D; the active-expert loop uses pointer
    //   arithmetic into the on-device sorted buffers populated by
    //   bucket_tokens.
    auto align_up_64h = [](size_t s) -> size_t {
      return (s + 63) & ~size_t(63);
    };
    size_t hsz_counts = align_up_64h(num_experts * sizeof(int32_t));
    // Grouped-WMMA path builds an active-expert id list on the host from the
    // D2H'd counts; reserve space for it alongside h_counts.
    size_t hsz_active = align_up_64h(num_experts * sizeof(int32_t));
    size_t total_host = hsz_counts + hsz_active;

    if (hipdnn_ep_state_ensure_qmoe_host_scratch(state, total_host) != 0) {
      fprintf(stderr, "wrap_qmoe: ensure_qmoe_host_scratch(%zu) failed\n",
              total_host);
      return -1;
    }
    char *host_base =
        static_cast<char *>(hipdnn_ep_state_get_qmoe_host_scratch(state));
    int32_t *h_counts = reinterpret_cast<int32_t *>(host_base);
    int32_t *h_active = reinterpret_cast<int32_t *>(host_base + hsz_counts);

    // Bucket tokens on the device: count per expert (atomicAdd), exclusive
    // prefix sum into d_expert_offsets, scatter (token_id, weight) pairs
    // into d_sorted_token_ids / d_sorted_weights ordered by expert.
    HIP_CHECK(hip_qmoe_bucket_tokens(stream, d_expert_indices, d_expert_weights,
                                     d_expert_counts, d_expert_offsets,
                                     d_sorted_token_ids, d_sorted_weights,
                                     num_tokens, num_experts, k, elem_size));

    // Only readback the counts (num_experts * int32, e.g. 32*4 = 128 bytes)
    // to drive the host-side per-expert dispatch loop. Offsets are computed
    // on the host from the prefix sum of counts (cheap, avoids a second D2H).
    HIP_CHECK(hipMemcpyAsync(h_counts, d_expert_counts,
                             num_experts * sizeof(int32_t),
                             hipMemcpyDeviceToHost, hip_stream));
    HIP_CHECK(hipStreamSynchronize(hip_stream));

    // Recompute offsets on the host (mirrors the on-device prefix sum); used
    // for pointer arithmetic into the sorted device buffers below.
    std::vector<int64_t> h_offsets(num_experts + 1, 0);
    for (int64_t e = 0; e < num_experts; e++) {
      h_offsets[e + 1] = h_offsets[e] + static_cast<int64_t>(h_counts[e]);
    }

    HIP_CHECK(hipMemsetAsync(output, 0, num_tokens * hidden_size * elem_size,
                             hip_stream));

    // Post-bucket fill-aware gate: choose grouped WMMA prefill vs the host loop
    // from the ACTUAL per-expert row distribution (h_counts), not from
    // num_experts (which mis-predicts: gpt-oss-120b has 128 experts but ~7.6
    // rows/expert, so the 64-row WMMA tile is mostly padding and grouped
    // regresses; Qwen3.6-35B-A3B fills the tile and wins). See debug_log.h.
    // One pass over h_counts computes: num_active (for launch amortization),
    // max_count (grouped launch is sized to the heaviest expert), and
    // dense_rows (routed rows in experts that fill the tile). Grouped is chosen
    // only when the shape is WMMA-serviceable, there are enough active experts,
    // and most routed rows land in dense experts; otherwise the host loop runs
    // (this is the pre-existing `main` behavior, so there is no regression vs
    // main).
    const int tile_rows = hipdnn_ep_qmoe_grouped_tile_rows();
    int num_active = 0;
    int max_count = 0;
    int64_t dense_rows = 0;
    for (int64_t e = 0; e < num_experts; e++) {
      int c = h_counts[e];
      if (c > 0) {
        num_active++;
        if (c > max_count)
          max_count = c;
        if (c >= tile_rows)
          dense_rows += c;
      }
    }
    const int64_t total_rows = num_tokens * k; // routed (token,expert) pairs
    const int dense_pct =
        total_rows > 0 ? static_cast<int>((dense_rows * 100) / total_rows) : 0;
    // Shape/scale portion of the gate was computed up front (grouped_shape_ok,
    // grouped_scale_ok) as the single source of truth used to size the
    // grouped-only scratch (pf_*). Only the fill test needs h_counts. Reusing
    // the pre-bucket predicates guarantees the scratch reservation and the
    // dispatch decision can never disagree (pf_* are always sized when this can
    // pick grouped). num_experts >= MIN_EXPERTS (inside grouped_shape_ok) is
    // retained as a hard override/off-switch (set it very high to force the
    // host loop for A/B).
    const bool fill_ok = num_active >= hipdnn_ep_qmoe_grouped_min_active() &&
                         dense_pct >= hipdnn_ep_qmoe_grouped_min_dense_pct();
    // grouped_scale_ok (num_tokens >= MIN_TOKENS) is the absolute work-scale
    // gate: the dense-fraction test is scale-free, so it would otherwise admit
    // grouped on short prefills (e.g. VLM image prefill ~1.8k tokens) where the
    // per-expert launch overhead is not amortized and grouped loses to the host
    // loop; below the threshold we fall back to the host loop (= main).
    const bool use_grouped_prefill =
        grouped_shape_ok && grouped_scale_ok && fill_ok;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_qmoe: prefill gate active=%d/%lld max_count=%d "
        "dense_pct=%d num_tokens=%lld (min_active=%d min_dense_pct=%d "
        "min_tokens=%d) -> %s\n",
        num_active, (long long)num_experts, max_count, dense_pct,
        (long long)num_tokens, hipdnn_ep_qmoe_grouped_min_active(),
        hipdnn_ep_qmoe_grouped_min_dense_pct(),
        hipdnn_ep_qmoe_grouped_min_tokens(),
        use_grouped_prefill ? "grouped WMMA" : "host loop");
    if (use_grouped_prefill) {
      // Build the active-expert id list (grouped launches one block group per
      // active expert, blockIdx.z = expert).
      int na = 0;
      for (int64_t e = 0; e < num_experts; e++) {
        if (h_counts[e] > 0)
          h_active[na++] = static_cast<int32_t>(e);
      }
      if (na > 0) {
        HIP_CHECK(hipMemcpyAsync(d_active_eids, h_active, na * sizeof(int32_t),
                                 hipMemcpyHostToDevice, hip_stream));
      }
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: grouped WMMA prefill "
                        "(active=%d, max_count=%d)\n",
                        na, max_count);
      int rc = hip_qmoe_grouped_prefill_wmma(
          stream, input, d_sorted_token_ids, d_sorted_weights, d_expert_offsets,
          d_active_eids, na, max_count, fc1_weights, fc1_scales,
          fc1_zero_points, fc1_bias, fc2_weights, fc2_scales, fc2_zero_points,
          fc2_bias, d_pf_act, d_pf_accum, output, num_tokens, hidden_size,
          inter_size, k, num_experts, block_size, activation_alpha,
          activation_beta, swiglu_limit, elem_size);
      if (rc != 0) {
        fprintf(stderr, "wrap_qmoe: grouped WMMA prefill failed (%d)\n", rc);
        result = -1;
      }
      goto cleanup;
    }

    // Per-session pointer-keyed zero_points unpack cache (qmoe-owned, lives on
    // RuntimeState). Each expert's fc1/fc2 zp is a distinct pointer into the
    // constants blob, so each gets its own entry; unpack cost is paid once per
    // expert across the session lifetime.
    hipdnn_ep_real::ZpUnpackCache *zpc =
        hipdnn_ep_real::get_or_create_zp_cache(state);

    for (int64_t e = 0; e < num_experts; e++) {
      int64_t count = static_cast<int64_t>(h_counts[e]);
      if (count == 0) {
        continue;
      }

      // Slices into the on-device sorted buffers populated by bucket_tokens.
      // No per-expert H2D needed: the gather/scatter kernels read these
      // directly via pointer arithmetic.
      int64_t off_e = h_offsets[e];
      int32_t *d_ids_e = d_sorted_token_ids + off_e;
      char *d_wts_e = d_sorted_weights + off_e * elem_size;

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: %lld tokens - gather\n",
                        (long long)e, (long long)count);
      HIP_CHECK(hip_qmoe_gather_tokens(stream, input, d_gather_buf, d_ids_e,
                                       hidden_size, count, elem_size));

      const char *fc1_w_e = static_cast<const char *>(fc1_weights) +
                            e * fusion_inter * k_blocks_fc1 * blob_size_fc1;
      const char *fc1_s_e = static_cast<const char *>(fc1_scales) +
                            e * fusion_inter * k_blocks_fc1 * elem_size;
      // Per-expert ZP slice: ONNX MatMulNBits with bits=4 stores
      // zero_points as uint8 packed nibbles (two 4-bit values per byte),
      // so the per-row size is ceil(k_blocks/2) bytes, NOT k_blocks. Using
      // the unpacked stride here makes expert e read ZPs from a region 2x
      // too large, overshooting into the next expert's bytes; downstream
      // dequantization grows ~10x per MoE layer until fp16 overflows
      // (router_probs becomes Inf -> SSLN T5LN emits NaN -> topk routing
      // returns -1 for every token -> 0/N experts active from layer ~3 on
      // -> garbage logits). Matches `convertZpToFp16`'s
      // `packed_cols = (groups_k + 1) / 2` in
      // lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip and the
      // hard-coded `zp_elem_size = 1` we pass to hip_matmul_nbits below.
      const void *fc1_zp_e =
          fc1_zero_points ? static_cast<const char *>(fc1_zero_points) +
                                e * fusion_inter * ((k_blocks_fc1 + 1) / 2)
                          : nullptr;
      const void *fc1_b_e = fc1_bias ? static_cast<const char *>(fc1_bias) +
                                           e * fusion_inter * elem_size
                                     : nullptr;

      // hip_matmul_nbits no longer unpacks zp internally — pre-unpack via the
      // per-session pointer-keyed cache (same path as wrap_matmul_nbits). Each
      // expert's fc1_zp_e is a distinct pointer into the constants blob, so
      // each expert gets its own cache entry; the cost is paid once per
      // expert across the lifetime of the session.
      const void *fc1_pre_zp_u8 = nullptr;
      const void *fc1_pre_zp_fp16 = nullptr;
      if (fc1_zp_e && expert_weight_bits == 4 && block_size > 0) {
        int ngk = static_cast<int>(k_blocks_fc1);
        fc1_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc1_zp_e, static_cast<int>(fusion_inter), ngk);
        if (!fc1_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
        bool wmma_data_format = (hidden_size % 32 == 0);
        if (wmma_data_format && count > 1) {
          fc1_pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
              *zpc, stream, fc1_zp_e, static_cast<int>(fusion_inter), ngk);
          if (!fc1_pre_zp_fp16) {
            result = -1;
            goto cleanup;
          }
        }
      }

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: fc1 matmul_nbits "
                        "[%lld x %lld] -> [%lld x %lld]\n",
                        (long long)e, (long long)count, (long long)hidden_size,
                        (long long)count, (long long)fusion_inter);
      HIP_CHECK(
          hip_matmul_nbits(stream, d_gather_buf, fc1_w_e, fc1_s_e, fc1_zp_e,
                           fc1_b_e, d_fc1_buf, count, fusion_inter, hidden_size,
                           1, expert_weight_bits, block_size, elem_size,
                           /*zp_elem_size=*/1, fc1_pre_zp_u8, fc1_pre_zp_fp16));

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: swiglu(alpha=%.3f, "
                        "beta=%.3f, limit=%.1f)\n",
                        (long long)e, (double)activation_alpha,
                        (double)activation_beta, (double)swiglu_limit);
      HIP_CHECK(hip_qmoe_swiglu(stream, d_fc1_buf, d_act_buf, count, inter_size,
                                activation_alpha, activation_beta, swiglu_limit,
                                elem_size));

      const char *fc2_w_e = static_cast<const char *>(fc2_weights) +
                            e * hidden_size * k_blocks_fc2 * blob_size_fc2;
      const char *fc2_s_e = static_cast<const char *>(fc2_scales) +
                            e * hidden_size * k_blocks_fc2 * elem_size;
      // See fc1_zp_e comment above -- same packed-nibble layout for fc2.
      const void *fc2_zp_e =
          fc2_zero_points ? static_cast<const char *>(fc2_zero_points) +
                                e * hidden_size * ((k_blocks_fc2 + 1) / 2)
                          : nullptr;
      const void *fc2_b_e = fc2_bias ? static_cast<const char *>(fc2_bias) +
                                           e * hidden_size * elem_size
                                     : nullptr;

      // Same pre-unpack as fc1; per-expert distinct pointer.
      const void *fc2_pre_zp_u8 = nullptr;
      const void *fc2_pre_zp_fp16 = nullptr;
      if (fc2_zp_e && expert_weight_bits == 4 && block_size > 0) {
        int ngk = static_cast<int>(k_blocks_fc2);
        fc2_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc2_zp_e, static_cast<int>(hidden_size), ngk);
        if (!fc2_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
        bool wmma_data_format = (inter_size % 32 == 0);
        if (wmma_data_format && count > 1) {
          fc2_pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
              *zpc, stream, fc2_zp_e, static_cast<int>(hidden_size), ngk);
          if (!fc2_pre_zp_fp16) {
            result = -1;
            goto cleanup;
          }
        }
      }

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: fc2 matmul_nbits "
                        "[%lld x %lld] -> [%lld x %lld]\n",
                        (long long)e, (long long)count, (long long)inter_size,
                        (long long)count, (long long)hidden_size);
      HIP_CHECK(hip_matmul_nbits(
          stream, d_act_buf, fc2_w_e, fc2_s_e, fc2_zp_e, fc2_b_e, d_fc2_buf,
          count, hidden_size, inter_size, 1, expert_weight_bits, block_size,
          elem_size, /*zp_elem_size=*/1, fc2_pre_zp_u8, fc2_pre_zp_fp16));

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: scatter_add\n",
                        (long long)e);
      HIP_CHECK(hip_qmoe_scatter_add(stream, output, d_fc2_buf, d_ids_e,
                                     d_wts_e, hidden_size, count, elem_size));
    }
  }

cleanup:
  // Sub-buffers above (d_expert_indices ... d_sorted_weights) are views into
  // the per-session RuntimeState::qmoe_scratch pool -- freed in
  // hipdnn_ep_state_cleanup. Do NOT hipFree them here.
  if (result == 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: completed successfully\n");
  }
  return result;
}
