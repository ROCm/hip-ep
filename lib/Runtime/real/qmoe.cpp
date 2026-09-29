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
#include <string>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

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

  int result = 0;

  int64_t fusion_inter = 2 * inter_size;
  int64_t k_blocks_fc1 = (hidden_size + block_size - 1) / block_size;
  int64_t k_blocks_fc2 = (inter_size + block_size - 1) / block_size;
  int64_t routed_rows = num_tokens * k;

  // Per-state grow-on-demand scratch in place of 8 hipMalloc/8 hipFree per
  // call. Sub-buffers are 64-byte aligned (matches GPU pool alignment, gives
  // each sub-buffer its own cache line). The buffer grows when num_tokens /
  // sizes exceed the cached capacity, never shrinks; freed in state cleanup.
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_fc1_buf = align_up_64(routed_rows * fusion_inter * elem_size);
  // Fused decode (num_tokens == 1) reuses act_buf and fc2_buf as the [k,
  // inter] activation slots and [k, hidden] per-expert output slots needed
  // by hip_qmoe_decode_fused (gather/scatter happen inline inside the
  // kernel, indexed by expert_indices). For num_tokens > 1 the multi-pass
  // path uses [num_tokens, ...] sizing. Take the max so the per-state
  // scratch is never under-sized regardless of which path runs.
  int64_t act_slots = std::max<int64_t>(routed_rows, k);
  size_t sz_act_buf = align_up_64(act_slots * inter_size * elem_size);
  size_t sz_fc2_buf = align_up_64(act_slots * hidden_size * elem_size);
  // Device bucket outputs: per-expert counts/offsets, sorted routed pairs,
  // inverse pair mapping, adaptive <=64-row task descriptors, and two queues.
  size_t sz_expert_counts = align_up_64(num_experts * sizeof(int32_t));
  size_t sz_expert_offsets = align_up_64((num_experts + 1) * sizeof(int32_t));
  size_t sz_sorted_token_ids = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_sorted_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_pair_to_sorted = align_up_64(routed_rows * sizeof(int32_t));
  size_t sz_row_groups = align_up_64(routed_rows * 4 * sizeof(int32_t));
  size_t sz_row_group_count = align_up_64(sizeof(int32_t));
  size_t sz_fc1_queue_head = align_up_64(sizeof(uint32_t));
  size_t sz_fc2_queue_head = align_up_64(sizeof(uint32_t));
  // dp4a decode scratch (fused decode only, env-gated): int8-quantized
  // activations + per-group fp32 scales for the fc1 input ([hidden]) and the
  // fc2 slot activations ([k, inter]). Sized unconditionally (a few KB) so the
  // offset layout is identical whether or not dp4a runs; the fp path ignores
  // them. k_blocks_fc1 == n_blk_in, k_blocks_fc2 == n_blk_mid.
  size_t sz_a_qb_in = align_up_64(hidden_size * sizeof(int8_t));
  size_t sz_a_scale_in = align_up_64(k_blocks_fc1 * sizeof(float));
  size_t sz_a_qb_mid = align_up_64(k * inter_size * sizeof(int8_t));
  size_t sz_a_scale_mid = align_up_64(k * k_blocks_fc2 * sizeof(float));

  size_t off_expert_indices = 0;
  size_t off_expert_weights = off_expert_indices + sz_expert_indices;
  size_t off_fc1_buf = off_expert_weights + sz_expert_weights;
  size_t off_act_buf = off_fc1_buf + sz_fc1_buf;
  size_t off_fc2_buf = off_act_buf + sz_act_buf;
  size_t off_expert_counts = off_fc2_buf + sz_fc2_buf;
  size_t off_expert_offsets = off_expert_counts + sz_expert_counts;
  size_t off_sorted_token_ids = off_expert_offsets + sz_expert_offsets;
  size_t off_sorted_weights = off_sorted_token_ids + sz_sorted_token_ids;
  size_t off_pair_to_sorted = off_sorted_weights + sz_sorted_weights;
  size_t off_row_groups = off_pair_to_sorted + sz_pair_to_sorted;
  size_t off_row_group_count = off_row_groups + sz_row_groups;
  size_t off_fc1_queue_head = off_row_group_count + sz_row_group_count;
  size_t off_fc2_queue_head = off_fc1_queue_head + sz_fc1_queue_head;
  size_t off_a_qb_in = off_fc2_queue_head + sz_fc2_queue_head;
  size_t off_a_scale_in = off_a_qb_in + sz_a_qb_in;
  size_t off_a_qb_mid = off_a_scale_in + sz_a_scale_in;
  size_t off_a_scale_mid = off_a_qb_mid + sz_a_qb_mid;
  size_t total_scratch = off_a_scale_mid + sz_a_scale_mid;

  if (hipdnn_ep_state_ensure_qmoe_scratch(state, total_scratch) != 0) {
    fprintf(stderr, "wrap_qmoe: ensure_qmoe_scratch(%zu) failed\n",
            total_scratch);
    return -1;
  }
  char *scratch_base =
      static_cast<char *>(hipdnn_ep_state_get_qmoe_scratch(state));
  void *d_expert_indices = scratch_base + off_expert_indices;
  void *d_expert_weights = scratch_base + off_expert_weights;
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
  int32_t *d_pair_to_sorted =
      reinterpret_cast<int32_t *>(scratch_base + off_pair_to_sorted);
  int32_t *d_row_groups =
      reinterpret_cast<int32_t *>(scratch_base + off_row_groups);
  int32_t *d_row_group_count =
      reinterpret_cast<int32_t *>(scratch_base + off_row_group_count);
  uint32_t *d_fc1_queue_head =
      reinterpret_cast<uint32_t *>(scratch_base + off_fc1_queue_head);
  uint32_t *d_fc2_queue_head =
      reinterpret_cast<uint32_t *>(scratch_base + off_fc2_queue_head);
  void *d_a_qb_in = scratch_base + off_a_qb_in;
  void *d_a_scale_in = scratch_base + off_a_scale_in;
  void *d_a_qb_mid = scratch_base + off_a_qb_mid;
  void *d_a_scale_mid = scratch_base + off_a_scale_mid;

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

  // Device-driven prefill path. The bucket launch creates every descriptor
  // consumed below and resets two independent persistent queues. FC1 gathers
  // source token rows through d_sorted_token_ids; FC2 consumes the contiguous
  // sorted activation rows. No expert metadata is copied to the host.
  {
    if (expert_weight_bits != 4) {
      fprintf(stderr,
              "wrap_qmoe: ragged prefill supports expert_weight_bits=4, got "
              "%lld\n",
              (long long)expert_weight_bits);
      return -1;
    }

    const void *fc1_pre_zp_u8 = nullptr;
    const void *fc2_pre_zp_u8 = nullptr;
    if (fc1_zero_points || fc2_zero_points) {
      hipdnn_ep_real::ZpUnpackCache *zpc =
          hipdnn_ep_real::get_or_create_zp_cache(state);
      if (!zpc) {
        fprintf(stderr, "wrap_qmoe: failed to create zero-point cache\n");
        return -1;
      }
      if (fc1_zero_points) {
        fc1_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc1_zero_points,
            static_cast<int>(num_experts * fusion_inter),
            static_cast<int>(k_blocks_fc1));
        if (!fc1_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
      }
      if (fc2_zero_points) {
        fc2_pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
            *zpc, stream, fc2_zero_points,
            static_cast<int>(num_experts * hidden_size),
            static_cast<int>(k_blocks_fc2));
        if (!fc2_pre_zp_u8) {
          result = -1;
          goto cleanup;
        }
      }
    }

    constexpr int64_t row_group_size = 64;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_qmoe: device ragged prefill rows=%lld experts=%lld\n",
        (long long)routed_rows, (long long)num_experts);
    HIP_CHECK(hip_qmoe_bucket_tokens_ragged(
        stream, d_expert_indices, d_expert_weights, d_expert_counts,
        d_expert_offsets, d_sorted_token_ids, d_sorted_weights,
        d_pair_to_sorted, d_row_groups, d_row_group_count, d_fc1_queue_head,
        d_fc2_queue_head, num_tokens, num_experts, k, row_group_size,
        elem_size));

    HIP_CHECK(hip_qmoe_ragged_matmul_nbits(
        stream, input, d_sorted_token_ids, d_row_groups, d_row_group_count,
        d_fc1_queue_head, fc1_weights, fc1_scales, fc1_pre_zp_u8, fc1_bias,
        d_fc1_buf, routed_rows, num_experts, fusion_inter, hidden_size,
        block_size, elem_size));

    HIP_CHECK(hip_qmoe_swiglu(stream, d_fc1_buf, d_act_buf, routed_rows,
                              inter_size, activation_alpha, activation_beta,
                              swiglu_limit, elem_size));

    HIP_CHECK(hip_qmoe_ragged_matmul_nbits(
        stream, d_act_buf, /*input_row_ids=*/nullptr, d_row_groups,
        d_row_group_count, d_fc2_queue_head, fc2_weights, fc2_scales,
        fc2_pre_zp_u8, fc2_bias, d_fc2_buf, routed_rows, num_experts,
        hidden_size, inter_size, block_size, elem_size));

    HIP_CHECK(hip_qmoe_reduce_sorted_pairs(
        stream, d_fc2_buf, d_pair_to_sorted, d_expert_indices, d_expert_weights,
        output, num_tokens, hidden_size, k, elem_size));
    return 0;
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
