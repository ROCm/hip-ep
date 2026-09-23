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

#include <cstdio>

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

  int result = 0;

  int64_t k_blocks_fc1 = (hidden_size + block_size - 1) / block_size;
  int64_t k_blocks_fc2 = (inter_size + block_size - 1) / block_size;

  // Per-state grow-on-demand scratch in place of 8 hipMalloc/8 hipFree per
  // call. Sub-buffers are 64-byte aligned (matches GPU pool alignment, gives
  // each sub-buffer its own cache line). The buffer grows when num_tokens /
  // sizes exceed the cached capacity, never shrinks; freed in state cleanup.
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  // Both fused paths use one activation/output slot per routing choice.
  // Decode has k slots; prefill has num_tokens*k slots.
  int64_t act_slots = num_tokens == 1 ? k : num_tokens * k;
  size_t sz_act_buf = align_up_64(act_slots * inter_size * elem_size);
  size_t sz_fc2_buf = align_up_64(act_slots * hidden_size * elem_size);
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
  size_t off_act_buf = off_expert_weights + sz_expert_weights;
  size_t off_fc2_buf = off_act_buf + sz_act_buf;
  size_t off_a_qb_in = off_fc2_buf + sz_fc2_buf;
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
  void *d_act_buf = scratch_base + off_act_buf;
  void *d_fc2_buf = scratch_base + off_fc2_buf;
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
  // scatter_add) sequence. d_act_buf is reused as the [k, inter]
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

  // Fixed three-launch prefill sequence. Every (token, routing-slot) row
  // selects its expert weights on device; FC1 fuses SwiGLU, FC2 writes a
  // routing-weighted slot, and the final kernel reduces k slots per token.
  // No bucketing, count D2H, host branch, memset, or stream synchronization.
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qmoe: fused prefill (tokens=%lld experts=%lld k=%lld)\n",
      (long long)num_tokens, (long long)num_experts, (long long)k);
  HIP_CHECK(hip_qmoe_prefill_fused(
      stream, input, d_expert_indices, d_expert_weights, fc1_weights,
      fc1_scales, fc1_zero_points, fc1_bias, fc2_weights, fc2_scales,
      fc2_zero_points, fc2_bias, d_fc2_buf, d_act_buf, output, num_tokens,
      hidden_size, inter_size, k, expert_weight_bits, block_size,
      activation_alpha, activation_beta, swiglu_limit, elem_size));

cleanup:
  // Sub-buffers above are views into the per-session
  // RuntimeState::qmoe_scratch pool, freed in state cleanup.
  if (result == 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: completed successfully\n");
  }
  return result;
}
