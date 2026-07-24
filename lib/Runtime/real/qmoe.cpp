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

  int64_t act_slots = std::max<int64_t>(num_tokens, k);

  hipdnn_ep_scratch_restore(state, 0);
  {
    size_t total = 0;
    auto align64 = [](size_t s) { return (s + 63) & ~size_t(63); };
    total += align64(num_tokens * k * sizeof(int32_t));
    total += align64(num_tokens * k * elem_size);
    total += align64(num_tokens * hidden_size * elem_size);
    total += align64(num_tokens * fusion_inter * elem_size);
    total += align64(act_slots * inter_size * elem_size);
    total += align64(act_slots * hidden_size * elem_size);
    total += align64(num_experts * sizeof(int32_t));
    total += align64((num_experts + 1) * sizeof(int32_t));
    total += align64(num_tokens * k * sizeof(int32_t));
    total += align64(num_tokens * k * elem_size);
    if (hipdnn_ep_scratch_reserve(state, total) != 0) {
      fprintf(stderr, "wrap_qmoe: scratch_reserve(%zu) failed\n", total);
      return -1;
    }
  }
  void *d_expert_indices =
      hipdnn_ep_scratch_alloc(state, num_tokens * k * sizeof(int32_t));
  void *d_expert_weights =
      hipdnn_ep_scratch_alloc(state, num_tokens * k * elem_size);
  void *d_gather_buf =
      hipdnn_ep_scratch_alloc(state, num_tokens * hidden_size * elem_size);
  void *d_fc1_buf =
      hipdnn_ep_scratch_alloc(state, num_tokens * fusion_inter * elem_size);
  void *d_act_buf =
      hipdnn_ep_scratch_alloc(state, act_slots * inter_size * elem_size);
  void *d_fc2_buf =
      hipdnn_ep_scratch_alloc(state, act_slots * hidden_size * elem_size);
  int32_t *d_expert_counts = static_cast<int32_t *>(
      hipdnn_ep_scratch_alloc(state, num_experts * sizeof(int32_t)));
  int32_t *d_expert_offsets = static_cast<int32_t *>(
      hipdnn_ep_scratch_alloc(state, (num_experts + 1) * sizeof(int32_t)));
  int32_t *d_sorted_token_ids = static_cast<int32_t *>(
      hipdnn_ep_scratch_alloc(state, num_tokens * k * sizeof(int32_t)));
  char *d_sorted_weights = static_cast<char *>(
      hipdnn_ep_scratch_alloc(state, num_tokens * k * elem_size));

  if (!d_expert_indices || !d_expert_weights || !d_gather_buf || !d_fc1_buf ||
      !d_act_buf || !d_fc2_buf || !d_expert_counts || !d_expert_offsets ||
      !d_sorted_token_ids || !d_sorted_weights) {
    fprintf(stderr, "wrap_qmoe: scratch_alloc failed\n");
    return -1;
  }

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
    size_t total_host = hsz_counts;

    if (hipdnn_ep_state_ensure_qmoe_host_scratch(state, total_host) != 0) {
      fprintf(stderr, "wrap_qmoe: ensure_qmoe_host_scratch(%zu) failed\n",
              total_host);
      return -1;
    }
    char *host_base =
        static_cast<char *>(hipdnn_ep_state_get_qmoe_host_scratch(state));
    int32_t *h_counts = reinterpret_cast<int32_t *>(host_base);

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

    int64_t active_experts = 0;
    for (int64_t e = 0; e < num_experts; e++) {
      if (h_counts[e] > 0) {
        active_experts++;
      }
    }
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: %lld/%lld experts active\n",
                      (long long)active_experts, (long long)num_experts);

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
