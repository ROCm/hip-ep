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

  // Per-state grow-on-demand scratch in place of 8 hipMalloc/8 hipFree per
  // call. Sub-buffers are 64-byte aligned (matches GPU pool alignment, gives
  // each sub-buffer its own cache line). The buffer grows when num_tokens /
  // sizes exceed the cached capacity, never shrinks; freed in state cleanup.
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_gather_buf = align_up_64(num_tokens * hidden_size * elem_size);
  size_t sz_fc1_buf = align_up_64(num_tokens * fusion_inter * elem_size);
  // Fused decode (num_tokens == 1) reuses act_buf and fc2_buf as the [k,
  // inter] activation slots and [k, hidden] per-expert output slots needed
  // by hip_qmoe_decode_fused (gather/scatter happen inline inside the
  // kernel, indexed by expert_indices). For num_tokens > 1 the multi-pass
  // path uses [num_tokens, ...] sizing. Take the max so the per-state
  // scratch is never under-sized regardless of which path runs.
  int64_t act_slots = std::max<int64_t>(num_tokens, k);
  size_t sz_act_buf = align_up_64(act_slots * inter_size * elem_size);
  size_t sz_fc2_buf = align_up_64(act_slots * hidden_size * elem_size);
  // bucket_tokens outputs (Phase 2): per-expert counts + exclusive prefix sum
  // offsets, plus tokens/weights re-grouped on-device into per-expert
  // contiguous slices. Replaces the old per-expert host h_ids/h_wts_e build +
  // H2D round-trip (2 hipMemcpyAsync per active expert per layer).
  size_t sz_expert_counts = align_up_64(num_experts * sizeof(int32_t));
  size_t sz_expert_offsets = align_up_64((num_experts + 1) * sizeof(int32_t));
  size_t sz_sorted_token_ids = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_sorted_weights = align_up_64(num_tokens * k * elem_size);
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
