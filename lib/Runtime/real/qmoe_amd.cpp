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
#include <cstdlib>
#include <vector>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

// com.amd QMoE (LatentMoE) real runtime implementation.
//
// Fully independent from lib/Runtime/real/qmoe.cpp (com.microsoft QMoE):
// different routing (sigmoid + correction-bias vs softmax), different
// activation (relu2 vs SwiGLU), and a mandatory latent projection +
// shared-expert branch that com.microsoft::QMoE does not have. Shares only
// the generic sub-kernels (hip_matmul_nbits, hip_qmoe_gather_tokens,
// hip_qmoe_scatter_add, hip_elementwise_add) -- none of qmoe.cpp's code is
// called or modified.
//
// There are no zero_points inputs in this op's schema (always symmetric
// zero-point 8, matching MatMulNBits' default), so every hip_matmul_nbits
// call below passes null zero_points/bias and skips the ZpUnpackCache
// machinery that com.microsoft::QMoE needs.
//
// Pipeline (Hip_QMoEAmdOp in include/hip/Dialect/IR/HipOps.td carries the
// full math):
//   1. hip_qmoe_amd_route            : sigmoid+correction-bias routing
//   2. fc1_latent_proj (matmul_nbits): hidden_states -> h        [latent]
//   3. routed branch -> acc                                     [latent]
//        decode/prefill (fused when supported): hip_qmoe_amd_decode_fused,
//        three launches per layer with device-side expert lookup
//          that index the expert weights on the device
//        otherwise: hip_qmoe_amd_bucket_tokens -> count readback -> per
//          active expert gather(h) -> fc1 -> relu2 -> fc2 -> weighted
//          scatter_add
//   4. fc2_latent_proj (matmul_nbits): acc -> y                 [hidden]
//   5. shared expert: hidden_states -> fc1 -> relu2 -> fc2 -> s  [hidden]
//   6. output = y + s (hip_elementwise_add)
int wrap_qmoe_amd(
    RuntimeState *state, const void *hidden_states,
    const void *fc1_experts_weights, const void *fc1_experts_scales,
    const void *fc2_experts_weights, const void *fc2_experts_scales,
    const void *fc1_latent_weights, const void *fc1_latent_scales,
    const void *fc2_latent_weights, const void *fc2_latent_scales,
    const void *shared_fc1_weights, const void *shared_fc1_scales,
    const void *shared_fc2_weights, const void *shared_fc2_scales,
    const void *router_weight, const void *correction_bias, void *output,
    int64_t num_tokens, int64_t hidden_size, int64_t latent_size,
    int64_t moe_intermediate_size, int64_t shared_intermediate_size,
    int64_t num_experts, int64_t k, int64_t expert_weight_bits,
    int64_t block_size, int64_t normalize_routing_weights,
    int64_t use_correction_bias, float routed_scaling_factor,
    int64_t activation_type, int64_t routing_type, int64_t elem_size) {
  OP_PROFILE(
      "qmoe_amd",
      [&] {
        char b[96];
        snprintf(b, sizeof(b),
                 "%lldx%lld,latent=%lld,moe_inter=%lld,shared_inter=%lld,e=%"
                 "lld",
                 (long long)num_tokens, (long long)hidden_size,
                 (long long)latent_size, (long long)moe_intermediate_size,
                 (long long)shared_intermediate_size, (long long)num_experts);
        return std::string(b);
      },
      state);

  if (!state || !hidden_states || !fc1_experts_weights || !fc1_experts_scales ||
      !fc2_experts_weights || !fc2_experts_scales || !fc1_latent_weights ||
      !fc1_latent_scales || !fc2_latent_weights || !fc2_latent_scales ||
      !shared_fc1_weights || !shared_fc1_scales || !shared_fc2_weights ||
      !shared_fc2_scales || !router_weight || !output) {
    fprintf(stderr, "wrap_qmoe_amd: null argument\n");
    return -1;
  }
  if (use_correction_bias && !correction_bias) {
    fprintf(stderr,
            "wrap_qmoe_amd: use_correction_bias=1 but correction_bias is "
            "null\n");
    return -1;
  }

  // The compiler constrains both modes before emitting this call, so these are
  // a backstop against a caller that bypassed it rather than the primary gate.
  if (activation_type != HIPDNN_EP_QMOE_AMD_ACTIVATION_RELU2) {
    fprintf(stderr,
            "wrap_qmoe_amd: unsupported activation_type=%lld (only relu2 is "
            "implemented)\n",
            (long long)activation_type);
    return -1;
  }
  if (routing_type != HIPDNN_EP_QMOE_AMD_ROUTING_SIGMOID) {
    fprintf(stderr,
            "wrap_qmoe_amd: unsupported routing_type=%lld (only sigmoid is "
            "implemented)\n",
            (long long)routing_type);
    return -1;
  }

  // fp16 only -- matches every QMoE sub-kernel this wrapper calls.
  if (elem_size != 2) {
    fprintf(stderr,
            "wrap_qmoe_amd: only fp16 (elem_size=2) supported, got "
            "%lld\n",
            (long long)elem_size);
    return -1;
  }
  // int4 only. hip_matmul_nbits itself accepts 2/3/4/8 bits, but the per-expert
  // weight strides computed below hardcode the 4-bit blob geometry
  // (block_size/2 bytes per row-block), so any other width would silently
  // index the wrong expert slice instead of failing.
  if (expert_weight_bits != 4) {
    fprintf(stderr,
            "wrap_qmoe_amd: only 4-bit expert weights supported, got "
            "expert_weight_bits=%lld\n",
            (long long)expert_weight_bits);
    return -1;
  }
  if (block_size <= 0 || (block_size & 1) != 0) {
    fprintf(stderr,
            "wrap_qmoe_amd: invalid block_size=%lld (must be a positive "
            "even value matching the weights' quant block layout)\n",
            (long long)block_size);
    return -1;
  }
  if (num_tokens <= 0 || hidden_size <= 0 || latent_size <= 0 ||
      moe_intermediate_size <= 0 || shared_intermediate_size <= 0 ||
      num_experts <= 0 || k <= 0 || k > num_experts) {
    fprintf(stderr,
            "wrap_qmoe_amd: invalid sizes (tokens=%lld hidden=%lld "
            "latent=%lld moe_inter=%lld shared_inter=%lld experts=%lld "
            "k=%lld)\n",
            (long long)num_tokens, (long long)hidden_size,
            (long long)latent_size, (long long)moe_intermediate_size,
            (long long)shared_intermediate_size, (long long)num_experts,
            (long long)k);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qmoe_amd(tokens=%lld, hidden=%lld, latent=%lld, "
      "moe_inter=%lld, shared_inter=%lld, experts=%lld, k=%lld, bits=%lld, "
      "block=%lld)\n",
      (long long)num_tokens, (long long)hidden_size, (long long)latent_size,
      (long long)moe_intermediate_size, (long long)shared_intermediate_size,
      (long long)num_experts, (long long)k, (long long)expert_weight_bits,
      (long long)block_size);

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_qmoe_amd: null stream\n");
    return -1;
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int result = 0;

  // Per-expert quant block geometry (K_blocks = ceil(K/block_size), blob =
  // block_size/2 bytes per row-block -- 4-bit packing, enforced above). See
  // MatMulNBits layout.
  int64_t k_blocks_e1 = (latent_size + block_size - 1) / block_size;
  int64_t k_blocks_e2 = (moe_intermediate_size + block_size - 1) / block_size;
  int64_t k_blocks_l1 = (hidden_size + block_size - 1) / block_size;
  int64_t k_blocks_l2 = (latent_size + block_size - 1) / block_size;
  int64_t k_blocks_s1 = (hidden_size + block_size - 1) / block_size;
  int64_t k_blocks_s2 =
      (shared_intermediate_size + block_size - 1) / block_size;
  int64_t blob_size = block_size / 2;

  // A single token selects k *distinct* experts, so its k routing slots
  // already ARE the active expert set: decode needs neither bucketing nor the
  // per-expert count readback, and the expert id can be looked up on the
  // device instead. That matters because the general path's readback costs a
  // full hipStreamSynchronize plus a 5-launch chain per active expert on every
  // decode step, which at this op's k is ~110 launches per layer.
  //
  // Prefill has two fast paths: slot-parallel decode-style fused (opt-in;
  // repeats expert weights per token) and expert-grouped fused (default),
  // which buckets on the GPU and GEMVs each expert once.
  //
  // Declared here rather than at the branch below because every `goto cleanup`
  // in between would otherwise jump into its scope.
  static const bool prefill_slot_fused = []() {
    const char *value = std::getenv("HIPDNN_EP_QMOE_AMD_PREFILL_FUSED");
    return value && std::atoi(value) != 0;
  }();
  static const bool prefill_grouped = []() {
    const char *value = std::getenv("HIPDNN_EP_QMOE_AMD_PREFILL_GROUPED");
    return !value || std::atoi(value) != 0;
  }();
  static const bool prefill_grouped_wmma = []() {
    const char *value = std::getenv("HIPDNN_EP_QMOE_AMD_PREFILL_GROUPED_WMMA");
    return !value || std::atoi(value) != 0;
  }();
  // Expert-grouped prefill can use hip_matmul_nbits per active expert (bulk
  // gather, one D2H/sync per layer) instead of the serial GEMV tiles in
  // qmoe_amd_prefill_fc{1,2}_kernel.
  static const bool prefill_grouped_matmul = []() {
    const char *value =
        std::getenv("HIPDNN_EP_QMOE_AMD_PREFILL_GROUPED_MATMUL");
    return value && std::atoi(value) != 0;
  }();
  const bool fused_supported =
      hip_qmoe_amd_decode_fused_supported(latent_size, moe_intermediate_size,
                                          block_size, expert_weight_bits,
                                          elem_size) != 0;
  const bool use_decode_fused = fused_supported && num_tokens == 1;
  const bool use_slot_prefill_fused =
      fused_supported && num_tokens > 1 && prefill_slot_fused;
  const bool use_fused_routed = use_decode_fused || use_slot_prefill_fused;
  const bool use_prefill_grouped = fused_supported && num_tokens > 1 &&
                                   prefill_grouped && !use_slot_prefill_fused;
  const bool use_prefill_grouped_wmma =
      use_prefill_grouped && prefill_grouped_wmma;
  const bool use_prefill_grouped_matmul = use_prefill_grouped &&
                                          !use_prefill_grouped_wmma &&
                                          prefill_grouped_matmul;
  const bool use_prefill_grouped_gemv = use_prefill_grouped &&
                                        !use_prefill_grouped_wmma &&
                                        !prefill_grouped_matmul;

  // Per-session grow-on-demand scratch (own qmoe_amd_scratch field --
  // independent from com.microsoft QMoE's qmoe_scratch). 64-byte aligned
  // sub-buffers, offsets recomputed per call.
  auto align_up_64 = [](size_t s) -> size_t { return (s + 63) & ~size_t(63); };
  // The general path uses the fc1/fc2 buffers for one active expert's rows
  // (at most num_tokens); the fused/grouped paths use [num_tokens * k, ...]
  // slot scratch for FC1/FC2 intermediates. The grouped WMMA path packs its
  // expert slices back-to-back in routing order, so it fits the same
  // num_tokens * k rows rather than a num_experts * num_tokens grid.
  const int64_t fc1_rows = (use_fused_routed || use_prefill_grouped)
                               ? (num_tokens * k)
                               : (num_tokens > k ? num_tokens : k);
  const int64_t fc2_rows = (use_fused_routed || use_prefill_grouped)
                               ? (num_tokens * k)
                               : (num_tokens > k ? num_tokens : k);
  const int64_t gather_rows =
      (use_prefill_grouped_matmul || use_prefill_grouped_wmma)
          ? (num_tokens * k)
          : num_tokens;
  size_t sz_expert_indices = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_expert_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_h_buf = align_up_64(num_tokens * latent_size * elem_size);
  size_t sz_acc_buf = align_up_64(num_tokens * latent_size * elem_size);
  size_t sz_gather_buf = align_up_64(gather_rows * latent_size * elem_size);
  size_t sz_fc1_buf = align_up_64(fc1_rows * moe_intermediate_size * elem_size);
  size_t sz_fc2_buf = align_up_64(fc2_rows * latent_size * elem_size);
  size_t sz_expert_counts = align_up_64(num_experts * sizeof(int32_t));
  size_t sz_expert_offsets = align_up_64((num_experts + 1) * sizeof(int32_t));
  size_t sz_sorted_token_ids = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_sorted_weights = align_up_64(num_tokens * k * elem_size);
  size_t sz_pair_to_padded = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_pair_to_slot = align_up_64(num_tokens * k * sizeof(int32_t));
  size_t sz_y_buf = align_up_64(num_tokens * hidden_size * elem_size);
  size_t sz_shared_buf =
      align_up_64(num_tokens * shared_intermediate_size * elem_size);

  size_t off_expert_indices = 0;
  size_t off_expert_weights = off_expert_indices + sz_expert_indices;
  size_t off_h_buf = off_expert_weights + sz_expert_weights;
  size_t off_acc_buf = off_h_buf + sz_h_buf;
  size_t off_gather_buf = off_acc_buf + sz_acc_buf;
  size_t off_fc1_buf = off_gather_buf + sz_gather_buf;
  size_t off_fc2_buf = off_fc1_buf + sz_fc1_buf;
  size_t off_expert_counts = off_fc2_buf + sz_fc2_buf;
  size_t off_expert_offsets = off_expert_counts + sz_expert_counts;
  size_t off_sorted_token_ids = off_expert_offsets + sz_expert_offsets;
  size_t off_sorted_weights = off_sorted_token_ids + sz_sorted_token_ids;
  size_t off_pair_to_padded = off_sorted_weights + sz_sorted_weights;
  size_t off_pair_to_slot = off_pair_to_padded + sz_pair_to_padded;
  size_t off_y_buf = off_pair_to_slot + sz_pair_to_slot;
  size_t off_shared_buf = off_y_buf + sz_y_buf;
  size_t total_scratch = off_shared_buf + sz_shared_buf;

  if (hipdnn_ep_state_ensure_qmoe_amd_scratch(state, total_scratch) != 0) {
    fprintf(stderr, "wrap_qmoe_amd: ensure_qmoe_amd_scratch(%zu) failed\n",
            total_scratch);
    return -1;
  }
  char *scratch_base =
      static_cast<char *>(hipdnn_ep_state_get_qmoe_amd_scratch(state));
  void *d_expert_indices = scratch_base + off_expert_indices;
  void *d_expert_weights = scratch_base + off_expert_weights;
  void *d_h_buf = scratch_base + off_h_buf;
  void *d_acc_buf = scratch_base + off_acc_buf;
  void *d_gather_buf = scratch_base + off_gather_buf;
  void *d_fc1_buf = scratch_base + off_fc1_buf;
  void *d_fc2_buf = scratch_base + off_fc2_buf;
  int32_t *d_expert_counts =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_counts);
  int32_t *d_expert_offsets =
      reinterpret_cast<int32_t *>(scratch_base + off_expert_offsets);
  int32_t *d_sorted_token_ids =
      reinterpret_cast<int32_t *>(scratch_base + off_sorted_token_ids);
  char *d_sorted_weights = scratch_base + off_sorted_weights;
  int32_t *d_pair_to_padded =
      reinterpret_cast<int32_t *>(scratch_base + off_pair_to_padded);
  int32_t *d_pair_to_slot =
      reinterpret_cast<int32_t *>(scratch_base + off_pair_to_slot);
  void *d_y_buf = scratch_base + off_y_buf;
  void *d_shared_buf = scratch_base + off_shared_buf;

  // 1. Routing: sigmoid(x @ router_weight), correction-biased top-k select,
  //    raw-prob weight, optional normalize, routed_scaling_factor.
  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: route(tokens=%lld, experts=%lld, "
                    "k=%lld)\n",
                    (long long)num_tokens, (long long)num_experts,
                    (long long)k);
  HIP_CHECK(hip_qmoe_amd_route(
      stream, hidden_states, router_weight, correction_bias, d_expert_indices,
      d_expert_weights,
      use_fused_routed ? static_cast<void *>(d_expert_counts) : nullptr,
      num_tokens, hidden_size, num_experts, k, normalize_routing_weights,
      use_correction_bias, routed_scaling_factor, elem_size));

  // 2. fc1_latent_proj: hidden_size -> latent_size (dense batch matmul).
  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: fc1_latent_proj [%lld x %lld] -> "
                    "[%lld x %lld]\n",
                    (long long)num_tokens, (long long)hidden_size,
                    (long long)num_tokens, (long long)latent_size);
  HIP_CHECK(hip_matmul_nbits(stream, hidden_states, fc1_latent_weights,
                             fc1_latent_scales, /*zero_points=*/nullptr,
                             /*bias=*/nullptr, d_h_buf, num_tokens, latent_size,
                             hidden_size, /*batch_count=*/1, expert_weight_bits,
                             block_size, elem_size, /*zp_elem_size=*/1,
                             /*pre_unpacked_zp_u8=*/nullptr,
                             /*pre_unpacked_zp_fp16=*/nullptr));

  // 3. Routed branch: decode slot-fused, prefill expert-grouped fused, or the
  //    legacy host per-expert dispatch loop.
  if (use_fused_routed) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: fused routed branch "
                      "(tokens=%lld k=%lld)\n",
                      (long long)num_tokens, (long long)k);
    // Writes every element of acc, so it needs no pre-zeroing.
    HIP_CHECK(hip_qmoe_amd_decode_fused(
        stream, d_h_buf, d_expert_indices, d_expert_weights,
        fc1_experts_weights, fc1_experts_scales, fc2_experts_weights,
        fc2_experts_scales, /*act_scratch=*/d_fc1_buf,
        /*slot_scratch=*/d_fc2_buf, /*acc=*/d_acc_buf, num_tokens, latent_size,
        moe_intermediate_size, k, expert_weight_bits, block_size, elem_size));
  } else if (use_prefill_grouped_wmma) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: prefill grouped WMMA branch "
                      "(tokens=%lld experts=%lld k=%lld)\n",
                      (long long)num_tokens, (long long)num_experts,
                      (long long)k);
    HIP_CHECK(hip_qmoe_amd_prefill_grouped_wmma(
        stream, d_h_buf, d_expert_indices, d_expert_weights,
        fc1_experts_weights, fc1_experts_scales, fc2_experts_weights,
        fc2_experts_scales, d_expert_counts, d_expert_offsets,
        d_sorted_token_ids, d_sorted_weights, d_pair_to_padded, d_pair_to_slot,
        /*packed_latent=*/d_gather_buf, /*packed_act=*/d_fc1_buf,
        /*slot_scratch=*/d_fc2_buf, /*acc=*/d_acc_buf, num_tokens, num_experts,
        latent_size, moe_intermediate_size, k, expert_weight_bits, block_size,
        elem_size));
  } else if (use_prefill_grouped_matmul) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: prefill grouped matmul branch "
                      "(tokens=%lld experts=%lld k=%lld)\n",
                      (long long)num_tokens, (long long)num_experts,
                      (long long)k);
    HIP_CHECK(hipMemsetAsync(d_acc_buf, 0, num_tokens * latent_size * elem_size,
                             hip_stream));

    HIP_CHECK(hip_qmoe_amd_bucket_tokens(
        stream, d_expert_indices, d_expert_weights, d_expert_counts,
        d_expert_offsets, d_sorted_token_ids, d_sorted_weights, num_tokens,
        num_experts, k, elem_size));

    const int64_t total_pairs = num_tokens * k;
    HIP_CHECK(hip_qmoe_gather_tokens(stream, d_h_buf, d_gather_buf,
                                     d_sorted_token_ids, latent_size,
                                     total_pairs, elem_size));

    size_t total_host = align_up_64(num_experts * sizeof(int32_t));
    if (hipdnn_ep_state_ensure_qmoe_amd_host_scratch(state, total_host) != 0) {
      fprintf(stderr,
              "wrap_qmoe_amd: ensure_qmoe_amd_host_scratch(%zu) failed\n",
              total_host);
      return -1;
    }
    int32_t *h_counts = static_cast<int32_t *>(
        hipdnn_ep_state_get_qmoe_amd_host_scratch(state));

    HIP_CHECK(hipMemcpyAsync(h_counts, d_expert_counts,
                             num_experts * sizeof(int32_t),
                             hipMemcpyDeviceToHost, hip_stream));
    HIP_CHECK(hipStreamSynchronize(hip_stream));

    std::vector<int64_t> h_offsets(num_experts + 1, 0);
    for (int64_t e = 0; e < num_experts; e++) {
      h_offsets[e + 1] = h_offsets[e] + static_cast<int64_t>(h_counts[e]);
    }

    const int64_t pair_stride_latent = latent_size * elem_size;
    const int64_t pair_stride_moe = moe_intermediate_size * elem_size;
    char *gather_base = static_cast<char *>(d_gather_buf);
    char *fc1_base = static_cast<char *>(d_fc1_buf);
    char *fc2_base = static_cast<char *>(d_fc2_buf);

    for (int64_t e = 0; e < num_experts; e++) {
      const int64_t count = static_cast<int64_t>(h_counts[e]);
      if (count == 0) {
        continue;
      }

      const int64_t off_e = h_offsets[e];
      const char *gather_e = gather_base + off_e * pair_stride_latent;
      char *fc1_e = fc1_base + off_e * pair_stride_moe;
      char *fc2_e = fc2_base + off_e * pair_stride_latent;
      int32_t *d_ids_e = d_sorted_token_ids + off_e;
      char *d_wts_e = d_sorted_weights + off_e * elem_size;

      const char *fc1_w_e = static_cast<const char *>(fc1_experts_weights) +
                            e * moe_intermediate_size * k_blocks_e1 * blob_size;
      const char *fc1_s_e = static_cast<const char *>(fc1_experts_scales) +
                            e * moe_intermediate_size * k_blocks_e1 * elem_size;
      HIP_CHECK(hip_matmul_nbits(
          stream, gather_e, fc1_w_e, fc1_s_e, /*zero_points=*/nullptr,
          /*bias=*/nullptr, fc1_e, count, moe_intermediate_size, latent_size,
          /*batch_count=*/1, expert_weight_bits, block_size, elem_size,
          /*zp_elem_size=*/1, /*pre_unpacked_zp_u8=*/nullptr,
          /*pre_unpacked_zp_fp16=*/nullptr));

      HIP_CHECK(hip_qmoe_amd_relu2(stream, fc1_e, fc1_e, count,
                                   moe_intermediate_size, elem_size));

      const char *fc2_w_e = static_cast<const char *>(fc2_experts_weights) +
                            e * latent_size * k_blocks_e2 * blob_size;
      const char *fc2_s_e = static_cast<const char *>(fc2_experts_scales) +
                            e * latent_size * k_blocks_e2 * elem_size;
      HIP_CHECK(hip_matmul_nbits(
          stream, fc1_e, fc2_w_e, fc2_s_e, /*zero_points=*/nullptr,
          /*bias=*/nullptr, fc2_e, count, latent_size, moe_intermediate_size,
          /*batch_count=*/1, expert_weight_bits, block_size, elem_size,
          /*zp_elem_size=*/1, /*pre_unpacked_zp_u8=*/nullptr,
          /*pre_unpacked_zp_fp16=*/nullptr));

      HIP_CHECK(hip_qmoe_scatter_add(stream, d_acc_buf, fc2_e, d_ids_e, d_wts_e,
                                     latent_size, count, elem_size));
    }
  } else if (use_prefill_grouped_gemv) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: prefill grouped gemv branch "
                      "(tokens=%lld experts=%lld k=%lld)\n",
                      (long long)num_tokens, (long long)num_experts,
                      (long long)k);
    HIP_CHECK(hip_qmoe_amd_prefill_grouped_fused(
        stream, d_h_buf, d_expert_indices, d_expert_weights,
        fc1_experts_weights, fc1_experts_scales, fc2_experts_weights,
        fc2_experts_scales, d_expert_counts, d_expert_offsets,
        d_sorted_token_ids, d_sorted_weights, /*act_scratch=*/d_fc1_buf,
        /*slot_scratch=*/d_fc2_buf, /*acc=*/d_acc_buf, num_tokens, num_experts,
        latent_size, moe_intermediate_size, k, expert_weight_bits, block_size,
        elem_size));
  } else {
    HIP_CHECK(hipMemsetAsync(d_acc_buf, 0, num_tokens * latent_size * elem_size,
                             hip_stream));

    // GPU-side bucketing + a small D2H readback of just the per-expert counts
    // to drive the host-side per-expert dispatch loop below.
    HIP_CHECK(hip_qmoe_amd_bucket_tokens(
        stream, d_expert_indices, d_expert_weights, d_expert_counts,
        d_expert_offsets, d_sorted_token_ids, d_sorted_weights, num_tokens,
        num_experts, k, elem_size));

    size_t total_host = align_up_64(num_experts * sizeof(int32_t));
    if (hipdnn_ep_state_ensure_qmoe_amd_host_scratch(state, total_host) != 0) {
      fprintf(stderr,
              "wrap_qmoe_amd: ensure_qmoe_amd_host_scratch(%zu) failed\n",
              total_host);
      return -1;
    }
    int32_t *h_counts = static_cast<int32_t *>(
        hipdnn_ep_state_get_qmoe_amd_host_scratch(state));

    HIP_CHECK(hipMemcpyAsync(h_counts, d_expert_counts,
                             num_experts * sizeof(int32_t),
                             hipMemcpyDeviceToHost, hip_stream));
    HIP_CHECK(hipStreamSynchronize(hip_stream));

    std::vector<int64_t> h_offsets(num_experts + 1, 0);
    for (int64_t e = 0; e < num_experts; e++) {
      h_offsets[e + 1] = h_offsets[e] + static_cast<int64_t>(h_counts[e]);
    }

    int64_t active_experts = 0;
    for (int64_t e = 0; e < num_experts; e++) {
      if (h_counts[e] > 0) {
        active_experts++;
      }
    }
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: %lld/%lld experts active\n",
                      (long long)active_experts, (long long)num_experts);

    // Per active expert: gather (from h_buf) -> fc1 -> relu2 -> fc2 ->
    // weighted scatter_add into acc_buf.
    for (int64_t e = 0; e < num_experts; e++) {
      int64_t count = static_cast<int64_t>(h_counts[e]);
      if (count == 0) {
        continue;
      }

      int64_t off_e = h_offsets[e];
      int32_t *d_ids_e = d_sorted_token_ids + off_e;
      char *d_wts_e = d_sorted_weights + off_e * elem_size;

      HIP_CHECK(hip_qmoe_gather_tokens(stream, d_h_buf, d_gather_buf, d_ids_e,
                                       latent_size, count, elem_size));

      const char *fc1_w_e = static_cast<const char *>(fc1_experts_weights) +
                            e * moe_intermediate_size * k_blocks_e1 * blob_size;
      const char *fc1_s_e = static_cast<const char *>(fc1_experts_scales) +
                            e * moe_intermediate_size * k_blocks_e1 * elem_size;
      HIP_CHECK(hip_matmul_nbits(
          stream, d_gather_buf, fc1_w_e, fc1_s_e, /*zero_points=*/nullptr,
          /*bias=*/nullptr, d_fc1_buf, count, moe_intermediate_size,
          latent_size, /*batch_count=*/1, expert_weight_bits, block_size,
          elem_size, /*zp_elem_size=*/1, /*pre_unpacked_zp_u8=*/nullptr,
          /*pre_unpacked_zp_fp16=*/nullptr));

      HIP_CHECK(hip_qmoe_amd_relu2(stream, d_fc1_buf, d_fc1_buf, count,
                                   moe_intermediate_size, elem_size));

      const char *fc2_w_e = static_cast<const char *>(fc2_experts_weights) +
                            e * latent_size * k_blocks_e2 * blob_size;
      const char *fc2_s_e = static_cast<const char *>(fc2_experts_scales) +
                            e * latent_size * k_blocks_e2 * elem_size;
      HIP_CHECK(hip_matmul_nbits(
          stream, d_fc1_buf, fc2_w_e, fc2_s_e, /*zero_points=*/nullptr,
          /*bias=*/nullptr, d_fc2_buf, count, latent_size,
          moe_intermediate_size, /*batch_count=*/1, expert_weight_bits,
          block_size, elem_size, /*zp_elem_size=*/1,
          /*pre_unpacked_zp_u8=*/nullptr, /*pre_unpacked_zp_fp16=*/nullptr));

      HIP_CHECK(hip_qmoe_scatter_add(stream, d_acc_buf, d_fc2_buf, d_ids_e,
                                     d_wts_e, latent_size, count, elem_size));
    }
  }

  // 4. fc2_latent_proj: latent_size -> hidden_size, on the routed
  //    accumulator.
  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: fc2_latent_proj [%lld x %lld] -> "
                    "[%lld x %lld]\n",
                    (long long)num_tokens, (long long)latent_size,
                    (long long)num_tokens, (long long)hidden_size);
  HIP_CHECK(hip_matmul_nbits(stream, d_acc_buf, fc2_latent_weights,
                             fc2_latent_scales, /*zero_points=*/nullptr,
                             /*bias=*/nullptr, d_y_buf, num_tokens, hidden_size,
                             latent_size, /*batch_count=*/1, expert_weight_bits,
                             block_size, elem_size, /*zp_elem_size=*/1,
                             /*pre_unpacked_zp_u8=*/nullptr,
                             /*pre_unpacked_zp_fp16=*/nullptr));

  // 5. Shared expert branch: runs for every token, independent of routing.
  //    Writes directly into `output` (== s) so step 6 only needs one add.
  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: shared_fc1 [%lld x %lld] -> [%lld "
                    "x %lld]\n",
                    (long long)num_tokens, (long long)hidden_size,
                    (long long)num_tokens, (long long)shared_intermediate_size);
  HIP_CHECK(hip_matmul_nbits(
      stream, hidden_states, shared_fc1_weights, shared_fc1_scales,
      /*zero_points=*/nullptr, /*bias=*/nullptr, d_shared_buf, num_tokens,
      shared_intermediate_size, hidden_size, /*batch_count=*/1,
      expert_weight_bits, block_size, elem_size, /*zp_elem_size=*/1,
      /*pre_unpacked_zp_u8=*/nullptr, /*pre_unpacked_zp_fp16=*/nullptr));

  HIP_CHECK(hip_qmoe_amd_relu2(stream, d_shared_buf, d_shared_buf, num_tokens,
                               shared_intermediate_size, elem_size));

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: shared_fc2 [%lld x %lld] -> [%lld "
                    "x %lld]\n",
                    (long long)num_tokens, (long long)shared_intermediate_size,
                    (long long)num_tokens, (long long)hidden_size);
  HIP_CHECK(hip_matmul_nbits(stream, d_shared_buf, shared_fc2_weights,
                             shared_fc2_scales, /*zero_points=*/nullptr,
                             /*bias=*/nullptr, output, num_tokens, hidden_size,
                             shared_intermediate_size, /*batch_count=*/1,
                             expert_weight_bits, block_size, elem_size,
                             /*zp_elem_size=*/1, /*pre_unpacked_zp_u8=*/nullptr,
                             /*pre_unpacked_zp_fp16=*/nullptr));

  // 6. output = y + s.
  HIP_CHECK(hip_elementwise_add(stream, d_y_buf, output, output,
                                num_tokens * hidden_size,
                                /*hip_dtype=*/HIP_DTYPE_FLOAT16));

cleanup:
  // Sub-buffers above are views into the per-session
  // RuntimeState::qmoe_amd_scratch pool -- freed in hipdnn_ep_state_cleanup.
  // Do NOT hipFree them here.
  if (result == 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe_amd: completed successfully\n");
  }
  return result;
}
