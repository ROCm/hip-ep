/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../debug_log.h"
#include "runtime_types.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t _err = (cmd);                                                   \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(_err));                                         \
      rc = -1;                                                                 \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define KERNEL_CHECK(call)                                                     \
  do {                                                                         \
    int _rc = (call);                                                          \
    if (_rc != 0) {                                                            \
      fprintf(stderr, "Kernel failed at %s:%d (rc=%d)\n", __FILE__, __LINE__,  \
              _rc);                                                            \
      rc = _rc;                                                                \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

struct TokenEntry {
  int32_t token_id;
  int32_t slot; // index into the per-token top-k list
};

int wrap_qmoe(RuntimeState* state,
              const void* input, const void* router_probs,
              const void* fc1_weights, const void* fc1_scales,
              const void* fc1_bias,
              const void* fc2_weights, const void* fc2_scales,
              const void* fc2_bias,
              const void* fc3_weights, const void* fc3_scales,
              const void* fc3_bias,
              const void* fc1_zero_points, const void* fc2_zero_points,
              const void* fc3_zero_points,
              void* output,
              int64_t num_tokens, int64_t hidden_size,
              int64_t inter_size, int64_t num_experts,
              int64_t k, int64_t expert_weight_bits,
              int64_t block_size, int64_t swiglu_fusion,
              int64_t activation_type,
              float activation_alpha, float activation_beta,
              float swiglu_limit,
              int64_t normalize_routing_weights,
              int64_t elem_size) {
  if (!state || !input || !router_probs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: null argument\n");
    return -1;
  }

  if (swiglu_fusion != 1) {
    fprintf(stderr, "wrap_qmoe: only swiglu_fusion=1 supported, got %lld\n",
            (long long)swiglu_fusion);
    return -1;
  }

  if (fc3_weights || fc3_scales || fc3_bias || fc3_zero_points) {
    fprintf(stderr,
            "wrap_qmoe: fc3 (unfused SwiGLU) not supported, "
            "use swiglu_fusion=1\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe(tokens=%lld, hidden=%lld, inter=%lld, "
                    "experts=%lld, k=%lld, bits=%lld, elem=%lld)\n",
                    (long long)num_tokens, (long long)hidden_size,
                    (long long)inter_size, (long long)num_experts,
                    (long long)k, (long long)expert_weight_bits,
                    (long long)elem_size);

  void* stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_qmoe: null stream\n");
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int rc = 0;

  int64_t fusion_inter = 2 * inter_size; // swiglu_fusion=1
  int64_t k_blocks_fc1 = (hidden_size + block_size - 1) / block_size;
  int64_t blob_size_fc1 = block_size / 2;
  int64_t k_blocks_fc2 = (inter_size + block_size - 1) / block_size;
  int64_t blob_size_fc2 = block_size / 2;

  // GPU temp buffers (allocated once, sized for worst-case: all tokens)
  void* d_expert_indices = nullptr;  // [num_tokens, k] int32
  void* d_expert_weights = nullptr;  // [num_tokens, k] elem_size
  void* d_gather_buf = nullptr;      // [num_tokens, hidden_size]
  void* d_fc1_buf = nullptr;         // [num_tokens, fusion_inter]
  void* d_act_buf = nullptr;         // [num_tokens, inter_size]
  void* d_fc2_buf = nullptr;         // [num_tokens, hidden_size]
  void* d_token_ids = nullptr;       // [num_tokens] int32
  void* d_token_wts = nullptr;       // [num_tokens] elem_size

  HIP_CHECK(hipMalloc(&d_expert_indices, num_tokens * k * sizeof(int32_t)));
  HIP_CHECK(hipMalloc(&d_expert_weights, num_tokens * k * elem_size));
  HIP_CHECK(hipMalloc(&d_gather_buf, num_tokens * hidden_size * elem_size));
  HIP_CHECK(hipMalloc(&d_fc1_buf, num_tokens * fusion_inter * elem_size));
  HIP_CHECK(hipMalloc(&d_act_buf, num_tokens * inter_size * elem_size));
  HIP_CHECK(hipMalloc(&d_fc2_buf, num_tokens * hidden_size * elem_size));
  HIP_CHECK(hipMalloc(&d_token_ids, num_tokens * sizeof(int32_t)));
  HIP_CHECK(hipMalloc(&d_token_wts, num_tokens * elem_size));

  // ---- Step 1: Top-k routing (kernel launch) ----
  RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: topk_routing(tokens=%lld, experts=%lld, "
                    "k=%lld, normalize=%lld)\n",
                    (long long)num_tokens, (long long)num_experts,
                    (long long)k, (long long)normalize_routing_weights);
  KERNEL_CHECK(hip_qmoe_topk_routing(stream, router_probs, d_expert_indices,
                                      d_expert_weights, num_tokens,
                                      num_experts, k,
                                      normalize_routing_weights, elem_size));

  // ---- Step 2: Copy routing results to host for control flow ----
  {
    std::vector<int32_t> h_indices(num_tokens * k);
    std::vector<char> h_weights(num_tokens * k * elem_size);

    HIP_CHECK(hipMemcpyAsync(h_indices.data(), d_expert_indices,
                             num_tokens * k * sizeof(int32_t),
                             hipMemcpyDeviceToHost, hip_stream));
    HIP_CHECK(hipMemcpyAsync(h_weights.data(), d_expert_weights,
                             num_tokens * k * elem_size,
                             hipMemcpyDeviceToHost, hip_stream));
    HIP_CHECK(hipStreamSynchronize(hip_stream));

    // ---- Step 3: Build per-expert token lists ----
    std::vector<std::vector<TokenEntry>> expert_tokens(num_experts);
    for (int64_t t = 0; t < num_tokens; t++) {
      for (int64_t s = 0; s < k; s++) {
        int32_t eid = h_indices[t * k + s];
        if (eid >= 0 && eid < num_experts) {
          expert_tokens[eid].push_back(
              {static_cast<int32_t>(t), static_cast<int32_t>(s)});
        }
      }
    }

    // ---- Step 4: Zero output ----
    HIP_CHECK(hipMemsetAsync(output, 0,
                             num_tokens * hidden_size * elem_size,
                             hip_stream));

    // ---- Step 5: Per-expert MLP ----
    int64_t active_experts = 0;
    for (int64_t e = 0; e < num_experts; e++)
      if (!expert_tokens[e].empty()) active_experts++;
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: %lld/%lld experts active\n",
                      (long long)active_experts, (long long)num_experts);

    for (int64_t e = 0; e < num_experts; e++) {
      int64_t count = static_cast<int64_t>(expert_tokens[e].size());
      if (count == 0) continue;

      // Build host-side token IDs and per-token weights for this expert
      std::vector<int32_t> h_ids(count);
      std::vector<char> h_wts_e(count * elem_size);
      for (int64_t i = 0; i < count; i++) {
        h_ids[i] = expert_tokens[e][i].token_id;
        int32_t slot = expert_tokens[e][i].slot;
        int64_t src_off = (h_ids[i] * k + slot) * elem_size;
        memcpy(h_wts_e.data() + i * elem_size,
               h_weights.data() + src_off, elem_size);
      }

      HIP_CHECK(hipMemcpyAsync(d_token_ids, h_ids.data(),
                               count * sizeof(int32_t),
                               hipMemcpyHostToDevice, hip_stream));
      HIP_CHECK(hipMemcpyAsync(d_token_wts, h_wts_e.data(),
                               count * elem_size,
                               hipMemcpyHostToDevice, hip_stream));

      // 5a. Gather input tokens for this expert
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: %lld tokens - gather\n",
                        (long long)e, (long long)count);
      KERNEL_CHECK(hip_qmoe_gather_tokens(stream, input, d_gather_buf,
                                           d_token_ids, hidden_size, count,
                                           elem_size));

      // 5b. FC1: gathered [count, hidden] → [count, fusion_inter]
      const char* fc1_w_e =
          static_cast<const char*>(fc1_weights) +
          e * fusion_inter * k_blocks_fc1 * blob_size_fc1;
      const char* fc1_s_e =
          static_cast<const char*>(fc1_scales) +
          e * fusion_inter * k_blocks_fc1 * elem_size;
      const void* fc1_zp_e =
          fc1_zero_points
              ? static_cast<const char*>(fc1_zero_points) +
                    e * fusion_inter * k_blocks_fc1
              : nullptr;
      const void* fc1_b_e =
          fc1_bias
              ? static_cast<const char*>(fc1_bias) + e * fusion_inter * elem_size
              : nullptr;

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: fc1 matmul_nbits "
                        "[%lld x %lld] -> [%lld x %lld]\n",
                        (long long)e, (long long)count, (long long)hidden_size,
                        (long long)count, (long long)fusion_inter);
      KERNEL_CHECK(hip_matmul_nbits(stream, d_gather_buf, fc1_w_e, fc1_s_e,
                                     fc1_zp_e, fc1_b_e, d_fc1_buf, count,
                                     fusion_inter, hidden_size, 1,
                                     expert_weight_bits, block_size,
                                     elem_size));

      // 5c. SwiGLU activation
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: swiglu(alpha=%.3f, "
                        "beta=%.3f, limit=%.1f)\n",
                        (long long)e, (double)activation_alpha,
                        (double)activation_beta, (double)swiglu_limit);
      KERNEL_CHECK(hip_qmoe_swiglu(stream, d_fc1_buf, d_act_buf, count,
                                    inter_size, activation_alpha,
                                    activation_beta, swiglu_limit, elem_size));

      // 5d. FC2: act [count, inter] → [count, hidden]
      const char* fc2_w_e =
          static_cast<const char*>(fc2_weights) +
          e * hidden_size * k_blocks_fc2 * blob_size_fc2;
      const char* fc2_s_e =
          static_cast<const char*>(fc2_scales) +
          e * hidden_size * k_blocks_fc2 * elem_size;
      const void* fc2_zp_e =
          fc2_zero_points
              ? static_cast<const char*>(fc2_zero_points) +
                    e * hidden_size * k_blocks_fc2
              : nullptr;
      const void* fc2_b_e =
          fc2_bias
              ? static_cast<const char*>(fc2_bias) + e * hidden_size * elem_size
              : nullptr;

      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: fc2 matmul_nbits "
                        "[%lld x %lld] -> [%lld x %lld]\n",
                        (long long)e, (long long)count, (long long)inter_size,
                        (long long)count, (long long)hidden_size);
      KERNEL_CHECK(hip_matmul_nbits(stream, d_act_buf, fc2_w_e, fc2_s_e,
                                     fc2_zp_e, fc2_b_e, d_fc2_buf, count,
                                     hidden_size, inter_size, 1,
                                     expert_weight_bits, block_size,
                                     elem_size));

      // 5e. Scatter-add weighted expert output to final output
      RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: expert %lld: scatter_add\n",
                        (long long)e);
      KERNEL_CHECK(hip_qmoe_scatter_add(stream, output, d_fc2_buf,
                                         d_token_ids, d_token_wts,
                                         hidden_size, count, elem_size));
    }
  }

cleanup:
  if (d_expert_indices) hipFree(d_expert_indices);
  if (d_expert_weights) hipFree(d_expert_weights);
  if (d_gather_buf) hipFree(d_gather_buf);
  if (d_fc1_buf) hipFree(d_fc1_buf);
  if (d_act_buf) hipFree(d_act_buf);
  if (d_fc2_buf) hipFree(d_fc2_buf);
  if (d_token_ids) hipFree(d_token_ids);
  if (d_token_wts) hipFree(d_token_wts);

  if (rc == 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_qmoe: completed successfully\n");
  }
  return rc;
}
