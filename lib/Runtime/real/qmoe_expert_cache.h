/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_QMOE_EXPERT_CACHE_H
#define HIPDNN_EP_REAL_QMOE_EXPERT_CACHE_H

#include <cstdint>

struct RuntimeState;

// Replaces the expert operands of one hip.qmoe call with a slot pool.
// `prepare` runs after routing, with the num_tokens * k device expert indices
// on the state stream. It makes those experts resident, rewrites the indices
// to slot rows in [0, num_slots), and returns the fc1 weights, scales, bias,
// fc2 weights, scales, bias pool bases (null for an absent bias), each laid
// out like the original operand with num_slots expert rows.
struct QMoEExpertHook {
  int64_t num_slots;
  int (*prepare)(void *ctx, void *d_expert_indices, int64_t num_tokens,
                 int64_t k, const void *pools[6]);
  void *ctx;
};

// wrap_qmoe with an optional expert hook; a null hook is wrap_qmoe.
int hipdnn_ep_qmoe_run(RuntimeState *state, const void *input,
                       const void *router_probs, const void *router_weights,
                       const void *fc1_weights, const void *fc1_scales,
                       const void *fc1_bias, const void *fc2_weights,
                       const void *fc2_scales, const void *fc2_bias,
                       const void *fc3_weights, const void *fc3_scales,
                       const void *fc3_bias, const void *fc1_zero_points,
                       const void *fc2_zero_points, const void *fc3_zero_points,
                       void *output, int64_t num_tokens, int64_t hidden_size,
                       int64_t inter_size, int64_t num_experts, int64_t k,
                       int64_t expert_weight_bits, int64_t block_size,
                       int64_t swiglu_fusion, int64_t activation_type,
                       float activation_alpha, float activation_beta,
                       float swiglu_limit, int64_t normalize_routing_weights,
                       int64_t elem_size, const void *correction_bias,
                       int64_t routing_type, float routed_scaling_factor,
                       float routing_norm_eps, const QMoEExpertHook *hook);

#endif // HIPDNN_EP_REAL_QMOE_EXPERT_CACHE_H
