/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gqa_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: GroupQueryAttention (GQA) Pattern Matching
 *
 * Matches com.microsoft.GroupQueryAttention patterns and replaces them with
 * ROCm custom ops backed by HIP FMHA kernels with causal masking.
 *
 * GQA inputs (7 of 12 used in Llama-3.1-8B):
 *   0: query      [B, S_q, num_heads * head_size]     fp16, required
 *   1: key        [B, S_q, kv_num_heads * head_size]   fp16, optional
 *   2: value      [B, S_q, kv_num_heads * head_size]   fp16, optional
 *   3: past_key   [B, kv_num_heads, past_S, head_size] fp16, optional (BNSH)
 *   4: past_value [B, kv_num_heads, past_S, head_size] fp16, optional (BNSH)
 *   5: seqlens_k  [B]                                  int32, required
 *   6: total_sequence_length  scalar                    int32, required
 *   7-11: cos_cache, sin_cache, position_ids, attention_bias, head_sink
 * (unused)
 *
 * GQA outputs (3):
 *   0: output         [B, S_q, num_heads * head_size]       fp16
 *   1: present_key    [B, kv_num_heads, total_S, head_size] fp16 (BNSH)
 *   2: present_value  [B, kv_num_heads, total_S, head_size] fp16 (BNSH)
 */
struct Level2RocmGqa {
  static constexpr const char *LOG_PREFIX = "[ROCm GQA L2]";

  Level2RocmGqa(IPass &self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass *self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char *)gqa_json));

    return Rule::create_rule(
        pattern, [=](Graph *graph, binder_t &binder) -> bool {
          auto query = binder["query"];
          auto key = binder["key"];
          auto value = binder["value"];
          auto past_key = binder["past_key"];
          auto past_value = binder["past_value"];
          auto seqlens_k = binder["seqlens_k"];
          auto total_seq_len = binder["total_sequence_length"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found GroupQueryAttention pattern";

          // Get the GQA node to extract attributes
          auto *gqa_node = output.node;
          if (!gqa_node) {
            ROCM_LOG(1) << LOG_PREFIX << " output node is null";
            return false;
          }

          // Extract required attributes
          int64_t num_heads = 0;
          int64_t kv_num_heads = 0;
          if (node_has_attr(*gqa_node, "num_heads")) {
            num_heads = node_get_attr_int(*gqa_node, "num_heads");
          } else {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Missing required attribute: num_heads";
            return false;
          }
          if (node_has_attr(*gqa_node, "kv_num_heads")) {
            kv_num_heads = node_get_attr_int(*gqa_node, "kv_num_heads");
          } else {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Missing required attribute: kv_num_heads";
            return false;
          }

          // Extract optional scale attribute via NodeConstRef
          // scale = 0.0f means "auto-compute as 1/sqrt(head_size)" in custom op
          auto gqa_ref = NodeConstRef::from_node(*graph, *gqa_node);
          float scale = gqa_ref.get_attr_float("scale", 0.0f);

          // Get shapes for dimension computation
          auto query_shape = node_arg_get_shape_i64(*query.node_arg);
          auto key_shape = node_arg_get_shape_i64(*key.node_arg);
          auto output_shape = node_arg_get_shape_i64(*output.node_arg);

          if (!query_shape || query_shape->size() < 3) {
            ROCM_LOG(1) << LOG_PREFIX << " Cannot get query shape";
            return false;
          }

          // Compute head_size from query shape: hidden_size / num_heads
          int64_t hidden_size = (*query_shape)[2];
          int64_t head_size = hidden_size / num_heads;

          // Compute default scale if not provided
          if (scale == 0.0f) {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
          }

          ROCM_LOG(1) << LOG_PREFIX << " num_heads=" << num_heads
                      << " kv_num_heads=" << kv_num_heads
                      << " head_size=" << head_size << " scale=" << scale;

          // Check for unsupported features
          // TODO: Forward cos_cache/sin_cache inputs when do_rotary=1
          // TODO: Forward position_ids input
          // TODO: Forward attention_bias input
          // TODO: Forward head_sink input
          // TODO: Handle output_qk (4th output)

          if (node_has_attr(*gqa_node, "do_rotary") &&
              node_get_attr_int(*gqa_node, "do_rotary") != 0) {
            ROCM_LOG(1) << LOG_PREFIX
                        << " do_rotary=1 not yet supported, skipping";
            return false;
          }

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("gqa");

          auto *gqa_params = rocm_param.mutable_gqa_params();
          gqa_params->set_num_heads(static_cast<int32_t>(num_heads));
          gqa_params->set_kv_num_heads(static_cast<int32_t>(kv_num_heads));
          gqa_params->set_head_size(static_cast<int32_t>(head_size));
          gqa_params->set_scale(scale);
          gqa_params->set_num_inputs(7); // query, key, value, past_key,
                                         // past_value, seqlens_k, total_seq_len
          gqa_params->set_num_outputs(3); // output, present_key, present_value

          // Store query shape
          if (query_shape) {
            for (auto dim : *query_shape) {
              gqa_params->add_query_shape(dim);
            }
          }

          // Store key shape
          if (key_shape) {
            for (auto dim : *key_shape) {
              gqa_params->add_key_shape(dim);
            }
          }

          // Store past_key shape (for past sequence length)
          auto past_key_shape = node_arg_get_shape_i64(*past_key.node_arg);
          if (past_key_shape) {
            for (auto dim : *past_key_shape) {
              gqa_params->add_past_key_shape(dim);
            }
          }

          // Store output shape
          if (output_shape) {
            for (auto dim : *output_shape) {
              gqa_params->add_output_shape(dim);
            }
          }

          // Compute present_key/value shapes
          // The present_key shape depends on runtime total_seq_len, which is
          // dynamic. We use the output shape from the graph which has this
          // information. present_key: output 1 of the GQA node
          auto present_key_out_shape = node_get_output_shape(*gqa_node, 1);
          auto present_value_out_shape = node_get_output_shape(*gqa_node, 2);

          for (auto dim : present_key_out_shape) {
            gqa_params->add_present_key_shape(dim);
          }
          for (auto dim : present_value_out_shape) {
            gqa_params->add_present_value_shape(dim);
          }

          // Build input/output name lists
          std::string query_name = node_arg_get_name(*query.node_arg);
          std::string key_name = node_arg_get_name(*key.node_arg);
          std::string value_name = node_arg_get_name(*value.node_arg);
          std::string past_key_name = node_arg_get_name(*past_key.node_arg);
          std::string past_value_name = node_arg_get_name(*past_value.node_arg);
          std::string seqlens_k_name = node_arg_get_name(*seqlens_k.node_arg);
          std::string total_seq_len_name =
              node_arg_get_name(*total_seq_len.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          // GQA node has 3 outputs - get names for present_key and
          // present_value The output binder refers to output 0, but we need
          // outputs 1 and 2 as well
          auto gqa_output_node_args = node_get_output_node_args(*gqa_node);

          std::vector<std::string> input_names{
              query_name,      key_name,       value_name,        past_key_name,
              past_value_name, seqlens_k_name, total_seq_len_name};

          // Output names: output(0), present_key(1), present_value(2)
          std::vector<std::string> output_names;
          if (gqa_output_node_args.size() >= 3) {
            for (size_t i = 0; i < 3; ++i) {
              output_names.push_back(
                  node_arg_get_name(*gqa_output_node_args[i]));
            }
          } else {
            output_names.push_back(output_name);
            // Fallback: construct names
            output_names.push_back(output_name + "_present_key");
            output_names.push_back(output_name + "_present_value");
          }

          // No constant initializers for GQA
          std::vector<std::string> constant_initializers;

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_gqa", input_names, output_names,
                             constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(self, *graph, *meta_def,
                                                   rocm_param, output_names[0],
                                                   LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass &self, Graph &graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for GQA patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass &self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmGqa, morphizen_pass_level2_rocm_gqa)
