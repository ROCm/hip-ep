/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gemm_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Gemm Pattern Matching (hipBLASLt)
 */
struct Level2RocmGemm {
  static constexpr const char *LOG_PREFIX = "[ROCm Gemm L2]";

  Level2RocmGemm(IPass &self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass *self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char *)gemm_json));

    return Rule::create_rule(
        pattern, [=](Graph *graph, binder_t &binder) -> bool {
          auto input_A = binder["input_A"];
          auto input_B = binder["input_B"];
          auto output = binder["output"];
          bool has_C = binder["input_C"].node_arg != nullptr;

          ROCM_LOG(1) << LOG_PREFIX << " Found Gemm pattern";

          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("gemm");

          auto *gemm_params = rocm_param.mutable_gemm_params();
          gemm_params->set_trans_a(0);
          gemm_params->set_trans_b(0);
          gemm_params->set_alpha(1.0f);
          gemm_params->set_beta(has_C ? 1.0f : 0.0f);
          gemm_params->set_has_bias(has_C);
          gemm_params->set_algorithm_index(-1);

          auto a_shape = node_arg_get_shape_i64(*input_A.node_arg);
          auto b_shape = node_arg_get_shape_i64(*input_B.node_arg);

          if (a_shape && a_shape->size() >= 2 && b_shape &&
              b_shape->size() >= 2) {
            gemm_params->set_m((*a_shape)[a_shape->size() - 2]);
            gemm_params->set_k((*a_shape)[a_shape->size() - 1]);
            gemm_params->set_n((*b_shape)[b_shape->size() - 1]);
          }

          // Extract and save weight tensor (B matrix) to cache
          auto pass_context = self->get_context();
          auto weight_ref =
              NodeArgConstRef::from_node_arg(*graph, *input_B.node_arg);
          auto weight_name = node_arg_get_name(*input_B.node_arg);

          // Check if weight is a constant initializer using new API
          if (weight_ref.is_constant()) {
            auto weight_data =
                node_arg_get_const_data_as_floats(*graph, *input_B.node_arg);
            std::string weight_filename =
                rocm_pass::generate_weight_filename("rocm_gemm", weight_name);

            if (rocm_pass::save_weight_to_cache(pass_context, weight_data,
                                                weight_filename, LOG_PREFIX)) {
              gemm_params->set_weight_file_path(weight_filename);
              gemm_params->set_weight_file_size(
                  static_cast<int64_t>(weight_data.size() * sizeof(float)));
            }
          } else {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Weight is not a constant: " << weight_name;
          }

          // Extract and save bias tensor (C matrix) if present
          if (has_C) {
            auto *bias_node_arg = binder["input_C"].node_arg;
            auto bias_ref =
                NodeArgConstRef::from_node_arg(*graph, *bias_node_arg);
            auto bias_name = node_arg_get_name(*bias_node_arg);

            if (bias_ref.is_constant()) {
              auto bias_data =
                  node_arg_get_const_data_as_floats(*graph, *bias_node_arg);
              std::string bias_filename = rocm_pass::generate_weight_filename(
                  "rocm_gemm_bias", bias_name);

              if (rocm_pass::save_weight_to_cache(pass_context, bias_data,
                                                  bias_filename, LOG_PREFIX)) {
                gemm_params->set_bias_file_path(bias_filename);
                gemm_params->set_bias_file_size(
                    static_cast<int64_t>(bias_data.size() * sizeof(float)));
              }
            }
          }

          // Store input/output names
          gemm_params->add_input_names(node_arg_get_name(*input_A.node_arg));
          gemm_params->add_input_names(weight_name);
          if (has_C) {
            gemm_params->add_input_names(
                node_arg_get_name(*binder["input_C"].node_arg));
          }
          gemm_params->add_output_names(node_arg_get_name(*output.node_arg));

          // Create fused op - only pass activation (A matrix) as runtime input
          // Weight (B) and bias (C) are constant initializers (saved to cache)
          std::vector<std::string> input_names;
          input_names.push_back(node_arg_get_name(*input_A.node_arg));
          // Don't add weight/bias as runtime inputs - they're cached

          std::vector<std::string> output_names;
          output_names.push_back(node_arg_get_name(*output.node_arg));

          // Add weight and bias as constant initializers
          std::vector<std::string> constant_initializers;
          constant_initializers.push_back(weight_name);
          if (has_C) {
            constant_initializers.push_back(
                node_arg_get_name(*binder["input_C"].node_arg));
          }

          // Get output name for param file and fused node naming
          std::string fused_output_name = node_arg_get_name(*output.node_arg);

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_gemm", input_names, output_names,
                             constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, fused_output_name,
                LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass &self, Graph &graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Gemm patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass &self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmGemm, morphizen_pass_level2_rocm_gemm)
