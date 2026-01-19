// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "rocm_pass_utils.hpp"
#include "gemm_pattern_json.hpp"

using namespace vaip_core;

/**
 * Level-2 Pass: Gemm Pattern Matching (hipBLASLt)
 */
struct Level2RocmGemm {
  static constexpr const char* LOG_PREFIX = "[ROCm Gemm L2]";
  
  Level2RocmGemm(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char*)gemm_json));

    return Rule::create_rule(
        pattern,
        [=](Graph* graph, binder_t& binder) -> bool {
          auto input_A = binder["input_A"];
          auto input_B = binder["input_B"];
          auto output = binder["output"];
          bool has_C = binder["input_C"].node_arg != nullptr;

          ROCM_LOG(1) << LOG_PREFIX << " Found Gemm pattern";

          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("gemm");
          
          auto* gemm_params = rocm_param.mutable_gemm_params();
          gemm_params->set_trans_a(0);
          gemm_params->set_trans_b(0);
          gemm_params->set_alpha(1.0f);
          gemm_params->set_beta(has_C ? 1.0f : 0.0f);
          gemm_params->set_has_bias(has_C);
          gemm_params->set_algorithm_index(-1);

          auto a_shape = node_arg_get_shape_i64(*input_A.node_arg);
          auto b_shape = node_arg_get_shape_i64(*input_B.node_arg);
          
          if (a_shape && a_shape->size() >= 2 && b_shape && b_shape->size() >= 2) {
            gemm_params->set_m((*a_shape)[a_shape->size() - 2]);
            gemm_params->set_k((*a_shape)[a_shape->size() - 1]);
            gemm_params->set_n((*b_shape)[b_shape->size() - 1]);
          }

          std::vector<std::string> input_names;
          input_names.push_back(node_arg_get_name(*input_A.node_arg));
          input_names.push_back(node_arg_get_name(*input_B.node_arg));
          if (has_C) {
            input_names.push_back(node_arg_get_name(*binder["input_C"].node_arg));
          }

          std::vector<std::string> output_names;
          output_names.push_back(node_arg_get_name(*output.node_arg));

          // Get output name for param file and fused node naming
          std::string fused_output_name = node_arg_get_name(*output.node_arg);
          
          auto [meta_def, error] = self->try_fuse(
              *graph, "rocm_gemm", input_names, output_names, {}, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, fused_output_name, LOG_PREFIX);
          }
          
          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass& self, Graph& graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Gemm patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level2RocmGemm, vaip_pass_level2_rocm_gemm)
