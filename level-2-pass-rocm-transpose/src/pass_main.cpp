/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "rocm_pass_utils.hpp"
#include "transpose_pattern_json.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Transpose Pattern Matching
 *
 * Matches Transpose patterns and replaces them with ROCm custom ops.
 * Extracts permutation from the 'perm' attribute.
 */
struct Level2RocmTranspose {
  static constexpr const char* LOG_PREFIX = "[ROCm Transpose L2]";

  Level2RocmTranspose(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char*)transpose_json));

    return Rule::create_rule(
        pattern, [=](Graph* graph, binder_t& binder) -> bool {
          auto input = binder["input"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found Transpose pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("transpose");

          auto* transpose_params = rocm_param.mutable_transpose_params();

          // Get input shape
          auto in_shape = node_arg_get_shape_i64(*input.node_arg);
          auto out_shape = node_arg_get_shape_i64(*output.node_arg);

          int64_t total_size = 1;
          int32_t ndim = 0;

          if (in_shape) {
            ndim = static_cast<int32_t>(in_shape->size());
            for (auto dim : *in_shape) {
              transpose_params->add_shape_in(dim);
              total_size *= dim;
            }
          }

          if (out_shape) {
            for (auto dim : *out_shape) {
              transpose_params->add_shape_out(dim);
            }
          }

          transpose_params->set_total_size(total_size);
          transpose_params->set_ndim(ndim);

          // Get permutation from node attribute
          auto* transpose_node = output.node;
          if (node_has_attr(*transpose_node, "perm")) {
            auto perm = node_get_attr_ints(*transpose_node, "perm");
            for (auto p : perm) {
              transpose_params->add_perm(static_cast<int32_t>(p));
            }

            // Check for common pattern [0, 2, 1, 3]
            if (perm.size() == 4 && perm[0] == 0 && perm[1] == 2 &&
                perm[2] == 1 && perm[3] == 3) {
              transpose_params->set_is_0213(true);
            } else {
              transpose_params->set_is_0213(false);
            }
          } else {
            // Default permutation is reverse
            transpose_params->set_is_0213(false);
            for (int i = ndim - 1; i >= 0; --i) {
              transpose_params->add_perm(i);
            }
          }

          // Store input/output names
          std::string input_name = node_arg_get_name(*input.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          transpose_params->add_input_names(input_name);
          transpose_params->add_output_names(output_name);

          // Create fused op
          std::vector<std::string> input_names{input_name};
          std::vector<std::string> output_names{output_name};
          std::vector<std::string> constant_initializers;

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_transpose", input_names,
                             output_names, constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, output_name, LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass& self, Graph& graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Transpose patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmTranspose, morphizen_pass_level2_rocm_transpose)
