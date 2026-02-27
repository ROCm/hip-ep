/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "reshape_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Reshape Pattern Matching
 *
 * Matches Reshape patterns and replaces them with ROCm custom ops.
 * Reshape is typically zero-copy when data is contiguous.
 */
struct Level2RocmReshape {
  static constexpr const char *LOG_PREFIX = "[ROCm Reshape L2]";

  Level2RocmReshape(IPass &self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass *self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char *)reshape_json));

    return Rule::create_rule(
        pattern, [=](Graph *graph, binder_t &binder) -> bool {
          auto input_data = binder["input_data"];
          auto input_shape = binder["input_shape"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found Reshape pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("reshape");

          auto *reshape_params = rocm_param.mutable_reshape_params();

          // Get input and output shapes
          auto in_shape = node_arg_get_shape_i64(*input_data.node_arg);
          auto out_shape = node_arg_get_shape_i64(*output.node_arg);

          int64_t total_size = 1;
          if (in_shape) {
            for (auto dim : *in_shape) {
              reshape_params->add_shape_in(dim);
              total_size *= dim;
            }
          }

          if (out_shape) {
            for (auto dim : *out_shape) {
              reshape_params->add_shape_out(dim);
            }
          }

          reshape_params->set_total_size(total_size);

          // Store input/output names
          std::string input_name = node_arg_get_name(*input_data.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          reshape_params->add_input_names(input_name);
          reshape_params->add_output_names(output_name);

          // Create fused op - only the data input is runtime
          // The shape input is typically a constant initializer
          std::vector<std::string> input_names{input_name};
          std::vector<std::string> output_names{output_name};
          std::vector<std::string> constant_initializers;

          // Shape tensor is usually constant
          std::string shape_name = node_arg_get_name(*input_shape.node_arg);
          constant_initializers.push_back(shape_name);

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_reshape", input_names, output_names,
                             constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, output_name, LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass &self, Graph &graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Reshape patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass &self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmReshape, morphizen_pass_level2_rocm_reshape)
