/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mul_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Mul (Element-wise Multiplication) Pattern Matching
 *
 * Matches Mul patterns and replaces them with ROCm custom ops.
 * Handles broadcasting when one input is smaller.
 */
struct Level2RocmMul {
  static constexpr const char* LOG_PREFIX = "[ROCm Mul L2]";

  Level2RocmMul(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char*)mul_json));

    return Rule::create_rule(
        pattern, [=](Graph* graph, binder_t& binder) -> bool {
          auto input_A = binder["input_A"];
          auto input_B = binder["input_B"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found Mul pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("mul");

          auto* mul_params = rocm_param.mutable_mul_params();

          // Get input shapes
          auto a_shape = node_arg_get_shape_i64(*input_A.node_arg);
          auto b_shape = node_arg_get_shape_i64(*input_B.node_arg);

          // Calculate total sizes
          int64_t a_size = 1, b_size = 1;
          if (a_shape) {
            for (auto dim : *a_shape) {
              a_size *= dim;
              mul_params->add_shape_a(dim);
            }
          }
          if (b_shape) {
            for (auto dim : *b_shape) {
              b_size *= dim;
              mul_params->add_shape_b(dim);
            }
          }

          mul_params->set_size_a(a_size);
          mul_params->set_size_b(b_size);

          // Determine if B is scalar or needs broadcasting
          mul_params->set_b_is_scalar(b_size == 1);

          // Check if B is a constant
          auto b_ref =
              NodeArgConstRef::from_node_arg(*graph, *input_B.node_arg);
          if (b_ref.is_constant()) {
            mul_params->set_b_is_constant(true);

            // If B is a scalar constant, extract the value
            if (b_size == 1) {
              auto b_data =
                  node_arg_get_const_data_as_floats(*graph, *input_B.node_arg);
              if (!b_data.empty()) {
                mul_params->set_scalar_value(b_data[0]);
              }
            }
          } else {
            mul_params->set_b_is_constant(false);
          }

          // Store input/output names
          std::string input_a_name = node_arg_get_name(*input_A.node_arg);
          std::string input_b_name = node_arg_get_name(*input_B.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          mul_params->add_input_names(input_a_name);
          mul_params->add_input_names(input_b_name);
          mul_params->add_output_names(output_name);

          // Output shape (for broadcasting, output matches larger tensor)
          auto out_shape = node_arg_get_shape_i64(*output.node_arg);
          if (out_shape) {
            for (auto dim : *out_shape) {
              mul_params->add_shape_y(dim);
            }
          }

          // Create fused op
          std::vector<std::string> input_names;
          input_names.push_back(input_a_name);
          if (!mul_params->b_is_constant()) {
            input_names.push_back(input_b_name);
          }

          std::vector<std::string> output_names;
          output_names.push_back(output_name);

          std::vector<std::string> constant_initializers;
          if (mul_params->b_is_constant()) {
            constant_initializers.push_back(input_b_name);
          }

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_mul", input_names, output_names,
                             constant_initializers, "ROCm_EP");

          if (meta_def) {
            return rocm_pass::finalize_level2_fuse(
                self, *graph, *meta_def, rocm_param, output_name, LOG_PREFIX);
          }

          ROCM_LOG(1) << LOG_PREFIX << " Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass& self, Graph& graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Mul patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmMul, morphizen_pass_level2_rocm_mul)
