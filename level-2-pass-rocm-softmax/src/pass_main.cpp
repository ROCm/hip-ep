/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "rocm_pass_utils.hpp"
#include "softmax_pattern_json.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Softmax Pattern Matching
 *
 * Matches Softmax patterns and replaces them with ROCm custom ops.
 * Softmax is applied along a specified axis (default: -1, last dim).
 */
struct Level2RocmSoftmax {
  static constexpr const char *LOG_PREFIX = "[ROCm Softmax L2]";

  Level2RocmSoftmax(IPass &self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass *self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char *)softmax_json));

    return Rule::create_rule(
        pattern, [=](Graph *graph, binder_t &binder) -> bool {
          auto input = binder["input"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found Softmax pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("softmax");

          auto *softmax_params = rocm_param.mutable_softmax_params();

          // Get input shape
          auto in_shape = node_arg_get_shape_i64(*input.node_arg);

          int64_t batch = 1;
          int64_t dim = 1;
          int32_t axis = -1;

          // Get axis attribute from node
          auto *softmax_node = output.node;
          if (node_has_attr(*softmax_node, "axis")) {
            axis =
                static_cast<int32_t>(node_get_attr_int(*softmax_node, "axis"));
          }

          if (in_shape && !in_shape->empty()) {
            int32_t ndim = static_cast<int32_t>(in_shape->size());

            // Normalize negative axis
            if (axis < 0) {
              axis = ndim + axis;
            }

            // Calculate batch (product of dims before axis) and dim (axis size)
            for (int i = 0; i < ndim; ++i) {
              softmax_params->add_shape((*in_shape)[i]);
              if (i < axis) {
                batch *= (*in_shape)[i];
              } else if (i == axis) {
                dim = (*in_shape)[i];
              } else {
                // dims after axis become part of "batch" for the kernel
                batch *= (*in_shape)[i];
              }
            }

            // Actually, for softmax we need:
            // batch = product of all dims except the axis dim
            // dim = size of axis dim
            // The batch elements are processed independently
            batch = 1;
            for (int i = 0; i < ndim; ++i) {
              if (i != axis) {
                batch *= (*in_shape)[i];
              }
            }
            dim = (*in_shape)[axis];
          }

          softmax_params->set_axis(axis);
          softmax_params->set_batch(batch);
          softmax_params->set_dim(dim);

          // Store input/output names
          std::string input_name = node_arg_get_name(*input.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          softmax_params->add_input_names(input_name);
          softmax_params->add_output_names(output_name);

          // Create fused op
          std::vector<std::string> input_names{input_name};
          std::vector<std::string> output_names{output_name};
          std::vector<std::string> constant_initializers;

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_softmax", input_names, output_names,
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
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Softmax patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass &self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmSoftmax, morphizen_pass_level2_rocm_softmax)
