/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "rocm_pass_utils.hpp"
#include "tile_pattern_json.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

/**
 * Level-2 Pass: Tile Pattern Matching
 *
 * Matches Tile patterns and replaces them with ROCm custom ops.
 * Tile repeats tensor along specified dimensions.
 */
struct Level2RocmTile {
  static constexpr const char *LOG_PREFIX = "[ROCm Tile L2]";

  Level2RocmTile(IPass &self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass *self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char *)tile_json));

    return Rule::create_rule(
        pattern, [=](Graph *graph, binder_t &binder) -> bool {
          auto input = binder["input"];
          auto repeats_arg = binder["repeats"];
          auto output = binder["output"];

          ROCM_LOG(1) << LOG_PREFIX << " Found Tile pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("tile");

          auto *tile_params = rocm_param.mutable_tile_params();

          // Get input shape
          auto in_shape = node_arg_get_shape_i64(*input.node_arg);
          auto out_shape = node_arg_get_shape_i64(*output.node_arg);

          int64_t in_size = 1;
          int64_t out_size = 1;
          int32_t ndim = 0;

          if (in_shape) {
            ndim = static_cast<int32_t>(in_shape->size());
            for (auto dim : *in_shape) {
              tile_params->add_shape_in(dim);
              in_size *= dim;
            }
          }

          if (out_shape) {
            for (auto dim : *out_shape) {
              tile_params->add_shape_out(dim);
              out_size *= dim;
            }
          }

          tile_params->set_in_size(in_size);
          tile_params->set_out_size(out_size);
          tile_params->set_ndim(ndim);

          // Compute repeats from input and output shapes
          // repeats[i] = out_shape[i] / in_shape[i]
          if (in_shape && out_shape && in_shape->size() == out_shape->size()) {
            for (size_t i = 0; i < in_shape->size(); ++i) {
              int64_t repeat = (*out_shape)[i] / (*in_shape)[i];
              tile_params->add_repeats(repeat);
            }
          } else {
            ROCM_LOG(1) << LOG_PREFIX << " Cannot infer repeats from shapes";
            return false;
          }

          // Store input/output names
          std::string input_name = node_arg_get_name(*input.node_arg);
          std::string output_name = node_arg_get_name(*output.node_arg);

          tile_params->add_input_names(input_name);
          tile_params->add_output_names(output_name);

          // Create fused op
          std::vector<std::string> input_names{input_name};
          std::vector<std::string> output_names{output_name};

          // Repeats tensor is constant
          std::vector<std::string> constant_initializers;
          constant_initializers.push_back(
              node_arg_get_name(*repeats_arg.node_arg));

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_tile", input_names, output_names,
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
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Tile patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass &self_;
};

DEFINE_MORPHIZEN_PASS(Level2RocmTile, morphizen_pass_level2_rocm_tile)
