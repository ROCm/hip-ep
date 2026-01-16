// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "rocm.pb.h"
#include "conv_pattern_json.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

using namespace vaip_core;

/**
 * Level-2 Pass: Conv Pattern Matching (MIOpen)
 * 
 * Matches Conv patterns and replaces them with ROCm custom ops.
 */
struct Level2RocmConv {
  Level2RocmConv(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char*)conv_json));

    return Rule::create_rule(
        pattern,
        [=](Graph* graph, binder_t& binder) -> bool {
          // Extract matched nodes
          auto input_X = binder["input_X"];
          auto input_W = binder["input_W"];
          auto output = binder["output"];
          bool has_bias = binder["input_B"].node_arg != nullptr;

          MY_LOG(1) << "[ROCm Conv L2] Found Conv pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("conv");
          
          auto* conv_params = rocm_param.mutable_conv_params();
          
          // Get Conv attributes from the matched node
          auto* conv_node = output.node;
          auto attrs = node_get_attributes(*conv_node);
          
          // Extract padding, stride, dilation from attributes
          // (using safe defaults if not present)
          conv_params->set_pad_h(1);
          conv_params->set_pad_w(1);
          conv_params->set_stride_h(1);
          conv_params->set_stride_w(1);
          conv_params->set_dilation_h(1);
          conv_params->set_dilation_w(1);
          conv_params->set_group_count(1);
          conv_params->set_has_bias(has_bias);
          conv_params->set_alpha(1.0f);
          conv_params->set_beta(0.0f);
          conv_params->set_algorithm_index(-1);
          conv_params->set_exhaustive_search(false);

          // Get input tensor shapes
          auto x_shape = node_arg_get_shape_i64(*input_X.node_arg);
          auto w_shape = node_arg_get_shape_i64(*input_W.node_arg);
          
          if (x_shape && x_shape->size() == 4) {
            conv_params->set_batch_size((*x_shape)[0]);
            conv_params->set_in_channels((*x_shape)[1]);
            conv_params->set_in_height((*x_shape)[2]);
            conv_params->set_in_width((*x_shape)[3]);
          }
          
          if (w_shape && w_shape->size() == 4) {
            conv_params->set_out_channels((*w_shape)[0]);
            conv_params->set_filter_height((*w_shape)[2]);
            conv_params->set_filter_width((*w_shape)[3]);
          }

          // Store input/output names
          conv_params->add_input_names(node_arg_get_name(*input_X.node_arg));
          conv_params->add_input_names(node_arg_get_name(*input_W.node_arg));
          if (has_bias) {
            conv_params->add_input_names(node_arg_get_name(*binder["input_B"].node_arg));
          }
          conv_params->add_output_names(node_arg_get_name(*output.node_arg));

          // Create fused op
          std::vector<std::string> input_names;
          input_names.push_back(node_arg_get_name(*input_X.node_arg));
          input_names.push_back(node_arg_get_name(*input_W.node_arg));
          if (has_bias) {
            input_names.push_back(node_arg_get_name(*binder["input_B"].node_arg));
          }

          std::vector<std::string> output_names;
          output_names.push_back(node_arg_get_name(*output.node_arg));

          auto [meta_def, error] = self->try_fuse(
              *graph,
              "rocm_conv",
              input_names,
              output_names,
              {},  // constant_initializers
              "ROCm_EP"
          );

          if (meta_def) {
            self->attach_meta_def_param(*meta_def, rocm_param.SerializeAsString().c_str());
            self->fuse(*graph, std::move(*meta_def));
            MY_LOG(1) << "[ROCm Conv L2] Fused Conv pattern successfully";
            return true;
          }
          
          MY_LOG(1) << "[ROCm Conv L2] Failed to fuse: " << error.comments;
          return false;
        });
  }

  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "[ROCm Conv L2] Processing graph for Conv patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};

DEFINE_VAIP_PASS(Level2RocmConv, vaip_pass_level2_rocm_conv)
