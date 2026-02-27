/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "conv_pattern_json.hpp"
#include "rocm_pass_utils.hpp"

using namespace morphizen;
using namespace morphizen_cxx;

namespace {
// Helper to calculate output dimension: (input + 2*pad - dilation*(filter-1) -
// 1) / stride + 1
int64_t calc_output_dim(int64_t input_dim, int64_t filter_dim, int32_t pad,
                        int32_t stride, int32_t dilation) {
  int64_t effective_filter = dilation * (filter_dim - 1) + 1;
  return (input_dim + 2 * pad - effective_filter) / stride + 1;
}
/**
 * Level-2 Pass: Conv Pattern Matching (MIOpen)
 *
 * Matches Conv patterns and replaces them with ROCm custom ops.
 * Extracts weight tensors and saves them to pass context cache.
 */
struct Level2RocmConv {
  static constexpr const char* LOG_PREFIX = "[ROCm Conv L2]";

  Level2RocmConv(IPass& self) : self_{self} {}

  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern =
        PatternBuilder().create_by_json(std::string((const char*)conv_json));

    return Rule::create_rule(
        pattern, [=](Graph* graph, binder_t& binder) -> bool {
          // Extract matched nodes
          auto input_X = binder["input_X"];
          auto input_W = binder["input_W"];
          auto output = binder["output"];
          bool has_bias = binder["input_B"].node_arg != nullptr;

          ROCM_LOG(1) << LOG_PREFIX << " Found Conv pattern";

          // Build RocmParamProto
          rocm::RocmParamProto rocm_param;
          rocm_param.set_op_type("conv");

          auto* conv_params = rocm_param.mutable_conv_params();

          // Get Conv attributes from the matched node using new API
          auto* conv_node = output.node;

          // Extract padding, stride, dilation from attributes
          int32_t pad_h = 0, pad_w = 0;
          int32_t stride_h = 1, stride_w = 1;
          int32_t dilation_h = 1, dilation_w = 1;
          int32_t group_count = 1;

          // Use new morphizen API to get attributes
          if (node_has_attr(*conv_node, "pads")) {
            auto pads = node_get_attr_ints(*conv_node, "pads");
            if (pads.size() >= 2) {
              pad_h = static_cast<int32_t>(pads[0]);
              pad_w = static_cast<int32_t>(pads[1]);
            }
          }
          if (node_has_attr(*conv_node, "strides")) {
            auto strides = node_get_attr_ints(*conv_node, "strides");
            if (strides.size() >= 2) {
              stride_h = static_cast<int32_t>(strides[0]);
              stride_w = static_cast<int32_t>(strides[1]);
            }
          }
          if (node_has_attr(*conv_node, "dilations")) {
            auto dilations = node_get_attr_ints(*conv_node, "dilations");
            if (dilations.size() >= 2) {
              dilation_h = static_cast<int32_t>(dilations[0]);
              dilation_w = static_cast<int32_t>(dilations[1]);
            }
          }
          if (node_has_attr(*conv_node, "group")) {
            group_count =
                static_cast<int32_t>(node_get_attr_int(*conv_node, "group"));
          }

          conv_params->set_pad_h(pad_h);
          conv_params->set_pad_w(pad_w);
          conv_params->set_stride_h(stride_h);
          conv_params->set_stride_w(stride_w);
          conv_params->set_dilation_h(dilation_h);
          conv_params->set_dilation_w(dilation_w);
          conv_params->set_group_count(group_count);
          conv_params->set_has_bias(has_bias);
          conv_params->set_alpha(1.0f);
          conv_params->set_beta(0.0f);
          conv_params->set_algorithm_index(-1);
          conv_params->set_exhaustive_search(false);
          conv_params->set_spatial_dim(2);

          // Get input tensor shapes
          auto x_shape = node_arg_get_shape_i64(*input_X.node_arg);
          auto w_shape = node_arg_get_shape_i64(*input_W.node_arg);

          int64_t batch_size = 1, in_channels = 3, in_height = 8, in_width = 8;
          int64_t out_channels = 16, filter_height = 3, filter_width = 3;

          if (x_shape && x_shape->size() == 4) {
            batch_size = (*x_shape)[0];
            in_channels = (*x_shape)[1];
            in_height = (*x_shape)[2];
            in_width = (*x_shape)[3];
            conv_params->set_batch_size(batch_size);
            conv_params->set_in_channels(in_channels);
            conv_params->set_in_height(in_height);
            conv_params->set_in_width(in_width);
          }

          if (w_shape && w_shape->size() == 4) {
            out_channels = (*w_shape)[0];
            filter_height = (*w_shape)[2];
            filter_width = (*w_shape)[3];
            conv_params->set_out_channels(out_channels);
            conv_params->set_filter_height(filter_height);
            conv_params->set_filter_width(filter_width);
          }

          // Calculate output dimensions
          int64_t out_height = calc_output_dim(in_height, filter_height, pad_h,
                                               stride_h, dilation_h);
          int64_t out_width = calc_output_dim(in_width, filter_width, pad_w,
                                              stride_w, dilation_w);
          conv_params->set_out_height(out_height);
          conv_params->set_out_width(out_width);

          ROCM_LOG(2) << LOG_PREFIX << " Input shape: [" << batch_size << ", "
                      << in_channels << ", " << in_height << ", " << in_width
                      << "]";
          ROCM_LOG(2) << LOG_PREFIX << " Weight shape: [" << out_channels
                      << ", " << in_channels << ", " << filter_height << ", "
                      << filter_width << "]";
          ROCM_LOG(2) << LOG_PREFIX << " Output shape: [" << batch_size << ", "
                      << out_channels << ", " << out_height << ", " << out_width
                      << "]";

          // Extract and save weight tensor to cache
          auto pass_context = self->get_context();
          auto weight_ref =
              NodeArgConstRef::from_node_arg(*graph, *input_W.node_arg);
          auto weight_name = node_arg_get_name(*input_W.node_arg);

          // Check if weight is a constant initializer using new API
          if (weight_ref.is_constant()) {
            auto weight_data =
                node_arg_get_const_data_as_floats(*graph, *input_W.node_arg);
            std::string weight_filename =
                rocm_pass::generate_weight_filename("rocm_conv", weight_name);

            if (rocm_pass::save_weight_to_cache(pass_context, weight_data,
                                                weight_filename, LOG_PREFIX)) {
              conv_params->set_weight_file_path(weight_filename);
              conv_params->set_weight_file_size(
                  static_cast<int64_t>(weight_data.size() * sizeof(float)));
            }
          } else {
            ROCM_LOG(1) << LOG_PREFIX
                        << " Weight is not a constant: " << weight_name;
          }

          // Extract and save bias tensor if present
          if (has_bias) {
            auto* bias_node_arg = binder["input_B"].node_arg;
            auto bias_ref =
                NodeArgConstRef::from_node_arg(*graph, *bias_node_arg);
            auto bias_name = node_arg_get_name(*bias_node_arg);

            if (bias_ref.is_constant()) {
              auto bias_data =
                  node_arg_get_const_data_as_floats(*graph, *bias_node_arg);
              std::string bias_filename = rocm_pass::generate_weight_filename(
                  "rocm_conv_bias", bias_name);

              if (rocm_pass::save_weight_to_cache(pass_context, bias_data,
                                                  bias_filename, LOG_PREFIX)) {
                conv_params->set_bias_file_path(bias_filename);
                conv_params->set_bias_file_size(
                    static_cast<int64_t>(bias_data.size() * sizeof(float)));
              }
            }
          }

          // Store input/output names
          conv_params->add_input_names(node_arg_get_name(*input_X.node_arg));
          conv_params->add_input_names(weight_name);
          if (has_bias) {
            conv_params->add_input_names(
                node_arg_get_name(*binder["input_B"].node_arg));
          }
          conv_params->add_output_names(node_arg_get_name(*output.node_arg));

          // Create fused op - only pass activation as runtime input
          // Weight and bias are constant initializers (saved to cache)
          std::vector<std::string> input_names;
          input_names.push_back(node_arg_get_name(*input_X.node_arg));
          // Don't add weight/bias as runtime inputs - they're cached

          std::vector<std::string> output_names;
          output_names.push_back(node_arg_get_name(*output.node_arg));

          // Add weight and bias as constant initializers
          std::vector<std::string> constant_initializers;
          constant_initializers.push_back(weight_name);
          if (has_bias) {
            constant_initializers.push_back(
                node_arg_get_name(*binder["input_B"].node_arg));
          }

          // Get the output name for naming the param file and fused node
          std::string fused_output_name = node_arg_get_name(*output.node_arg);

          auto [meta_def, error] =
              self->try_fuse(*graph, "rocm_conv", input_names, output_names,
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

  void process(IPass& self, Graph& graph) {
    ROCM_LOG(1) << LOG_PREFIX << " Processing graph for Conv patterns...";
    create_rule(&self)->apply(&graph);
  }

  IPass& self_;
};
}  // namespace

DEFINE_MORPHIZEN_PASS(Level2RocmConv, morphizen_pass_level2_rocm_conv)
