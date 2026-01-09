/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "hipdnn.pb.h"
#include "hipdnn_pattern_json.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <memory>
#include <unordered_map>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace {
using namespace vaip_core;
using namespace vaip_cxx;

// Helper function to compute strides from shape (NCHW layout)
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Convert ONNX data type to hipDNN data type
std::optional<hipdnn_frontend::DataType> ToHipDNNDataType(int32_t onnx_dtype) {
  using hipdnn_frontend::DataType;
  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10
  switch (onnx_dtype) {
    case 1:  // FLOAT
      return DataType::FLOAT;
    case 10: // FLOAT16
      return DataType::HALF;
    default:
      return std::nullopt;
  }
}

// Determine compute data type based on input data types
std::optional<hipdnn_frontend::DataType> GetComputeDataType(
    hipdnn_frontend::DataType x_dtype,
    hipdnn_frontend::DataType w_dtype) {
  using hipdnn_frontend::DataType;

  // Both must be float types (FLOAT or HALF)
  bool x_is_float = (x_dtype == DataType::FLOAT || x_dtype == DataType::HALF);
  bool w_is_float = (w_dtype == DataType::FLOAT || w_dtype == DataType::HALF);

  if (x_is_float && w_is_float) {
    // Use float32 for compute when inputs are float types
    return DataType::FLOAT;
  }

  return std::nullopt;
}

using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

// Build and validate hipDNN graph for Conv operation
bool BuildAndValidateGraph(
    const Node& conv_node,
    const NodeArgConstRef& input_ref,
    const NodeArgConstRef& weight_ref,
    const NodeArgConstRef& output_ref) {
  using hipdnn_frontend::ConvolutionMode;
  using HipDNNGraph = hipdnn_frontend::graph::Graph;
  using hipdnn_frontend::graph::TensorAttributes;
  using hipdnn_frontend::graph::ConvFpropAttributes;

  try {
    MY_LOG(1) << "Building hipDNN graph for Conv operation";

    // Create hipDNN graph
    auto graph = std::make_unique<HipDNNGraph>();
    int64_t next_uid = 1;
    std::unordered_map<std::string, TensorAttrPtr> symbol_table;

    // Get input shapes and types
    auto input_shape = node_arg_get_shape_i64(input_ref);
    auto weight_shape = node_arg_get_shape_i64(weight_ref);
    auto output_shape = node_arg_get_shape_i64(output_ref);

    if (!input_shape || !weight_shape || !output_shape) {
      MY_LOG(1) << "Cannot build graph: missing static shapes";
      return false;
    }

    // Get data types
    auto input_dtype = ToHipDNNDataType(node_arg_get_element_type(input_ref));
    auto weight_dtype = ToHipDNNDataType(node_arg_get_element_type(weight_ref));
    auto output_dtype = ToHipDNNDataType(node_arg_get_element_type(output_ref));

    if (!input_dtype.has_value() || !weight_dtype.has_value() || !output_dtype.has_value()) {
      MY_LOG(1) << "Cannot build graph: unsupported data types";
      return false;
    }

    // Create input tensor attribute
    auto x_attr = std::make_shared<TensorAttributes>();
    x_attr->set_uid(next_uid++)
        .set_name(node_arg_get_name(input_ref))
        .set_data_type(input_dtype.value())
        .set_dim(*input_shape)
        .set_stride(ComputeStrides(*input_shape))
        .set_is_virtual(false);
    symbol_table[node_arg_get_name(input_ref)] = x_attr;

    // Create weight tensor attribute
    auto w_attr = std::make_shared<TensorAttributes>();
    w_attr->set_uid(next_uid++)
        .set_name(node_arg_get_name(weight_ref))
        .set_data_type(weight_dtype.value())
        .set_dim(*weight_shape)
        .set_stride(ComputeStrides(*weight_shape))
        .set_is_virtual(false);
    symbol_table[node_arg_get_name(weight_ref)] = w_attr;

    // Extract Conv attributes
    auto pads = node_get_attr_ints(conv_node, "pads");
    auto strides = node_get_attr_ints(conv_node, "strides");
    auto dilations = node_get_attr_ints(conv_node, "dilations");

    // Convert gsl::span to std::vector and normalize padding format
    std::vector<int64_t> pads_vec(pads.begin(), pads.end());
    std::vector<int64_t> strides_vec(strides.begin(), strides.end());
    std::vector<int64_t> dilations_vec(dilations.begin(), dilations.end());

    if (pads_vec.empty()) {
      pads_vec = {0, 0, 0, 0};
    } else if (pads_vec.size() == 2) {
      pads_vec = {pads_vec[0], pads_vec[1], pads_vec[0], pads_vec[1]};
    } else if (pads_vec.size() != 4) {
      MY_LOG(1) << "Invalid pads size: " << pads_vec.size();
      return false;
    }

    if (strides_vec.empty()) {
      strides_vec = {1, 1};
    }
    if (dilations_vec.empty()) {
      dilations_vec = {1, 1};
    }

    // Determine compute data type
    auto compute_dtype = GetComputeDataType(input_dtype.value(), weight_dtype.value());
    if (!compute_dtype.has_value()) {
      MY_LOG(1) << "Cannot determine compute data type";
      return false;
    }

    // Create convolution attributes
    ConvFpropAttributes conv_attrs;
    conv_attrs.set_padding({pads_vec[0], pads_vec[1]})
        .set_stride({strides_vec[0], strides_vec[1]})
        .set_dilation({dilations_vec[0], dilations_vec[1]})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
        .set_compute_data_type(compute_dtype.value());

    // Add convolution operation to graph and get output tensor
    auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);
    
    // Set output properties
    y_attr->set_uid(next_uid++)
        .set_name(node_arg_get_name(output_ref))
        .set_data_type(output_dtype.value())
        .set_dim(*output_shape)
        .set_stride(ComputeStrides(*output_shape))
        .set_is_virtual(false);

    // Validate the graph by checking if output tensor was created successfully
    if (!y_attr) {
      MY_LOG(1) << "hipDNN graph validation failed: conv_fprop returned null";
      return false;
    }

    MY_LOG(1) << "hipDNN graph validation succeeded";
    return true;

  } catch (const std::exception& ex) {
    MY_LOG(1) << "Exception building hipDNN graph: " << ex.what();
    return false;
  }
}

struct Level1HipDnn {
  Level1HipDnn(IPass& self) : self_{self} {}
  std::unique_ptr<Rule> create_rule(IPass* self) {
    std::shared_ptr<Pattern> pattern_ =
        vaip_core::PatternBuilder().create_by_json(
            std::string((const char*)hipdnn_json));
    CHECK(pattern_ != nullptr) << "Pattern hipdnn not found";
    return Rule::create_rule(
        pattern_, [=](onnxruntime::Graph* ort_graph, binder_t& binder) -> bool {
          auto input = vaip_cxx::NodeArgConstRef::from_node_arg(
              *ort_graph, *binder["input"].node_arg);
          auto output = vaip_cxx::NodeArgConstRef::from_node_arg(
              *ort_graph, *binder["output"].node_arg);
          auto conv_node = binder["hipdnn_op"].node;
          
          // Get Conv node inputs (input data and weight)
          auto conv_inputs = node_get_inputs(*conv_node);
          if (conv_inputs.size() < 2) {
            MY_LOG(1) << "Conv node must have at least 2 inputs (data and weight)";
            return false;
          }
          
          // Validate Conv operation for hipDNN compatibility
          auto input_data = vaip_cxx::NodeArgConstRef::from_node_arg(*ort_graph, *conv_inputs[0].node_arg);
          auto weight_data = vaip_cxx::NodeArgConstRef::from_node_arg(*ort_graph, *conv_inputs[1].node_arg);
          
          bool graph_valid = BuildAndValidateGraph(
              *conv_node, 
              input_data,
              weight_data,
              output);
          
          if (!graph_valid) {
            MY_LOG(1) << "hipDNN graph validation failed, skipping fusion";
            return false;
          }
          
          MY_LOG(1) << "hipDNN graph validation succeeded, proceeding with fusion";
          auto unique_id = output.name();
          auto [meta_def, fuse_error] =
              self_.try_fuse(*ort_graph, unique_id, {input.name()}, {output.name()},
                             {}, "HIPDNN");
          if (meta_def == nullptr) {
            MY_LOG(1) << "fuse error: " << fuse_error.comments;
            return false;
          } else {
            MY_LOG(1) << "merge hipdnn operation";
            auto hipdnn_param = hipdnn::HipdnnParamProto();
            
            // Set device and kernel type
            hipdnn_param.set_device_id("0");
            hipdnn_param.set_kernel_type("conv");
            
            // Extract Conv attributes from the node
            hipdnn_param.set_op_type(node_op_type(*conv_node));
            auto pads_attr = node_get_attr_ints(*conv_node, "pads");
            if (!pads_attr.empty()) {
              for (auto pad : pads_attr) {
                hipdnn_param.add_pads(pad);
              }
            }
            auto strides_attr = node_get_attr_ints(*conv_node, "strides");
            if (!strides_attr.empty()) {
              for (auto stride : strides_attr) {
                hipdnn_param.add_strides(stride);
              }
            }
            auto dilations_attr = node_get_attr_ints(*conv_node, "dilations");
            if (!dilations_attr.empty()) {
              for (auto dilation : dilations_attr) {
                hipdnn_param.add_dilations(dilation);
              }
            }
            auto group_attr = node_get_attr_int(*conv_node, "group");
            hipdnn_param.set_group(group_attr);
            
            auto hipdnn_json_str = std::string();
            auto status = google::protobuf::util::MessageToJsonString(
                hipdnn_param, &hipdnn_json_str);
            self->attach_meta_def_param(*meta_def, hipdnn_json_str.c_str());
            self->fuse(*ort_graph, std::move(*meta_def));
          }
          return true; // return true if graph is modified.
        });
  }
  void process(IPass& self, Graph& ort_graph) { create_rule(&self)->apply(&ort_graph); }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Level1HipDnn, vaip_pass_level1_hipdnn)
