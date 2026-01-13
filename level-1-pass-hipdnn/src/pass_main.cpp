/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "hipdnn.pb.h"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <hipdnn_frontend.hpp>
#include <memory>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
DEF_ENV_PARAM(MORPHIZEN_MAX_FUSED_SUBGRAPH_NUM, "65535")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace {
using namespace vaip_core;
using namespace vaip_cxx;

//=============================================================================
// Helper Functions
//=============================================================================

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

// Save FlatBuffer to file
void SaveGraphToFile(const flatbuffers::DetachedBuffer& buffer, 
                     const std::string& filepath) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }
  
  file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  file.close();
  
  if (!file.good()) {
    throw std::runtime_error("Failed to write graph to file: " + filepath);
  }
}

using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

//=============================================================================
// Operation Support Checking (similar to ep.cc)
//=============================================================================

// Check if a Conv node is supported
static bool IsSupportedConv(const Node& node) {
  try {
    auto inputs = node_get_inputs(node);
    auto outputs = node_get_output_node_args(node);

    // Conv requires at least 2 inputs (X, W) and optionally bias
    if (inputs.size() < 2 || outputs.size() != 1) {
      return false;
    }

    // Check data types - we support float and float16
    auto x_dtype = node_arg_get_element_type(*inputs[0].node_arg);
    auto w_dtype = node_arg_get_element_type(*inputs[1].node_arg);
    auto y_dtype = node_arg_get_element_type(*outputs[0]);

    bool supported_type =
        (x_dtype == 1 || x_dtype == 10) &&  // FLOAT or FLOAT16
        x_dtype == w_dtype && x_dtype == y_dtype;

    if (!supported_type) {
      return false;
    }

    // Check if it's a 2D convolution (4D tensors: NCHW)
    auto x_shape = node_arg_get_shape_i64(*inputs[0].node_arg);
    auto w_shape = node_arg_get_shape_i64(*inputs[1].node_arg);

    if (!x_shape || !w_shape) {
      return false;  // Dynamic shapes not supported yet
    }

    if (x_shape->size() != 4 || w_shape->size() != 4) {
      return false;  // Only 2D conv supported
    }

    // Check auto_pad - only NOTSET supported (explicit padding)
    if (node_has_attr(node, "auto_pad")) {
      auto auto_pad = node_get_attr_string(node, "auto_pad");
      if (auto_pad != "NOTSET") {
        return false;
      }
    }

    // Check group - only 1 supported (no grouped/depthwise convolutions)
    auto group = node_get_attr_int_with_default(node, "group", 1);
    if (group != 1) {
      return false;
    }

    // Check dilations - only [1,1] supported (no dilated convolutions)
    if (node_has_attr(node, "dilations")) {
      auto dilations = node_get_attr_ints(node, "dilations");
      if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
        return false;
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if an op is supported
static bool IsSupportedOp(const Node& node) {
  std::string op_type = node_op_type(node);

  if (op_type == "Conv") {
    return IsSupportedConv(node);
  }

  // Add more operations here as we implement them
  return false;
}

//=============================================================================
// Graph Building and Serialization
//=============================================================================

// Build hipDNN graph and serialize it to file
bool BuildAndSerializeGraph(
    const Node& conv_node,
    const NodeArgConstRef& input_ref,
    const NodeArgConstRef& weight_ref,
    const NodeArgConstRef& output_ref,
    std::string& out_filename) {
  using hipdnn_frontend::ConvolutionMode;
  using HipDNNGraph = hipdnn_frontend::graph::Graph;
  using hipdnn_frontend::graph::TensorAttributes;
  using hipdnn_frontend::graph::ConvFpropAttributes;

  try {
    MY_LOG(1) << "Building hipDNN graph for Conv operation";

    // Create hipDNN graph
    auto graph = std::make_unique<HipDNNGraph>();
    graph->set_name("Conv_" + node_arg_get_name(output_ref));
    
    int64_t next_uid = 1;

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
        .set_is_virtual(false);  // Graph input

    // Create weight tensor attribute
    auto w_attr = std::make_shared<TensorAttributes>();
    w_attr->set_uid(next_uid++)
        .set_name(node_arg_get_name(weight_ref))
        .set_data_type(weight_dtype.value())
        .set_dim(*weight_shape)
        .set_stride(ComputeStrides(*weight_shape))
        .set_is_virtual(false);  // Graph input

    // Extract Conv attributes (all are optional, use defaults if not present)
    
    // Get pads (default: [0, 0, 0, 0])
    std::vector<int64_t> pads_vec;
    if (node_has_attr(conv_node, "pads")) {
      auto pads = node_get_attr_ints(conv_node, "pads");
      pads_vec.assign(pads.begin(), pads.end());
    }
    
    if (pads_vec.empty()) {
      pads_vec = {0, 0, 0, 0};
    } else if (pads_vec.size() == 2) {
      // Expand [pad_h, pad_w] to [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
      pads_vec = {pads_vec[0], pads_vec[1], pads_vec[0], pads_vec[1]};
    } else if (pads_vec.size() != 4) {
      MY_LOG(1) << "Invalid pads size: " << pads_vec.size();
      return false;
    }

    // Get strides (default: [1, 1])
    std::vector<int64_t> strides_vec;
    if (node_has_attr(conv_node, "strides")) {
      auto strides = node_get_attr_ints(conv_node, "strides");
      strides_vec.assign(strides.begin(), strides.end());
    }
    
    if (strides_vec.empty()) {
      strides_vec = {1, 1};
    }

    // Get dilations (default: [1, 1])
    std::vector<int64_t> dilations_vec;
    if (node_has_attr(conv_node, "dilations")) {
      auto dilations = node_get_attr_ints(conv_node, "dilations");
      dilations_vec.assign(dilations.begin(), dilations.end());
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
        .set_is_virtual(false);  // Graph output

    // Serialize the graph
    flatbuffers::DetachedBuffer buffer = graph->buildFlatbufferOperationGraph();
    
    // Generate unique filename
    out_filename = "hipdnn_graph_" + node_arg_get_name(output_ref) + ".bin";
    
    // Save to file
    SaveGraphToFile(buffer, out_filename);
    
    MY_LOG(1) << "Serialized graph to: " << out_filename 
              << " (" << buffer.size() << " bytes)";
    
    return true;

  } catch (const std::exception& ex) {
    MY_LOG(1) << "Exception building/serializing hipDNN graph: " << ex.what();
    return false;
  }
}

//=============================================================================
// Pass Implementation
//=============================================================================

struct Level1HipDnn {
  Level1HipDnn(IPass& self) : self_{self} {}
  
  void process(IPass& self, Graph& ort_graph) {
    // Iterate through all nodes in reverse order looking for supported operations
    auto node_indices = graph_get_node_in_topoligical_order(ort_graph);

    auto count_fused_subgraph = 0;
    
    for (auto it = node_indices.rbegin(); it != node_indices.rend(); ++it) {

      if (count_fused_subgraph >= ENV_PARAM(MORPHIZEN_MAX_FUSED_SUBGRAPH_NUM)) {
        MY_LOG(1) << "Max fused subgraph num reached, skipping";
        break;
      }

      auto node_idx = *it;
      auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
      auto node_ref = NodeConstRef::from_node(ort_graph, *node);
      MY_LOG(1) << "node_idx: " << node_idx << " node_name:" << node_ref.name();
      
      // Check if operation is supported
      if (!IsSupportedOp(*node)) {
        continue;
      }
      
      MY_LOG(1) << "Found supported Conv node: " << node_ref;
      
      // Get Conv node inputs (input data and weight)
      auto conv_inputs = node_get_inputs(*node);
      if (conv_inputs.size() < 2) {
        MY_LOG(1) << "Conv node must have at least 2 inputs (data and weight)";
        continue;
      }
      
      // Get Conv node outputs
      auto conv_output_node_args = node_get_output_node_args(*node);
      if (conv_output_node_args.size() != 1) {
        MY_LOG(1) << "Conv node must have exactly 1 output";
        continue;
      }
      
      auto input_data = vaip_cxx::NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[0].node_arg);
      auto weight_data = vaip_cxx::NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[1].node_arg);
      auto output_data = vaip_cxx::NodeArgConstRef::from_node_arg(ort_graph, *conv_output_node_args[0]);
      
      // Build and serialize graph
      std::string graph_filename;
      bool success = BuildAndSerializeGraph(
          *node, 
          input_data,
          weight_data,
          output_data,
          graph_filename);
      
      if (!success) {
        MY_LOG(1) << "Failed to build/serialize graph, skipping fusion";
        continue;
      }
      
      MY_LOG(1) << "Graph serialization succeeded, proceeding with fusion";
      
      // Create fused node
      // Check if weight is a constant initializer in the original graph
      bool weight_is_constant = weight_data.is_constant();
      MY_LOG(1) << "Weight " << weight_data.name() << " is_constant=" << weight_is_constant;
      
      auto unique_id = output_data.name();
      std::vector<std::string> inputs_list;
      std::vector<std::string> constants_list;
      
      if (weight_is_constant) {
        // Weight is a constant, only pass data as runtime input
        inputs_list = {input_data.name()};
        constants_list = {weight_data.name()};
        MY_LOG(1) << "Fusing with weight as constant initializer";
      } else {
        // Weight is a runtime input (shouldn't happen for conv, but handle it)
        inputs_list = {input_data.name(), weight_data.name()};
        constants_list = {};
        MY_LOG(1) << "Fusing with weight as runtime input";
      }
      
      auto [meta_def, fuse_error] =
          self_.try_fuse(ort_graph, unique_id, 
                        inputs_list,
                        {output_data.name()},
                        constants_list, "HIPDNN");
      
      if (meta_def == nullptr) {
        MY_LOG(1) << "fuse error: " << fuse_error.comments;
        continue;
      }
      
      MY_LOG(1) << "Creating fused HIPDNN operation";
      MY_LOG(1) << "  meta_def inputs: " << meta_def->inputs_size();
      MY_LOG(1) << "  meta_def constants: " << meta_def->constant_initializers_size();
      
      // Create proto with graph filename
      auto hipdnn_param = hipdnn::HipdnnParamProto();
      hipdnn_param.set_graph_file_name(graph_filename);
      
      // Save constant initializer data to files
      if (weight_is_constant) {
        // Get weight data
        auto& weight_tensor = node_arg_get_const_data_as_tensor(ort_graph, weight_data);
        auto weight_raw = vaip_core::api()->tensor_proto_as_raw(ort_graph, weight_tensor);
        
        // Create weight data filename
        std::string weight_filename = graph_filename + ".weight0.bin";
        
        // Write weight data to file
        std::ofstream weight_file(weight_filename, std::ios::binary);
        if (!weight_file) {
          MY_LOG(1) << "Failed to create weight file: " << weight_filename;
          continue;
        }
        weight_file.write(weight_raw.data(), static_cast<std::streamsize>(weight_raw.size()));
        weight_file.close();
        
        MY_LOG(1) << "Saved weight data: " << weight_filename << " (" << weight_raw.size() << " bytes)";
        
        // Add to proto
        hipdnn_param.add_constant_names(weight_data.name());
        hipdnn_param.add_constant_data_files(weight_filename);
      }
      
      // Serialize proto to JSON
      auto hipdnn_json_str = std::string();
      auto status = google::protobuf::util::MessageToJsonString(
          hipdnn_param, &hipdnn_json_str);
      
      if (!status.ok()) {
        MY_LOG(1) << "Failed to serialize proto: " << status.ToString();
        continue;
      }
      
      // Attach parameter and fuse
      self_.attach_meta_def_param(*meta_def, hipdnn_json_str.c_str());
      self_.fuse(ort_graph, std::move(*meta_def));
      
      MY_LOG(1) << "Successfully fused Conv operation";

      count_fused_subgraph++;
    }
    MY_LOG(1) << "Total fused subgraph num: " << count_fused_subgraph;
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Level1HipDnn, vaip_pass_level1_hipdnn)
