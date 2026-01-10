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
#include <unordered_map>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
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

    if (!x_shape.has_value() || !w_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (x_shape->size() != 4 || w_shape->size() != 4) {
      return false;  // Only 2D conv supported
    }

    // Check auto_pad - only NOTSET supported (explicit padding)
    auto auto_pad_attr = node_try_get_attr_string(node, "auto_pad");
    if (auto_pad_attr.has_value() && auto_pad_attr.value() != "NOTSET") {
      return false;
    }

    // Check group - only 1 supported (no grouped/depthwise convolutions)
    auto group = node_get_attr_int_with_default(node, "group", 1);
    if (group != 1) {
      return false;
    }

    // Check dilations - only [1,1] supported (no dilated convolutions)
    auto dilations = node_get_attr_ints(node, "dilations");
    if (!dilations.empty()) {
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
// Tensor Attribute Creation
//=============================================================================

// Create TensorAttributes from NodeArg
bool CreateTensorAttr(
    const Graph& ort_graph,
    const NodeArg& node_arg,
    int64_t uid,
    TensorAttrPtr& out_attr) {
  using hipdnn_frontend::graph::TensorAttributes;

  auto node_arg_ref = NodeArgConstRef::from_node_arg(ort_graph, node_arg);
  std::string name = node_arg_get_name(node_arg_ref);

  auto shape = node_arg_get_shape_i64(node_arg_ref);
  if (!shape.has_value()) {
    MY_LOG(1) << "Value must have static shape: " << name;
    return false;
  }

  auto dtype = ToHipDNNDataType(node_arg_get_element_type(node_arg_ref));
  if (!dtype.has_value()) {
    MY_LOG(1) << "Unsupported data type for value: " << name;
    return false;
  }

  out_attr = std::make_shared<TensorAttributes>();
  out_attr->set_uid(uid)
      .set_name(name)
      .set_data_type(dtype.value())
      .set_dim(shape.value())
      .set_stride(ComputeStrides(shape.value()));

  return true;
}

//=============================================================================
// Operation Node Addition (similar to kernel.cc)
//=============================================================================

// Add Conv operation to hipDNN graph
bool AddConvNode(
    const Graph& ort_graph,
    hipdnn_frontend::graph::Graph& graph,
    const Node& node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;

  if (input_attrs.size() < 2) {
    MY_LOG(1) << "Conv requires at least 2 input tensor attributes";
    return false;
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

  // Extract Conv attributes
  auto pads = node_get_attr_ints(node, "pads");
  auto strides = node_get_attr_ints(node, "strides");
  auto dilations = node_get_attr_ints(node, "dilations");

  // Convert to vectors and normalize
  std::vector<int64_t> pads_vec(pads.begin(), pads.end());
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  std::vector<int64_t> dilations_vec(dilations.begin(), dilations.end());

  // Normalize padding format
  if (pads_vec.empty()) {
    pads_vec = {0, 0, 0, 0};
  } else if (pads_vec.size() == 2) {
    pads_vec = {pads_vec[0], pads_vec[1], pads_vec[0], pads_vec[1]};
  } else if (pads_vec.size() != 4) {
    MY_LOG(1) << "Conv pads must have 2 or 4 elements";
    return false;
  }

  if (strides_vec.empty()) {
    strides_vec = {1, 1};
  }
  if (dilations_vec.empty()) {
    dilations_vec = {1, 1};
  }

  // Determine compute data type from input data types
  auto compute_dtype = GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    MY_LOG(1) << "Unsupported data type combination for Conv compute";
    return false;
  }

  // Create convolution attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads_vec[0], pads_vec[1]})  // Use begin padding
      .set_stride({strides_vec[0], strides_vec[1]})
      .set_dilation({dilations_vec[0], dilations_vec[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());

  // Add convolution to graph - returns output tensor attributes
  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);

  return true;
}

// Dispatch to appropriate Add*Node based on op_type
bool AddNode(
    const Graph& ort_graph,
    hipdnn_frontend::graph::Graph& graph,
    const Node& node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs) {
  std::string op_type = node_op_type(node);

  if (op_type == "Conv") {
    TensorAttrPtr y_attr;
    if (!AddConvNode(ort_graph, graph, node, input_attrs, y_attr)) {
      return false;
    }
    output_attrs.push_back(y_attr);
    return true;
  }

  MY_LOG(1) << "Unsupported op type: " << op_type;
  return false;
}

//=============================================================================
// Generic Graph Building (similar to kernel.cc BuildAndCompile)
//=============================================================================

// Build hipDNN graph generically using symbol table pattern
bool BuildAndSerializeGraph(
    const Graph& ort_graph,
    const Node& fused_node,
    const std::vector<NodeInput>& graph_inputs,
    const std::vector<const NodeArg*>& graph_outputs,
    std::string& out_filename) {
  using hipdnn_frontend::graph::Graph;

  try {
    MY_LOG(1) << "Building hipDNN graph for fused node";

    // Create hipDNN graph
    auto graph = std::make_unique<Graph>();
    
    // Use output name for graph name
    if (graph_outputs.empty()) {
      MY_LOG(1) << "Graph must have at least one output";
      return false;
    }
    auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *graph_outputs[0]);
    graph->set_name("Graph_" + node_arg_get_name(output_ref));
    
    int64_t next_uid = 1;
    std::unordered_map<std::string, TensorAttrPtr> symbol_table;
    std::vector<int64_t> input_uids;
    std::vector<int64_t> output_uids;

    // Create TensorAttributes for all graph inputs and add to symbol table
    input_uids.reserve(graph_inputs.size());
    for (const auto& input : graph_inputs) {
      TensorAttrPtr attr;
      if (!CreateTensorAttr(ort_graph, *input.node_arg, next_uid++, attr)) {
        return false;
      }
      attr->set_is_virtual(false);
      auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *input.node_arg);
      symbol_table[node_arg_get_name(input_ref)] = attr;
      input_uids.push_back(attr->get_uid());
    }

    // Process each node in the fused subgraph
    auto subgraph_nodes = node_get_inputs(fused_node);
    for (const auto& input : subgraph_nodes) {
      if (!input.node) continue;
      
      const Node& node = *input.node;
      
      // Look up input TensorAttributes from symbol table
      auto node_inputs = node_get_inputs(node);
      std::vector<TensorAttrPtr> input_attrs;
      input_attrs.reserve(node_inputs.size());

      for (const auto& node_input : node_inputs) {
        auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *node_input.node_arg);
        std::string name = node_arg_get_name(input_ref);
        auto it = symbol_table.find(name);
        if (it == symbol_table.end()) {
          MY_LOG(1) << "Input not found in symbol table: " << name;
          return false;
        }
        input_attrs.push_back(it->second);
      }

      // Add the node to hipDNN graph
      std::vector<TensorAttrPtr> output_attrs;
      if (!AddNode(ort_graph, *graph, node, input_attrs, output_attrs)) {
        return false;
      }

      // Set UID, name on output TensorAttributes and add to symbol table
      auto node_outputs = node_get_output_node_args(node);
      if (output_attrs.size() != node_outputs.size()) {
        MY_LOG(1) << "Output count mismatch for node " << node_get_op_type(node) 
                  << ": expected " << node_outputs.size() 
                  << ", got " << output_attrs.size();
        return false;
      }

      for (size_t i = 0; i < output_attrs.size(); ++i) {
        auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *node_outputs[i]);
        std::string name = node_arg_get_name(output_ref);

        // Get output data type
        auto dtype = ToHipDNNDataType(node_arg_get_element_type(output_ref));
        if (!dtype.has_value()) {
          MY_LOG(1) << "Unsupported data type for output: " << name;
          return false;
        }

        // Get output shape for strides
        auto shape = node_arg_get_shape_i64(output_ref);
        if (!shape.has_value()) {
          MY_LOG(1) << "Output must have static shape: " << name;
          return false;
        }

        output_attrs[i]->set_uid(next_uid++)
            .set_name(name)
            .set_data_type(dtype.value())
            .set_dim(shape.value())
            .set_stride(ComputeStrides(shape.value()));
        symbol_table[name] = output_attrs[i];
      }
    }

    // Mark graph outputs as non-virtual and store their UIDs
    output_uids.reserve(graph_outputs.size());
    for (const auto* output_arg : graph_outputs) {
      auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *output_arg);
      std::string name = node_arg_get_name(output_ref);
      auto it = symbol_table.find(name);
      if (it == symbol_table.end()) {
        MY_LOG(1) << "Graph output not found in symbol table: " << name;
        return false;
      }
      it->second->set_is_virtual(false);
      output_uids.push_back(it->second->get_uid());
    }

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
    // Iterate through all nodes looking for supported operations
    auto node_indices = graph_get_node_in_topoligical_order(ort_graph);
    
    for (auto node_idx : node_indices) {
      auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
      auto node_ref = NodeConstRef::from_node(ort_graph, *node);
      
      // Check if operation is supported
      if (!IsSupportedOp(*node)) {
        continue;
      }
      
      std::string op_type = node_op_type(*node);
      MY_LOG(1) << "Found supported " << op_type << " node: " << node_ref;
      
      // Get node inputs and outputs
      auto node_inputs = node_get_inputs(*node);
      auto node_outputs = node_get_output_node_args(*node);
      
      if (node_inputs.empty() || node_outputs.empty()) {
        MY_LOG(1) << "Node must have inputs and outputs";
        continue;
      }
      
      // Prepare input/output names for fusion
      std::vector<std::string> input_names;
      std::vector<std::string> output_names;
      
      input_names.reserve(node_inputs.size());
      for (const auto& input : node_inputs) {
        auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *input.node_arg);
        input_names.push_back(node_arg_get_name(input_ref));
      }
      
      output_names.reserve(node_outputs.size());
      for (const auto* output : node_outputs) {
        auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *output);
        output_names.push_back(node_arg_get_name(output_ref));
      }
      
      // Build and serialize graph
      std::string graph_filename;
      bool success = BuildAndSerializeGraph(
          ort_graph,
          *node,
          node_inputs,
          node_outputs,
          graph_filename);
      
      if (!success) {
        MY_LOG(1) << "Failed to build/serialize graph, skipping fusion";
        continue;
      }
      
      MY_LOG(1) << "Graph serialization succeeded, proceeding with fusion";
      
      // Create fused node
      auto unique_id = output_names[0];
      auto [meta_def, fuse_error] =
          self_.try_fuse(ort_graph, unique_id, 
                        input_names, 
                        output_names,
                        {}, "HIPDNN");
      
      if (meta_def == nullptr) {
        MY_LOG(1) << "fuse error: " << fuse_error.comments;
        continue;
      }
      
      MY_LOG(1) << "Creating fused HIPDNN operation";
      
      // Create proto with only the graph filename
      auto hipdnn_param = hipdnn::HipdnnParamProto();
      hipdnn_param.set_graph_file_name(graph_filename);
      
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
      
      MY_LOG(1) << "Successfully fused " << op_type << " operation";
    }
  }

  IPass& self_;
};

} // namespace

DEFINE_VAIP_PASS(Level1HipDnn, vaip_pass_level1_hipdnn)
