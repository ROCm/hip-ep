/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 * 
 * MIOpen-based implementation (migrated from hipDNN)
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "hipdnn.pb.h"
#include <glog/logging.h>
#include <memory>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
DEF_ENV_PARAM(MORPHIZEN_MAX_FUSED_SUBGRAPH_NUM, "65535")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace {
using namespace morphizen;
using namespace morphizen_cxx;

//=============================================================================
// Operation Support Checking (MIOpen version)
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

    // Check data types
    auto x_type = node_arg_get_element_type(*inputs[0].node_arg);
    auto w_type = node_arg_get_element_type(*inputs[1].node_arg);
    
    // Only support FLOAT (1) and FLOAT16 (10)
    if ((x_type != 1 && x_type != 10) || (w_type != 1 && w_type != 10)) {
      return false;
    }
    
    // If bias present, check its type
    if (inputs.size() >= 3) {
      auto b_type = node_arg_get_element_type(*inputs[2].node_arg);
      if (b_type != x_type) {
        return false;  // Bias must match input type
      }
      
      // Check bias shape - should be 1D
      auto b_shape = node_arg_get_shape_i64(*inputs[2].node_arg);
      if (!b_shape || b_shape->size() != 1) {
        return false;
      }
    }

    // Check shapes - need static shapes
    auto x_shape = node_arg_get_shape_i64(*inputs[0].node_arg);
    auto w_shape = node_arg_get_shape_i64(*inputs[1].node_arg);
    auto y_shape = node_arg_get_shape_i64(*outputs[0]);
    
    if (!x_shape || !w_shape || !y_shape) {
      return false;  // Dynamic shapes not supported
    }
    
    // Check for 2D convolution (4D tensors: NCHW)
    if (x_shape->size() != 4 || w_shape->size() != 4 || y_shape->size() != 4) {
      return false;
    }
    
    return true;
    
  } catch (const std::exception& ex) {
    MY_LOG(1) << "Exception in IsSupportedConv: " << ex.what();
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
// Metadata Generation (MIOpen version - replaces graph serialization)
//=============================================================================

// Helper to add shape to proto
void AddShapeToProto(hipdnn::TensorShape* shape_proto, const std::vector<int64_t>& shape) {
  for (auto dim : shape) {
    shape_proto->add_dims(dim);
  }
}

// Generate Conv metadata directly into proto
bool GenerateConvMetadata(
    const Node& conv_node,
    const NodeArgConstRef& input_ref,
    const NodeArgConstRef& weight_ref,
    const NodeArgConstRef& output_ref,
    std::vector<std::string>& constant_names,
    hipdnn::HipdnnParamProto& out_proto) {
  try {
    MY_LOG(1) << "Generating Conv metadata (MIOpen)";
    
    // Get shapes
    auto x_shape = node_arg_get_shape_i64(input_ref);
    auto w_shape = node_arg_get_shape_i64(weight_ref);
    auto y_shape = node_arg_get_shape_i64(output_ref);
    
    if (!x_shape || !w_shape || !y_shape) {
      MY_LOG(1) << "Missing shape information";
      return false;
    }
    
    // Get data types
    auto x_dtype = node_arg_get_element_type(input_ref);
    auto w_dtype = node_arg_get_element_type(weight_ref);
    auto y_dtype = node_arg_get_element_type(output_ref);
    
    // Get Conv attributes
    std::vector<int64_t> pads_vec;
    std::vector<int64_t> strides_vec;
    std::vector<int64_t> dilations_vec;
    int64_t group = 1;
    
    if (node_has_attr(conv_node, "pads")) {
      auto pads = node_get_attr_ints(conv_node, "pads");
      pads_vec.assign(pads.begin(), pads.end());
    }
    if (node_has_attr(conv_node, "strides")) {
      auto strides = node_get_attr_ints(conv_node, "strides");
      strides_vec.assign(strides.begin(), strides.end());
    }
    if (node_has_attr(conv_node, "dilations")) {
      auto dilations = node_get_attr_ints(conv_node, "dilations");
      dilations_vec.assign(dilations.begin(), dilations.end());
    }
    if (node_has_attr(conv_node, "group")) {
      group = node_get_attr_int(conv_node, "group");
    }
    
    // Normalize defaults
    if (pads_vec.empty()) {
      pads_vec = {0, 0, 0, 0};
    } else if (pads_vec.size() == 2) {
      pads_vec = {pads_vec[0], pads_vec[1], pads_vec[0], pads_vec[1]};
    }
    
    if (strides_vec.empty()) {
      strides_vec = {1, 1};
    }
    
    if (dilations_vec.empty()) {
      dilations_vec = {1, 1};
    }
    
    // Check for bias (3rd input)
    auto inputs = node_get_inputs(conv_node);
    bool has_bias = inputs.size() >= 3;
    std::vector<int64_t> b_shape_vec;
    if (has_bias) {
      auto b_shape = node_arg_get_shape_i64(*inputs[2].node_arg);
      if (b_shape) {
        b_shape_vec = *b_shape;
      }
    }
    
    // Build proto directly
    out_proto.set_op_type("Conv");
    out_proto.set_version("miopen_1.0");
    
    // Input shapes
    AddShapeToProto(out_proto.add_input_shapes(), *x_shape);
    AddShapeToProto(out_proto.add_input_shapes(), *w_shape);
    if (has_bias) {
      AddShapeToProto(out_proto.add_input_shapes(), b_shape_vec);
    }
    
    // Output shapes
    AddShapeToProto(out_proto.add_output_shapes(), *y_shape);
    
    // Input data types
    out_proto.add_input_data_types(x_dtype);
    out_proto.add_input_data_types(w_dtype);
    if (has_bias) {
      out_proto.add_input_data_types(node_arg_get_element_type(*inputs[2].node_arg));
    }
    
    // Output data types
    out_proto.add_output_data_types(y_dtype);
    
    // Conv attributes (using oneof node_attrs)
    auto* conv_attrs = out_proto.mutable_conv_attrs();
    for (auto p : pads_vec) conv_attrs->add_pads(p);
    for (auto s : strides_vec) conv_attrs->add_strides(s);
    for (auto d : dilations_vec) conv_attrs->add_dilations(d);
    conv_attrs->set_group(group);
    conv_attrs->set_has_bias(has_bias);
    
    // Store constant names for custom op
    constant_names.push_back(node_arg_get_name(weight_ref));  // Weight
    if (has_bias) {
      constant_names.push_back(node_arg_get_name(*inputs[2].node_arg));  // Bias
    }
    
    MY_LOG(1) << "Generated Conv metadata proto";
    
    return true;
    
  } catch (const std::exception& ex) {
    MY_LOG(1) << "Exception generating Conv metadata: " << ex.what();
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
      auto node = MORPHIZEN_ORT_API(graph_get_node)(ort_graph, node_idx);
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
      
      auto input_data = NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[0].node_arg);
      auto weight_data = NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[1].node_arg);
      auto output_data = NodeArgConstRef::from_node_arg(ort_graph, *conv_output_node_args[0]);
      
      // Generate metadata directly into proto
      hipdnn::HipdnnParamProto hipdnn_param;
      std::vector<std::string> constant_names;
      bool success = GenerateConvMetadata(
          *node, 
          input_data,
          weight_data,
          output_data,
          constant_names,
          hipdnn_param);
      
      if (!success) {
        MY_LOG(1) << "Failed to generate metadata, skipping fusion";
        continue;
      }
      
      MY_LOG(1) << "Metadata generation succeeded, proceeding with fusion";
      
      // Build inputs list - include ALL inputs (input, weight, bias)
      // This handles both constant and dynamic weights/bias correctly
      std::vector<std::string> inputs_list;
      inputs_list.push_back(input_data.name());   // Input (always dynamic)
      inputs_list.push_back(weight_data.name());  // Weight (may be constant or dynamic)
      
      // Add bias if present
      bool has_bias = conv_inputs.size() >= 3;
      if (has_bias) {
        auto bias_data = NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[2].node_arg);
        inputs_list.push_back(bias_data.name());  // Bias (may be constant or dynamic)
      }
      
      // Identify which inputs are actually constants (for ORT optimization)
      // Note: constant_names is now a subset of inputs_list
      std::vector<std::string> actual_constant_names;
      if (weight_data.is_constant()) {
        actual_constant_names.push_back(weight_data.name());
        MY_LOG(1) << "Weight is a constant initializer";
      } else {
        MY_LOG(1) << "Weight is a dynamic tensor (output from previous op)";
      }
      
      if (has_bias) {
        auto bias_data = NodeArgConstRef::from_node_arg(ort_graph, *conv_inputs[2].node_arg);
        if (bias_data.is_constant()) {
          actual_constant_names.push_back(bias_data.name());
          MY_LOG(1) << "Bias is a constant initializer";
        } else {
          MY_LOG(1) << "Bias is a dynamic tensor (output from previous op)";
        }
      }
      
      // Create fused node with ALL inputs
      auto unique_id = output_data.name();
      
      auto [meta_def, fuse_error] =
          self_.try_fuse(ort_graph, unique_id, 
                        inputs_list,              // ALL inputs (dynamic + constant)
                        {output_data.name()},
                        actual_constant_names,    // Subset that are constants
                        "HIPDNN");
      
      if (meta_def == nullptr) {
        MY_LOG(1) << "fuse error: " << fuse_error.comments;
        continue;
      }
      
      MY_LOG(1) << "Creating fused HIPDNN operation (MIOpen)";
      MY_LOG(1) << "  meta_def inputs: " << meta_def->inputs_size();
      MY_LOG(1) << "  meta_def constants: " << meta_def->constant_initializers_size();
      
      // Serialize proto to JSON (proto already populated by GenerateConvMetadata)
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
      
      MY_LOG(1) << "Successfully fused Conv operation (MIOpen)";

      count_fused_subgraph++;
    }
    MY_LOG(1) << "Total fused subgraph num: " << count_fused_subgraph;
  }

  IPass& self_;
};
} // namespace

DEFINE_MORPHIZEN_PASS(Level1HipDnn, morphizen_pass_level1_hipdnn)
