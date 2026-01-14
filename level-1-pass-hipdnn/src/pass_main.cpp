/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 * 
 * MIOpen-based implementation (migrated from hipDNN)
 */
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "hipdnn.pb.h"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <memory>
#include <nlohmann/json.hpp>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
DEF_ENV_PARAM(MORPHIZEN_MAX_FUSED_SUBGRAPH_NUM, "65535")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace {
using namespace vaip_core;
using namespace vaip_cxx;

//=============================================================================
// Helper Functions (MIOpen version)
//=============================================================================

// Save JSON metadata to file
void SaveMetadataToFile(const nlohmann::json& metadata, 
                        const std::string& filepath) {
  std::ofstream file(filepath);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }
  
  file << metadata.dump(2);  // Pretty print with 2-space indent
  file.close();
  
  if (!file.good()) {
    throw std::runtime_error("Failed to write metadata to file: " + filepath);
  }
}

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

// Generate Conv metadata and save to JSON file
bool GenerateConvMetadata(
    const Node& conv_node,
    const NodeArgConstRef& input_ref,
    const NodeArgConstRef& weight_ref,
    const NodeArgConstRef& output_ref,
    std::vector<std::string>& constant_names,
    std::string& out_filename) {
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
    
    // Build JSON metadata (format expected by custom_op.cpp)
    nlohmann::json metadata;
    metadata["op_type"] = "Conv";
    metadata["version"] = "miopen_1.0";
    
    // Shapes as arrays
    metadata["input_shapes"] = nlohmann::json::array();
    metadata["input_shapes"].push_back(*x_shape);
    metadata["input_shapes"].push_back(*w_shape);
    if (has_bias) {
      metadata["input_shapes"].push_back(b_shape_vec);
    }
    
    metadata["output_shapes"] = nlohmann::json::array();
    metadata["output_shapes"].push_back(*y_shape);
    
    // Data types as arrays
    metadata["input_data_types"] = nlohmann::json::array();
    metadata["input_data_types"].push_back(x_dtype);
    metadata["input_data_types"].push_back(w_dtype);
    if (has_bias) {
      metadata["input_data_types"].push_back(node_arg_get_element_type(*inputs[2].node_arg));
    }
    
    metadata["output_data_types"] = nlohmann::json::array();
    metadata["output_data_types"].push_back(y_dtype);
    
    // Conv attributes (directly at top level)
    metadata["pads"] = pads_vec;
    metadata["strides"] = strides_vec;
    metadata["dilations"] = dilations_vec;
    metadata["group"] = group;
    metadata["has_bias"] = has_bias;
    
    // Store constant names for custom op
    constant_names.push_back(node_arg_get_name(weight_ref));  // Weight
    if (has_bias) {
      constant_names.push_back(node_arg_get_name(*inputs[2].node_arg));  // Bias
    }
    
    // Generate filename
    out_filename = "hipdnn_meta_" + node_arg_get_name(output_ref) + ".json";
    
    // Save metadata
    SaveMetadataToFile(metadata, out_filename);
    
    MY_LOG(1) << "Saved metadata to: " << out_filename;
    
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
      
      // Generate metadata (MIOpen version)
      std::string metadata_filename;
      std::vector<std::string> constant_names;
      bool success = GenerateConvMetadata(
          *node, 
          input_data,
          weight_data,
          output_data,
          constant_names,
          metadata_filename);
      
      if (!success) {
        MY_LOG(1) << "Failed to generate metadata, skipping fusion";
        continue;
      }
      
      MY_LOG(1) << "Metadata generation succeeded, proceeding with fusion";
      
      // Prepare constant data files
      std::vector<std::string> constant_data_files;
      std::filesystem::path meta_dir = std::filesystem::path(metadata_filename).parent_path();
      
      // Save weight constant data (always exists)
      if (!constant_names.empty() && weight_data.is_constant()) {
        const auto& const_name = constant_names[0];
        auto& tensor = node_arg_get_const_data_as_tensor(ort_graph, weight_data);
        auto raw_data = vaip_core::api()->tensor_proto_as_raw(ort_graph, tensor);
        
        std::string data_filename = "hipdnn_const_" + const_name + ".bin";
        std::filesystem::path data_path = meta_dir / data_filename;
        
        std::ofstream file(data_path.string(), std::ios::binary);
        if (file) {
          file.write(raw_data.data(), static_cast<std::streamsize>(raw_data.size()));
          file.close();
          constant_data_files.push_back(data_path.string());
          MY_LOG(1) << "Saved weight constant: " << data_path.string() 
                    << " (" << raw_data.size() << " bytes)";
        }
      }
      
      // Save bias constant data (if exists)
      if (constant_names.size() > 1 && conv_inputs.size() > 2) {
        auto bias_node_arg = conv_inputs[2].node_arg;
        if (bias_node_arg) {
          auto bias_data = vaip_cxx::NodeArgConstRef::from_node_arg(ort_graph, *bias_node_arg);
          if (bias_data.is_constant()) {
            const auto& const_name = constant_names[1];
            auto& tensor = node_arg_get_const_data_as_tensor(ort_graph, bias_data);
            auto raw_data = vaip_core::api()->tensor_proto_as_raw(ort_graph, tensor);
            
            std::string data_filename = "hipdnn_const_" + const_name + ".bin";
            std::filesystem::path data_path = meta_dir / data_filename;
            
            std::ofstream file(data_path.string(), std::ios::binary);
            if (file) {
              file.write(raw_data.data(), static_cast<std::streamsize>(raw_data.size()));
              file.close();
              constant_data_files.push_back(data_path.string());
              MY_LOG(1) << "Saved bias constant: " << data_path.string() 
                        << " (" << raw_data.size() << " bytes)";
            }
          }
        }
      }
      
      // Create fused node
      auto unique_id = output_data.name();
      std::vector<std::string> inputs_list = {input_data.name()};  // Only runtime input
      
      auto [meta_def, fuse_error] =
          self_.try_fuse(ort_graph, unique_id, 
                        inputs_list,
                        {output_data.name()},
                        constant_names,  // All constants
                        "HIPDNN");
      
      if (meta_def == nullptr) {
        MY_LOG(1) << "fuse error: " << fuse_error.comments;
        continue;
      }
      
      MY_LOG(1) << "Creating fused HIPDNN operation (MIOpen)";
      MY_LOG(1) << "  meta_def inputs: " << meta_def->inputs_size();
      MY_LOG(1) << "  meta_def constants: " << meta_def->constant_initializers_size();
      
      // Create proto with metadata filename
      auto hipdnn_param = hipdnn::HipdnnParamProto();
      hipdnn_param.set_graph_file_name(metadata_filename);
      
      for (const auto& name : constant_names) {
        hipdnn_param.add_constant_names(name);
      }
      
      for (const auto& file : constant_data_files) {
        hipdnn_param.add_constant_data_files(file);
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
      
      MY_LOG(1) << "Successfully fused Conv operation (MIOpen)";

      count_fused_subgraph++;
    }
    MY_LOG(1) << "Total fused subgraph num: " << count_fused_subgraph;
  }

  IPass& self_;
};
} // namespace

DEFINE_VAIP_PASS(Level1HipDnn, vaip_pass_level1_hipdnn)
