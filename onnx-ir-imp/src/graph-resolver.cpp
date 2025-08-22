/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./graph-resolver.hpp"
#include "./graph.hpp"
#include "./node-arg-index.hpp"
#include "./staging-graph.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <morphizen-utils/morphizen-utils.hpp>
#include <string>
#include <string_view>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER, "0");
DEF_ENV_PARAM_2(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR, "", std::string);
#define MY_LOG(n)                                                              \
  if (ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER) >= (n))                        \
  LOG(INFO)

namespace morphizen {

// Helper function to dump an ONNX graph proto to a file
static void dump_onnx_model(const morphizen_onnx::GraphProto& graph_proto,
                            const std::filesystem::path& filename) {
  try {
    // Create a temporary model proto to wrap the graph proto
    morphizen_onnx::ModelProto model_proto;
    model_proto.set_ir_version(7); // ONNX IR version
    model_proto.set_producer_name("MorphiZen");
    model_proto.set_producer_version("1.0");

    // Add standard ONNX opset import
    auto* opset_import = model_proto.add_opset_import();
    opset_import->set_domain("");  // Default domain
    opset_import->set_version(17); // ONNX opset version

    // Copy the graph proto into the model
    auto* graph_copy = model_proto.mutable_graph();
    graph_copy->CopyFrom(graph_proto);

    // Ensure the directory exists
    std::filesystem::create_directories(filename.parent_path());

    // Write the model to file
    std::ofstream output_file(filename, std::ios::binary);
    if (output_file.is_open()) {
      std::string serialized_model;
      if (model_proto.SerializeToString(&serialized_model)) {
        output_file.write(serialized_model.data(), serialized_model.size());
        output_file.close();
        MY_LOG(1) << "Successfully saved ONNX model to " << filename.string();
      } else {
        MY_LOG(1) << "Failed to serialize model proto to string";
      }
    } else {
      MY_LOG(1) << "Failed to open output file: " << filename.string();
    }
  } catch (const std::exception& e) {
    MY_LOG(1) << "Exception while saving ONNX model to " << filename.string()
              << ": " << e.what();
  }
}

GraphResolver::GraphResolver() : new_graph_id_{GraphId::from_raw(0)} {
  // Empty constructor implementation
}

GraphResolver::~GraphResolver() {
  // Empty destructor implementation
}

void GraphResolver::initialize_private_variables(Graph& graph,
                                                 GraphId new_graph_id) {
  // Initialize basic graph pointers and ID
  origin_graph_ = &graph;
  staging_graph_ = graph.get_staging_graph();
  new_graph_id_ = new_graph_id;

  // Clear and initialize containers
  resolved_graph_proto_.Clear();
  name_on_staging_graph_.clear();

  // Initialize deletion tracking flags
  // Size these based on the staging graph if it exists, otherwise the original
  // graph
  CHECK(staging_graph_ != nullptr)
      << "Staging graph must be set before resolving.";
  const auto& origin_proto = origin_graph_->get_graph_proto();
  initializer_deleted_flags_.assign(origin_proto.initializer_size(), false);
  node_deleted_flags_.assign(origin_proto.node_size(), false);

  MY_LOG(1) << "GraphResolver::initialize_private_variables() - Initialized "
               "with new graph ID: "
            << new_graph_id_.to_string();
}

morphizen_onnx::GraphProto
GraphResolver::resolve(Graph& graph, GraphId new_graph_id,
                       std::unordered_map<std::string, int>& opset) {
  CHECK(graph.need_resolve())
      << "Graph does not need resolution, cannot resolve.";
  // Initialize all private variables
  initialize_private_variables(graph, new_graph_id);
  if (!ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR).empty()) {
    // dump onnx model before and after shape inference
    auto dump_dir = std::filesystem::path(
        ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR));
    auto dump_file = dump_dir / "origin_graph.onnx";
    LOG(INFO) << "save origin graph to resolve to" << dump_file
              << " graph id = " << origin_graph_->get_graph_id().to_string();
    dump_onnx_model(origin_graph_->graph_proto_, dump_file);
    dump_file = dump_dir / "staing_graph_resolved.onnx";
    LOG(INFO) << "save staging graph  to resolve to" << dump_file
              << " graph id = " << staging_graph_->get_graph_id().to_string();
    dump_onnx_model(staging_graph_->get_graph_proto(), dump_file);
  }

  print_log_message();

  // Collect information from staging graph and process deletions
  collect_staging_graph_names();
  process_meta_node_delete();
  maybe_mark_delete_initializers();

  // Proceed with main resolution steps
  resolve_name();
  resolve_opset(opset);
  resolve_doc_string();
  resolve_inputs();
  resolve_constant_initializers();
  resolve_outputs();
  resolve_nodes();
  resolve_value_info();

  if (!ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR).empty()) {
    // dump onnx model before and after shape inference
    auto dump_dir = std::filesystem::path(
        ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR));
    auto dump_file = dump_dir / "before_shape_infer.onnx";
    LOG(INFO) << "save graph before resolve to " << dump_file;
    dump_onnx_model(resolved_graph_proto_, dump_file);
  }

  morphizen_onnx::shape_inference::InferShapes(&resolved_graph_proto_, opset);

  if (!ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR).empty()) {
    // dump onnx model before and after shape inference
    auto dump_dir = std::filesystem::path(
        ENV_PARAM(MORPHIZEN_DEBUG_GRAPH_RESOLVER_DUMP_DIR));
    auto dump_file = dump_dir / "after_shape_infer.onnx";
    LOG(INFO) << "save graph after resolve to " << dump_file;
    dump_onnx_model(resolved_graph_proto_, dump_file);
  }

  return resolved_graph_proto_;
}

void GraphResolver::resolve_name() {
  MY_LOG(1) << "GraphResolver::resolve_name() - Starting name resolution";

  // Copy the name from the original graph proto
  const auto& orig_proto = origin_graph_->get_graph_proto();
  resolved_graph_proto_.set_name(orig_proto.name());

  // If staging graph has a different name, use that instead
  const auto& staging_proto = staging_graph_->get_graph_proto();
  if (!staging_proto.name().empty() &&
      staging_proto.name() != orig_proto.name()) {
    MY_LOG(2) << "Using staging graph name: " << staging_proto.name();
    resolved_graph_proto_.set_name(staging_proto.name());
  }

  MY_LOG(1) << "GraphResolver::resolve_name() - Completed with name: "
            << resolved_graph_proto_.name();
}

void GraphResolver::resolve_doc_string() {
  MY_LOG(1)
      << "GraphResolver::resolve_doc_string() - Starting doc string resolution";

  // Copy the doc string from the original graph proto
  const auto& orig_proto = origin_graph_->get_graph_proto();
  resolved_graph_proto_.set_doc_string(orig_proto.doc_string());

  // If staging graph has a different doc string, use that instead
  const auto& staging_proto = staging_graph_->get_graph_proto();
  if (!staging_proto.doc_string().empty() &&
      staging_proto.doc_string() != orig_proto.doc_string()) {
    MY_LOG(2) << "Using staging graph doc_string: "
              << staging_proto.doc_string();
    resolved_graph_proto_.set_doc_string(staging_proto.doc_string());
  }

  MY_LOG(1) << "GraphResolver::resolve_doc_string() - Completed";
}

void GraphResolver::resolve_inputs() {
  // just use the staging graph inputs, because when staging graph is
  // created, it already has the inputs from the original graph
  const auto& staging_proto = staging_graph_->get_graph_proto();
  resolved_graph_proto_.mutable_input()->Assign(staging_proto.input().begin(),
                                                staging_proto.input().end());
  auto input_index = -1;
  for (const auto& input : resolved_graph_proto_.input()) {
    input_index = input_index + 1;
    // Add to node_args_map for tracking
    node_args_map_[input.name()] =
        NodeArgIndex::graph_input(input_index, new_graph_id_);
  }
}

void GraphResolver::resolve_outputs() {
  // just use the staging graph outputs, because when staging graph is
  // created, it already has the outputs from the original graph
  const auto& staging_proto = staging_graph_->get_graph_proto();
  resolved_graph_proto_.mutable_output()->Assign(staging_proto.output().begin(),
                                                 staging_proto.output().end());
  auto output_index = -1;
  for (const auto& output : resolved_graph_proto_.output()) {
    output_index = output_index + 1;
    // Add to node_args_map for tracking
    node_args_map_[output.name()] =
        NodeArgIndex::graph_output(output_index, new_graph_id_);
  }
}

void GraphResolver::resolve_constant_initializers() {
  MY_LOG(1) << "GraphResolver::resolve_constant_initializers() - Starting "
               "initializer resolution";

  // Copy all initializers from the original graph proto
  const auto& orig_proto = origin_graph_->get_graph_proto();
  for (const auto& initializer : orig_proto.initializer()) {
    // when initializer has no name, it is deleted
    if (initializer.name().empty()) {
      MY_LOG(2) << "Skipping initializer with empty name";
      continue;
    }
    if (name_on_staging_graph_.count(initializer.name())) {
      MY_LOG(2) << "Skipping deleted initializer: " << initializer.name();
      continue;
    }
    auto* new_initializer = resolved_graph_proto_.add_initializer();
    new_initializer->Swap(
        const_cast<morphizen_onnx::TensorProto*>(&initializer));
  }

  // Merge any additional initializers from staging graph
  const auto& staging_proto = staging_graph_->get_graph_proto();
  for (const auto& staging_init : staging_proto.initializer()) {
    // Check if this initializer already exists in resolved graph
    MY_LOG(2) << "Adding new initializer from staging: " << staging_init.name();
    auto* new_initializer = resolved_graph_proto_.add_initializer();
    new_initializer->Swap(
        const_cast<morphizen_onnx::TensorProto*>(&staging_init));
  }
  auto initializer_index = -1;
  for (auto& initializer : resolved_graph_proto_.initializer()) {
    initializer_index = initializer_index + 1;
    // Ensure the initializer has a name
    if (initializer.name().empty()) {
      LOG(ERROR)
          << "Initializer with empty name found, this should not happen.";
    }
    // Add to node_args_map for tracking
    node_args_map_[initializer.name()] =
        NodeArgIndex::initializer(initializer_index, new_graph_id_);
  }
  MY_LOG(1)
      << "GraphResolver::resolve_constant_initializers() - Completed with "
      << resolved_graph_proto_.initializer_size() << " initializers";
}

void GraphResolver::troubleshooting(int staging_node_index,
                                    int origin_node_index) const {
  auto find_all = [](const std::string& name, int index,
                     const morphizen_onnx::GraphProto& graph) -> bool {
    for (auto& output : graph.node(index).output()) {
      if (output == name) {
        return true;
      }
    }
    return false;
  };
  auto search_in_graph = [&](const std::string& name, int index,
                             const morphizen_onnx::GraphProto& graph) -> int {
    for (int i = index + 1; i < graph.node_size(); ++i) {
      if (find_all(name, i, graph)) {
        return i;
      }
    }
    return -1;
  };

  auto& staging_graph_proto = staging_graph_->get_graph_proto();
  auto& origin_graph_proto = origin_graph_->get_graph_proto();
  auto search_for_node = [&](int index,
                             const morphizen_onnx::GraphProto& graph) {
    for (auto& input : graph.node(index).input()) {
      LOG(INFO) << "searching for input: " << input
                << " in graph node: " << graph.node(index).name()
                << " at index: " << index;
      auto found_in_staging_graph =
          search_in_graph(input, staging_node_index, staging_graph_proto);
      auto found_in_origin_graph =
          search_in_graph(input, origin_node_index, origin_graph_proto);
      if (found_in_origin_graph >= 0) {
        LOG(INFO)
            << "Input: " << input
            << " found in origin graph at node index: " << found_in_origin_graph
            << "\n"
            << origin_graph_proto.node(found_in_origin_graph).DebugString();
        ;
      }
      if (found_in_staging_graph >= 0) {
        LOG(INFO)
            << "Input: " << input << " found in staging graph at node index: "
            << found_in_staging_graph << "\n"
            << staging_graph_proto.node(found_in_staging_graph).DebugString();
      }
    }
  };
  LOG(INFO) << "searching for staging node";
  search_for_node(staging_node_index, staging_graph_proto);
  LOG(INFO) << "searching for origin node";
  search_for_node(origin_node_index, origin_graph_proto);
}
static std::string proto_debug_string(const morphizen_onnx::NodeProto& node) {
  auto ss = std::ostringstream();
  ss << node.name() << " (" << node.op_type() << ")";
  if (!node.domain().empty()) {
    ss << " [domain: " << node.domain() << "]";
  }
  ss << "input: [";
  int c = 0;
  for (const auto& input : node.input()) {
    if (c++ != 0) {
      ss << ",";
    }
    ss << "\"" << input << "\"";
  }
  ss << "],";
  c = 0;
  ss << "outputs: [";

  for (const auto& output : node.output()) {
    if (c++ != 0) {
      ss << ",";
    }
    ss << "\"" << output << "\"";
  }
  ss << "],";
  return ss.str();
};
void GraphResolver::resolve_nodes() {
  MY_LOG(1) << "GraphResolver::resolve_nodes() - Starting node resolution";
  auto output_node_args = origin_graph_->get_outputs_unsafe();
  auto output_nodes = std::vector<NodeIndex>{};
  output_nodes.reserve(output_node_args.size());
  for (auto& output_node_arg : output_node_args) {
    auto node_index = output_node_arg.get_producer_node();
    // the graph output maybe is a graph_initializer, not a node's output
    // test case : PSI_v3_0
    // graph output : [
    // "output_convert_QuantizeLinear_Output",
    // "interim_embeddings",
    // "output_exposed_scale_Output",   # graph initializer
    // "output_exposed_zero_point_Output"  # graph initializer
    // ]
    // so here remove the check all output_node_arg is node output
    // CHECK(node_index.is_valid());
    output_nodes.push_back(node_index);
  }

  origin_graph_->reverse_dfs_from_preemp(
      output_nodes, /*enter*/
      [this](const NodeIndex& node) {
        MY_LOG(1) << "enter :" << proto_debug_string(node.get_node_proto());
        return false;
      },
      [this](const NodeIndex& node) -> bool {
        auto graph_id = node.get_graph_id();
        auto& node_proto = node.get_node_proto();

        if (graph_id.is_staging()) {
          MY_LOG(1) << "merge from staging graph node: " << node.to_string()
                    << " at index: " << node.get_index() << " with "
                    << proto_debug_string(node_proto);

        } else {
          MY_LOG(1) << "merge from origin graph node: " << node.to_string()
                    << " at index: " << node.get_index() << " with "
                    << proto_debug_string(node_proto);
        }
        resolved_graph_proto_.add_node()->Swap(
            const_cast<morphizen_onnx::NodeProto*>(&node_proto));
        auto outputs = node.get_output_node_args();
        for (auto& output_node_arg_index : outputs) {
          if (output_node_arg_index.is_node_output()) {
            // Add to node_args_map for tracking
            *resolved_graph_proto_.mutable_value_info()->Add() =
                output_node_arg_index.get_value_info();
          } else if (output_node_arg_index.is_graph_output()) {
            // suppose graph output already have value info.
          } else if (output_node_arg_index.is_initializer()) {
            // create a new node to replace initializer.
            // import value info
            auto initializer = output_node_arg_index.get_const_data_as_tensor(
                *this->origin_graph_);
            CHECK(initializer != nullptr) << " cannot find initializer.";
            auto new_value_info =
                resolved_graph_proto_.mutable_value_info()->Add();
            new_value_info->set_name(initializer->name());
            new_value_info->mutable_type()
                ->mutable_tensor_type()
                ->set_elem_type(initializer->data_type());
            auto shape = new_value_info->mutable_type()
                             ->mutable_tensor_type()
                             ->mutable_shape();
            for (auto dim : initializer->dims()) {
              shape->mutable_dim()->Add()->set_dim_value(dim);
            }
          }
        }
        return false; // we do not stop travales.
      },
      nullptr /*compare*/, nullptr /* if*/, true /*include_staging_graph*/);
  MY_LOG(1) << "GraphResolver::resolve_nodes() - Completed with "
            << resolved_graph_proto_.node_size() << " total nodes";
}

void GraphResolver::resolve_value_info() {
  MY_LOG(1)
      << "GraphResolver::resolve_value_info() - Starting value_info resolution";

  // const auto& orig_proto = origin_graph_->get_graph_proto();
  // const auto& staging_proto = staging_graph_->get_graph_proto();

  //// Create a map to track value_info by name for deduplication
  // std::unordered_map<std::string, int> value_info_map;

  //// First, add all value_info from the original graph
  // for (const auto& value_info : orig_proto.value_info()) {
  //   if (!value_info_map.count(value_info.name())) {
  //     MY_LOG(2) << "Adding original value_info: " << value_info.name();
  //     // Add to the map if it doesn't already exist
  //     int index = resolved_graph_proto_.value_info_size();
  //     resolved_graph_proto_.add_value_info();
  //     value_info_map[value_info.name()] = index;
  //   }
  //   resolved_graph_proto_.mutable_value_info()->at(
  //       value_info_map[value_info.name()]) = value_info;
  //   MY_LOG(2) << "Added original value_info: " << value_info.name();
  // }

  //// Secondly, add value_info from staging graph (overriding original ones
  /// with / same name)
  // for (const auto& value_info : staging_proto.value_info()) {
  //   if (!value_info_map.count(value_info.name())) {
  //     MY_LOG(2) << "Adding staging value_info: " << value_info.name();
  //     // Add to the map if it doesn't already exist
  //     int index = resolved_graph_proto_.value_info_size();
  //     resolved_graph_proto_.add_value_info();
  //     value_info_map[value_info.name()] = index;
  //   }
  //   resolved_graph_proto_.mutable_value_info()->at(
  //       value_info_map[value_info.name()]) = value_info;

  //  MY_LOG(2) << "Added staging value_info: " << value_info.name();
  //}

  MY_LOG(1) << "GraphResolver::resolve_value_info() - Completed with "
            << resolved_graph_proto_.value_info_size() << " value_info entries";
}

void GraphResolver::print_log_message() {
  MY_LOG(1) << "GraphResolver::print_log_message() - Printing log messages";
  for (auto& log_message : staging_graph_->log_messages_) {
    MY_LOG(1) << "  -- " << log_message;
  }
}

bool GraphResolver::is_meta_node(const morphizen_onnx::NodeProto& node) {
  // Check if the node has the graph meta domain, indicating it's a meta node
  // used for graph manipulation operations (like deletion, fusion, etc.)
  return node.domain() == GRAPH_META_DOMAIN;
}

// Node resolution helper methods
bool GraphResolver::all_input_is_availabele(
    const morphizen_onnx::NodeProto& node,
    const std::unordered_map<std::string, NodeArgIndex>& node_args_map) {
  // check if all inputs for the node are available in
  // node_args_map
  MY_LOG(3) << "Checking if all inputs are available for node: " << node.name();

  for (const auto& input : node.input()) {
    if (input.empty()) {
      // Empty input names are allowed (optional inputs)
      continue;
    }

    if (node_args_map.find(input) == node_args_map.end()) {
      MY_LOG(3) << "Input not available: " << input
                << " for node: " << node.name();
      return false;
    }
  }

  return true;
}

void GraphResolver::add_node(const morphizen_onnx::NodeProto& node_v,
                             const Graph* source) {
  // Add a node to the resolved graph and update
  // node_args_map
  MY_LOG(3) << "Adding node to resolved graph: " << node_v.DebugString();

  // Add the node to the resolved graph proto
  auto* new_node = resolved_graph_proto_.add_node();
  new_node->Swap(const_cast<morphizen_onnx::NodeProto*>(&node_v));

  // Update node_args_map with the node's outputs
  for (int output_idx = 0; output_idx < new_node->output_size(); ++output_idx) {
    const auto& output_name = new_node->output(output_idx);
    if (output_name.empty()) {
      // Empty output names are allowed (optional outputs)
      continue;
    }
    // if the output name already exists in the map, we should not
    // overwrite it, because it may be used by other nodes
    if (source == origin_graph_) {
      if (name_on_staging_graph_.count(output_name)) {
        MY_LOG(3) << "Output name already exists in staging graph: "
                  << "\"" << output_name << "\"";
        new_node->set_output(output_idx, "");
        continue; // Skip if this output is already in the staging graph
      }
    }
    auto invalid_index = (unsigned int)(-1); // temporary invalid index
    // need to resolve it after shape inference
    auto node_arg_index =
        NodeArgIndex::node_output(invalid_index, new_graph_id_);
    node_args_map_[output_name] = node_arg_index;
    MY_LOG(3) << "Added node output to map: " << output_name
              << " with index: " << node_arg_index.to_string();
  }
}

bool GraphResolver::is_node_deleted(const morphizen_onnx::NodeProto& node,
                                    int index) {
  // Check if the node is marked for deletion using the index
  MY_LOG(3) << "Checking if node is deleted: " << node.name() << " at index "
            << index;

  // First check if the index is valid and the node is marked for deletion in
  // flags
  CHECK(index >= 0 && index < static_cast<int>(node_deleted_flags_.size()))
      << "Node index out of bounds: " << index << " vs size "
      << node_deleted_flags_.size();
  if (node_deleted_flags_[index]) {
    MY_LOG(3) << "Node is deleted due to deletion flag at index: " << index;
    return true;
  }

  // Check if all outputs of the node is either empty or found on the staging
  // graph
  return std::all_of(node.output().begin(), node.output().end(),
                     [this](const std::string& output) {
                       // Empty output names are allowed (optional outputs)
                       if (output.empty()) {
                         return true;
                       }
                       // Check if the output is found on the staging graph
                       return name_on_staging_graph_.count(output) > 0;
                     });
}

// Additional helper methods declared in header but not yet implemented
void GraphResolver::collect_staging_graph_names() {
  // Collect all node output names and initializer names from the staging graph
  // This populates name_on_staging_graph_ with all output names from staging
  // graph nodes and all initializer names
  name_on_staging_graph_.clear();

  if (!staging_graph_) {
    MY_LOG(2) << "GraphResolver::collect_staging_graph_names() - No staging "
                 "graph available";
    return;
  }

  const auto& staging_proto = staging_graph_->get_graph_proto();

  // Iterate through all nodes in the staging graph to collect output names
  for (const auto& node : staging_proto.node()) {
    // Add all output names from this node to the set
    for (const auto& output : node.output()) {
      if (!output.empty()) { // Skip empty output names (optional outputs)
        name_on_staging_graph_.insert(output);
      }
    }
  }

  // Iterate through all initializers in the staging graph to collect their
  // names
  for (const auto& initializer : staging_proto.initializer()) {
    if (!initializer.name().empty()) { // Skip initializers with empty names
      name_on_staging_graph_.insert(initializer.name());
    }
  }

  MY_LOG(2) << "GraphResolver::collect_staging_graph_names() - Collected "
            << name_on_staging_graph_.size()
            << " names (outputs + initializers) from staging graph";
}

void GraphResolver::process_meta_node_delete() {
  // Mark nodes for deletion based on meta nodes in the staging graph
  const auto& staging_proto = staging_graph_->get_graph_proto();

  // Look for meta nodes with "delete" op_type in the staging graph
  for (int i = 0; i < staging_proto.node_size(); ++i) {
    const auto& node = staging_proto.node(i);

    // Check if this is a meta node for deletion
    if (is_meta_node(node)) {
      if (node.op_type() == "delete_node") {
        // Find the target_node_index attribute
        for (const auto& attr : node.attribute()) {
          if (attr.name() == "target_node_index" &&
              attr.type() == morphizen_onnx::AttributeProto::INT) {
            int64_t target_index = attr.i();

            // Mark the target node for deletion if within bounds
            CHECK(target_index >= 0 &&
                  target_index <
                      static_cast<int64_t>(node_deleted_flags_.size()))
                << "Invalid target_node_index: " << target_index << " vs size "
                << node_deleted_flags_.size();
            node_deleted_flags_[target_index] = true;
            MY_LOG(2)
                << "GraphResolver::process_meta_node_delete() - Marked node "
                << target_index << " for deletion";
            break;
          }
        }
      } else if (node.op_type() == "remove_initializer") {
        // Handle initializer deletion
        for (const auto& attr : node.attribute()) {
          if (attr.name() == "index" &&
              attr.type() == morphizen_onnx::AttributeProto::INT) {
            int64_t index = attr.i();
            CHECK_GE(index, 0) << "Initializer index must be non-negative: ";
            CHECK_LT(index,
                     static_cast<int64_t>(initializer_deleted_flags_.size()))
                << "Initializer index out of bounds: " << index << " vs size "
                << initializer_deleted_flags_.size();
            initializer_deleted_flags_[index] = true;
            break;
          }
        }
      } else {
        LOG(FATAL) << "GraphResolver::process_meta_node_delete() - Skipping "
                      "non-delete meta node: "
                   << node.name() << " with op_type: " << node.op_type();
      }
    }
  }
}

void GraphResolver::maybe_mark_delete_initializers() {
  // Mark all initializers whose names are found in name_on_staging_graph_ as
  // deleted on original graph
  const auto& orig_proto = origin_graph_->get_graph_proto();
  for (int i = 0; i < orig_proto.initializer_size(); ++i) {
    const auto& initializer = orig_proto.initializer(i);
    if (name_on_staging_graph_.count(initializer.name())) {
      // Check bounds before marking for deletion
      CHECK(i >= 0 && i < static_cast<int>(initializer_deleted_flags_.size()))
          << "Initializer index out of bounds: " << i << " vs size "
          << initializer_deleted_flags_.size();
      initializer_deleted_flags_[i] = true;
      MY_LOG(2) << "GraphResolver::maybe_mark_delete_initializers() - Marked "
                   "initializer "
                << initializer.name() << " at index " << i << " for deletion";
    }
  }
}

void GraphResolver::resolve_opset(std::unordered_map<std::string, int>& opset) {
  MY_LOG(1) << "GraphResolver::resolve_opset() - Starting opset resolution";

  // Note: GraphProto does not contain opset_import fields directly.
  // Opset imports are typically stored at the ModelProto level.
  // however we can iterate through all nodes and check their opset
  // information if needed, and since_version
  for (auto& node : staging_graph_->get_graph_proto().node()) {
    // Check if the node has an opset_version attribute
    auto version =
        1; // this is not a good design, let's fix it when it is broken.
    auto it = opset.find(node.domain());
    if (it != opset.end()) {
      // If domain already exists, update the version if it's higher
      if (version > it->second) {
        MY_LOG(2) << "Updating opset version for domain: " << node.domain()
                  << " from " << it->second << " to " << version;
        it->second = version;
      }
    } else {
      // Otherwise, add the new domain and version
      MY_LOG(2) << "Adding new opset domain: " << node.domain()
                << " with version: " << version;
      opset[node.domain()] = version;
    }
  }

  MY_LOG(1)
      << "GraphResolver::resolve_opset() - Completed (no-op for GraphProto)";
}

} // namespace morphizen
