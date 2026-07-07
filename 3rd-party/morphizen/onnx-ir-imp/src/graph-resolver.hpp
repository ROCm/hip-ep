/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./node-arg-index.hpp"
#include "graph-id.hpp"
#include "onnx-deps.hpp"
#include <cstdint>
#include <optional>
#include <unordered_map>
namespace morphizen {

// String constants for graph meta operations
static constexpr const char* GRAPH_META_DOMAIN = "com.morphizen.meta";

// Forward declarations
class Graph;
class StagingGraph;
class NodeArgIndex;

/**
 * @brief GraphResolver class - handles Graph::resolve operations
 *
 * This class provides functionality for resolving graph inconsistencies
 * and managing graph state transitions.
 */
class GraphResolver {
public:
  // Constructor
  GraphResolver();

  // Destructor
  ~GraphResolver();

  // Copy constructor (deleted for now)
  GraphResolver(const GraphResolver&) = delete;

  // Copy assignment operator (deleted for now)
  GraphResolver& operator=(const GraphResolver&) = delete;

  // Move constructor (deleted for now)
  GraphResolver(GraphResolver&&) = delete;

  // Move assignment operator (deleted for now)
  GraphResolver& operator=(GraphResolver&&) = delete;

  // Main resolve method
  morphizen_onnx::GraphProto
  resolve(Graph& graph, GraphId new_graph_id,
          std::unordered_map<std::string, int>& opset);

  // Public static helper functions
  static bool is_meta_node(const morphizen_onnx::NodeProto& node);

private:
  void initialize_private_variables(Graph& graph, GraphId new_graph_id);

  // Helper methods for graph resolution
  // Collects all node output names and initializer names from the staging graph
  void collect_staging_graph_names();
  void maybe_mark_delete_initializers();
  void resolve_name();
  void resolve_opset(std::unordered_map<std::string, int>& opset);
  void resolve_doc_string();
  void resolve_inputs();
  void resolve_outputs();
  void resolve_constant_initializers();
  void resolve_nodes();
  void troubleshooting(int staging_node_index, int origin_node_index) const;
  void resolve_value_info();

  void print_log_message();

  // Node resolution helper methods
  bool all_input_is_availabele(
      const morphizen_onnx::NodeProto& node,
      const std::unordered_map<std::string, NodeArgIndex>& node_args_map);
  void add_node(const morphizen_onnx::NodeProto& node, const Graph* from);
  bool is_node_deleted(const morphizen_onnx::NodeProto& node, int index);
  std::optional<NodeArgIndex> find_node_arg_index(const std::string& name);
  void process_meta_node_delete();

  // Private members
  Graph* origin_graph_;         // Pointer to the original graph being resolved
  StagingGraph* staging_graph_; // Pointer to the staging graph being resolved
  GraphId new_graph_id_;        // New graph ID to assign to resolved graph
  std::unordered_map<std::string, NodeArgIndex>
      node_args_map_;           // Map to track node args for resolution
  morphizen_onnx::GraphProto
      resolved_graph_proto_;    // Resolved graph proto after processing
  std::vector<bool> initializer_deleted_flags_; // Flags to track deleted nodes
  std::vector<bool> node_deleted_flags_;        // Flags to track deleted nodes
  std::unordered_set<std::string>
      name_on_staging_graph_; // all node output names and initializer names on
                              // staging graph
};

} // namespace morphizen
