/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "./graph-id.hpp"
#include "./node-arg-index.hpp"
#include "./node-arg-producer-map.hpp"
#include "./node-arg.hpp"
#include "./node-index.hpp"
#include "./node.hpp"
#include <filesystem>
#include <gsl/gsl>
#include <map>
#include <onnx/onnx_pb.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace google {
namespace protobuf {
template <typename T> class RepeatedPtrField;
}
} // namespace google

namespace morphizen {

// Forward declarations
class Model;
class Graph;
class NodeArg;
class GraphResolver;

/**
 * @class StagingGraph
 * @brief Represents a staging graph for temporary graph modifications
 *
 * StagingGraph provides a sandbox environment for graph transformations where
 * modifications can be made without affecting the main graph until changes are
 * committed. It maintains similar API to Graph but operates on a staging
 * GraphProto.
 *
 * Design rationale:
 * - Provides isolation for graph modifications during transformations
 * - Maintains similar interface to Graph for consistency
 * - Supports rollback capabilities for failed transformations
 * - Uses GraphId for type-safe staging graph identification
 */
class StagingGraph {
private:
  // === Private Tag for Constructor ===
  struct PrivateTag {};
  friend class Graph;
  friend class NodeArgIndex;

public:
  // === Static Factory Methods ===
  static std::unique_ptr<StagingGraph>
  create_from_graph(const Graph& main_graph);

  // === Constructor & Destructor ===
  explicit StagingGraph(PrivateTag, const Graph& main_graph);
  ~StagingGraph();

  // Disable copy constructor and assignment operator
  StagingGraph(const StagingGraph&) = delete;
  StagingGraph& operator=(const StagingGraph&) = delete;
  StagingGraph(StagingGraph&& other) = delete;
  StagingGraph& operator=(StagingGraph&& other) = delete;

  // === Const Methods (Read-Only Operations) ===

  GraphId get_graph_id() const;
  const Graph& get_main_graph() const;

  const morphizen_onnx::GraphProto& get_graph_proto() const;
  morphizen_onnx::GraphProto& get_graph_proto();

  void remove_node(NodeIndex node_index);
  void add_initialized_tensor(const morphizen_onnx::TensorProto& tensor);
  void remove_initialized_tensor(unsigned int index,
                                 const std::string& tensor_name);
  void set_inputs(gsl::span<NodeArgIndex> inputs);
  void set_outputs(gsl::span<const NodeArgIndex> outputs);

  NodeIndex
  add_node(const std::string& name, const std::string& op_type,
           const std::string& description,
           const std::vector<NodeArgIndex>& input_args,
           const std::vector<NodeArgIndex>& output_args,
           ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*
               attributes,
           const std::string& domain);

  NodeArgIndex node_arg_new(const std::string& name,
                            const std::vector<int64_t>* shape,
                            int element_type);

private:
  // Logging and debugging
  void log_remove_node(const NodeIndex& node_index,
                       const morphizen_onnx::NodeProto& node_proto);
  void log_add_initialized_tensor(const morphizen_onnx::TensorProto& tensor);
  void log_remove_initialized_tensor(unsigned int index,
                                     const std::string& tensor_name);
  void log_set_inputs(gsl::span<const NodeArgIndex> inputs);
  void log_set_outputs(gsl::span<const NodeArgIndex> outputs);
  void log_add_node(const std::string& name, const std::string& op_type,
                    const std::string& description, const std::string& domain,
                    const std::vector<NodeArgIndex>& input_args,
                    const std::vector<NodeArgIndex>& output_args);
  void log_node_arg_new(const std::string& name,
                        const std::vector<int64_t>* shape, int element_type);
  int get_graph_output_index(const std::string& name) const;

  // Node addition helper methods
  void validate_add_node_parameters(
      const std::string& name, const std::string& op_type,
      const std::vector<NodeArgIndex>& input_args,
      const std::vector<NodeArgIndex>& output_args) const;

  void configure_node_proto(
      morphizen_onnx::NodeProto* new_node, const std::string& name,
      const std::string& op_type, const std::string& description,
      const std::string& domain,
      ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*
          attributes) const;

  void process_input_arguments(morphizen_onnx::NodeProto* new_node,
                               const std::vector<NodeArgIndex>& input_args);

  void process_output_arguments(morphizen_onnx::NodeProto* new_node,
                                const std::vector<NodeArgIndex>& output_args);

  NodeIndex update_staging_nodes_structures(
      const std::vector<NodeArgIndex>& input_node_arg_indices,
      std::vector<NodeArgIndex>& output_node_arg_indices);

  void
  update_producers_for_new_node(NodeIndex node_index,
                                gsl::span<const NodeArgIndex> output_node_args);

  void
  update_consumers_for_new_node(NodeIndex node_index,
                                gsl::span<const NodeArgIndex> input_node_args);

private:
  friend class GraphResolver;
  friend class NodeIndex;

private:
  const Graph& main_graph_; ///< Reference to the main graph
  morphizen_onnx::GraphProto
      graph_proto_;         ///< Staging graph proto for modifications
  // Graph structure (similar to Graph class)
  std::unordered_map<std::string, NodeArgIndex> node_args_map_;
  NodeArgProducer producer_map_;
  std::vector<Node> nodes_;
  std::unordered_map<std::string, const morphizen_onnx::TensorProto*>
      initializers_map_;
  // State tracking
  std::vector<std::string> log_messages_; ///< Log messages for debugging
};

} // namespace morphizen
