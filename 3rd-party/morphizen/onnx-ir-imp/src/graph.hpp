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
class NodeArg;
class StagingGraph;
class GraphResolver;
class GraphShapeInfer;

class Graph {

  // === Graph API Methods ===
private:
  // === Private Tag for Constructor ===
  struct PrivateTag {};
  friend class NodeArgIndex;
  friend class GraphResolver;

public:
  // === Static Factory Methods ===
  static std::unique_ptr<Graph>
  create_main_graph(morphizen_onnx::ModelProto &model_proto,
                    const Model *parent_model);

  // === Constructor & Destructor ===
  explicit Graph(PrivateTag, morphizen_onnx::GraphProto &graph_proto,
                 const Model *parent_model, uint32_t proposed_graph_index,
                 const Graph *parent_graph);
  ~Graph();

  // === Const Methods (Read-Only Operations) ===

  // Basic graph information
  GraphId get_graph_id() const;
  const std::string &get_name() const;
  const Model &get_model() const;
  const morphizen_onnx::GraphProto &get_graph_proto() const;
  StagingGraph *get_staging_graph() const;
  const std::filesystem::path &get_model_path() const;
  bool need_resolve() const;

  // Node access and traversal
  std::vector<NodeIndex> nodes_unsafe() const;
  const NodeIndex producer_node(const std::string &node_arg_name) const;
  std::vector<NodeIndex>
  get_consumer_nodes(const std::string &node_arg_name) const;

  // Input/Output access
  std::vector<NodeArgIndex> get_inputs_unsafe() const;
  std::vector<NodeArgIndex> get_outputs_unsafe() const;
  const NodeArgIndex get_node_arg(const std::string &name) const;

  // Initializers access
  const std::unordered_map<std::string, const morphizen_onnx::TensorProto *> &
  get_all_initialized_tensors() const;

  // Graph traversal
  void reverse_dfs_from_preemp(
      gsl::span<const NodeIndex> from,
      const std::function<bool(const NodeIndex &)> &enter,
      const std::function<bool(const NodeIndex &)> &leave,
      const std::function<bool(const NodeIndex &, const NodeIndex &)> &comp,
      const std::function<bool(const NodeIndex &, const NodeIndex &)> &stop,
      bool include_staging_graph) const;

  // I/O operations
  void save(const std::string &filename, const std::string &dat_filename,
            size_t external_data_threshold) const;
  std::string save_string() const;

  // === Const Methods that Modify Staging Graph (Logical Const) ===
  // Note: These are const because they don't modify the current graph state,
  // but work with the mutable staging graph
  void remove_node(NodeIndex node_index) const;
  void add_initialized_tensor(const morphizen_onnx::TensorProto &tensor) const;
  void set_inputs(gsl::span<NodeArgIndex> inputs) const;
  void set_outputs(gsl::span<const NodeArgIndex> outputs) const;
  void set_graph_name(const char *name) const;

  NodeIndex
  add_node(const std::string &name, const std::string &op_type,
           const std::string &description,
           const std::vector<NodeArgIndex> &input_args,
           const std::vector<NodeArgIndex> &output_args,
           ::google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>
               *attributes,
           const std::string &domain) const;
  NodeArgIndex node_arg_new(const std::string &name,
                            const std::vector<int64_t> *shape,
                            int element_type) const;
  // NodeArg related methods
  // it is only used by this pass
  // to be removed.
  // clang-format off
/*
morphizen_pass_graph_output_add_node/src/graph_output_add_node.cpp:71:            MORPHIZEN_ORT_API(node_arg_clone)(*graph, *output.node_arg, name);
*/
  // clang-format on
  void *node_arg_clone(const NodeArg &node_arg, const std::string &name) const;
  // === Non-Const Methods (Modify Graph State) ===

  // Direct graph proto access
  morphizen_onnx::GraphProto &get_graph_proto();

  // Graph structure modification
  void remove_initialized_tensor(const std::string &tensor_name) const;

  // Advanced graph operations
  NodeIndex fuse(const std::string &name, const std::string &op_type,
                 const std::vector<size_t> &nodes,
                 const std::vector<std::string> &inputs,
                 const std::vector<std::string> &outputs,
                 const std::vector<std::string> &constant_initializers);
  int resolve(bool force);

private:
  /**
   * @brief Create a Graph instance from ONNX GraphProto
   * @param graph_proto The ONNX GraphProto to take ownership of
   * @param parent_model Pointer to the parent Model instance
   * @return Unique pointer to Graph instance
   */
  static std::unique_ptr<Graph>
  create_graph(morphizen_onnx::GraphProto &graph_proto,
               const Model *parent_model, uint32_t proposed_graph_index = 0,
               const Graph *parent_graph = nullptr);

private:
  // === Private Initialization Methods ===
  void initialize();
  void initialize_map();
  void initialize_nodes();
  void initialize_consumer_map();
  void initialize_initializers();

  // === Private Const Helper Methods ===

  // DFS traversal implementation
  void reverse_dfs_from_impl(
      gsl::span<const NodeIndex> from,
      const std::function<bool(const NodeIndex &)> &enter,
      const std::function<bool(const NodeIndex &)> &leave,
      const std::function<bool(const NodeIndex &, const NodeIndex &)> &comp,
      const std::function<bool(const NodeIndex &, const NodeIndex &)> &stop,
      bool include_staging_graph) const;

  // Staging graph management
  void ensure_enter_into_inconsistent_state() const;

  // Graph structure queries
  int get_graph_output_index(const std::string &name) const;
  void validate_add_node_parameters(
      const std::string &name, const std::string &op_type,
      const std::vector<NodeArgIndex> &input_args,
      const std::vector<NodeArgIndex> &output_args) const;

  // === Private Non-Const Helper Methods ===

  // Graph ID management
  GraphId allocate_new_graph_index_and_release_old();

  // Shape inference
  void infer_shapes();

private:
  friend class StagingGraph; // for searching for node_arg_map_ in ::remove_node
  friend class NodeIndex;
  friend class Model;

private:
  morphizen_onnx::GraphProto &graph_proto_;
  const Model *parent_model_; // Reference to the parent model
  const Graph *parent_graph_; // Pointer to the parent graph (for subgraphs,
                              // nullptr for main graph)
  // cache_[graph_index_] === this
  uint32_t graph_index_;
  // node_args_map_[node_arg_name] = NodeArgIndex
  // include "graph_input", "initializer", "node_output"
  // 1. for "graph_input", node_arg_index.index is index to graph_proto_.input
  // 2. for "initializer", node_arg_index.index is index to
  // graph_proto_.initializer
  // 3. for "node_output", node_arg_index.index is index to
  // graph_proto_.value_info
  // 4. for "graph_output", node_arg_index.index is index to
  // graph_proto_.output
  std::unordered_map<std::string, NodeArgIndex> node_args_map_;
  // nodes_[node_arg_index] === Node, return invalid NodeIndex if not found,
  // e.g. "graph_input", or "initializer"
  NodeArgProducer producer_map_; // map from node_arg to producer node  //
                                 // consumer_map_[node_arg_index] === vector of
                                 // NodeIndex which consume this
  // node_arg
  std::unordered_map<NodeArgIndex, std::vector<NodeIndex>>
      consumer_map_; // map from node_arg to producer node  // dont use this map
                     // directly, it is only used for save the topological
  // structure
  std::vector<Node> nodes_;
  // Cached initialized tensors map
  std::unordered_map<std::string, const morphizen_onnx::TensorProto *>
      initializers_map_;
  // Staging graph for modifications
  mutable std::unique_ptr<StagingGraph> staging_graph_;
};

} // namespace morphizen
