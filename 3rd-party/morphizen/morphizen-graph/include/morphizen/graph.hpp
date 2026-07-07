/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/// @file graph.hpp
/// @brief C++ wrapper utilities for graph operations over MORPHIZEN_ORT_API
///
/// This file provides C++ wrappers over the MORPHIZEN_ORT_API interface,
/// offering type-safe, RAII-friendly graph manipulation APIs that work
/// with any backend (onnx-ir-imp, mlir-imp, etc.).
///
/// The wrappers encapsulate the low-level function pointer interface
/// and provide a clean C++ object model for:
/// - Graph manipulation (GraphRef, GraphConstRef)
/// - Node operations (via node.hpp)
/// - NodeArg handling (via node_arg.hpp)
/// - Node building (NodeBuilder for high-level construction)
///
/// All operations are forwarded to the active backend implementation
/// via the MORPHIZEN_ORT_API function pointer table, allowing runtime
/// backend selection while maintaining a consistent API.

#pragma once
#include "./_sanity_check.hpp"
#include "./node.hpp"
#include "./node_arg.hpp"
#include "./node_attr.hpp"
#include <cassert>
#include <filesystem>
#include <functional>
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#include <optional>

namespace morphizen {

#ifndef MORPHIZEN_USE_DEPRECATED_API
[[deprecated("This API will be removed in the future release version. Please "
             "use NodeBuilder instead.")]]
#endif
MORPHIZEN_DLL_SPEC Node &
graph_add_node(Graph &graph, const std::string &name,
               const std::string &op_type, const std::string &description,
               const std::vector<const NodeArg *> &input_args,
               const std::vector<const NodeArg *> &output_args,
               NodeAttributesPtr attributes, const std::string &domain);

MORPHIZEN_DLL_SPEC std::vector<const NodeArg *>
node_inputs_2_node_args(const std::vector<NodeInput> &inputs);

MORPHIZEN_DLL_SPEC std::vector<const NodeArg *>
graph_get_outputs(const Graph &graph);

MORPHIZEN_DLL_SPEC void graph_set_name(Graph &graph, const std::string &name);

// NOTE: NodeBuilder has been moved to morphizen-core (node_builder.hpp)
// It depends on IPass and AnchorPoint which are high-level morphizen-core
// concepts. This file contains only low-level graph wrappers over
// MORPHIZEN_ORT_API.

const Model &graph_get_model(const Graph &graph);

std::vector<const Node *> graph_nodes(const Graph &graph);

std::vector<const NodeArg *> graph_get_inputs(const Graph &graph);

MORPHIZEN_DLL_SPEC std::vector<const Node *>
graph_get_output_nodes(const Graph &graph);

/** @brief get indices of all nodes in topoligial order
 *
 * @param graph
 * @return vector of node indices.
 *
 * each element is a node index, for example, we can print all nodes
 * in topological order of nodes
 *
 *    auto nodes = graph_get_node_in_topoligical_order(graph);
 *    for (auto node_idx : nodes) {
 *        auto node = MORPHIZEN_ORT_API(graph_get_node)(graph, node_idx);
 *        if (node == nullptr) { // should never goes here
 *             cout << node_as_string(*node);
 *        }
 *    }
 */
MORPHIZEN_DLL_SPEC std::vector<size_t>
graph_get_node_in_topoligical_order(const Graph &graph);

/** @brief dump a graph as a string for debugging purpose
 *
 * @param graph
 * @return a string
 *
 * When environment variable ENABLE_SAVE_GRAPH_TXT=1,
 * `graph_as_string` is used to dump many text files in the cache
 * directory, after each pass, it is handy for troubleshooting, because
 *
 *  1. it is a text file, and it is easy for searching
 *  2. it contains shape information.
 *
 */
MORPHIZEN_DLL_SPEC std::string graph_as_string(const Graph &graph);

std::vector<const Node *>
graph_get_consumer_nodes(const Graph &graph, const std::string &node_arg_name);

/** @brief garbage collection by removing dangling nodes.
 *
 *  @param graph
 *
 *  usuaully a pass writer does not need to invoke this function
 *  explicitly, in morphizen_config.json, we can set `enableGc=true` to
 *  automatically apply garbage collection.
 *
 *  Sometime it is useful to disable gc for troubleshooting.
 *
 */
MORPHIZEN_DLL_SPEC void graph_gc(Graph &graph);

/** @brief rebuild graph data structure.
 *
 *  @param graph
 *  @param force
 *
 *  this function is very heavy, because firstly it cleans up all
 *  internal data structure and rebuild everything from sratch.
 *
 *     1. shape infer
 *     2. build edge/node relationship
 *
 *  After invoke MORPHIZEN_ORT_API(add_node) or MORPHIZEN_ORT_API(remove_node),
 *  the internal data structure becomes invalid, sometimes,
 *  `graph_get_consumer_nodes` or other funtions cannot return proper
 *  values until we inoke `graph_resolve`.
 *
 *  `Pass::apply(...)` and `Pass::fuse` invoke `graph_resolve`.
 *
 */
MORPHIZEN_DLL_SPEC void graph_resolve(Graph &graph, bool force = false);

// NOTE: graph_replace_node_arg has been moved to morphizen-core
// It depends on IPass which is a morphizen-core type

/** @brief Get the producer node of a node argument
 *
 * @param graph The graph to query
 * @param node_arg_name Name of the node argument
 * @return Pointer to the producer node, or nullptr if not found
 *
 * Returns the node that produces the given node argument as output.
 */
MORPHIZEN_DLL_SPEC const Node *
graph_producer_node(const Graph &graph, const std::string &node_arg_name);

/** @brief Get a node argument by name
 *
 * @param graph The graph to query
 * @param name Name of the node argument
 * @return Pointer to the node argument, or nullptr if not found
 */
MORPHIZEN_DLL_SPEC const NodeArg *graph_get_node_arg(const Graph &graph,
                                                     const std::string &name);

/** @brief Get the name of a graph
 *
 * @param graph The graph to query
 * @return The graph name
 */
MORPHIZEN_DLL_SPEC const std::string &graph_get_name(const Graph &graph);

/** @brief Add an initialized tensor to the graph
 *
 * @param graph The graph to modify
 * @param tensor The tensor proto to add as an initializer
 *
 * Adds a tensor as a constant initializer to the graph. This is used to add
 * constant weights and parameters to the model.
 */
MORPHIZEN_DLL_SPEC void graph_add_initialized_tensor(Graph &graph,
                                                     const TensorProto &tensor);

/** @brief Set graph inputs
 *
 * @param graph The graph to modify
 * @param inputs Vector of node arguments to set as graph inputs
 */
MORPHIZEN_DLL_SPEC void graph_set_inputs(Graph &graph,
                                         const std::vector<NodeArg *> &inputs);

/** @brief Set graph outputs
 *
 * @param graph The graph to modify
 * @param outputs Vector of node arguments to set as graph outputs
 */
MORPHIZEN_DLL_SPEC void
graph_set_outputs(Graph &graph, const std::vector<NodeArg *> &outputs);

/** @brief Save graph to file
 *
 * @param graph The graph to save
 * @param model_path Path to save the model file
 * @param external_data_path Path for external data file
 * @param threshold Threshold for external data
 */
MORPHIZEN_DLL_SPEC void graph_save(Graph &graph, const std::string &model_path,
                                   const std::string &external_data_path,
                                   size_t threshold);

/** @brief Perform reverse DFS traversal from a node
 *
 * @param graph The graph to traverse
 * @param node_index Starting node index
 * @param enter Callback invoked when entering a node (return false to skip
 * subtree)
 * @param leave Callback invoked when leaving a node
 * @param comp Comparator for traversal order (optional)
 * @param subgraph_sensitive Whether to respect subgraph boundaries
 *
 * Traverses the graph in reverse DFS order starting from the given node.
 */
MORPHIZEN_DLL_SPEC void graph_reverse_dfs_from(
    const Graph &graph, size_t node_index,
    const std::function<bool(const Node *)> &enter,
    const std::function<void(const Node *)> &leave,
    const std::function<bool(const Node *, const Node *)> &comp = nullptr,
    bool subgraph_sensitive = false);

/** @brief Perform reverse DFS traversal from multiple nodes
 *
 * @param graph The graph to traverse
 * @param from Starting nodes (multiple)
 * @param enter Callback invoked when entering a node
 * @param leave Callback invoked when leaving a node (optional)
 * @param stop Callback to determine if traversal should stop along an edge
 *
 * Traverses the graph in reverse DFS order starting from multiple nodes.
 * This is a low-level wrapper used by morphizen-core for fusion analysis.
 */
MORPHIZEN_DLL_SPEC void graph_reverse_dfs_from_multi(
    const Graph &graph, gsl::span<const Node *const> from,
    const std::function<void(const Node *)> &enter,
    const std::function<void(const Node *)> &leave,
    const std::function<bool(const Node *, const Node *)> &stop);

/** @brief Fuse nodes in the graph
 *
 * @param graph The graph to modify
 * @param name Name for the fused node
 * @param op_type Operation type for the fused node
 * @param nodes Nodes to fuse
 * @param inputs Input node arguments
 * @param outputs Output node arguments
 * @param attributes Node attributes (optional)
 *
 * Fuses multiple nodes into a single node, typically used for optimization.
 */
MORPHIZEN_DLL_SPEC void
graph_fuse(Graph &graph, const std::string &name, const std::string &op_type,
           const std::vector<const Node *> &nodes,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs,
           const std::vector<std::string> &constant_initializers = {});

/** @brief Fuse nodes in the graph (low-level wrapper)
 *
 * @param graph The graph to modify
 * @param name Name for the fused node
 * @param op_type Operation type for the fused node
 * @param nodes Node indices to fuse
 * @param inputs Input node argument names
 * @param outputs Output node argument names
 * @param constant_initializers Constant initializer names
 * @return Reference to the newly created fused node
 *
 * This is a low-level wrapper that takes node indices instead of node pointers.
 * It directly wraps MORPHIZEN_ORT_API(graph_fuse) for use by morphizen-core.
 */
MORPHIZEN_DLL_SPEC Node &
graph_fuse(Graph &graph, const std::string &name, const std::string &op_type,
           const std::vector<size_t> &nodes,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs,
           const std::vector<std::string> &constant_initializers);

// Model operations (wrappers for model-level MORPHIZEN_ORT_API calls)
// These allow morphizen-core's Model class to use morphizen-graph wrappers
// instead of calling MORPHIZEN_ORT_API directly

/** @brief Get the main graph from a model
 *
 * @param model The model to query
 * @return Reference to the main graph
 */
MORPHIZEN_DLL_SPEC Graph &model_main_graph(Model &model);

/** @brief Get metadata from a model
 *
 * @param model The model to query
 * @param key Metadata key
 * @return Metadata value (returned by value to avoid dangling reference)
 */
MORPHIZEN_DLL_SPEC std::string model_get_meta_data(const Model &model,
                                                   const std::string &key);

/** @brief Check if model has metadata
 *
 * @param model The model to query
 * @param key Metadata key to check
 * @return true if metadata exists, false otherwise
 */
MORPHIZEN_DLL_SPEC bool model_has_meta_data(const Model &model,
                                            const std::string &key);

/** @brief Clone a model
 *
 * @param model The model to clone
 * @return Pointer to the cloned model
 */
MORPHIZEN_DLL_SPEC Model *model_clone(const Model &model);

} // namespace morphizen

namespace morphizen_cxx {
class Subgraph;
class NodeRef;
/**
 * @class GraphConstRef
 * @brief A reference wrapper to a constant `onnxruntime::Graph` object. It is
 * CopyConstructiable and MoveConstructiable, so that for example it can be put
 * into a vector.
 */
class MORPHIZEN_DLL_SPEC GraphConstRef {
public:
  /**
   * @brief Constructs a `GraphConstRef` object.
   *
   * @param graph The underlying `morphizen::Graph` object.
   */
  GraphConstRef(const morphizen::Graph &graph) : graph_(graph) {}

  /**
   * @brief Destroys the `GraphConstRef` object.
   */
  ~GraphConstRef();

  /**
   * @brief Checks if the current GraphConstRef object is equal to another
   * GraphConstRef object.
   *
   * @param other The other GraphConstRef object to compare with.
   * @return true if the two GraphConstRef objects are equal, false otherwise.
   */
  bool operator==(const GraphConstRef &other) const {
    return &graph_ == &other.graph_;
  }
  /**
   * @brief Gets the name of the graph.
   *
   * @return The name of the graph.
   */
  const std::string &name() const;
  /**
   * Returns the path to the model.
   *
   * @return An optional containing the path to the model, or an empty path
   * if model is loaded from memory.
   */
  const std::filesystem::path &model_path() const;
  /**
   * @brief Conversion operator to convert to a const reference of
   * `onnxruntime::Graph`.
   *
   * @return A const reference to the underlying `onnxruntime::Graph` object.
   */
  operator const onnxruntime::Graph &() const { return graph_; }

  /**
   * @brief Returns a vector of NodeArg objects representing the inputs of the
   * graph.
   *
   * @return A vector of NodeArg objects representing the inputs of the graph.
   */
  std::vector<NodeArgConstRef> inputs() const;

  /**
   * @brief Returns a vector of NodeArg objects representing the outputs of the
   * graph.
   *
   * @return A vector of NodeArg objects representing the outputs of the graph.
   */
  std::vector<NodeArgConstRef> outputs() const;

  /**
   * @brief Returns a vector of NodeArg objects representing the constant
   * initializers of the graph.
   *
   * @return A vector of NodeArg objects representing the constant initializers
   * of the graph.
   */
  std::vector<NodeArgConstRef> constant_initializers() const;
  /**
   * @brief Returns a vector of Node objects representing the
   * nodes of the graph.
   *
   * @return A vector of Node objects representing
   * @note it is faster and no sorting.
   */
  std::vector<NodeConstRef> nodes() const;
  /**
   * @brief Save the graph to a file.
   * @param filename The name of the file to save the graph to.
   */
  void save(const std::filesystem::path &filename) const;
  /**
   * @brief Save the graph to a ONNX model file with exteranl data.
   *
   * @param filename The name of the file to save the graph to.
   * @param external_data_file The name of the external data file.  Supported
   * relative path, absolute path, and empty path. If the external_data_file is
   * empty, the external data will not be saved. If the external_data_file is a
   * relative path, the external data will be saved to the same directory as the
   * model file. If the external_data_file is an absolute path, the external
   * data will be saved to the specified directory. relative path is relative to
   * the save onnx file directory.
   * @param threshold The threshold value. If a size of constant initializer is
   * larger than the threshold it will be saved into the external data file.
   * Note : If the threshold is max size_t, all constant initializers will be
   * saved into ONNX model file.
   */
  void save(const std::filesystem::path &filename,
            const std::filesystem::path &external_data_file,
            size_t threshold) const;
  /**
   * @brief Save the graph to a string.
   *
   */
  morphizen::DllSafe<std::string> save_string() const;
  /**
   * @brief Retrieves a constant reference to the node at the specified index.
   *
   * @param index The index of the node to retrieve.
   * @return A constant reference to the node at the specified index.
   */
  NodeConstRef node(size_t index) const;
  /**
   * Returns a vector of NodeConstRef objects representing the nodes in the
   * graph in topological order.
   *
   * @return A vector of NodeConstRef objects in topological order.
   * @note the graph must be resolved before getting nodes
   * `nodes_in_topological_order`
   */
  std::vector<NodeConstRef> nodes_in_topological_order() const;

  /**
   * Finds the consumers of the current node arg.
   *
   * @return A vector of NodeConstRef objects representing the consumers of the
   * current node.
   */
  std::vector<NodeConstRef> find_consumers(const std::string &name) const;

  /**
   * Finds a node with the given node arg name in the graph.
   *
   * @param name The name of the one of the node's outputs
   * @return An optional reference to the found node, or std::nullopt if the
   * node is not found.
   */
  std::optional<NodeConstRef> find_node(const std::string &name) const;

  /**
   * @brief Finds a node argument with the given name.
   *
   * This function searches for a node argument with the specified name in the
   * graph. If a matching node argument is found, it is returned as an optional
   * value. If no matching node argument is found, an empty optional is
   * returned.
   *
   * @param name The name of the node argument to find.
   * @return An optional reference to the found node argument, or an empty
   * optional if not found.
   */
  std::optional<NodeArgConstRef> find_node_arg(const std::string &name) const;
  // NOTE: try_fuse and virtual_fuse have been moved to morphizen-core
  // They depend on MetaDefProto and TryFuseError which are morphizen-core types
  /**
   * @brief Converts the graph to a string representation, for debugging purpose
   *
   * @return The string representation of the graph.
   */
  std::string to_string() const;
  MORPHIZEN_DLL_SPEC friend std::ostream &
  operator<<(std::ostream &os, const GraphConstRef &graph);

  /**
   * @brief Gets the model that contains this graph
   *
   * @return Reference to the model
   */
  const morphizen::Model &model() const;

  /**
   * @brief Gets the output nodes of the graph
   *
   * @return Vector of nodes that produce graph outputs
   *
   * Returns the nodes that produce the graph's output node arguments.
   * Useful for identifying leaf nodes in the computation graph.
   */
  std::vector<NodeConstRef> output_nodes() const;

  /**
   * @brief Performs reverse DFS traversal from a node
   *
   * @param node_index Starting node index
   * @param enter Callback invoked when entering a node (return false to skip
   * subtree)
   * @param leave Callback invoked when leaving a node
   * @param comp Comparator for traversal order (optional)
   * @param subgraph_sensitive Whether to respect subgraph boundaries
   *
   * Traverses the graph in reverse DFS order starting from the given node.
   * This is useful for walking dependencies backwards from a node.
   */
  void reverse_dfs_from(
      size_t node_index,
      const std::function<bool(const morphizen::Node *)> &enter,
      const std::function<void(const morphizen::Node *)> &leave,
      const std::function<bool(const morphizen::Node *,
                               const morphizen::Node *)> &comp = nullptr,
      bool subgraph_sensitive = false) const;

  /**
   * @brief Performs reverse DFS traversal from multiple nodes
   *
   * @param nodes Vector of starting nodes
   * @param enter Callback invoked when entering a node (return false to skip
   * subtree)
   * @param leave Callback invoked when leaving a node
   * @param comp Comparator for traversal order (optional)
   * @param stop Stop condition (return true to stop traversal)
   */
  void reverse_dfs_from_multi(
      gsl::span<const NodeConstRef> nodes,
      const std::function<bool(NodeConstRef)> &enter,
      const std::function<bool(NodeConstRef)> &leave,
      const std::function<bool(NodeConstRef, NodeConstRef)> &comp,
      const std::function<bool(NodeConstRef, NodeConstRef)> &stop) const;

protected:
  /**
   * @brief Returns a non-const reference to the underlying
   * `onnxruntime::Graph` object.
   *
   * @return A non-const reference to the underlying `onnxruntime::Graph`
   * object.
   */
  onnxruntime::Graph &self() {
    return const_cast<onnxruntime::Graph &>(graph_);
  }

private:
  const morphizen::Graph &graph_;
};
/**
 * @brief A mutable version of GraphConstRef
 *
 * This class provides a wrapper around the `onnxruntime::Graph` class and
 * allows access to the underlying graph object. It also provides a conversion
 * operator to convert the `Graph` object to a const or non-const reference of
 * the `onnxruntime::Graph` class.
 *
 * This is a light weight value like object, it is not a shared object. It does
 * not own any resources. It is safe to copy and move. it is more like a
 * reference.
 */
class MORPHIZEN_DLL_SPEC GraphRef : public GraphConstRef {
public:
  /**
   * @brief Constructs a `Graph` object.
   *
   * @param graph The underlying `onnxruntime::Graph` object.
   */
  GraphRef(morphizen::Graph &graph);

  /**
   * @brief Destroys the `Graph` object.
   */
  ~GraphRef();

  /**
   * @brief Sets the name of the graph.
   *
   * This function assigns a new name to the graph object.
   *
   * @param name The new name to be set for the graph.
   */
  void set_name(const std::string &name);
  /**
   * @brief Conversion operator to convert to a reference of
   * `onnxruntime::Graph`.
   *
   * @return A reference to the underlying `onnxruntime::Graph` object.
   */
  operator onnxruntime::Graph &() { return self(); }
  /**
   * @brief Conversion operator to convert to a const reference of
   * `onnxruntime::Graph`.
   *
   * @return A const reference to the underlying `onnxruntime::Graph` object.
   */
  operator const onnxruntime::Graph &() const {
    return GraphConstRef::operator const onnxruntime::Graph &();
  }

  /**
   * @brief Resolves the graph.
   *
   * This function resolves the graph by performing necessary computations and
   * updates.
   *
   * @param force If set to true, the resolution will be forced even if it's
   * not necessary.
   * @return True if the resolution is successful, false otherwise.
   *
   * @note before a graph is properly resolved, some functions like
   * get_consumers get_producer topological_sorted_nodes() are not functional.
   * It is a heavy calculation that includes:
   *
   * 1. Build edge/node relationship
   * 2. Clean up all internal data structure and rebuild everything from
   * scratch.
   * 3. Other backend-specific internal operations
   */
  bool resolve(bool force = false);

  // NOTE: fuse and node_builder have been moved to morphizen-core
  // They depend on MetaDefProto and IPass which are morphizen-core types
  /** @brief save a graph to a file
   * this function is not a const member function, because when
   * `filter_out_special_tensor` is true, the constant initializers might be
   * replaced with the regular tensor proto, i.e. revert the optimization of
   * model clone, i.e. no weights sharing.
   */
  void mut_save(const std::filesystem::path &file_path,
                const std::filesystem::path &external_data_file,
                size_t threshold, bool filter_out_special_tensor);

  /** @brief save a graph to a string
   * this function is not a const member function, because when
   * `filter_out_special_tensor` is true, the constant initializers might be
   * replaced with the regular tensor proto, i.e. revert the optimization of
   * model clone, i.e. no weights sharing.
   */
  morphizen::DllSafe<std::string>
  mut_save_string(bool filter_out_special_tensor);
  /**
   * @brief Performs garbage collection.
   *
   * This function is responsible for deleting orphon nodes which are not used
   * by any other nodes.
   */
  void gc();
  /**
   * Creates a new constant initializer with an int8_t value.
   *
   * @param value The int8_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i8(int8_t value,
                                         const std::string &name = "");
  /**
   * Creates a new constant initializer with a uint8_t value.
   *
   * @param value The uint8_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u8(uint8_t value,
                                         const std::string &name = "");

  /**
   * Creates a new constant initializer with an int16_t value.
   *
   * @param value The int16_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i16(int16_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a uint16_t value.
   *
   * @param value The uint16_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u16(uint16_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with an int32_t value.
   *
   * @param value The int32_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i32(int32_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a uint32_t value.
   *
   * @param value The uint32_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u32(uint32_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with an int64_t value.
   *
   * @param value The int64_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i64(int64_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a uint64_t value.
   *
   * @param value The uint64_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u64(uint64_t value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a float value.
   *
   * @param value The float value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_f32(float value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a double value.
   *
   * @param value The double value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_f64(double value,
                                          const std::string &name = "");

  /**
   * Creates a new constant initializer with a bf16_t value.
   *
   * This function creates a new constant initializer with the specified bf16_t
   * value. The initializer can be used to initialize a node in a graph.
   *
   * @param value The bf16_t value to be used as the initializer.
   * @param name The name of the initializer (optional).
   * @return A reference to the created NodeArgRef object.
   */
  NodeArgRef new_constant_initializer_bf16(bf16_t value,
                                           const std::string &name = "");
  /**
   * Creates a new constant initializer with a 16-bit floating-point value.
   *
   * This function creates a new constant initializer with the specified 16-bit
   * floating-point value. The initializer can be used to initialize a node in a
   * graph.
   *
   * @param value The 16-bit floating-point value to use for the initializer.
   * @param name The name of the initializer (optional).
   * @return A reference to the created NodeArgRef object.
   */
  NodeArgRef new_constant_initializer_fp16(fp16_t value,
                                           const std::string &name = "");
  // Function declarations for creating new constant initializers with gsl::span
  // for various data types

  /**
   * @brief Creates a new constant initializer for int8_t values.gsl::span<const
   * int16_t> values,
   *
   * @param values gsl::span of int8_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef new_constant_initializer_i8_span(gsl::span<const int8_t> values,
                                              const std::vector<int64_t> &shape,
                                              const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for uint8_t values.
   *
   * @param values gsl::span of uint8_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef new_constant_initializer_u8_span(gsl::span<const uint8_t> values,
                                              const std::vector<int64_t> &shape,
                                              const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for int16_t values.
   *
   * @param values gsl::span of int16_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */

  NodeArgRef
  new_constant_initializer_i16_span(gsl::span<const int16_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for uint16_t values.
   *
   * @param values gsl::span of uint16_t values to initialize the constant with.
   * @param name Optional name for the initializer.values, const
   * std::vector<int64_t>& shape,
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_u16_span(gsl::span<const uint16_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for int32_t values.
   *
   * @param values gsl::span of int32_t values to initialize the constanvalues,
   * const std::vector<int64_t>& shape,
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_i32_span(gsl::span<const int32_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for uint32_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of uint32_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_u32_span(gsl::span<const uint32_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for int64_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of int64_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_i64_span(gsl::span<const int64_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for uint64_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of uint64_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_u64_span(gsl::span<const uint64_t> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for float values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of float values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_f32_span(gsl::span<const float> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");

  /**
   * @brief Creates a new constant initializer for double values.
   *
   * @param values gsl::span of double values to initialize thevalues, const
   * std::vector<int64_t>& shape,nt with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_f64_span(gsl::span<const double> values,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name = "");
  /**
   * Creates a new constant initializer for a graph node with a span of bf16_t
   * values.
   *
   * This function creates a new constant initializer for a graph node with a
   * span of bf16_t values. The initializer is used to initialize the node with
   * the provided values.
   *
   * @param values The span of bf16_t values to initialize the node with.
   * @param shape The shape of the node.
   * @param name The name of the node (optional).
   * @return A reference to the created NodeArg object.
   */
  NodeArgRef
  new_constant_initializer_bf16_span(gsl::span<const bf16_t> values,
                                     const std::vector<int64_t> &shape,
                                     const std::string &name = "");
  /**
   * Creates a new constant initializer for a graph node with a span of fp16_t
   * values.
   *
   * This function creates a new constant initializer for a graph node with a
   * span of fp16_t values. The initializer is used to initialize the node with
   * the given values and shape.
   *
   * @param values The span of fp16_t values to initialize the node with.
   * @param shape The shape of the node.
   * @param name The name of the node (optional).
   * @return A reference to the created NodeArg.
   */
  NodeArgRef
  new_constant_initializer_fp16_span(gsl::span<const fp16_t> values,
                                     const std::vector<int64_t> &shape,
                                     const std::string &name = "");

  /**
   * @brief Sets the inputs for the graph.
   *
   * This function sets the inputs for the graph by taking a vector of
   * `NodeConstRef` objects as input. The `NodeConstRef` objects represent the
   * input nodes of the graph.
   *
   * @param inputs A vector of `NodeConstRef` objects representing the input
   * nodes of the graph.
   */
  void set_inputs(const std::vector<NodeArgConstRef> &inputs);
  /**
   * @brief Sets the outputs of the graph.
   *
   * This function sets the outputs of the graph to the specified vector of
   * nodes. The outputs represent the nodes in the graph that produce the final
   * results.
   *
   * @param outputs The vector of nodes to set as the outputs of the graph.
   */
  void set_outputs(const std::vector<NodeArgConstRef> &outputs);
  /**
   * Creates a new NodeArgRef object with the specified name, shape, and data
   * type.
   *
   * @param name The name of the NodeArgRef object.
   * @param shape The shape of the NodeArgRef object.
   * @param data_type The data type of the NodeArgRef object.
   * @return A NodeArgRef object with the specified name, shape, and data type.
   */
  NodeArgConstRef new_node_arg(const std::string &name,
                               const std::vector<int64_t> &shape,
                               ONNX_NAMESPACE::TensorProto_DataType data_type);
  /**
   * Creates a new NodeArgRef object with the specified name, data type and
   * unknown shape.
   *
   * @param name The name of the NodeArgRef object.
   * @param data_type The data type of the NodeArgRef object.
   * @return A NodeArgRef object with the specified name, data type and unknown
   * shape.
   */
  NodeArgConstRef new_node_arg(const std::string &name,
                               ONNX_NAMESPACE::TensorProto_DataType data_type);
  /**
   * Adds a node to the graph.
   *
   * @param name The name of the node.
   * @param op_domain The domain of the operator associated with the node.
   * @param op_type The type of the operator associated with the node.
   * @param description The description of the node.
   * @param inputs The vector of input nodes connected to this node.
   * @param outputs The vector of output nodes connected to this node.
   * @param attributes The attributes associated with the node.
   * @return A reference to the newly added node.
   */
  NodeRef add_node(const std::string &name, const std::string &op_domain,
                   const std::string &op_type, const std::string &description,
                   const std::vector<std::optional<NodeArgConstRef>> &inputs,
                   const std::vector<std::optional<NodeArgConstRef>> &outputs,
                   morphizen::NodeAttributesPtr attributes);

  /** prune_special_tensor_proto
   */
  void prune_special_tensor_proto();

  /**
   * @brief Sets the graph inputs
   *
   * @param inputs Vector of node arguments to set as graph inputs
   */
  void set_inputs(const std::vector<morphizen::NodeArg *> &inputs);

  /**
   * @brief Sets the graph outputs
   *
   * @param outputs Vector of node arguments to set as graph outputs
   */
  void set_outputs(const std::vector<morphizen::NodeArg *> &outputs);

  /**
   * @brief Adds an initialized tensor to the graph
   *
   * @param tensor The tensor proto to add as an initializer
   */
  void add_initialized_tensor(const morphizen::TensorProto &tensor);
};
class Subgraph {
public:
  Subgraph(const std::vector<NodeArgConstRef> &inputs,
           const std::vector<NodeArgConstRef> &outputs,
           const std::vector<NodeConstRef> &nodes,
           const std::vector<NodeArgConstRef> &constant_initializers)
      : inputs_(inputs), outputs_(outputs), nodes_(nodes),
        constant_initializers_(constant_initializers) {}

  const std::vector<NodeArgConstRef> &inputs() const { return inputs_; }
  const std::vector<NodeArgConstRef> &outputs() const { return outputs_; }
  const std::vector<NodeConstRef> &nodes() const { return nodes_; }
  const std::vector<NodeArgConstRef> &constant_initializers() const {
    return constant_initializers_;
  }

  friend class GraphConstRef;

private:
  const std::vector<NodeArgConstRef> inputs_;
  const std::vector<NodeArgConstRef> outputs_;
  const std::vector<NodeConstRef> nodes_;
  const std::vector<NodeArgConstRef> constant_initializers_;
};
} // namespace morphizen_cxx
