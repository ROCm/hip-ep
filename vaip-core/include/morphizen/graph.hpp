/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "./_sanity_check.hpp"
#include "./anchor_point.hpp"
#include "./node.hpp"
#include "./node_arg.hpp"
#include "./node_attr.hpp"
#include <cassert>
#include <filesystem>
#include <functional>
#include <optional>
#include <vaip/my_ort.h>
#include <vaip/vaip_gsl.h>
namespace vaip_core {

#ifndef VAIP_USE_DEPRECATED_API
[[deprecated("This API will be removed in the future release version. Please "
             "use NodeBuilder instread.")]]
#endif
VAIP_DLL_SPEC Node&
graph_add_node(Graph& graph, const std::string& name,
               const std::string& op_type, const std::string& description,
               const std::vector<const NodeArg*>& input_args,
               const std::vector<const NodeArg*>& output_args,
               NodeAttributesPtr attributes, const std::string& domain);

VAIP_DLL_SPEC std::vector<const NodeArg*>
node_inputs_2_node_args(const std::vector<NodeInput>& inputs);
VAIP_DLL_SPEC void graph_set_name(Graph& graph, const std::string& name);
/**
 * @brief The NodeBuilder class is responsible for building nodes in a graph.
 *
 * The NodeBuilder class provides methods for constructing nodes with various
 * properties, such as operation type, input nodes, attributes, shape, and data
 * type. It also supports adding multiple outputs and optional outputs.
 */
struct NodeBuilder {
public:
  /**
   * @brief Constructs a NodeBuilder object.
   *
   * @param graph The graph to which the node belongs.
   * @param pass The pass to which the node belongs.
   */
  VAIP_DLL_SPEC explicit NodeBuilder(Graph& graph, IPass& pass);

  /**
   * @brief Builds and returns the constructed node.
   *
   * @return The constructed node.
   */
  VAIP_DLL_SPEC const Node& build();

  /**
   * @brief Builds and returns the constructed node as a constant reference.
   *
   * @return The constructed node as a constant reference.
   */
  VAIP_DLL_SPEC vaip_cxx::NodeConstRef build_ex();

  /**
   * @brief Clones the given node and sets it as the current node being built.
   *
   * @param node The node to clone.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_node(const Node& node);

  /**
   * @brief Clones the operation type of the given node and sets it as the
   * current node being built.
   *
   * @param node The node from which to clone the operation type.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_op_type(const Node& node);

  /**
   * @brief Sets the operation type of the current node being built.
   *
   * @param op_type The operation type.
   * @param domain The domain of the operation type (default: "com.xilinx").
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder&
  set_op_type(const std::string& op_type,
              const std::string& domain = "com.xilinx");

  /**
   * @brief Clones the inputs of the given node and sets them as the inputs of
   * the current node being built.
   *
   * @param node The node from which to clone the inputs.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_inputs(const Node& node);

  /**
   * @brief Sets the input nodes of the current node being built.
   *
   * @param input_nodes The input nodes.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder&
  set_input_nodes(const std::vector<const Node*>& input_nodes);

  /**
   * @brief Sets the input node arguments of the current node being built.
   *
   * @param input_args The input node arguments.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder&
  set_input_node_args(const std::vector<const NodeArg*>& input_args);

  /**
   * @brief Sets the input node arguments for the NodeBuilder.
   *
   * This function sets the input node arguments for the NodeBuilder. The input
   * arguments are provided as a vector of NodeArgConstRef objects.
   *
   * @param input_args The vector of NodeArgConstRef objects representing the
   * input node arguments.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_input_node_args_ex(
      const std::vector<vaip_cxx::NodeArgConstRef>& input_args);
  /**
   * @brief Appends the given node as an input to the current node being built.
   *
   * @param node The node to append as an input.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& append_input(const Node& node);

  /**
   * @brief Clones the attributes of the given node and sets them as the
   * attributes of the current node being built.
   *
   * @param node The node from which to clone the attributes.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_attrs(const Node& node);

  /**
   * @brief Returns the NodeAttributesBuilder object for modifying the
   * attributes of the current node being built.
   *
   * @return The NodeAttributesBuilder object.
   */
  VAIP_DLL_SPEC NodeAttributesBuilder& get_attrs_builder();

  /**
   * @brief Clones the shape of the given node and sets it as the shape of the
   * current node being built.
   *
   * @param node The node from which to clone the shape.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_shape(const Node& node);

  /**
   * @brief Sets the shape of the current node being built.
   *
   * @param shape The shape of the node.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_shape(const gsl::span<const int64_t>& shape);

  /**
   * @brief Clones the shape of the given node argument and sets it as the shape
   * of the current node being built.
   *
   * @param node_arg The node argument from which to clone the shape.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_shape(const NodeArg& node_arg);

  /**
   * @brief Clones the data type of the given node and sets it as the data type
   * of the current node being built.
   *
   * @param node The node from which to clone the data type.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_data_type(const Node& node);

  /**
   * @brief Clones the data type of the given node argument and sets it as the
   * data type of the current node being built.
   *
   * @param node_arg The node argument from which to clone the data type.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& clone_data_type(const NodeArg& node);

  /**
   * @brief Sets the data type of the current node being built.
   *
   * @param data_type The data type.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_data_type(const std::string& data_type);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node.
   *
   * @param node The node to set as the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_anchor_point1(const Node& node);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node argument.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_anchor_point1(const NodeArg& node);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node argument and description.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder&
  set_anchor_point2(const NodeArg& node_arg,
                    const AnchorPoint::Description& description);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node, description, and shape.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @param shape The shape of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder&
  set_anchor_point3(const NodeArg& node_arg,
                    const AnchorPoint::Description& description,
                    const std::vector<int64_t>& shape);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node, description, shape, and data type.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @param shape The shape of the anchor point.
   * @param data_type The data type of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& set_anchor_point4(
      const NodeArg& node_arg, const AnchorPoint::Description& description,
      const std::vector<int64_t>& shape, const std::string& data_type);

  /**
   * @brief Adds an attribute with the given name and value to the current node
   * being built.
   *
   * @tparam T The type of the attribute value.
   * @param name The name of the attribute.
   * @param value The value of the attribute.
   * @return A reference to the NodeBuilder object.
   */
  template <typename T> NodeBuilder& add(const std::string& name, T&& value) {
    if (name == "data_type" || name == "shape") {
      assert(false &&
             "data_type and shape are deprecated, please use set_anchor_point, "
             "clone_shape, set_shape, clone_data_type, set_data_type instead");
    }
    attrs_builder_.add(name, std::forward<T>(value));
    return *this;
  }

  /**
   * @brief Adds an output to the current node being built.
   *
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& add_output();

  /**
   * @brief Skips the optional output of the current node being built.
   *
   * @return A reference to the NodeBuilder object.
   */
  VAIP_DLL_SPEC NodeBuilder& skip_optional_output();

private:
  Graph& graph_;
  IPass* pass_;
  std::string op_type_;
  std::string description_;
  std::vector<const NodeArg*> input_args_;
  NodeAttributesPtr attrs_;
  NodeAttributesBuilder attrs_builder_;
  std::string domain_;
  size_t num_of_outputs_ = 1u;
  std::vector<std::vector<int64_t>> shape_;
  std::vector<std::string> data_type_;
  std::vector<std::unique_ptr<AnchorPoint>> anchor_point_;
  std::vector<std::optional<vaip_cxx::NodeArgConstRef>> anchor_node_arg_;
  // Cache producer nodes
  // After node addition in MLIR, original nodes become inaccessible via
  // graph_producer_node
  std::vector<std::optional<vaip_cxx::NodeConstRef>> anchor_producer_node_;
};

const Model& graph_get_model(const Graph& graph);
std::vector<const Node*> graph_nodes(const Graph& graph);
std::vector<const NodeArg*> graph_get_inputs(const Graph& graph);
VAIP_DLL_SPEC std::vector<const NodeArg*> graph_get_outputs(const Graph& graph);
VAIP_DLL_SPEC std::vector<const Node*>
graph_get_output_nodes(const Graph& graph);

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
 *        auto node = VAIP_ORT_API(graph_get_node)(graph, node_idx);
 *        if (node == nullptr) { // should never goes here
 *             cout << node_as_string(*node);
 *        }
 *    }
 */
VAIP_DLL_SPEC std::vector<size_t>
graph_get_node_in_topoligical_order(const Graph& graph);

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
VAIP_DLL_SPEC std::string graph_as_string(const Graph& graph);
std::vector<const Node*>
graph_get_consumer_nodes(const Graph& graph, const std::string& node_arg_name);

/** @brief garbage collection by removing dangling nodes.
 *
 *  @param graph
 *
 *  usuaully a pass writer does not need to invoke this function
 *  explicitly, in vaip_config.json, we can set `enableGc=true` to
 *  automatically apply garbage collection.
 *
 *  Sometime it is useful to disable gc for troubleshooting.
 *
 */
VAIP_DLL_SPEC void graph_gc(Graph& graph);

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
 *  After invoke VAIP_ORT_API(add_node) or VAIP_ORT_API(remove_node),
 *  the internal data structure becomes invalid, sometimes,
 *  `graph_get_consumer_nodes` or other funtions cannot return proper
 *  values until we inoke `graph_resolve`.
 *
 *  `Pass::apply(...)` and `Pass::fuse` invoke `graph_resolve`.
 *
 */
VAIP_DLL_SPEC void graph_resolve(Graph& graph, bool force = false);

/** @brief replace a node arg.
 *
 *  this function searchese for all consumers of node arg `from`, and
 *  make these consumers use `to` instead of `from`, it changes the
 *  topological structure of a graph and potentially introduces cyclic
 *  dependency. `graph_resolve` is invoked afterward, `graph_resolve`
 *  throw an exception if a cyclic dependency is detected.
 *
 *  @note this function is not stable. please use it with caution.
 */
VAIP_DLL_SPEC void graph_replace_node_arg(const Graph& graph, const IPass& pass,
                                          const NodeArg& from,
                                          const NodeArg& to);

} // namespace vaip_core

namespace vaip_cxx {
class Subgraph;
class NodeRef;
/**
 * @class GraphConstRef
 * @brief A reference wrapper to a constant `onnxruntime::Graph` object. It is
 * CopyConstructiable and MoveConstructiable, so that for example it can be put
 * into a vector.
 */
class VAIP_DLL_SPEC GraphConstRef {
public:
  /**
   * @brief Constructs a `GraphConstRef` object.
   *
   * @param graph The underlying `vaip_core::Graph` object.
   */
  GraphConstRef(const vaip_core::Graph& graph) : graph_(graph) {}

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
  bool operator==(const GraphConstRef& other) const {
    return &graph_ == &other.graph_;
  }
  /**
   * @brief Gets the name of the graph.
   *
   * @return The name of the graph.
   */
  const std::string& name() const;
  /**
   * Returns the path to the model.
   *
   * @return An optional containing the path to the model, or an empty path
   * if model is loaded from memory.
   */
  const std::filesystem::path& model_path() const;
  /**
   * @brief Conversion operator to convert to a const reference of
   * `onnxruntime::Graph`.
   *
   * @return A const reference to the underlying `onnxruntime::Graph` object.
   */
  operator const onnxruntime::Graph&() const { return graph_; }

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
  void save(const std::filesystem::path& filename) const;
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
  void save(const std::filesystem::path& filename,
            const std::filesystem::path& external_data_file,
            size_t threshold) const;
  /**
   * @brief Save the graph to a string.
   *
   */
  vaip_core::DllSafe<std::string> save_string() const;
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
  std::vector<NodeConstRef> find_consumers(const std::string& name) const;

  /**
   * Finds a node with the given node arg name in the graph.
   *
   * @param name The name of the one of the node's outputs
   * @return An optional reference to the found node, or std::nullopt if the
   * node is not found.
   */
  std::optional<NodeConstRef> find_node(const std::string& name) const;

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
  std::optional<NodeArgConstRef> find_node_arg(const std::string& name) const;
  /**
   * Tries to fuse the specified operation into the graph.
   *
   * @param name The name of the operation to fuse.
   * @param inputs The names of the input tensors.
   * @param outputs The names of the output tensors.
   * @param constant_initializers The names of the constant initializers.
   * @param device The device to execute the fused operation on.
   * @return A pair containing a unique pointer to the fused operation's
   * metadata definition, i.e. MetaDefProto and an error code indicating the
   * result of the fusion attempt.
   *
   * If MetaDefProto is nullptr, vaip_core::TryFuseError contains more details
   * about why it is failed.
   *
   * MetaDefProto can be passed to GraphRef::fuse() to change the actual graph.
   *
   */
  std::pair<std::unique_ptr<vaip_core::MetaDefProto>, vaip_core::TryFuseError>
  try_fuse(const std::string& name, const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::vector<std::string>& constant_initializers,
           const std::string& device) const;
  /**
   * @brief Fuses the subgraph based on the given meta definition.
   *
   * This function takes a `MetaDefProto` object as input and fuses the subgraph
   * based on the provided meta definition. The fused subgraph is returned as a
   * `Subgraph` object.
   *
   * @param meta_def The meta definition used for fusing the subgraph.
   * @return The fused subgraph as a `Subgraph` object.
   */
  Subgraph virtual_fuse(const vaip_core::MetaDefProto& meta_def) const;
  /**
   * @brief Converts the graph to a string representation, for debugging purpose
   *
   * @return The string representation of the graph.
   */
  std::string to_string() const;
  VAIP_DLL_SPEC friend std::ostream& operator<<(std::ostream& os,
                                                const GraphConstRef& graph);

protected:
  /**
   * @brief Returns a non-const reference to the underlying
   * `onnxruntime::Graph` object.
   *
   * @return A non-const reference to the underlying `onnxruntime::Graph`
   * object.
   */
  onnxruntime::Graph& self() { return const_cast<onnxruntime::Graph&>(graph_); }

private:
  const vaip_core::Graph& graph_;
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
class VAIP_DLL_SPEC GraphRef : public GraphConstRef {
public:
  /**
   * @brief Constructs a `Graph` object.
   *
   * @param graph The underlying `onnxruntime::Graph` object.
   */
  GraphRef(vaip_core::Graph& graph);

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
  void set_name(const std::string& name);
  /**
   * @brief Conversion operator to convert to a reference of
   * `onnxruntime::Graph`.
   *
   * @return A reference to the underlying `onnxruntime::Graph` object.
   */
  operator onnxruntime::Graph&() { return self(); }
  /**
   * @brief Conversion operator to convert to a const reference of
   * `onnxruntime::Graph`.
   *
   * @return A const reference to the underlying `onnxruntime::Graph` object.
   */
  operator const onnxruntime::Graph&() const {
    return GraphConstRef::operator const onnxruntime::Graph&();
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
   * It  is a heavy calculation includes
   *
   * 1. Shape inference
   * 2. Build edge/node relationship
   * 3. clean up all internal data structure and rebuild everything from
   * sratch.
   * 4. Others
   */
  bool resolve(bool force = false);

  /**
   * Fuses the given `meta_def` into a `NodeRef`.
   *
   * @param meta_def The `MetaDefProto` to fuse.
   * @return The fused `NodeRef`.
   *
   * MetaDefProto is a protobuf message that represents a subgraph in the graph.
   */
  NodeRef fuse(const vaip_core::MetaDefProto& meta_def);
  /**
   * @brief Creates a NodeBuilder object.
   *
   * This function creates and returns a NodeBuilder object, which is used to
   * build nodes in the graph.
   *
   * @param pass The pass object to use for building the node.
   * @return The created NodeBuilder object.
   */
  vaip_core::NodeBuilder node_builder(vaip_core::IPass& pass);
  /** @brief save a graph to a file
   * this function is not a const member function, because when
   * `filter_out_special_tensor` is true, the constant initializers might be
   * replaced with the regular tensor proto, i.e. revert the optimization of
   * model clone, i.e. no weights sharing.
   */
  void mut_save(const std::filesystem::path& file_path,
                const std::filesystem::path& external_data_file,
                size_t threshold, bool filter_out_special_tensor);

  /** @brief save a graph to a string
   * this function is not a const member function, because when
   * `filter_out_special_tensor` is true, the constant initializers might be
   * replaced with the regular tensor proto, i.e. revert the optimization of
   * model clone, i.e. no weights sharing.
   */
  vaip_core::DllSafe<std::string>
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
                                         const std::string& name = "");
  /**
   * Creates a new constant initializer with a uint8_t value.
   *
   * @param value The uint8_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u8(uint8_t value,
                                         const std::string& name = "");

  /**
   * Creates a new constant initializer with an int16_t value.
   *
   * @param value The int16_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i16(int16_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with a uint16_t value.
   *
   * @param value The uint16_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u16(uint16_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with an int32_t value.
   *
   * @param value The int32_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i32(int32_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with a uint32_t value.
   *
   * @param value The uint32_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u32(uint32_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with an int64_t value.
   *
   * @param value The int64_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_i64(int64_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with a uint64_t value.
   *
   * @param value The uint64_t value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_u64(uint64_t value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with a float value.
   *
   * @param value The float value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_f32(float value,
                                          const std::string& name = "");

  /**
   * Creates a new constant initializer with a double value.
   *
   * @param value The double value to initialize the constant with.
   * @param name The name of the constant initializer. The name must be unique.
   * If it is empty, a unique name is automatically generated.
   * @return A NodeArgRef representing the new constant initializer.
   */
  NodeArgRef new_constant_initializer_f64(double value,
                                          const std::string& name = "");

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
                                           const std::string& name = "");
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
                                           const std::string& name = "");
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
                                              const std::vector<int64_t>& shape,
                                              const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for uint8_t values.
   *
   * @param values gsl::span of uint8_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef new_constant_initializer_u8_span(gsl::span<const uint8_t> values,
                                              const std::vector<int64_t>& shape,
                                              const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for int16_t values.
   *
   * @param values gsl::span of int16_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */

  NodeArgRef
  new_constant_initializer_i16_span(gsl::span<const int16_t> values,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

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
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

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
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for uint32_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of uint32_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_u32_span(gsl::span<const uint32_t> values,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for int64_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of int64_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_i64_span(gsl::span<const int64_t> values,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for uint64_t values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of uint64_t values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_u64_span(gsl::span<const uint64_t> values,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

  /**
   * @brief Creates a new constant initializer for float values.
   *values, const std::vector<int64_t>& shape,
   * @param values gsl::span of float values to initialize the constant with.
   * @param name Optional name for the initializer.
   * @return NodeArgRef Reference to the created node argument.
   */
  NodeArgRef
  new_constant_initializer_f32_span(gsl::span<const float> values,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");

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
                                    const std::vector<int64_t>& shape,
                                    const std::string& name = "");
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
                                     const std::vector<int64_t>& shape,
                                     const std::string& name = "");
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
                                     const std::vector<int64_t>& shape,
                                     const std::string& name = "");

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
  void set_inputs(const std::vector<NodeArgConstRef>& inputs);
  /**
   * @brief Sets the outputs of the graph.
   *
   * This function sets the outputs of the graph to the specified vector of
   * nodes. The outputs represent the nodes in the graph that produce the final
   * results.
   *
   * @param outputs The vector of nodes to set as the outputs of the graph.
   */
  void set_outputs(const std::vector<NodeArgConstRef>& outputs);
  /**
   * Creates a new NodeArgRef object with the specified name, shape, and data
   * type.
   *
   * @param name The name of the NodeArgRef object.
   * @param shape The shape of the NodeArgRef object.
   * @param data_type The data type of the NodeArgRef object.
   * @return A NodeArgRef object with the specified name, shape, and data type.
   */
  NodeArgConstRef new_node_arg(const std::string& name,
                               const std::vector<int64_t>& shape,
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
  NodeArgConstRef new_node_arg(const std::string& name,
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
  NodeRef add_node(const std::string& name, const std::string& op_domain,
                   const std::string& op_type, const std::string& description,
                   const std::vector<std::optional<NodeArgConstRef>>& inputs,
                   const std::vector<std::optional<NodeArgConstRef>>& outputs,
                   vaip_core::NodeAttributesPtr attributes);

  /** prune_special_tensor_proto
   */
  void prune_special_tensor_proto();
};
class Subgraph {
public:
  const std::vector<NodeArgConstRef>& inputs() const { return inputs_; }
  const std::vector<NodeArgConstRef>& outputs() const { return outputs_; }
  const std::vector<NodeConstRef>& nodes() const { return nodes_; }
  const std::vector<NodeArgConstRef>& constant_initializers() const {
    return constant_initializers_;
  }

private:
  Subgraph(const std::vector<NodeArgConstRef>& inputs,
           const std::vector<NodeArgConstRef>& outputs,
           const std::vector<NodeConstRef>& nodes,
           const std::vector<NodeArgConstRef>& constant_initializers)
      : inputs_(inputs), outputs_(outputs), nodes_(nodes),
        constant_initializers_(constant_initializers) {}

  friend class GraphConstRef;

private:
  const std::vector<NodeArgConstRef> inputs_;
  const std::vector<NodeArgConstRef> outputs_;
  const std::vector<NodeConstRef> nodes_;
  const std::vector<NodeArgConstRef> constant_initializers_;
};
} // namespace vaip_cxx
