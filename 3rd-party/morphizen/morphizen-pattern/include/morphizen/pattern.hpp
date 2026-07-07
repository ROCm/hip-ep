/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/**
 * @file pattern.hpp
 * @brief Usage Guide for pattern.hpp
 *
 * Overview:
 * This file is part of the MorphiZen library, focusing on pattern matching and
 * manipulation within computational graphs. It provides mechanisms to define,
 * build, and match patterns against nodes in a graph.
 *
 * Key Components:
 * - Pattern: Represents a single pattern that can be matched against graph
 * nodes.
 * - PatternBuilder: Facilitates the construction of complex patterns from
 * simpler components or JSON definitions.
 *
 * Usage:
 *
 * Creating a Pattern:
 * Patterns can be created directly or through a PatternBuilder. A
 * PatternBuilder allows for more complex pattern constructions.
 *
 * @code
 * morphizen::PatternBuilder builder;
 * auto pattern = builder.create_by_json("{...JSON representation of the
 * pattern...}");
 * @endcode
 *
 * Matching a Pattern:
 * Once a pattern is created, it can be used to match against nodes in a graph.
 *
 * @code
 * auto match_result = pattern->match(graph, node);
 * if (match_result) {
 *     // Pattern matched successfully
 * }
 * @endcode
 *
 * Python Integration:
 * If compiled with ENABLE_PYTHON, patterns can also be created from Python
 * code.
 *
 * @code
 * auto pattern = builder.create_by_py("python_code_as_string");
 * @endcode
 *
 * Note: This file also includes necessary utilities and definitions for pattern
 * matching, such as node inputs and argument handling.
 */
#pragma once
#include "morphizen/_sanity_check.hpp"
#include "morphizen/graph.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_input.hpp"
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <morphizen/my_ort.h>
#include <unordered_map>
namespace morphizen {
class RootPatternProto;
class PatternProto;
/**
 * @class Binder
 * @brief Represents matched node inputs used in pattern matching.
 *
 * Pattern::match() returns this object if a pattern is matched successfully.
 *
 * The `Binder` class is responsible for storing and retrieving node inputs
 * based on their indices or pattern names. It is used in pattern matching
 * operations to bind node inputs to specific indices or names.
 */
class MORPHIZEN_DLL_SPEC Binder {

private:
  Binder() = delete;
  Binder(const Binder&) = delete;
  Binder(Binder&&) = delete;

public:
  /**
   * Retrieves the NodeInput associated with the given pattern ID.
   *
   * @param pattern_id The ID of the pattern to retrieve.
   * @return The NodeInput associated with the pattern ID, or a
   * default-constructed NodeInput if the pattern ID is not found.
   */
  NodeInput operator[](size_t pattern_id) const {
    auto it = store_.find((int)pattern_id);
    auto ret = NodeInput{nullptr, nullptr};
    if (it == store_.end()) {
      ret = NodeInput{nullptr, nullptr};
    } else {
      ret = it->second;
    }
    return ret;
  }
  std::optional<morphizen_cxx::NodeInput> operator()(size_t pattern_id) const;

  /**
   * Retrieves the Node Name associated with the given pattern key.
   *
   * @param value The key of the pattern to retrieve.
   * @return The Node binder name associated with the pattern key, or a
   * default-constructed string if the pattern ID is not found.
   */

  std::optional<std::string> GetNodeNameFromKey(int value) const {
    if (!name_to_ids_)
      return std::nullopt; // Check if the map is valid

    for (const auto& pair : *name_to_ids_) {
      if (pair.second == value) {
        return pair.first; // Found the corresponding string
      }
    }
    return std::nullopt; // Value not found
  }

  /**
   * Retrieves the NodeInput associated with the specified pattern name.
   *
   * @param pattern_name The name of the pattern.
   * @return The NodeInput associated with the pattern name, or a default
   * NodeInput if the pattern name is not found.
   * @note PatternBuilder::bind() associate a name with a pattern. It is
   * recommended to use a unique name for each pattern. Patten with the same
   * name shaddow the previous one.
   */
  NodeInput operator[](const std::string& pattern_name) const {
    auto it = name_to_ids_->find(pattern_name);
    return it == name_to_ids_->end() ? NodeInput{nullptr, nullptr}
                                     : (*this)[it->second];
  }
  std::optional<morphizen_cxx::NodeInput>
  operator()(const std::string& pattern_name) const;
  /**
   * Returns an iterator pointing to the beginning of the map.
   *
   * @return An iterator pointing to the beginning of the map.
   * @note Togeterh with end(), it can be used to iterate over all the with
   * `for-each` statement in c++.
   * @code
   * for (const auto& [id, node_input] : binder) {
   *    ...
   * }
   * @endcond
   */
  std::map<int, NodeInput>::const_iterator begin() const {
    return store_.begin();
  };
  /**
   * Returns an iterator pointing to the past-the-end element in the container.
   *
   * This function returns an iterator to the element that follows the last
   * element of the container. It is used to indicate the end of a range when
   * iterating over the container.
   *
   * @return An iterator to the past-the-end element in the container.
   */
  std::map<int, NodeInput>::const_iterator end() const { return store_.end(); };

private:
  explicit Binder(
      std::map<int, NodeInput>&& store,
      std::shared_ptr<std::unordered_map<std::string, int>> name_to_ids,
      morphizen_cxx::GraphConstRef graph)
      : store_(store), name_to_ids_(name_to_ids), graph_{graph} {}
  std::optional<morphizen_cxx::NodeInput>
  create_morphizen_cxx_node_input(NodeInput node_input) const;

private:
  std::map<int, NodeInput> store_;
  std::shared_ptr<std::unordered_map<std::string, int>> name_to_ids_;
  morphizen_cxx::GraphConstRef graph_;
  friend class BinderBuilder;
};
using binder_t = Binder;
using binder_ptr_t = std::unique_ptr<binder_t>;
/** @class BinderBuilder
 * @brief only for internal use.
 */
class BinderBuilder;
using BinderBuilderPtr = std::unique_ptr<BinderBuilder>;
class BinderBuilder {
public:
  // only used by unique_ptr
  ~BinderBuilder();

private:
  BinderBuilder(const void* map, morphizen_cxx::GraphConstRef graph)
      : map_{map}, graph_{graph} {};
  BinderBuilder() = delete;
  binder_ptr_t build(
      const std::shared_ptr<std::unordered_map<std::string, int>>& name_to_ids)
      const;
  BinderBuilderPtr add(int id, const NodeInput& node_input) const;
  BinderBuilderPtr clone() const;
  NodeInput find(int id) const;
  friend class Pattern;
  friend class PatternWildcard;
  friend class PatternSequence;
  friend class PatternConstant;
  friend class PatternNode;
  friend class PatternCommutableNode;
  friend class PatternOr;
  friend class PatternWhere;
  friend class PatternGraphInput;
  friend class PatternNodeOutputArg;
  friend class PatternGraphOutput;

private:
  const void* map_;
  morphizen_cxx::GraphConstRef graph_;
};

/**
 * @class Pattern
 * @brief Represents a pattern used for matching nodes in a graph.
 *
 * The `Pattern` class provides a base class for defining patterns that can be
 * used to match nodes in a graph. It contains methods for enabling trace,
 * getting the pattern ID, and matching the pattern against a graph and a node.
 * Subclasses of `Pattern` can override the `match_uncached` method to define
 * their own matching logic.
 */
class Pattern {
public:
  /**
   * @brief Constructs a `Pattern` object with the specified ID.
   * @param id The unique ID of the pattern.
   * @todo change id type from `int` to `size_t` because `size_t` is more
   * friendly to be use as a index to an array, some compilers ,e.g. clang
   * reports a type-conversion warning when `int` is used an index.
   */
  explicit Pattern(int id);

  /**
   * @brief Destructor for the `Pattern` class.
   */
  virtual ~Pattern();

  /**
   * @brief Enables trace for the pattern.
   *
   * after `enable_trace`, a very verbose matching log will be printed to the
   * console. it is only used for debugging purpose.
   *
   * @param n The trace level to enable.
   */
  MORPHIZEN_DLL_SPEC static void enable_trace(int n);

  /**
   * @brief Gets the ID of the pattern.
   * @return The ID of the pattern.
   */
  int get_id() const { return id_; }

  /**
   * @brief Generates a debug string representation of the pattern.
   * @return A debug string representation of the pattern.
   * @note only used for debugging purpose.
   */
  virtual std::string debug_string() const;

  /**
   * @brief Matches the pattern against a graph and a node.
   * @param graph The graph to match against.
   * @param node The node to match against.
   * @return A `binder_ptr_t` object representing the match result. It is
   * nullptr if pattern is not matched.
   */
  MORPHIZEN_DLL_SPEC binder_ptr_t match(const onnxruntime::Graph& graph,
                                        const onnxruntime::Node& node) const;
  /**
   * @brief Matches the pattern against a NodeConstRef.
   * @param node The node to match against.
   * @return A `binder_ptr_t` object representing the match result. It is
   * nullptr if pattern is not matched.
   */
  MORPHIZEN_DLL_SPEC binder_ptr_t match(morphizen_cxx::NodeConstRef node) const;

  /**
   * Converts the object to a binary representation.
   *
   * @return A vector of characters representing the binary data.
   */
  MORPHIZEN_DLL_SPEC std::string to_binary() const;
  /**
   * Converts the object to a JSON representation.
   *
   * @return A string representing the JSON data.
   */
  MORPHIZEN_DLL_SPEC std::string to_json() const;

  /**
   * Extarct all the ops name present in pattern using recursion
   *
   * @return A vector of all ops name.
   *
   * @note it is only used by PatternInfo in morphizen. TODO: to be clarified.
   */
  MORPHIZEN_DLL_SPEC std::vector<std::string> get_ops_list_name() const;

protected:
  /**
   * Matches the given `node_input` against the `cached_binder` and returns a
   * `binder_ptr_t` object.
   *
   * This function is used to match the `node_input` against the
   * `cached_binder` in the context of the provided `graph`.
   *
   * Let's say we have two graphs as below:
   *
   *     A                 A1  A2
   *   /  \                |   |
   *  B    C               B   C
   *   \  /                \  /
   *    D                   D
   *
   *   (graph A)           (graph B)
   *
   * And we have a pattern as below:
   *
   *   A = wildcard()
   *   B = node("B", [A])
   *   C = node("C", [A])
   *   D1 = node("D", [B,C])
   *
   * D1 matches only graph A, but not graph B, because every pattern is
   * matched only once. Next match attempts will check if the matching node
   * is exactly the same as the previously matched node. If it is the exactly
   * same node, it returns true, otherwise false.
   *
   *   A1 = wildcard()
   *   A2 = wildcard()
   *   B = node("B", [A1])
   *   C = node("C", [A2])
   *   D2 = node("D", [B,C])
   *
   * D2 matches both graph A and graph B. In the case of Graph A,
   * `binder[A1->get_id()]` and `binder[A2->get_id()]` return the same
   * `NodeInput` object, however `A1->get_id()` is not equal to `A2->get_id()`.
   *
   * @param graph The onnxruntime graph to match against.
   * @param node_input The node input to match.
   * @param cached_binder The cached binder to match against.
   * @return A `BinderBuilderPtr` object representing the matched binder, or
   * `nullptr` if no match is found.
   */
  BinderBuilderPtr match_cached(const onnxruntime::Graph& graph,
                                const NodeInput& node_input,
                                const BinderBuilder& cached_binder) const;

  /**
   * @brief Generates a virtual label for the pattern.
   * @return A virtual label for the pattern.
   */
  virtual std::string virtualize_label() const;

  /**
   * @brief Matches the pattern against a graph and a node without using a
   * cached binder.
   * @param graph The graph to match against.
   * @param node_input The input node to match against.
   * @param cached_binder The cached binder to use for matching.
   * @return A `binder_ptr_t` object representing the match result.
   */
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& cached_binder) const = 0;

private:
  PatternProto* dump_to_proto(RootPatternProto& pattern_proto) const;
  virtual void dump_to_proto_imp(RootPatternProto& pattern_proto,
                                 PatternProto& this_proto) const;
  virtual void fill_ops_name(std::vector<std::string>& list_of_ops_name) const;

private:
  int id_;                      // The ID of the pattern.
  std::shared_ptr<std::unordered_map<std::string, int>>
      name_to_ids_;             // A shared pointer to a map of names to IDs.
  friend struct PatternBuilder; // Friend struct for pattern building.
  friend class PatternNode;
  friend class PatternSequence;
  friend class PatternCommutableNode;
  friend class PatternOr;
  friend class PatternWhere;
  friend class PatternNodeOutputArg;
  friend class PatternGraphOutput;
};

/**
 * @brief The PatternBuilder struct is responsible for creating and managing
 * patterns.
 *
 * The PatternBuilder struct provides methods for creating different types of
 * patterns, such as patterns created from JSON or Python, wildcard patterns,
 * node patterns, etc. It also allows binding patterns to names and retrieving
 * patterns by name or ID.
 *
 * @note When constructing a composite pattern, raw pointers are used to access
 * the sub-patterns.
 */
struct PatternBuilder {
  /**
   * @brief Constructs a new PatternBuilder object.
   */
  MORPHIZEN_DLL_SPEC PatternBuilder();

  /**
   * @brief Creates a pattern from a JSON string.
   *
   * @param pattern The JSON string representing the pattern.
   * @return std::shared_ptr<Pattern> The created pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  create_by_json(const std::string& pattern);

  /**
   * @brief Creates a pattern from a Python string.
   *
   * @param pattern The Python string representing the pattern.
   * @return std::shared_ptr<Pattern> The created pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  create_by_py(const std::string& pattern);

  /**
   * @brief Creates a Pattern object from binary data.
   *
   * This function creates a shared pointer to a Pattern object using the
   * provided binary data and its size.
   *
   * @param data A pointer to the binary data.
   * @param size The size of the binary data in bytes.
   * @return A shared pointer to the created Pattern object.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  create_from_binary(const char* data, size_t size);

  /**
   * @brief Creates a wildcard pattern.
   *
   * @return std::shared_ptr<Pattern> The created wildcard pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> wildcard();

  /**
   * @brief Creates a node pattern with two arguments.
   *
   * @param op_type_and_domain The type and domain of the node, seperated by
   * ":", for example, "com.microsoft:EPContext"
   * @param args The vector of arguments for the node.
   * @return std::shared_ptr<Pattern> The created node pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  node2(const std::string& op_type_and_domain,
        const std::vector<std::shared_ptr<Pattern>>& args);

  /**
   * @brief Creates a pattern for matching nodes with exactly 3 input arguments.
   *
   * This function constructs a pattern that matches ONNX nodes of a specific
   * operation type with exactly three input patterns. It's part of the pattern
   * matching system for identifying specific node configurations in
   * computational graphs.
   *
   * @param op_type The operation type to match (e.g., "Add", "Conv", "MatMul")
   * @param args Vector containing exactly 3 shared pointers to Pattern objects
   * representing the input patterns that must match the node's inputs
   * @param op_domain The operation domain namespace, defaults to empty for
   * standard ONNX operators
   *
   * @return std::shared_ptr<Pattern> A shared pointer to the created Pattern
   * object that can be used for graph pattern matching and transformation
   * operations
   *
   * @note This is a DLL-exported function (MORPHIZEN_DLL_SPEC) making it
   * available across module boundaries
   * @see Pattern class for more details on pattern matching capabilities
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  node2_with_optional_domain(const std::string& op_type,
                             const std::vector<std::shared_ptr<Pattern>>& args,
                             const std::string& op_domain = "");
  /**
   * @brief Creates a node pattern with three arguments.
   *
   * @param op_type_and_domain The type and domain of the node, seperated by
   * ":", for example, "com.microsoft:EPContext"
   * @param args The vector of arguments for the node.
   * @param optional_args The vector indicating whether each argument is
   * optional.
   * @return std::shared_ptr<Pattern> The created node pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  node3(const std::string& op_type_and_domain,
        const std::vector<std::shared_ptr<Pattern>>& args,
        const std::vector<bool>& optional_args);

  /**
   * @brief Creates a pattern node with specified operation type and arguments.
   *
   * @param op_type The operation type string identifying the node operation
   * @param args Vector of shared pointers to Pattern objects representing the
   * input arguments
   * @param optional_args Vector of boolean values indicating which arguments
   * are optional (true = optional, false = required)
   * @param op_domain The operation domain, defaults to empty
   * @return std::shared_ptr<Pattern> A shared pointer to the created Pattern
   * node
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  node3_with_optional_domain(const std::string& op_type,
                             const std::vector<std::shared_ptr<Pattern>>& args,
                             const std::vector<bool>& optional_args,
                             const std::string& op_domain = "");

#if MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT
  /**
   * @brief Creates a node pattern using named arguments.
   *
   * This method creates node patterns where inputs are identified by name
   * rather than position, providing clearer and more flexible pattern
   * specifications. The operation domain defaults to empty.
   *
   * Named arguments allow specification of only the inputs you need to match,
   * making patterns more maintainable when operators have many optional inputs.
   *
   * @param op_type The operation type (e.g., "Conv", "Add", "MatMul").
   * @param named_args Map of argument names to their Pattern objects.
   *                   Use nullptr to indicate an optional argument that matches
   * any input.
   * @param op_domain The operation domain (default: empty).
   * @return Shared pointer to the created node Pattern.
   *
   * @note Unspecified arguments are treated as optional. Examples:
   * @code
   * // These patterns are equivalent - both treat bias 'B' as optional
   * auto conv1 = builder.node_with_named_args("Conv", {{"X", input}, {"W",
   * weight}}); auto conv2 = builder.node_with_named_args("Conv", {{"X", input},
   * {"W", weight}, {"B", nullptr}});
   *
   * // Suffix '*' explicitly marks an argument as optional
   * auto conv3 = builder.node_with_named_args("Conv", {{"X", input}, {"W",
   * weight}, {"B*", bias_pattern}});
   * @endcode
   * @note Named argument matching needs ONNX schema to resolve argument names
   * to positions. If you don't need this feature, use positional APIs (node2,
   * node3) to avoid ONNX dependency.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> node_with_named_args(
      const std::string& op_type,
      const std::map<std::string, std::shared_ptr<Pattern>>& named_args,
      const std::string& op_domain = "");
#endif // MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT

  /**
   * @brief Creates patterns for nodes with multiple outputs.
   *
   * Many ONNX operators produce multiple outputs (e.g., LayerNormalization,
   * BatchNormalization, Split, TopK). This method creates a pattern for such
   * nodes and returns a vector of Pattern objects, one for each output,
   * allowing you to reference and use each output independently in subsequent
   * patterns.
   *
   * @param op_type The operation type string (e.g., "LayerNormalization",
   * "BatchNormalization")
   * @param args Vector of shared pointers to Pattern objects representing the
   * input arguments to the node
   * @param optional_args Vector of boolean values indicating which arguments
   * are optional (true = optional, false = required)
   * @param op_domain The operation domain (e.g., "" for standard ONNX
   * operators)
   * @param num_of_outputs The number of outputs this node produces
   *
   * @return std::vector<std::shared_ptr<Pattern>> A vector containing Pattern
   * objects for each output. The vector size equals num_of_outputs, with
   * patterns[0] representing the first output, patterns[1] the second, etc.
   *
   * @note The returned patterns can be used individually in subsequent pattern
   * matching operations.
   *
   * Example usage:
   * @code
   * morphizen::PatternBuilder builder;
   *
   * // Create input patterns
   * auto input = builder.wildcard();
   * auto scale = builder.wildcard();
   * auto bias = builder.wildcard();
   *
   * // LayerNormalization has 3 outputs: output, mean, inv_std_var
   * auto ln_outputs = builder.node_with_multiple_outputs(
   *     "LayerNormalization",
   *     {input, scale, bias},
   *     {false, false, false},  // All inputs are required
   *     "",                      // Standard ONNX domain
   *     3                        // Number of outputs
   * );
   *
   * // Use the outputs individually
   * auto ln_output = ln_outputs[0];      // Main normalized output
   * auto ln_mean = ln_outputs[1];        // Mean output
   * auto ln_inv_std = ln_outputs[2];     // Inverse std deviation output
   *
   * // Create subsequent patterns using specific outputs
   * auto next_op = builder.node2("Add", {ln_mean, builder.wildcard()});
   *
   * // Match the pattern
   * auto match_result = next_op->match(graph, node);
   * if (match_result) {
   *     // Pattern matched successfully
   *     // Access matched nodes through binder
   * }
   * @endcode
   */
  MORPHIZEN_DLL_SPEC std::vector<std::shared_ptr<Pattern>>
  node_with_multiple_outputs(const std::string& op_type,
                             const std::vector<std::shared_ptr<Pattern>>& args,
                             const std::vector<bool>& optional_args,
                             const std::string& op_domain,
                             const size_t num_of_outputs);

  /**
   * Creates a commutable node pattern.
   *
   * This function creates a commutable node pattern with the specified
   * operator type and arguments. A commutable node pattern represents a node
   * in a pattern that can be commuted, meaning the order of the arguments
   * can be swapped without changing the meaning of the pattern.
   *
   * it means that the following two patterns are equivalent:
   *
   *   1. node("Add", [A, B])
   *   2. node("Add", [B, A])
   *
   * @param op_type The operator type of the commutable node pattern.
   * @param arg1 The first argument of the commutable node pattern.
   * @param arg2 The second argument of the commutable node pattern.
   * @return A shared pointer to the created commutable node pattern.
   *
   * @note if potenally both A and B can match a same set of nodes, it
   * is recommended that set the longer pattern as the first argument
   * pattern. for example,
   *
   *     A= node("Sin", {*})
   *     B1= node("Cos", {*})
   *     B2= node("Sin", {B2})
   *
   *     P1 = node("Add", [B2, A])
   *     P2 = node("Add", [A, B2])
   *
   *     P1 is recommended.
   *
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  commutable_node(const std::string& op_type, std::shared_ptr<Pattern> arg1,
                  std::shared_ptr<Pattern> arg2);
  /**
   * @brief Creates a pattern by combining multiple patterns with an OR
   * operator.
   *
   * @param args The vector of patterns to be combined.
   * @return std::shared_ptr<Pattern> The created pattern.
   * @exprimental DO NOT USE THIS METHOD
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  Or(const std::vector<std::shared_ptr<Pattern>>& args);

  /**
   * @brief Creates a constant pattern.
   *
   * @return std::shared_ptr<Pattern> The created constant pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> constant();

  /**
   * @brief Creates a graph input pattern.
   *
   * @return std::shared_ptr<Pattern> The created graph input pattern.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> graph_input();

  /**
   * @brief Creates a pattern that matches a node used as any graph output.
   *
   * This method wraps a pattern to constrain it to match only nodes whose
   * outputs are used as graph outputs. It checks if the matched node produces
   * any of the graph's output tensors, regardless of which output index or
   * name.
   *
   * @param arg The pattern to constrain. Must match a node that produces a
   * graph output.
   *
   * @return std::shared_ptr<Pattern> A pattern that succeeds only if arg
   * matches and the matched node produces a graph output.
   *
   * Example usage:
   * @code
   * morphizen::PatternBuilder builder;
   *
   * // Match any Softmax node that is used as a graph output
   * auto softmax_input = builder.wildcard();
   * auto softmax = builder.node2("Softmax", {softmax_input});
   * auto graph_output_softmax = builder.is_graph_output(softmax);
   *
   * // This will only match if the Softmax node's output is one of the
   * // graph's output tensors
   * auto match_result = graph_output_softmax->match(graph, node);
   * if (match_result) {
   *     // The Softmax node is confirmed to be a graph output
   * }
   * @endcode
   *
   * @note This is useful for identifying nodes that produce final results,
   * which often have different optimization or preservation requirements.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  is_graph_output(const std::shared_ptr<Pattern>& arg);

  /**
   * @brief Creates a pattern that matches a node at a specific graph output
   * index.
   *
   * This method constrains a pattern to match only nodes whose outputs are used
   * as a specific graph output, identified by its index position in the graph's
   * output list.
   *
   * @param arg The pattern to constrain. Must match a node that produces the
   * graph output at the specified index.
   * @param graph_output_index The zero-based index of the graph output to
   * match. For example, 0 for the first output, 1 for the second, etc.
   *
   * @return std::shared_ptr<Pattern> A pattern that succeeds only if arg
   * matches and the matched node is at the specified graph output index.
   *
   * Example usage:
   * @code
   * morphizen::PatternBuilder builder;
   *
   * // Suppose a graph has multiple outputs: [output0, output1, output2]
   * // Match a Conv node that specifically produces the second graph output
   * // (index 1)
   * auto conv_input = builder.wildcard();
   * auto conv_weight = builder.wildcard();
   * auto conv = builder.node2("Conv", {conv_input, conv_weight});
   * auto second_output = builder.is_graph_output(conv, 1);
   *
   * // This will only match if the Conv node's output is the graph's second
   * // output (index 1)
   * auto match_result = second_output->match(graph, node);
   * if (match_result) {
   *     // The Conv node produces the second graph output
   * }
   * @endcode
   *
   * @note Useful when you need to handle specific outputs differently, such as
   * applying different quantization schemes to different outputs.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  is_graph_output(const std::shared_ptr<Pattern>& arg,
                  size_t graph_output_index);

  /**
   * @brief Creates a pattern that matches a node with a specific graph output
   * name.
   *
   * This method constrains a pattern to match only nodes whose outputs are used
   * as a graph output with a specific name. This is the most precise way to
   * identify graph outputs when output names are known.
   *
   * @param arg The pattern to constrain. Must match a node that produces the
   * graph output with the specified name.
   * @param graph_output_name The name of the graph output to match. This should
   * match the exact name as defined in the ONNX graph's output list.
   *
   * @return std::shared_ptr<Pattern> A pattern that succeeds only if arg
   * matches and the matched node produces the named graph output.
   *
   * Example usage:
   * @code
   * morphizen::PatternBuilder builder;
   *
   * // Match a Sigmoid node that produces a graph output named "probabilities"
   * auto sigmoid_input = builder.wildcard();
   * auto sigmoid = builder.node2("Sigmoid", {sigmoid_input});
   * auto prob_output = builder.is_graph_output(sigmoid, "probabilities");
   *
   * // This will only match if:
   * // 1. The pattern matches a Sigmoid node
   * // 2. The Sigmoid node's output is a graph output
   * // 3. That graph output is named "probabilities"
   * auto match_result = prob_output->match(graph, node);
   * if (match_result) {
   *     // The Sigmoid node produces the "probabilities" output
   * }
   * @endcode
   *
   * @note This is the most robust method when working with models that have
   * well-defined output names, as it's immune to changes in output ordering.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  is_graph_output(const std::shared_ptr<Pattern>& arg,
                  const std::string& graph_output_name);

  /**
   * Creates a pattern that represents a sequence of other patterns.
   *
   * @param patterns A span of shared pointers to patterns that make up the
   * sequence.
   *
   * @return A shared pointer to the created sequence pattern.
   *
   * psudeo code for implementation:
   *
   * @code
   *    auto ret = patterns[0].match(graph, current_node)
   *    for(auto i = 1; i < patterns.size(); ++i) {
   *        auto found = false;
   *        for(auto node: graph.nodes()) {
   *             found = patterns[i].match(graph, node);
   *             if(found){
   *                 break;
   *             }
   *        }
   *        ret = ret && found;
   *        if(!ret) {
   *            return false;
   *        }
   *    }
   *     return ret;
   * @endcode
   *
   * So we can see that this function is rather slow potentially, especially
   * when a graph contains many nodes and the patterns are too many also.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  sequence(gsl::span<const std::shared_ptr<Pattern>> patterns);
  /**
   * @brief Creates an XIR constant operation pattern.
   *
   * @return std::shared_ptr<Pattern> The created XIR constant operation
   * pattern.
   * @exprimental DO NOT USE THIS METHOD
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> xir_const_op();

  /**
   * @brief Binds a pattern to a name.
   *
   * @param name The name to bind the pattern to.
   * @param pat The pattern to be bound.
   */
  MORPHIZEN_DLL_SPEC void bind(const std::string& name,
                               const std::shared_ptr<Pattern>& pat);

  /**
   * @brief Gets the ID of a pattern by its name.
   *
   * @param name The name of the pattern.
   * @return int The ID of the pattern.
   */
  MORPHIZEN_DLL_SPEC int get_id(const std::string& name) const;

  /**
   * @brief Gets a pattern by its name.
   *
   * @param name The name of the pattern.
   * @return std::shared_ptr<Pattern> The pattern with the specified name.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern>
  get_pattern(const std::string& name) const;

  /**
   * @brief Gets a pattern by its ID.
   *
   * @param id The ID of the pattern.
   * @return std::shared_ptr<Pattern> The pattern with the specified ID.
   */
  MORPHIZEN_DLL_SPEC std::shared_ptr<Pattern> get_pattern(int id) const;

  /**
   * @brief Gets the bindings of patterns.
   *
   * @return std::unordered_map<std::string, int> The map of pattern names to
   * their IDs.
   */
  std::unordered_map<std::string, int> bindings() const;

private:
  /**
   * @brief Creates a pattern internally using a function.
   *
   * @param f The function to create the pattern.
   * @return std::shared_ptr<Pattern> The created pattern.
   */
  std::shared_ptr<Pattern>
  create_internal(const std::function<Pattern*(int id)>& f);

  /**
   * @brief Creates a pattern that matches a specific output of a multi-output
   * node.
   *
   * Many ONNX operators produce multiple outputs (e.g., LayerNormalization,
   * BatchNormalization, Split). This method allows you to constrain matching
   * to a specific output index.
   *
   * @param arg The node pattern whose output to constrain.
   *            Must be a Node pattern (not NodeArg).
   * @param output_arg_index The index of the output to match.
   *
   * @return A pattern that matches the specified output of the node.
   *
   * Error Handling:
   * - If output_arg_index is out of bounds, match will fail
   * - If arg is not a Node-type pattern, CHECK will fail
   * - If the node doesn't produce the specified output, match returns nullptr
   *
   * @note This method is essential when working with multi-output
   * operators to avoid ambiguity about which output is being matched.
   */
  std::shared_ptr<Pattern>
  get_node_output_arg_by_index(const std::shared_ptr<Pattern>& arg,
                               size_t output_arg_index);

private:
  std::vector<std::shared_ptr<Pattern>> patterns_;
  std::shared_ptr<std::unordered_map<std::string, int>> id_map_;

  friend struct PatternBuilderHelper;
};
} // namespace morphizen
