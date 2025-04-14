/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
/**
 * @file pattern.hpp
 * @brief Usage Guide for pattern.hpp
 *
 * Overview:
 * This file is part of the VAIP library, focusing on pattern matching and
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
 * vaip_core::PatternBuilder builder;
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
#include "./_sanity_check.hpp"
#include "graph.hpp"
#include "node.hpp"
#include "node_input.hpp"
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vaip/my_ort.h>
namespace vaip_core {
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
class VAIP_DLL_SPEC Binder {

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
  std::optional<vaip_cxx::NodeInput> operator()(size_t pattern_id) const;

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
  std::optional<vaip_cxx::NodeInput>
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
      vaip_cxx::GraphConstRef graph)
      : store_(store), name_to_ids_(name_to_ids), graph_{graph} {}
  std::optional<vaip_cxx::NodeInput>
  create_vaip_cxx_node_input(NodeInput node_input) const;

private:
  std::map<int, NodeInput> store_;
  std::shared_ptr<std::unordered_map<std::string, int>> name_to_ids_;
  vaip_cxx::GraphConstRef graph_;
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
  BinderBuilder(const void* map, vaip_cxx::GraphConstRef graph)
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

private:
  const void* map_;
  vaip_cxx::GraphConstRef graph_;
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
  VAIP_DLL_SPEC static void enable_trace(int n);

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
  VAIP_DLL_SPEC binder_ptr_t match(const onnxruntime::Graph& graph,
                                   const onnxruntime::Node& node) const;
  /**
   * @brief Matches the pattern against a NodeConstRef.
   * @param node The node to match against.
   * @return A `binder_ptr_t` object representing the match result. It is
   * nullptr if pattern is not matched.
   */
  VAIP_DLL_SPEC binder_ptr_t match(vaip_cxx::NodeConstRef node) const;

  /**
   * Converts the object to a binary representation.
   *
   * @return A vector of characters representing the binary data.
   */
  VAIP_DLL_SPEC std::string to_binary() const;
  /**
   * Converts the object to a JSON representation.
   *
   * @return A string representing the JSON data.
   */
  VAIP_DLL_SPEC std::string to_json() const;

  /**
   * Extarct all the ops name present in pattern using recursion
   *
   * @return A vector of all ops name.
   */
  VAIP_DLL_SPEC std::vector<std::string> get_ops_list_name() const;

  /**
   * @brief Matches the pattern against a graph and a node using a cached
   * binder.
   * @param graph The graph to match against.
   * @param node_input The input node to match against.
   * @param cached_binder The cached binder to use for matching.
   * @return A `binder_ptr_t` object representing the match result.
   */
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
  VAIP_DLL_SPEC PatternBuilder();

  /**
   * @brief Creates a pattern from a JSON string.
   *
   * @param pattern The JSON string representing the pattern.
   * @return std::shared_ptr<Pattern> The created pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  create_by_json(const std::string& pattern);

  /**
   * @brief Creates a pattern from a Python string.
   *
   * @param pattern The Python string representing the pattern.
   * @return std::shared_ptr<Pattern> The created pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
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
  VAIP_DLL_SPEC std::shared_ptr<Pattern> create_from_binary(const char* data,
                                                            size_t size);

  /**
   * @brief Creates a wildcard pattern.
   *
   * @return std::shared_ptr<Pattern> The created wildcard pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern> wildcard();

  /**
   * @brief Creates a node pattern with two arguments.
   *
   * @param op_type The type of the node.
   * @param args The vector of arguments for the node.
   * @return std::shared_ptr<Pattern> The created node pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  node2(const std::string& op_type,
        const std::vector<std::shared_ptr<Pattern>>& args);

  /**
   * @brief Creates a node pattern with three arguments.
   *
   * @param op_type The type of the node.
   * @param args The vector of arguments for the node.
   * @param optional_args The vector indicating whether each argument is
   * optional.
   * @return std::shared_ptr<Pattern> The created node pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  node3(const std::string& op_type,
        const std::vector<std::shared_ptr<Pattern>>& args,
        const std::vector<bool>& optional_args);

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
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
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
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  Or(const std::vector<std::shared_ptr<Pattern>>& args);

  /**
   * @brief Creates a constant pattern.
   *
   * @return std::shared_ptr<Pattern> The created constant pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern> constant();

  /**
   * @brief Creates a graph input pattern.
   *
   * @return std::shared_ptr<Pattern> The created graph input pattern.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern> graph_input();
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
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  sequence(gsl::span<const std::shared_ptr<Pattern>> patterns);
  /**
   * @brief Creates an XIR constant operation pattern.
   *
   * @return std::shared_ptr<Pattern> The created XIR constant operation
   * pattern.
   * @exprimental DO NOT USE THIS METHOD
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern> xir_const_op();

  /**
   * @brief Binds a pattern to a name.
   *
   * @param name The name to bind the pattern to.
   * @param pat The pattern to be bound.
   */
  VAIP_DLL_SPEC void bind(const std::string& name,
                          const std::shared_ptr<Pattern>& pat);

  /**
   * @brief Gets the ID of a pattern by its name.
   *
   * @param name The name of the pattern.
   * @return int The ID of the pattern.
   */
  VAIP_DLL_SPEC int get_id(const std::string& name) const;

  /**
   * @brief Gets a pattern by its name.
   *
   * @param name The name of the pattern.
   * @return std::shared_ptr<Pattern> The pattern with the specified name.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern>
  get_pattern(const std::string& name) const;

  /**
   * @brief Gets a pattern by its ID.
   *
   * @param id The ID of the pattern.
   * @return std::shared_ptr<Pattern> The pattern with the specified ID.
   */
  VAIP_DLL_SPEC std::shared_ptr<Pattern> get_pattern(int id) const;

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

private:
  std::vector<std::shared_ptr<Pattern>> patterns_;
  std::shared_ptr<std::unordered_map<std::string, int>> id_map_;
};
} // namespace vaip_core