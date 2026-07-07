/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file node-index.hpp
 * @brief Efficient node indexing system for ONNX graph manipulation
 *
 * This file defines the NodeIndex class, which provides a compact and efficient
 * way to reference nodes within ONNX computational graphs. The design
 * prioritizes:
 *
 * - **Performance**: 64-bit compact representation for cache efficiency
 * - **Safety**: Graph-aware indexing prevents cross-graph reference errors
 * - **Usability**: Rich API for accessing node properties and relationships
 * - **Compatibility**: Support for legacy morphizen core integration
 *
 * Key Design Principles:
 * - Zero-cost abstractions: No heap allocation, minimal overhead
 * - Type safety: Strong typing prevents index misuse
 * - Graph context: Every node reference knows its graph
 * - Extensibility: Clean interface for future enhancements
 *
 * @author MorphiZen Development Team
 * @version 1.0
 * @since 2023
 */

#pragma once
#include "graph-id.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

// Forward declarations for minimal header dependencies
namespace google {
namespace protobuf {
template <typename T> class RepeatedPtrField;
}
} // namespace google

namespace morphizen_onnx {
class AttributeProto;
class NodeProto;
} // namespace morphizen_onnx

namespace morphizen {

// Forward declarations
class NodeArgIndex;
class Graph;
class Node;
/**
 * @brief Index for referencing nodes in a graph
 *
 * NodeIndex provides a lightweight, efficient way to reference nodes within
 * ONNX graphs. It stores both a node index and graph ID in a compact 64-bit
 * representation, enabling fast lookups while maintaining graph context.
 *
 * Key features:
 * - Compact 64-bit storage (31-bit index + 1-bit validity + 32-bit graph ID)
 * - Graph-aware referencing to prevent cross-graph confusion
 * - Efficient equality comparison and hashing for use in containers
 * - Rich API for accessing node properties and relationships
 * - Support for both valid and invalid/sentinel values
 *
 * Usage example:
 * @code
 *   Graph graph = ...;
 *   auto nodes = graph.nodes_unsafe();
 *   NodeIndex first_node = nodes[0];
 *   if (first_node.is_valid(graph)) {
 *     std::string op_type = first_node.get_node_op_type();
 *     auto inputs = first_node.get_input_node_args();
 *   }
 * @endcode
 */
class NodeIndex {
public:
  // === Static Factory Methods ===

  /**
   * @brief Create NodeIndex from legacy morphizen core node pointer
   * @param ptr Legacy void pointer from morphizen core system
   * @return NodeIndex corresponding to the legacy pointer
   * @note Used for backward compatibility with existing morphizen core APIs
   */
  static NodeIndex from_morphizen_core_node_ptr(const void *ptr);

  /**
   * @brief Create NodeIndex from legacy morphizen core node index
   * @param index Legacy size_t index from morphizen core system
   * @return NodeIndex corresponding to the legacy index
   * @note Used for backward compatibility with existing morphizen core APIs
   * that use size_t indices
   */
  static NodeIndex from_morphizen_core_node_index(size_t index);

  /**
   * @brief Create an invalid NodeIndex
   * @return NodeIndex with invalid state, suitable as sentinel value
   */
  static NodeIndex invalid();

  // === Constructors and Destructor ===

  /**
   * @brief Default constructor - creates invalid NodeIndex
   */
  NodeIndex();

  /**
   * @brief Construct NodeIndex with specific index and graph ID
   * @param index Node index within the graph (0-based)
   * @param graph_id ID of the graph containing this node (default: 0)
   */
  explicit NodeIndex(unsigned int index, GraphId graph_id);

  // Rule of Five (copy/move semantics)
  NodeIndex(const NodeIndex &other) = default;
  NodeIndex(NodeIndex &&other) noexcept = default;
  NodeIndex &operator=(const NodeIndex &other) = default;
  NodeIndex &operator=(NodeIndex &&other) noexcept = default;
  ~NodeIndex() = default;

  // === Validity and Comparison ===

  /**
   * @brief Check if this NodeIndex is valid within a specific graph
   * @param graph The graph to validate against
   * @return true if index is valid and references a node in the graph
   */
  bool is_valid(const Graph &graph) const;

  /**
   * @brief Check if this NodeIndex has valid internal state
   * @return true if the validity flag is set (basic sanity check)
   */
  bool is_valid() const;

  /**
   * @brief Equality comparison
   * @param other NodeIndex to compare with
   * @return true if both index and graph_id match
   */
  bool operator==(const NodeIndex &other) const;

  /**
   * @brief Inequality comparison
   * @param other NodeIndex to compare with
   * @return true if either index or graph_id differs
   */
  bool operator!=(const NodeIndex &other) const;

  /**
   * @brief Compute hash value for use in hash tables
   * @return Hash value based on index and graph_id
   */
  std::size_t hash() const;

  // === Basic Property Accessors ===

  /**
   * @brief Get the node index within its graph
   * @return 0-based index of the node
   */
  unsigned int get_index() const;

  /**
   * @brief Get the ID of the graph containing this node
   * @return Graph ID for context
   */
  GraphId get_graph_id() const;

  /**
   * @brief Check if this node is a fused node (contains subgraph)
   * @return true if node represents a fused operation with embedded graph
   */
  bool is_fused_node() const;

  // === Legacy Compatibility ===

  /**
   * @brief Convert to legacy morphizen core node pointer
   * @return void pointer compatible with legacy morphizen core APIs
   * @note Used for backward compatibility
   */
  const void *to_morphizen_core_node_ptr() const;

  // === Node Property Access ===
  /**
   * @brief Get the operation type of this node
   * @return ONNX operation type (e.g., "Conv", "Relu", "MatMul")
   */
  const std::string &get_node_op_type() const;

  /**
   * @brief Get the operation domain of this node
   * @return Domain string (e.g., "ai.onnx", "com.microsoft")
   */
  const std::string &get_node_op_domain() const;

  /**
   * @brief Get the name of this node
   * @return Node name as specified in the ONNX graph
   */
  const std::string &get_name() const;

  /**
   * @brief Get the description/documentation of this node
   * @return Human-readable description from doc_string field
   */
  const std::string &get_description() const;

  // === Node Relationship Access ===

  /**
   * @brief Get all input node arguments for this node
   * @return Vector of NodeArgIndex representing inputs
   * @note Returns empty vector if node has no inputs
   */
  const std::vector<NodeArgIndex> &get_input_node_args() const;

  /**
   * @brief Get all output node arguments produced by this node
   * @return Vector of NodeArgIndex representing outputs
   * @note Returns empty vector if node has no outputs
   */
  const std::vector<NodeArgIndex> &get_output_node_args() const;

  /**
   * @brief Get ONNX attributes associated with this node
   * @return Pointer to protobuf repeated field of attributes, or nullptr if
   * none
   * @note Attributes contain operation-specific parameters (e.g., kernel_size
   * for Conv)
   */
  const ::google::protobuf::RepeatedPtrField<::morphizen_onnx::AttributeProto> *
  get_attributes() const;
  /**
   * @brief Retrieves the function body associated with the current node.
   *
   * @return A pointer to the Graph object representing the function body,
   *         or nullptr if no function body is present.
   */
  const Graph *get_function_body() const;
  // === Debug and String Representation ===

  /**
   * @brief Create string representation for debugging
   * @return Human-readable string with index and graph ID
   * @note Format: "NodeIndex(index=X, graph_id=Y)" or "NodeIndex(invalid)"
   */
  std::string to_string() const;

public:
  /**
   * @brief Get reference to the underlying node protobuf
   * @return Reference to the NodeProto for this node
   * @throws std::runtime_error if node index is invalid or graph not found
   */
  const morphizen_onnx::NodeProto &get_node_proto() const;

private:
  /**
   * @brief Get reference to the underlying Node object
   * @return Reference to the Node for this node
   * @throws std::runtime_error if node index is invalid or graph not found
   */
  const Node &get_node() const;

  /**
   * @brief Compact storage for node index and metadata
   *
   * Uses a union to efficiently pack data into 64 bits:
   * - index_: 31 bits for node index (supports ~2B nodes per graph)
   * - is_valid_: 1 bit validity flag
   * - graph_id_: 32 bits for graph ID (supports ~4B graphs)
   *
   * This design provides:
   * - Cache-friendly 8-byte size
   * - Fast copying and comparison
   * - No heap allocation
   * - Type safety through the union
   */
  union {
    struct {
      unsigned int index_ : 31;    // Node index within the graph (0-based)
      unsigned int is_valid_ : 1;  // Validity flag (0=invalid, 1=valid)
      unsigned int graph_id_ : 32; // Graph ID for context identification
    } fields_;
    uint64_t value_; // Raw 64-bit value for fast operations
  };
};

/**
 * @brief Compile-time size validation
 *
 * Ensures NodeIndex maintains exactly 64-bit size for:
 * - Optimal cache performance
 * - Predictable memory layout
 * - Fast copying and comparison
 * - ABI stability across platforms
 */
static_assert(sizeof(NodeIndex) == 8,
              "NodeIndex should be exactly 8 bytes (64 bits)");

} // namespace morphizen

/**
 * @brief Standard library hash specialization for NodeIndex
 *
 * Enables NodeIndex to be used as a key in standard hash containers
 * like std::unordered_map, std::unordered_set, etc.
 *
 * Example usage:
 * @code
 *   std::unordered_map<NodeIndex, std::string> node_names;
 *   std::unordered_set<NodeIndex> visited_nodes;
 * @endcode
 */
namespace std {
template <> struct hash<morphizen::NodeIndex> {
  /**
   * @brief Hash function implementation
   * @param index NodeIndex to hash
   * @return Hash value suitable for hash table use
   */
  std::size_t operator()(const morphizen::NodeIndex &index) const {
    return index.hash();
  }
};
} // namespace std
