/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <cstdint>
#include <string>
namespace morphizen_onnx {
class GraphProto;
}
namespace morphizen {
class Graph;
/**
 * @class GraphId
 * @brief Represents a graph identifier with staging graph indication
 *
 * This class encapsulates a 32-bit graph identifier where:
 * - The highest bit (bit 31) indicates whether it's a staging graph
 * - The remaining 31 bits (bits 0-30) represent the graph index
 *
 * Design rationale:
 * - Provides type safety for graph identifiers
 * - Efficiently encodes staging graph information in a single uint32_t
 * - Supports up to 2^31-1 unique graph indices
 * - Clear separation between staging and non-staging graphs
 *
 * Usage example:
 * @code
 * GraphId main_graph = GraphId::create_main_graph(42);
 * GraphId staging_graph = GraphId::create_staging_graph(42);
 *
 * if (staging_graph.is_staging()) {
 *     // Handle staging graph logic
 * }
 *
 * uint32_t index = main_graph.get_index();  // Returns 42
 * @endcode
 */
class GraphId {

public:
  /**
   * @brief Create a main (non-staging) graph ID
   * @param index The graph index (must be <= MAX_INDEX)
   * @return GraphId instance for main graph
   * @throws std::invalid_argument if index > MAX_INDEX
   */
  static GraphId create_main_graph(uint32_t index);

  /**
   * @brief Create a staging graph ID
   * @param index The graph index (must be <= MAX_INDEX)
   * @return GraphId instance for staging graph
   * @throws std::invalid_argument if index > MAX_INDEX
   */
  static GraphId create_staging_graph(uint32_t index);

  /**
   * @brief Create GraphId from raw uint32_t value
   * @param value The raw 32-bit value
   * @return GraphId instance
   */
  static GraphId from_raw(uint32_t value);

  /**
   * @brief Check if this is a staging graph
   * @return true if staging graph, false otherwise
   */
  bool is_staging() const;
  Graph* get_graph() const;
  const morphizen_onnx::GraphProto* get_graph_proto() const;
  /**
   * @brief Get the graph index (lower 31 bits)
   * @return The graph index
   */
  uint32_t get_index() const;

  /**
   * @brief Get the raw uint32_t value
   * @return The raw 32-bit value
   */
  uint32_t get_raw() const;

  /**
   * @brief Get string representation
   * @return String representation of the graph ID
   */
  std::string to_string() const;

  // Comparison operators
  bool operator==(const GraphId& other) const;
  bool operator!=(const GraphId& other) const;
  bool operator<(const GraphId& other) const;
  bool operator<=(const GraphId& other) const;
  bool operator>(const GraphId& other) const;
  bool operator>=(const GraphId& other) const;

private:
  /**
   * @brief Private constructor from raw value
   * @param value The raw 32-bit value
   */
  explicit GraphId(uint32_t value);
  union {
    struct {
      unsigned int is_staging_ : 1;
      unsigned int index_ : 31;
    } fields_;
    uint32_t value_;
  };
};

} // namespace morphizen

// Hash specialization for std::unordered_map
namespace std {
template <> struct hash<morphizen::GraphId> {
  size_t operator()(const morphizen::GraphId& graph_id) const;
};
} // namespace std
