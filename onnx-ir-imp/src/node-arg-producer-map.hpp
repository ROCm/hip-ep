/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "graph-id.hpp"
#include "node-arg-index.hpp"
#include "node-index.hpp"
#include <optional>
#include <vector>

namespace morphizen {

// Forward declarations
class NodeIndex;
class NodeArgIndex;
class NodeArgProducer;

/**
 * @brief Helper class for implementing non-const operator[] with assignment
 * support
 *
 * This class acts as a proxy that allows the non-const operator[] to return a
 * reference that can be both read from and assigned to, while properly calling
 * set_producer() when assignment occurs.
 */
class NodeArgProducerProxy {
public:
  NodeArgProducerProxy(NodeArgProducer& producer_map,
                       const NodeArgIndex node_arg_index);

  // Assignment operator - calls set_producer when assigned to
  NodeArgProducerProxy& operator=(const NodeIndex& producer_index);

  // Conversion operator - allows reading the current value
  operator NodeIndex() const;

  // Copy assignment from another proxy
  NodeArgProducerProxy& operator=(const NodeArgProducerProxy& other);

private:
  NodeArgProducer& producer_map_;
  const NodeArgIndex node_arg_index_;
};

/**
 * @class NodeArgProducer
 * @brief Efficient mapping from NodeArgIndex to NodeIndex using vector-based
 * storage
 *
 * This class provides a fast way to map NodeArgIndex instances to their
 * producer NodeIndex instances. It uses a vector-based approach for maximum
 * performance when dealing with contiguous node argument indices.
 *
 * The class supports different types of node arguments:
 * - Node outputs (most common case, uses vector for O(1) access)
 * - Graph outputs (uses vector for O(1) access)
 * - Graph inputs (no producer, returns invalid NodeIndex)
 * - Initializers (no producer, returns invalid NodeIndex)
 *
 * Design rationale:
 * - Uses std::vector for all storage to achieve O(1) access time
 * - Optimized for the common case where NodeArgIndex instances are mostly
 * contiguous
 * - Thread-safe for read operations, requires external synchronization for
 * writes
 *
 * Usage example:
 * @code
 * NodeArgProducer producer_map;
 *
 * // Set a producer relationship
 * NodeArgIndex arg_idx = NodeArgIndex::node_output(42, graph_id);
 * NodeIndex producer_idx(10, graph_id);
 * producer_map[arg_idx] = producer_idx;
 *
 * // Query the producer
 * NodeIndex found_producer = producer_map[arg_idx];
 * if (found_producer.is_valid()) {
 *     // Process the producer node
 * }
 *
 * // Use const access
 * const NodeArgProducer& const_map = producer_map;
 * NodeIndex producer = const_map[arg_idx];
 * @endcode
 */
class NodeArgProducer {
public:
  /**
   * @brief Constructor with graph ID
   * @param graph_id The graph ID this producer map belongs to
   *
   * Creates a NodeArgProducer bound to a specific graph.
   */
  explicit NodeArgProducer(GraphId graph_id);

  /**
   * @brief Copy constructor
   */
  NodeArgProducer(const NodeArgProducer& other) = default;

  /**
   * @brief Move constructor
   */
  NodeArgProducer(NodeArgProducer&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  NodeArgProducer& operator=(const NodeArgProducer& other) = default;

  /**
   * @brief Move assignment operator
   */
  NodeArgProducer& operator=(NodeArgProducer&& other) noexcept = default;

  /**
   * @brief Destructor
   */
  ~NodeArgProducer() = default;

  /**
   * @brief Get the producer node for a given node argument (const access)
   * @param node_arg_index The node argument index to query
   * @return The producer NodeIndex, or invalid NodeIndex if no producer exists
   *
   * This is equivalent to get_producer() but provides convenient array-like
   * syntax. For graph inputs and initializers, this will always return an
   * invalid NodeIndex.
   *
   * Example:
   * @code
   * const NodeArgProducer& producer_map = ...;
   * NodeIndex producer = producer_map[node_arg_index];
   * @endcode
   */
  NodeIndex operator[](const NodeArgIndex& node_arg_index) const;

  /**
   * @brief Get/set the producer node for a given node argument (mutable access)
   * @param node_arg_index The node argument index to access
   * @return Proxy object that supports both reading and assignment
   *
   * Provides convenient array-like syntax for both reading and writing.
   * The returned proxy automatically calls set_producer() when assigned to.
   *
   * @note For graph inputs and initializers, this will log a warning when
   * written to as they typically don't have producer nodes.
   *
   * Example:
   * @code
   * NodeArgProducer producer_map;
   * producer_map[node_arg_index] = producer_node_index;  // Writing
   * NodeIndex producer = producer_map[node_arg_index];   // Reading
   * @endcode
   */
  NodeArgProducerProxy operator[](const NodeArgIndex& node_arg_index);

  /**
   * @brief Reserves memory for node arguments and graph outputs.
   *
   * Pre-allocates internal storage to accommodate the specified number of node
   * arguments and graph outputs, optimizing performance by reducing
   * reallocations.
   *
   * @param num_of_node_args The number of node arguments to reserve space for.
   * @param num_of_graph_outputs The number of graph outputs to reserve space
   * for.
   */
  void reserve(size_t num_of_node_args, size_t num_of_graph_outputs);

  /**
   * @brief Clear all producer mappings
   *
   * Removes all producer mappings from both node output and graph output
   * vectors. After calling this method, all producer queries will return
   * invalid NodeIndex.
   */
  void clear();

  /**
   * @brief Check if the producer map is empty
   * @return true if there are no producer mappings, false otherwise
   *
   * Returns true if both node output and graph output vectors are empty,
   * indicating that no producer mappings have been established.
   */
  bool empty() const;

  /**
   * @brief Get the graph ID this producer map belongs to
   * @return The graph ID
   */
  GraphId get_graph_id() const { return graph_id_; }
  void set_graph_id(GraphId id) { graph_id_ = id; }

private:
  /**
   * @brief Set the producer node for a given node argument (internal use)
   * @param node_arg_index The node argument index
   * @param producer_index The producer node index
   *
   * Establishes a mapping from the node argument to its producer node.
   * If the node argument already has a producer, it will be overwritten.
   *
   * @note For graph inputs and initializers, this method will log a warning
   *       as they typically don't have producer nodes.
   */
  void set_producer(const NodeArgIndex& node_arg_index,
                    const NodeIndex& producer_index);

  /**
   * @brief Get the producer node for a given node argument (internal use)
   * @param node_arg_index The node argument index to query
   * @return The producer NodeIndex, or invalid NodeIndex if no producer exists
   *
   * Returns the producer node for the specified node argument. For graph inputs
   * and initializers, this will always return an invalid NodeIndex since they
   * don't have producer nodes.
   */
  NodeIndex get_producer(const NodeArgIndex& node_arg_index) const;

  /**
   * @brief Ensure the node output vector has sufficient capacity for the given
   * index
   * @param index The vector index that needs to be accessible
   *
   * Resizes the node output vector if necessary to accommodate the specified
   * index. New entries are initialized with invalid NodeIndex instances.
   */
  void ensure_node_output_capacity(size_t index);

  /**
   * @brief Ensure the graph output vector has sufficient capacity for the given
   * index
   * @param index The vector index that needs to be accessible
   *
   * Resizes the graph output vector if necessary to accommodate the specified
   * index. New entries are initialized with invalid NodeIndex instances.
   */
  void ensure_graph_output_capacity(size_t index);

private:
  friend class NodeArgProducerProxy;

private:
  // Graph ID this producer map belongs to
  GraphId graph_id_;

  // Primary storage for node output producers (most common case)
  // Index corresponds to the NodeArgIndex's internal index for NODE_OUTPUT type
  std::vector<NodeIndex> node_output_producers_;

  // Storage for graph output producers
  // Index corresponds to the NodeArgIndex's internal index for GRAPH_OUTPUT
  // type
  std::vector<NodeIndex> graph_output_producers_;
  std::unordered_map<NodeArgIndex, NodeIndex> node_arg_producer_map_;
};

} // namespace morphizen
