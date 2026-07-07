/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace morphizen {

// Forward declaration
class Graph;

/**
 * @brief Graph storage and cache management utilities
 *
 * This class provides static methods for managing the global graph cache,
 * including allocation and release of graph IDs, and tracking of graph
 * instances.
 */
class GraphStore {
public:
  /**
   * @brief Get a graph by its ID
   * @param graph_id The ID of the graph to retrieve
   * @return Pointer to the graph, or nullptr if not found
   */
  static Graph* get_graph_by_id(uint32_t graph_id);

  /**
   * @brief Allocate a new graph ID for a graph instance
   * @param graph Pointer to the graph instance
   * @param proposed_graph_index Preferred graph ID (0 means auto-assign)
   * @return The allocated graph ID
   */
  static uint32_t allocate_graph_id(Graph* graph,
                                    uint32_t proposed_graph_index = 0);

  /**
   * @brief Release a graph ID and remove the graph from cache
   * @param graph Pointer to the graph instance
   * @param old_graph_id The graph ID to release
   */
  static void release_graph_id(Graph* graph, uint32_t old_graph_id);

  /**
   * @brief Get a string representation of all cached graph IDs for debugging
   * @return String containing graph cache information
   */
  static std::string graph_cache_ids_to_string();

private:
  /**
   * @brief Get reference to the global graphs cache
   * @return Reference to the vector storing graph pointers
   */
  static std::vector<Graph*>& get_graphs_cache();
};

} // namespace morphizen
