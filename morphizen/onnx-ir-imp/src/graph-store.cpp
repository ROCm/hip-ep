/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./graph-store.hpp"
#include "./graph.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <sstream>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_GRAPH, "0");
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_GRAPH) >= (n))

namespace morphizen {

std::vector<Graph *> &GraphStore::get_graphs_cache() {
  // This function is a placeholder for the actual implementation that would
  // retrieve the cached graphs. For now, it returns an empty vector.
  static auto the_store = std::vector<Graph *>();
  return the_store;
}

std::string GraphStore::graph_cache_ids_to_string() {
  auto &graphs_cache = get_graphs_cache();
  std::ostringstream oss;
  oss << "IDS: [";
  for (size_t i = 0; i < graphs_cache.size(); ++i) {
    if (graphs_cache[i]) {
      oss << "(" << i << "," << (void *)graphs_cache[i] << ")";
    }
  }
  oss << "]";
  return oss.str();
}

uint32_t GraphStore::allocate_graph_id(Graph *graph,
                                       uint32_t proposed_graph_index) {
  MY_LOG(1) << "Before allocating graph ID for graph: " << (void *)graph
            << " with proposed ID: " << proposed_graph_index << " "
            << graph_cache_ids_to_string();
  auto &graphs_cache = get_graphs_cache();
  auto it = std::find(graphs_cache.begin() + proposed_graph_index,
                      graphs_cache.end(), nullptr);
  auto ret = 0u;
  if (it != graphs_cache.end()) {
    *it = graph;
    ret = static_cast<uint32_t>(std::distance(graphs_cache.begin(), it));
  } else {
    // If no empty slot found, append the graph to the end of the cache
    ret = static_cast<uint32_t>(graphs_cache.size());
    graphs_cache.push_back(graph);
  }
  MY_LOG(1) << "After allocated graph ID: " << ret
            << " for graph: " << (void *)graph << " "
            << graph_cache_ids_to_string();
  return ret;
}

void GraphStore::release_graph_id(Graph *graph, uint32_t old_graph_id) {
  // invalidate all old node arg index and node index
  auto old_graph_ptr = get_graph_by_id(old_graph_id);
  CHECK_EQ(old_graph_ptr, graph)
      << "Old graph pointer must be the same as current graph pointer";
  get_graphs_cache()[old_graph_id] = nullptr;
  MY_LOG(1) << "Released graph ID: " << old_graph_id
            << " for graph: " << (void *)graph << " "
            << graph_cache_ids_to_string();
}

Graph *GraphStore::get_graph_by_id(uint32_t graph_id) {
  auto &graphs_cache = get_graphs_cache();
  if (graph_id < graphs_cache.size()) {
    return graphs_cache[graph_id];
  }
  return nullptr; // Return nullptr if graph_id is out of bounds
}

} // namespace morphizen
