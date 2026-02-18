/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-graph-store.hpp"
#include "./mlir-graph.hpp"
#include "morphizen-foundation/env_config.hpp"
#include <glog/logging.h>
#include <sstream>
#include <vector>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH_STORE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_GRAPH_STORE) >= n)

namespace morphizen {
namespace mlir_impl {

std::vector<MLIRGraph*>& GraphStore::get_graphs_cache() {
  static std::vector<MLIRGraph*> graphs_cache;
  return graphs_cache;
}

MLIRGraph* GraphStore::get_graph_by_id(uint32_t graph_id) {
  auto& graphs_cache = get_graphs_cache();
  if (graph_id < graphs_cache.size()) {
    return graphs_cache[graph_id];
  }
  return nullptr; // Return nullptr if graph_id is out of bounds
}

uint32_t GraphStore::allocate_graph_id(MLIRGraph* graph,
                                       uint32_t proposed_graph_index) {
  auto& graphs_cache = get_graphs_cache();
  uint32_t ret = 0;

  if (proposed_graph_index > 0 && proposed_graph_index < graphs_cache.size()) {
    // Check if the proposed index is available
    if (graphs_cache[proposed_graph_index] == nullptr) {
      ret = proposed_graph_index;
    } else {
      // Find the next available slot
      for (size_t i = 1; i < graphs_cache.size(); ++i) {
        if (graphs_cache[i] == nullptr) {
          ret = static_cast<uint32_t>(i);
          break;
        }
      }
      // If no available slot found, expand the cache
      if (ret == 0) {
        ret = static_cast<uint32_t>(graphs_cache.size());
        graphs_cache.push_back(nullptr);
      }
    }
  } else {
    // Auto-assign: find first available slot or expand
    for (size_t i = 1; i < graphs_cache.size(); ++i) {
      if (graphs_cache[i] == nullptr) {
        ret = static_cast<uint32_t>(i);
        break;
      }
    }
    // If no available slot found, expand the cache
    if (ret == 0) {
      ret = static_cast<uint32_t>(graphs_cache.size());
      graphs_cache.push_back(nullptr);
    }
  }

  // Ensure the cache is large enough
  while (ret >= graphs_cache.size()) {
    graphs_cache.push_back(nullptr);
  }

  graphs_cache[ret] = graph;
  MY_LOG(1) << "Allocated graph ID: " << ret << " for graph: " << (void*)graph
            << " " << graph_cache_ids_to_string();
  return ret;
}

void GraphStore::release_graph_id(MLIRGraph* graph, uint32_t old_graph_id) {
  // invalidate all old node arg index and node index
  auto old_graph_ptr = get_graph_by_id(old_graph_id);
  CHECK_EQ(old_graph_ptr, graph) << "Old graph pointer must be the same as "
                                    "current graph pointer, old_grapgh_id="
                                 << old_graph_id;
  get_graphs_cache()[old_graph_id] = nullptr;
  MY_LOG(1) << "Released graph ID: " << old_graph_id
            << " for graph: " << (void*)graph << " "
            << graph_cache_ids_to_string();
}

std::string GraphStore::graph_cache_ids_to_string() {
  auto& graphs_cache = get_graphs_cache();
  std::ostringstream oss;
  oss << "GraphStore cache [";
  for (size_t i = 0; i < graphs_cache.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << i << ": " << (graphs_cache[i] ? "used" : "free");
  }
  oss << "]";
  return oss.str();
}

MLIRGraph*
GraphStore::get_graph_by_symbol_name(const std::string& symbol_name) {
  auto& graphs_cache = get_graphs_cache();
  for (size_t i = 0; i < graphs_cache.size(); ++i) {
    if (graphs_cache[i] && graphs_cache[i]->get_symbol_name() == symbol_name) {
      return graphs_cache[i];
    }
  }
  return nullptr; // Return nullptr if no graph with matching symbol name found
}

} // namespace mlir_impl
} // namespace morphizen
