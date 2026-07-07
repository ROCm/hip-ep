/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "node-arg-producer-map.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <algorithm>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_NODE_PRODUCER, "0");
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_NODE_PRODUCER) >= (n))
namespace morphizen {

// Implementation of NodeArgProducerProxy methods
NodeArgProducerProxy::NodeArgProducerProxy(NodeArgProducer& producer_map,
                                           const NodeArgIndex node_arg_index)
    : producer_map_(producer_map), node_arg_index_(node_arg_index) {}

NodeArgProducerProxy&
NodeArgProducerProxy::operator=(const NodeIndex& producer_index) {
  producer_map_.set_producer(node_arg_index_, producer_index);
  return *this;
}

NodeArgProducerProxy::operator NodeIndex() const {
  return producer_map_.get_producer(node_arg_index_);
}

NodeArgProducerProxy&
NodeArgProducerProxy::operator=(const NodeArgProducerProxy& other) {
  producer_map_.set_producer(node_arg_index_, other);
  return *this;
}

// Implementation of NodeArgProducer methods

NodeArgProducer::NodeArgProducer(GraphId graph_id) : graph_id_(graph_id) {
  // Constructor body left empty since reserve() was removed
  // Vectors will grow automatically as needed
}

void NodeArgProducer::set_producer(const NodeArgIndex& node_arg_index,
                                   const NodeIndex& producer_index) {
  // Validate input parameters
  CHECK(producer_index.is_valid())
      << "Setting invalid producer for node argument";
  // it is possbile that
  // staging_graph_.producer[node_arg_index_on_staging_graph] =
  // node_indx_on_original_graph because of set output.
  // so that we cannot CHECK(producer_index.get_graph_id() == graph_id_)
  // we only check that get_index is same.
  auto producer_graph_id = producer_index.get_graph_id();

  CHECK(producer_graph_id.get_index() == graph_id_.get_index())
      << "Producer NodeIndex graph ID ("
      << producer_index.get_graph_id().to_string()
      << ") does not match producer map graph ID (" << graph_id_.to_string()
      << ")";
  CHECK(!node_arg_index.is_graph_input())
      << "Setting producer for graph input - this is unusual";

  if (node_arg_index.is_initializer()) {
    // it is OK to override a
    // constant initializer, for
    // example, convert a constant initializer to Constant Node.
    CHECK(!node_arg_index.get_graph_id().is_staging())
        << "Cannot set producer for initializer on staging graph - "
           "initializers "
           "should be set on the original graph only";
  } else {
    CHECK(node_arg_index.is_node_output() || node_arg_index.is_graph_output())
        << "NodeArgIndex must be either a node output or graph output"
        << " (got " << node_arg_index.to_string() << ")";
  }

  auto node_arg_graph_id = node_arg_index.get_graph_id();
  CHECK_EQ(node_arg_graph_id.get_index(), node_arg_graph_id.get_index())
      << "NodeArgIndex graph ID (" << node_arg_graph_id.to_string()
      << ") does not match producer map graph ID (" << graph_id_.to_string()
      << ")";

  if (graph_id_.is_staging() && !node_arg_graph_id.is_staging()) {
    node_arg_producer_map_[node_arg_index] = producer_index;
    MY_LOG(1) << "Set producer " << producer_index.to_string()
              << " on the staging graph for node argument "
              << node_arg_index.to_string() << " on the original graph";
  } else if (node_arg_index.is_node_output()) {
    // Check if NodeArgIndex belongs to the correct graph (if graph is not a
    // staging graph)
    CHECK(graph_id_ == node_arg_index.get_graph_id())
        << "NodeArgIndex graph ID ("
        << node_arg_index.get_graph_id().to_string()
        << ") does not match producer map graph ID (" << graph_id_.to_string()
        << ")";
    // Use node output vector
    unsigned int index = node_arg_index.get_index();
    ensure_node_output_capacity(index);

    // Check if we're overwriting an existing entry
    CHECK(!node_output_producers_[index].is_valid())
        << "Overwriting existing producer for node output at index " << index;
    node_output_producers_[index] = producer_index;
    MY_LOG(1) << "Set producer " << producer_index.to_string()
              << " for node output at index " << index;
  } else if (node_arg_index.is_graph_output()) {
    // Check if NodeArgIndex belongs to the correct graph (if graph is not a
    // staging graph)
    CHECK(graph_id_ == node_arg_index.get_graph_id())
        << "NodeArgIndex graph ID ("
        << node_arg_index.get_graph_id().to_string()
        << ") does not match producer map graph ID (" << graph_id_.to_string()
        << ")";
    // Use graph output vector
    unsigned int index = node_arg_index.get_index();
    ensure_graph_output_capacity(index);
    if (graph_output_producers_[index] == producer_index) {
      // ignore , write the same value more than once.
    } else {
      // Check if we're overwriting an existing entry
      CHECK(!graph_output_producers_[index].is_valid())
          << "Overwriting existing producer for graph output at index "
          << index;
      graph_output_producers_[index] = producer_index;
      MY_LOG(1) << "Set producer " << producer_index.to_string()
                << " for graph output at index " << index;
    }
  } else {
    // For other cases (graph inputs, initializers, etc.), we should not reach
    // here due to the CHECK statements above, but log an error if we do
    LOG(FATAL) << "Unexpected NodeArgIndex type in set_producer";
  }
}

NodeIndex
NodeArgProducer::get_producer(const NodeArgIndex& node_arg_index) const {
  auto node_arg_graph_id = node_arg_index.get_graph_id();
  if (node_arg_graph_id.get_index() != graph_id_.get_index()) {
    LOG(WARNING) << "NodeArgIndex graph ID (" << node_arg_graph_id.to_string()
                 << ") does not match producer map graph ID ("
                 << graph_id_.to_string() << ")";
    return NodeIndex::invalid(); // Invalid NodeIndex
  }
  // Graph inputs and initializers don't have producers
  if (!graph_id_.is_staging()) {
    if (node_arg_index.is_graph_input() || node_arg_index.is_initializer()) {
      return NodeIndex::invalid(); // Invalid NodeIndex
    }
  } else {
    // for staging graph, it is possible to create a new node to replace
    // initializer and graph inputs.
  }

  // Use appropriate vector storage based on NodeArgIndex type
  if (graph_id_.is_staging() && !node_arg_graph_id.is_staging()) {
    auto it = node_arg_producer_map_.find(node_arg_index);
    if (it != node_arg_producer_map_.end()) {
      return it->second;           // Return the producer NodeIndex
    } else {
      return NodeIndex::invalid(); // Invalid NodeIndex
    }
  } else if (node_arg_index.is_node_output()) {
    // Use node output vector
    unsigned int index = node_arg_index.get_index();
    if (index < node_output_producers_.size()) {
      return node_output_producers_[index];
    }
    return NodeIndex::invalid(); // Invalid NodeIndex
  } else if (node_arg_index.is_graph_output()) {
    // Use graph output vector
    unsigned int index = node_arg_index.get_index();
    if (index < graph_output_producers_.size()) {
      return graph_output_producers_[index];
    }
    return NodeIndex::invalid(); // Invalid NodeIndex
  }

  // For other cases, return invalid NodeIndex
  // LOG(ERROR) << "Unexpected NodeArgIndex type in get_producer";
  return NodeIndex::invalid();
}
void NodeArgProducer::reserve(size_t num_of_node_args,
                              size_t num_of_graph_outputs) {
  node_output_producers_.reserve(num_of_node_args);
  graph_output_producers_.reserve(num_of_graph_outputs);
}

void NodeArgProducer::clear() {
  node_output_producers_.clear();
  graph_output_producers_.clear();
  MY_LOG(1) << "Cleared all producer mappings";
}

bool NodeArgProducer::empty() const {
  return node_output_producers_.empty() && graph_output_producers_.empty();
}

void NodeArgProducer::ensure_node_output_capacity(size_t index) {
  if (index >= node_output_producers_.size()) {
    // Resize vector and initialize new entries with invalid NodeIndex
    size_t old_size = node_output_producers_.size();
    node_output_producers_.resize(index + 1);

    // Initialize new entries (already done by NodeIndex default constructor)
    MY_LOG(1) << "Resized node output producer vector from " << old_size
              << " to " << (index + 1);
  }
}

void NodeArgProducer::ensure_graph_output_capacity(size_t index) {
  if (index >= graph_output_producers_.size()) {
    // Resize vector and initialize new entries with invalid NodeIndex
    size_t old_size = graph_output_producers_.size();
    graph_output_producers_.resize(index + 1);

    // Initialize new entries (already done by NodeIndex default constructor)
    MY_LOG(1) << "Resized graph output producer vector from " << old_size
              << " to " << (index + 1);
  }
}

NodeIndex
NodeArgProducer::operator[](const NodeArgIndex& node_arg_index) const {
  return get_producer(node_arg_index);
}

NodeArgProducerProxy
NodeArgProducer::operator[](const NodeArgIndex& node_arg_index) {
  return NodeArgProducerProxy(*this, node_arg_index);
}

} // namespace morphizen
