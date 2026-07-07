/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./node-index.hpp"
#include "./graph-id.hpp"
#include "./graph.hpp"
#include "./staging-graph.hpp"
#include <glog/logging.h>
#include <sstream>

namespace morphizen {

NodeIndex::NodeIndex() : value_(0) {}

NodeIndex::NodeIndex(unsigned int index, GraphId graph_id) {
  fields_.index_ = index & 0x7FFFFFFF;
  fields_.is_valid_ = 1;
  fields_.graph_id_ = graph_id.get_raw();
}

// Method implementations
bool NodeIndex::is_valid(const Graph& graph) const {
  if (!fields_.is_valid_) {
    return false; // Invalid flag is set
  }
  auto graph_id = get_graph_id();
  auto graph_ptr = graph_id.get_graph();
  if (graph_ptr == nullptr) {
    return false; // Graph does not exist
  }
  if (graph_ptr != &graph) {
    return false; // Graph mismatch
  }

  const morphizen_onnx::GraphProto* graph_proto = nullptr;
  if (graph_id.is_staging()) {
    if (graph_ptr->get_staging_graph() == nullptr) {
      return false; // Staging graph does not exist
    }
    graph_proto = &graph_ptr->get_staging_graph()->get_graph_proto();
  } else {
    graph_proto = &graph_ptr->get_graph_proto();
  }
  // Check if the index is within bounds of the graph's nodes
  if (fields_.index_ >= static_cast<unsigned int>(graph_proto->node_size())) {
    return false; // Index out of bounds
  }
  return true;
}
bool NodeIndex::is_valid() const {
  if (!fields_.is_valid_) {
    return false; // Invalid flag is set
  }
  auto graph_id = get_graph_id();
  auto graph_proto_ptr = graph_id.get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return false; // Graph does not exist
  }

  // Check if the index is within bounds of the graph's nodes
  if (fields_.index_ >=
      static_cast<unsigned int>(graph_proto_ptr->node_size())) {
    return false; // Index out of bounds
  }
  return true;
}

bool NodeIndex::operator==(const NodeIndex& other) const {
  static_assert(sizeof(*this) == sizeof(value_),
                "NodeIndex fields size must match value size");
  return value_ == other.value_;
}

bool NodeIndex::operator!=(const NodeIndex& other) const {
  return !(*this == other);
}

std::size_t NodeIndex::hash() const {
  // Use the union value directly for hashing since it contains all fields
  return static_cast<std::size_t>(value_);
}

unsigned int NodeIndex::get_index() const { return fields_.index_; }

GraphId NodeIndex::get_graph_id() const {
  return GraphId::from_raw(fields_.graph_id_);
}

const morphizen_onnx::NodeProto& NodeIndex::get_node_proto() const {
  CHECK(is_valid()) << "NodeIndex is invalid, cannot get node proto: "
                    << to_string();

  auto graph_id = get_graph_id();
  const morphizen_onnx::GraphProto* graph_proto = nullptr;
  auto* graph = graph_id.get_graph();
  CHECK(graph != nullptr) << "Graph not found for NodeIndex: " << to_string();
  if (graph_id.is_staging()) {
    auto* staging_graph = graph->get_staging_graph();
    CHECK(staging_graph != nullptr)
        << "Staging graph not found for NodeIndex: " << to_string();
    graph_proto = &staging_graph->get_graph_proto();
  } else {
    graph_proto = &graph->get_graph_proto();
  }

  CHECK(fields_.index_ < static_cast<unsigned int>(graph_proto->node_size()))
      << "NodeIndex index out of bounds: " << fields_.index_
      << ", graph has only " << graph_proto->node_size() << " nodes.";

  return graph_proto->node(fields_.index_);
}

const Node& NodeIndex::get_node() const {
  CHECK(is_valid()) << "NodeIndex is invalid, cannot get node: " << to_string();

  auto graph_id = get_graph_id();
  auto* graph = graph_id.get_graph();
  CHECK(graph != nullptr) << "Graph not found for NodeIndex: " << to_string();
  const Node* ret = nullptr;
  if (graph_id.is_staging()) {
    auto* staging_graph = graph->get_staging_graph();
    CHECK(staging_graph != nullptr)
        << "Staging graph not found for NodeIndex: " << to_string();
    // check out of bounds of staging_graph->nodes_.size()
    CHECK(fields_.index_ <
          static_cast<unsigned int>(staging_graph->nodes_.size()))
        << "NodeIndex index out of bounds: " << fields_.index_
        << ", staging graph has only " << staging_graph->nodes_.size()
        << " nodes.";
    ret = &staging_graph->nodes_[fields_.index_];
  } else {
    // check out of bounds of node_.size()
    CHECK(fields_.index_ < static_cast<unsigned int>(graph->nodes_.size()))
        << "NodeIndex index out of bounds: " << fields_.index_
        << ", graph has only " << graph->nodes_.size() << " nodes.";

    ret = &graph->nodes_[fields_.index_];
  }
  CHECK(ret != nullptr) << "Node not found for NodeIndex: " << to_string();
  return *ret;
}

bool NodeIndex::is_fused_node() const {
  const auto& node_proto = get_node_proto();
  // Check if the node has a fused_node_index attribute
  const auto& attributes = node_proto.attribute();
  auto it =
      std::find_if(attributes.begin(), attributes.end(), [](const auto& attr) {
        return attr.name() == "fused_node_index";
      });
  if (it == attributes.end() || it->i() < 0) {
    return false; // No valid fused_node_index attribute found
  }
  return true;
}
NodeIndex NodeIndex::invalid() { return NodeIndex(); }

const std::string& NodeIndex::get_node_op_type() const {
  const auto& node_proto = get_node_proto();
  return node_proto.op_type();
}

const std::string& NodeIndex::get_node_op_domain() const {
  const auto& node_proto = get_node_proto();
  return node_proto.domain();
}

const std::string& NodeIndex::get_name() const {
  const auto& node_proto = get_node_proto();
  return node_proto.name();
}

const std::string& NodeIndex::get_description() const {
  const auto& node_proto = get_node_proto();
  return node_proto.doc_string();
}

const std::vector<NodeArgIndex>& NodeIndex::get_input_node_args() const {
  const auto& node = get_node();
  return node.get_inputs();
}
const std::vector<NodeArgIndex>& NodeIndex::get_output_node_args() const {
  const auto& node = get_node();
  return node.get_outputs();
}

const ::google::protobuf::RepeatedPtrField<::morphizen_onnx::AttributeProto>*
NodeIndex::get_attributes() const {
  if (!is_valid()) {
    return nullptr; // Return nullptr if the node is not valid
  }

  try {
    const auto& node_proto = get_node_proto();
    return &node_proto.attribute();
  } catch (...) {
    // If any error occurs (graph not found, staging graph issues, etc.), return
    // nullptr
    return nullptr;
  }
}

std::string NodeIndex::to_string() const {
  std::ostringstream oss;
  auto graph_id = get_graph_id();
  oss << "NodeIndex(index: " << fields_.index_
      << ", graph_id: " << graph_id.to_string()
      << ", valid: " << (fields_.is_valid_ ? "true" : "false") << ")";
  return oss.str();
}

const void* NodeIndex::to_morphizen_core_node_ptr() const {
  // Convert NodeIndex to morphizen::Node pointer
  return reinterpret_cast<const void*>(static_cast<uintptr_t>(value_));
}

NodeIndex NodeIndex::from_morphizen_core_node_ptr(const void* ptr) {
  auto ret = NodeIndex::invalid();
  if (ptr == nullptr) {
    return ret;
  }
  ret.value_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  return ret;
}

NodeIndex NodeIndex::from_morphizen_core_node_index(size_t index) {
  static_assert(sizeof(size_t) == sizeof(value_),
                "size_t must be 64-bit to match NodeIndex value_ field");

  auto ret = NodeIndex::invalid();
  // Convert size_t index directly to NodeIndex value_
  ret.value_ = static_cast<uint64_t>(index);
  return ret;
}

const Graph* NodeIndex::get_function_body() const {
  if (!is_valid()) {
    return nullptr; // Return nullptr if the node is not valid
  }
  try {
    const auto& node_proto = get_node_proto();
    const auto& attributes = node_proto.attribute();
    auto it = std::find_if(
        attributes.begin(), attributes.end(),
        [](const auto& attr) { return attr.name() == "fused_graph_id"; });
    if (it == attributes.end()) {
      LOG(ERROR) << "No fused_graph_id attribute found in NodeIndex: "
                 << "node proto=\n"
                 << node_proto.DebugString() << " " << to_string();
      return nullptr; // No fused graph ID found
    }
    auto& fused_graph_id_attr = *it;
    if (fused_graph_id_attr.type() != morphizen_onnx::AttributeProto::INT) {
      LOG(ERROR) << "Invalid fused_graph_id attribute type in NodeIndex: "
                 << to_string();
      return nullptr; // Invalid attribute type
    }
    auto fused_graph_id = GraphId::from_raw((uint32_t)fused_graph_id_attr.i());
    const auto* graph = fused_graph_id.get_graph();
    if (graph == nullptr) {
      LOG(ERROR) << "Graph not found for fused_graph_id: "
                 << fused_graph_id.to_string();
      return nullptr; // Graph not found
    }
    return graph;
  } catch (std::exception& e) {
    LOG(ERROR) << "Error getting function body for NodeIndex: " << to_string()
               << ", error: " << e.what();
    return nullptr;
  } catch (...) {
    LOG(ERROR) << "Unknown error getting function body for NodeIndex: "
               << to_string();
    return nullptr; // Catch-all for any other exceptions
  }
}
} // namespace morphizen
