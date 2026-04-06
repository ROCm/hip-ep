/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./node-arg-index.hpp"
#include "./graph-id.hpp"
#include "./graph.hpp"
#include "./node-index.hpp"
#include "./staging-graph.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <glog/logging.h>
#include <memory>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_NODE_ARG_INDEX, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_NODE_ARG_INDEX) >= n)
namespace morphizen {

// Constructor implementations
NodeArgIndex::NodeArgIndex() : fields_{0, 0, 0} {}

NodeArgIndex::NodeArgIndex(unsigned int index, Type type)
    : fields_{index & 0x1FFFFFFF, static_cast<unsigned int>(type) & 0x7, 0} {}

NodeArgIndex::NodeArgIndex(unsigned int index, Type type, GraphId graph_id)
    : fields_{index & 0x1FFFFFFF, static_cast<unsigned int>(type) & 0x7, 0} {
  fields_.graph_id_ = graph_id.get_raw();
}

// Method implementations
bool NodeArgIndex::is_valid() const { return get_type() != Type::INVALID; }

bool NodeArgIndex::is_valid(const Graph& graph) const {
  if (get_type() == Type::INVALID) {
    return false; // Invalid type
  }
  auto id = get_graph_id();
  if (id.get_index() != graph.get_graph_id().get_index()) {
    return false;
  }
  if (id.is_staging()) {
    if (graph.get_staging_graph() == nullptr) {
      return false;
    }
  }
  return true;
}
bool NodeArgIndex::is_valid_graph_input() const {
  auto ret = get_type() == Type::GRAPH_INPUT;
  if (ret) {
    auto graph_proto = get_graph_id().get_graph_proto();
    if (graph_proto == nullptr) {
      return false; // Graph does not exist
    }
    ret = fields_.index_ < static_cast<unsigned int>(graph_proto->input_size());
  }
  return ret;
}
bool NodeArgIndex::is_valid_initializer() const {
  auto ret = get_type() == Type::INITIALIZER;
  if (ret) {
    auto graph_proto = get_graph_id().get_graph_proto();
    if (graph_proto == nullptr) {
      return false; // Graph does not exist
    }
    ret = fields_.index_ <
          static_cast<unsigned int>(graph_proto->initializer_size());
  }
  return ret;
}
bool NodeArgIndex::is_valid_node_output() const {
  auto ret = get_type() == Type::NODE_OUTPUT;
  if (ret) {
    auto graph_proto = get_graph_id().get_graph_proto();
    if (graph_proto == nullptr) {
      return false; // Graph does not exist
    }
    ret = fields_.index_ <
          static_cast<unsigned int>(graph_proto->value_info_size());
  }
  return ret;
}
bool NodeArgIndex::is_valid_graph_output() const {
  auto ret = get_type() == Type::GRAPH_OUTPUT;
  if (ret) {
    auto graph_proto = get_graph_id().get_graph_proto();
    if (graph_proto == nullptr) {
      return false; // Graph does not exist
    }
    ret =
        fields_.index_ < static_cast<unsigned int>(graph_proto->output_size());
  }
  return ret;
}

bool NodeArgIndex::operator==(const NodeArgIndex& other) const {
  return this->value_ == other.value_;
}

bool NodeArgIndex::operator!=(const NodeArgIndex& other) const {
  return !(*this == other);
}

std::size_t NodeArgIndex::hash() const {
  // Combine index, type, and graph_id into a single hash value
  // Use the full 64-bit value for better hash distribution
  return static_cast<std::size_t>(value_);
}

unsigned int NodeArgIndex::get_index() const { return fields_.index_; }

NodeArgIndex::Type NodeArgIndex::get_type() const {
  return static_cast<Type>(fields_.type_);
}

GraphId NodeArgIndex::get_graph_id() const {
  return GraphId::from_raw(fields_.graph_id_);
}

bool NodeArgIndex::exists() const {
  if (!is_valid()) {
    return false; // Invalid type
  }
  return is_valid_graph_input() || is_valid_initializer() ||
         is_valid_node_output() || is_valid_graph_output();
}

std::vector<int64_t>* NodeArgIndex::extract_shape_from_tensor_type(
    const morphizen_onnx::TensorShapeProto& shape) {
  auto result = std::make_unique<std::vector<int64_t>>();
  result->reserve(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    const auto& dim = shape.dim(i);
    if (dim.has_dim_value()) {
      result->push_back(dim.dim_value());
    } else {
      result->push_back(-1); // Unknown dimension
    }
  }
  return result.release();
}

std::vector<int64_t>* NodeArgIndex::extract_shape_from_value_info(
    const morphizen_onnx::ValueInfoProto& value_info) {
  CHECK(value_info.has_type())
      << "ValueInfo must have type information: " << value_info.DebugString();
  CHECK(value_info.type().has_tensor_type())
      << "Only tensor_type is supported for shape extraction";

  if (value_info.type().tensor_type().has_shape()) {
    const auto& shape = value_info.type().tensor_type().shape();
    return extract_shape_from_tensor_type(shape);
  }
  return nullptr; // No shape information available, please run shape inference
}

std::vector<int64_t>* NodeArgIndex::extract_shape_from_initializer(
    const morphizen_onnx::TensorProto& initializer) {
  auto result = std::make_unique<std::vector<int64_t>>();
  result->reserve(initializer.dims_size());
  for (int i = 0; i < initializer.dims_size(); ++i) {
    result->push_back(initializer.dims(i));
  }
  return result.release();
}

std::vector<int64_t>* NodeArgIndex::get_shape_i64_unsafe() const {
  // Check if this NodeArgIndex is valid and exists
  if (!exists()) {
    return nullptr; // Return nullptr if invalid
  }

  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return nullptr;
  }

  // Get the graph proto from the graph
  const auto& graph_proto = *graph_proto_ptr;
  // Based on the type, look in different places for shape information
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    // Look in graph inputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    const auto& input = graph_proto.input(fields_.index_);
    return extract_shape_from_value_info(input);
  }
  case Type::INITIALIZER: {
    // Look in graph initializers - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    const auto& initializer = graph_proto.initializer(fields_.index_);
    return extract_shape_from_initializer(initializer);
  }
  case Type::NODE_OUTPUT: {
    // Look in value_info for output shapes - fatal error if index is out of
    // bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    const auto& value_info = graph_proto.value_info(fields_.index_);
    return extract_shape_from_value_info(value_info);
  }
  case Type::GRAPH_OUTPUT: {
    // Look in graph outputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    const auto& output = graph_proto.output(fields_.index_);
    return extract_shape_from_value_info(output);
  }
  case Type::INVALID:
  default:
    break;
  }

  // Return nullptr if no shape information found
  return nullptr;
}

const std::string* NodeArgIndex::get_name_unsafe() const {
  // Check if this NodeArgIndex is valid and exists
  if (!exists()) {
    return nullptr; // Return nullptr if invalid
  }
  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return nullptr;
  }

  // Get the graph proto from the graph
  const auto& graph_proto = *graph_proto_ptr;

  // Based on the type, look in different places for name information
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    // Look in graph inputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    const auto& input = graph_proto.input(fields_.index_);
    return &input.name();
  }
  case Type::INITIALIZER: {
    // Look in graph initializers - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    const auto& initializer = graph_proto.initializer(fields_.index_);
    return &initializer.name();
  }
  case Type::NODE_OUTPUT: {
    // Look in value_info for output names - fatal error if index is out of
    // bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    const auto& value_info = graph_proto.value_info(fields_.index_);
    return &value_info.name();
  }
  case Type::GRAPH_OUTPUT: {
    // Look in graph outputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    const auto& output = graph_proto.output(fields_.index_);
    return &output.name();
  }
  case Type::INVALID: {
    CHECK(false) << "Invalid NodeArgIndex type cannot have a name";
    break;
  }
  default:
    break;
  }

  // Return nullptr if no name information found
  return nullptr;
}
std::string NodeArgIndex::get_name() const {
  auto name_ptr = get_name_unsafe();
  auto ret = std::string();
  if (name_ptr) {
    ret = *name_ptr;
  } else {
    return "n/a" + to_string();
  }
  return ret;
}

// Type checking methods - simply check the type without validity
bool NodeArgIndex::is_graph_input() const {
  return get_type() == Type::GRAPH_INPUT;
}

bool NodeArgIndex::is_initializer() const {
  return get_type() == Type::INITIALIZER;
}

bool NodeArgIndex::is_node_output() const {
  return get_type() == Type::NODE_OUTPUT;
}

bool NodeArgIndex::is_graph_output() const {
  return get_type() == Type::GRAPH_OUTPUT;
}

std::string NodeArgIndex::to_string() const {
  std::string type_str;
  switch (get_type()) {
  case Type::INVALID:
    type_str = "INV";
    break;
  case Type::GRAPH_INPUT:
    type_str = "GI";
    break;
  case Type::INITIALIZER:
    type_str = "INIT";
    break;
  case Type::NODE_OUTPUT:
    type_str = "NO";
    break;
  case Type::GRAPH_OUTPUT:
    type_str = "GO";
    break;
  default:
    type_str = "UNK";
    break;
  }
  auto graph_id = get_graph_id();
  return "[" + type_str + ":" + std::to_string(get_index()) + ":g" +
         (graph_id.is_staging() ? "S" : "M") +
         std::to_string(graph_id.get_index()) + "]";
}
// Static factory method implementations
NodeArgIndex NodeArgIndex::invalid() {
  return NodeArgIndex(0, Type::INVALID, GraphId::from_raw(0));
}

NodeArgIndex NodeArgIndex::graph_input(unsigned int index, GraphId graph_id) {
  return NodeArgIndex(index, Type::GRAPH_INPUT, graph_id);
}

NodeArgIndex NodeArgIndex::initializer(unsigned int index, GraphId graph_id) {
  return NodeArgIndex(index, Type::INITIALIZER, graph_id);
}

NodeArgIndex NodeArgIndex::node_output(unsigned int index, GraphId graph_id) {
  return NodeArgIndex(index, Type::NODE_OUTPUT, graph_id);
}
NodeArgIndex NodeArgIndex::graph_output(unsigned int index, GraphId graph_id) {
  return NodeArgIndex(index, Type::GRAPH_OUTPUT, graph_id);
}

NodeArgIndex NodeArgIndex::from_morphizen_core_node_arg_ptr(const void* ptr) {
  auto ret = NodeArgIndex::invalid();
  if (ptr == nullptr) {
    return ret; // Return invalid index if pointer is null
  }

  // Convert the pointer to an index value
  // This creates a direct mapping between the pointer and the index
  auto ptr_value = reinterpret_cast<uintptr_t>(ptr);
  // Store the pointer value in the 64-bit union, preserving the pointer
  ret.value_ = static_cast<uint64_t>(ptr_value);
  return ret;
}

NodeIndex NodeArgIndex::get_producer_node() const {
  /*if (!is_valid_graph_output() && !is_valid_node_output()) {
    return NodeIndex::invalid();
  }*/
  // constant_initializer and graph input potentially has producer also because
  // it is possible to add a new node to replace the constant initializer and
  // graph input
  auto graph_id = get_graph_id();
  const auto* graph = graph_id.get_graph();
  CHECK(graph != nullptr) << "Graph not found for NodeArgIndex";
  for (const auto* g = graph; g != nullptr; g = g->parent_graph_) {
    auto* staging_graph = g->get_staging_graph();
    if (staging_graph) {
      NodeIndex node_index = staging_graph->producer_map_[*this];
      if (node_index.is_valid()) {
        return node_index;
      }
    }
    NodeIndex node_index = g->producer_map_[*this];
    if (node_index.is_valid()) {
      return node_index;
    }
  }
  return NodeIndex::invalid();
}

const void* NodeArgIndex::to_morphizen_core_node_arg_ptr() const {
  if (!exists()) {
    return nullptr;
  }
  // Convert the value back to a pointer
  // This reverses the mapping done in from_morphizen_core_node_arg_ptr
  return reinterpret_cast<const void*>(static_cast<uintptr_t>(value_));
}
const morphizen_onnx::ValueInfoProto& NodeArgIndex::get_value_info() const {
  if (!is_valid()) {
    throw std::runtime_error("Invalid NodeArgIndex");
  }
  // Get the graph associated with this node argument
  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    LOG(ERROR) << "Graph does not exist for NodeArgIndex"
               << " (graph_id: " << get_graph_id().to_string() << ")";
    throw std::runtime_error("Graph does not exist for NodeArgIndex");
  }
  auto& graph_proto = *graph_proto_ptr;
  const morphizen_onnx::ValueInfoProto* ret = nullptr;
  if (is_graph_input()) {
    // Graph input
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    ret = &graph_proto.input(fields_.index_);
  } else if (is_initializer()) {
    // Initializer
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    throw std::runtime_error("Initializers do not have ValueInfoProto, use "
                             "get_const_data_as_tensor() instead");
  } else if (is_node_output()) {
    // Node output
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    ret = &graph_proto.value_info(fields_.index_);
  } else if (is_graph_output()) {
    // Graph output
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    ret = &graph_proto.output(fields_.index_);
  }
  if (ret == nullptr) {
    throw std::runtime_error("Invalid NodeArgIndex type for get_value_info");
  }
  return *ret;
}
std::vector<std::string>* NodeArgIndex::extract_denotation_from_value_info(
    const morphizen_onnx::ValueInfoProto& value_info) {
  // Check if the value_info has type and tensor_type information
  if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
    return nullptr; // No tensor type information available
  }

  const auto& tensor_type = value_info.type().tensor_type();

  // Check if tensor_type has shape information
  if (!tensor_type.has_shape()) {
    return nullptr;
  }

  const auto& shape = tensor_type.shape();

  // Create result vector to store denotations
  auto result = std::make_unique<std::vector<std::string>>();
  result->reserve(shape.dim_size());

  // Extract denotation from each dimension
  for (int i = 0; i < shape.dim_size(); ++i) {
    const auto& dim = shape.dim(i);
    if (dim.has_denotation() && !dim.denotation().empty()) {
      result->push_back(dim.denotation());
    } else {
      result->push_back(""); // Empty string for dimensions without denotation
    }
  }
  return result.release();
}

std::vector<std::string>* NodeArgIndex::get_denotation_unsafe() const {
  // Check if this NodeArgIndex is valid and exists
  if (!is_valid()) {
    return nullptr; // Return nullptr if invalid
  }

  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return nullptr;
  }

  // Get the graph proto from the graph
  const auto& graph_proto = *graph_proto_ptr;

  // Based on the type, look in different places for denotation information
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    // Look in graph inputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    const auto& input = graph_proto.input(fields_.index_);
    return extract_denotation_from_value_info(input);
  }
  case Type::INITIALIZER: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    const auto& initializer = graph_proto.initializer(fields_.index_);
    auto result = std::make_unique<std::vector<std::string>>();
    result->reserve(initializer.dims_size());
    for (int i = 0; i < initializer.dims_size(); ++i) {
      result->push_back(""); // Placeholder denotation
    }
    return result.release();
  }
  case Type::NODE_OUTPUT: {
    // Look in value_info for output denotations - fatal error if index is out
    // of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    const auto& value_info = graph_proto.value_info(fields_.index_);
    return extract_denotation_from_value_info(value_info);
  }
  case Type::GRAPH_OUTPUT: {
    // Look in graph outputs - fatal error if index is out of bounds
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    const auto& output = graph_proto.output(fields_.index_);
    return extract_denotation_from_value_info(output);
  }
  case Type::INVALID:
  default:
    break;
  }

  // Return nullptr if no denotation information found
  return nullptr;
}

int NodeArgIndex::get_element_type() const {
  // Check if this NodeArgIndex is valid and exists
  if (!is_valid()) {
    return -1; // Return -1 if invalid
  }

  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return -1;
  }

  // Get the graph proto from the graph
  const auto& graph_proto = *graph_proto_ptr;

  // Based on the type, look in different places for element type information
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    const auto& input = graph_proto.input(fields_.index_);
    return input.type().tensor_type().elem_type();
  }
  case Type::INITIALIZER: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    const auto& initializer = graph_proto.initializer(fields_.index_);
    return initializer.data_type();
  }
  case Type::NODE_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    const auto& value_info = graph_proto.value_info(fields_.index_);
    return value_info.type().tensor_type().elem_type();
  }
  case Type::GRAPH_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    const auto& output = graph_proto.output(fields_.index_);
    return output.type().tensor_type().elem_type();
  }
  case Type::INVALID:
  default:
    break;
  }

  // Return -1 if no element type information found
  return -1;
}

void NodeArgIndex::set_shape_i64(const std::vector<int64_t>& shape) {
  // Check if this NodeArgIndex is valid and exists
  if (!is_valid()) {
    throw std::runtime_error("Invalid NodeArgIndex");
  }

  auto graph_id = get_graph_id();
  auto* graph = graph_id.get_graph();
  if (graph == nullptr) {
    throw std::runtime_error("Graph not found for NodeArgIndex");
  }

  // Get the graph proto from the graph
  auto& graph_proto = graph->get_graph_proto();

  // Based on the type, set the shape in different places
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    auto* input = graph_proto.mutable_input(fields_.index_);
    auto* shape_proto =
        input->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape_proto->clear_dim(); // Clear existing dimensions first
    for (const auto& dim : shape) {
      shape_proto->add_dim()->set_dim_value(dim);
    }
    // Ensure the graph is in an inconsistent state to trigger updates
    // let graph.need_resolve() return true
    graph->ensure_enter_into_inconsistent_state();
    break;
  }
  case Type::INITIALIZER: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    LOG(FATAL) << "Setting shape for initializers is not supported, please use "
                  "set_initializer_shape_i64 instead";
    break;
  }
  case Type::NODE_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    auto* value_info = graph_proto.mutable_value_info(fields_.index_);
    auto* shape_proto =
        value_info->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape_proto->clear_dim(); // Clear existing dimensions first
    for (const auto& dim : shape) {
      shape_proto->add_dim()->set_dim_value(dim);
    }
    LOG(WARNING) << "Setting shape for node outputs is not recommended, the "
                    "shape will override by next "
                    "shape inference pass";
    break;
  }
  case Type::GRAPH_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    auto* output = graph_proto.mutable_output(fields_.index_);
    auto* shape_proto =
        output->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape_proto->clear_dim(); // Clear existing dimensions first
    for (const auto& dim : shape) {
      shape_proto->add_dim()->set_dim_value(dim);
    }
    LOG(WARNING) << "Setting shape for graph outputs is not recommended, the "
                    "shape will override by next "
                    "shape inference pass";
    break;
  }
  case Type::INVALID:
  default:
    break;
  }
}

void NodeArgIndex::set_denotation(const std::vector<std::string>& denotation) {
  // Check if this NodeArgIndex is valid and exists
  if (!is_valid()) {
    throw std::runtime_error("Invalid NodeArgIndex");
  }

  auto graph_id = get_graph_id();
  auto* graph = graph_id.get_graph();
  if (graph == nullptr) {
    throw std::runtime_error("Graph not found for NodeArgIndex");
  }

  // Get the graph proto from the graph
  auto& graph_proto = graph->get_graph_proto();

  // Based on the type, set the denotation in different places
  switch (get_type()) {
  case Type::GRAPH_INPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.input_size()))
        << "Graph input index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.input_size() << ")";
    auto* input = graph_proto.mutable_input(fields_.index_);
    auto* shape_proto =
        input->mutable_type()->mutable_tensor_type()->mutable_shape();

    // Ensure we have the right number of dimensions
    if (shape_proto->dim_size() != static_cast<int>(denotation.size())) {
      LOG(WARNING) << "Denotation size (" << denotation.size()
                   << ") doesn't match shape dimensions ("
                   << shape_proto->dim_size() << ")";
      return;
    }

    // Set denotation for existing dimensions
    for (auto i = 0u; i < denotation.size(); ++i) {
      shape_proto->mutable_dim(i)->set_denotation(denotation[i]);
    }
    break;
  }
  case Type::INITIALIZER: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.initializer_size()))
        << "Initializer index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.initializer_size() << ")";
    LOG(FATAL) << "Setting denotation for initializers is not supported";
    break;
  }
  case Type::NODE_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.value_info_size()))
        << "Value info index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.value_info_size() << ")";
    auto* value_info = graph_proto.mutable_value_info(fields_.index_);
    auto* shape_proto =
        value_info->mutable_type()->mutable_tensor_type()->mutable_shape();

    // Ensure we have the right number of dimensions
    if (shape_proto->dim_size() != static_cast<int>(denotation.size())) {
      LOG(WARNING) << "Denotation size (" << denotation.size()
                   << ") doesn't match shape dimensions ("
                   << shape_proto->dim_size() << ")";
      return;
    }

    // Set denotation for existing dimensions
    for (auto i = 0u; i < denotation.size(); ++i) {
      shape_proto->mutable_dim(i)->set_denotation(denotation[i]);
    }
    break;
  }
  case Type::GRAPH_OUTPUT: {
    CHECK_LT(fields_.index_,
             static_cast<unsigned int>(graph_proto.output_size()))
        << "Graph output index " << fields_.index_
        << " is out of bounds (size: " << graph_proto.output_size() << ")";
    auto* output = graph_proto.mutable_output(fields_.index_);
    auto* shape_proto =
        output->mutable_type()->mutable_tensor_type()->mutable_shape();

    // Ensure we have the right number of dimensions
    if (shape_proto->dim_size() != static_cast<int>(denotation.size())) {
      LOG(WARNING) << "Denotation size (" << denotation.size()
                   << ") doesn't match shape dimensions ("
                   << shape_proto->dim_size() << ")";
      return;
    }

    // Set denotation for existing dimensions
    for (auto i = 0u; i < denotation.size(); ++i) {
      shape_proto->mutable_dim(i)->set_denotation(denotation[i]);
    }
    break;
  }
  case Type::INVALID:
  default:
    throw std::runtime_error("Invalid NodeArgIndex type for set_denotation");
  }
}

int NodeArgIndex::external_location(std::string& external_file, size_t& offset,
                                    size_t& size, size_t& checksum) const {

  // Check if this NodeArgIndex is valid and exists
  if (!is_valid_initializer()) {
    return 0; // Return 0 if invalid
  }

  auto graph_proto_ptr = get_graph_id().get_graph_proto();
  if (graph_proto_ptr == nullptr) {
    return 0;
  }

  // Get the graph proto from the graph
  const auto& graph_proto = *graph_proto_ptr;
  const auto& initializer = graph_proto.initializer(fields_.index_);
  // Extract external file, offset, size, and checksum
  if (!initializer.has_data_location()) {
    return 0; // No data location specified
  }
  if (initializer.data_location() != morphizen_onnx::TensorProto_DataLocation::
                                         TensorProto_DataLocation_EXTERNAL) {
    return 0; // Not an external data location
  }
  auto ret = 0;
  auto& external_data = initializer.external_data();
  auto external_data_size = external_data.size();
  for (auto i = 0; i < external_data_size; ++i) {
    const auto& entry = external_data.Get(i);
    if (entry.has_key() && entry.key() == "location") {
      external_file = entry.value();
      ret = 1;
    } else if (entry.has_key() && entry.key() == "offset") {
      offset = std::stoull(entry.value());
    } else if (entry.has_key() && entry.key() == "size") {
      size = std::stoull(entry.value());
    } else if (entry.has_key() && entry.key() == "checksum") {
      checksum = std::stoull(entry.value());
    }
  }
  return ret;
}

const morphizen_onnx::TensorProto*
NodeArgIndex::get_const_data_as_tensor(const Graph& g) const {

  // Check if this NodeArgIndex is valid
  if (!is_valid(g)) {
    MY_LOG(1) << "Invalid NodeArgIndex: " << to_string();
    return nullptr;
  }
  // Only initializers have tensor data
  if (!is_initializer()) {
    MY_LOG(1) << "NodeArgIndex is not an initializer: " << to_string();
    return nullptr;
  }
  // Get the graph proto from the graph
  const auto& graph_proto = g.get_graph_proto();

  // Check bounds
  if (fields_.index_ >=
      static_cast<unsigned int>(graph_proto.initializer_size())) {
    MY_LOG(1) << "NodeArgIndex is out of bounds: " << to_string();
    return nullptr;
  }

  // Return pointer to the tensor proto
  return &graph_proto.initializer(fields_.index_);
}

} // namespace morphizen
