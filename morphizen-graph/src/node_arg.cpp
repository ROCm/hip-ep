/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>
//
#include "morphizen/graph.hpp"
#include "morphizen/node_arg.hpp"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>

namespace vaip_core {

static std::string
shape_proto_as_string(const std::vector<int64_t>& shape,
                      const std::vector<std::string>& denotation) {
  CHECK((&denotation) != nullptr);
  std::ostringstream str;
  auto size = shape.size();
  auto is_empty_denotation = denotation.empty();
  // The denotation is not support in MLIR-backend now
  // CHECK_EQ(size, denotation.size());
  str << "[";
  for (auto i = 0u; i < size; ++i) {
    if (i != 0) {
      str << ",";
    }
    auto has_denotation = is_empty_denotation || denotation[i].empty();
    if (!has_denotation) {
      str << denotation[i] << "=" << shape[i];
    } else {
      str << shape[i];
    }
  }
  str << "]";
  return str.str();
}

static std::string type_proto_as_string(const NodeArg& node_arg) {
  std::ostringstream str;
  auto element_type = VAIP_ORT_API(node_arg_get_element_type)(node_arg);
  str << "(";
  if (element_type >= 0) {
    auto shape = node_arg_get_shape_i64(node_arg);
    auto denotation = node_arg_get_denotation(node_arg);
    str << "ty=" << element_type << ",shape="
        << ((shape != nullptr) ? shape_proto_as_string(*shape, *denotation)
                               : std::string("UNKWN"));
  } else {
    str << "UNKNOWN_TYPE";
  }
  str << ")";
  return str.str();
}

VAIP_DLL_SPEC bool node_arg_exists(const NodeArg& node_arg) {
  return VAIP_ORT_API(node_arg_is_exists)(node_arg);
}

VAIP_DLL_SPEC std::string node_arg_as_string(const NodeArg& node_arg) {
  std::ostringstream str;
  if (node_arg_exists(node_arg)) {
    auto name = node_arg_get_name(node_arg);
    // node_arg name == "" means node input is optional
    if (name != "") {
      str << node_arg_get_name(node_arg) << ":"
          << type_proto_as_string(node_arg);
    }
  } else {
    str << "N/A";
  }
  return str.str();
}

VAIP_DLL_SPEC const std::string& node_arg_get_name(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return VAIP_ORT_API(node_arg_get_name_unsafe)(node_arg);
}

VAIP_DLL_SPEC
std::unique_ptr<std::vector<int64_t>>
node_arg_get_shape_i64(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto shape = VAIP_ORT_API(node_arg_get_shape_i64_unsafe)(node_arg);
  if (nullptr == shape.get()) {
    return std::unique_ptr<std::vector<int64_t>>();
  }
  return std::make_unique<std::vector<int64_t>>(*shape);
}
VAIP_DLL_SPEC std::unique_ptr<std::vector<std::string>>
node_arg_get_denotation(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto denotation = VAIP_ORT_API(node_arg_get_denotation_unsafe)(node_arg);
  if (nullptr == denotation.get()) {
    return std::unique_ptr<std::vector<std::string>>();
  }
  return std::make_unique<std::vector<std::string>>(*denotation);
}

VAIP_DLL_SPEC int node_arg_get_element_type(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  auto element_type = VAIP_ORT_API(node_arg_get_element_type)(node_arg);
  CHECK_GE(element_type, 0) << "only support TypeProto Tensor!";
  return element_type;
}

VAIP_DLL_SPEC bool node_arg_is_unknown_shape(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto shape = node_arg_get_shape_i64(node_arg);
  return nullptr == shape;
}
VAIP_DLL_SPEC bool node_arg_is_scalar(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto shape = node_arg_get_shape_i64(node_arg);
  if (nullptr == shape)
    return false;

  return shape->empty();
}
VAIP_DLL_SPEC bool node_arg_is_zero_shape(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto shape = node_arg_get_shape_i64(node_arg);
  if (nullptr == shape)
    return false;

  return !std::all_of(shape->begin(), shape->end(),
                      [](int64_t v) { return v != 0; });
}
VAIP_DLL_SPEC bool node_arg_is_dynamic_shape(const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";

  auto shape = node_arg_get_shape_i64(node_arg);
  if (nullptr == shape)
    return false;

  return !std::all_of(shape->begin(), shape->end(), [](int64_t v) {
    // xilinx op does not support shape = 0
    return v >= 0;
  });
}
#if VAIP_ORT_API_MAJOR >= 7
static Graph* get_original_graph(const std::string& ptr) {
  uintptr_t ptr1 = std::stoull(ptr);
  return (Graph*)ptr1;
}
#endif
VAIP_DLL_SPEC const TensorProto&
node_arg_get_const_data_as_tensor(const Graph& graph, const NodeArg& node_arg) {
#if VAIP_ORT_API_MAJOR >= 7
  std::string location = "";
  location.reserve(1024);
  size_t size = 0;
  size_t offset = 0;
  size_t checksum = 0;
  int external_data = VAIP_ORT_API(node_arg_external_location)(
      graph, node_arg, location, size, offset, checksum);
  if (external_data && !location.empty() && location.front() == '<') {
    auto original_graph = get_original_graph(location.substr(1));
    return node_arg_get_const_data_as_tensor(*original_graph, node_arg);
  }
  CHECK_LE(location.size(), 1024)
      << "External data location is too long: " << location.size();
#endif
  return VAIP_ORT_API(node_arg_get_const_data_as_tensor)(graph, node_arg);
}

// NOTE: node_arg_get_const_data_as_* functions moved to
// vaip-core/src/node_arg_const_data.cpp These high-level functions depend on
// tensor_proto (vaip-core component)

bool node_arg_is_constant(const Graph& graph, const NodeArg& node_arg) {
  return VAIP_ORT_API(node_arg_is_constant)(graph, node_arg);
}

NodeArg& node_arg_new(Graph& graph, const std::string& name,
                      const std::vector<int64_t>* shape, int element_type) {
  return VAIP_ORT_API(node_arg_new)(graph, name, shape, element_type);
}

} // namespace vaip_core
namespace vaip_cxx {

bool NodeArgConstRef::is_graph_input() const {
  auto g = GraphConstRef(graph_);
  auto inputs = g.inputs();
  return std::find(inputs.begin(), inputs.end(), *this) != inputs.end();
}
bool NodeArgConstRef::is_graph_output() const {
  auto g = GraphConstRef(graph_);
  auto outputs = g.outputs();
  return std::find(outputs.begin(), outputs.end(), *this) != outputs.end();
}
std::vector<NodeConstRef> NodeArgConstRef::find_consumers() const {
  return GraphConstRef(graph_).find_consumers(name());
}
std::optional<NodeConstRef> NodeArgConstRef::find_producer() const {
  return GraphConstRef(graph_).find_node(name());
}

// NOTE: NodeArgConstRef::const_data_as_* methods moved to
// vaip-core/src/node_arg_const_data.cpp These methods depend on
// node_arg_get_const_data_as_* and tensor_proto functions

} // namespace vaip_cxx
