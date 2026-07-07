/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-node-arg-index.hpp"
#include "./mlir-graph-id.hpp"
#include "./mlir-graph.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include <glog/logging.h>
#include <memory>

// For now, provide minimal implementations of the MLIRNodeArgIndex methods
// These can be expanded later when the full Graph infrastructure is available

namespace morphizen {
namespace mlir_impl {

// Constructor implementations
MLIRNodeArgIndex::MLIRNodeArgIndex() : fields_{0, 0, 0} {}

MLIRNodeArgIndex::MLIRNodeArgIndex(unsigned int index, Type type)
    : fields_{index & 0x1FFFFFFF, static_cast<unsigned int>(type) & 0x7, 0} {}

MLIRNodeArgIndex::MLIRNodeArgIndex(unsigned int index, Type type,
                                   GraphId graph_id)
    : fields_{index & 0x1FFFFFFF, static_cast<unsigned int>(type) & 0x7,
              graph_id.get_raw()} {}

// Static factory methods
MLIRNodeArgIndex MLIRNodeArgIndex::invalid() {
  return MLIRNodeArgIndex(0, Type::INVALID, GraphId::from_raw(0));
}

MLIRNodeArgIndex MLIRNodeArgIndex::graph_input(unsigned int index,
                                               GraphId graph_id) {
  return MLIRNodeArgIndex(index, Type::GRAPH_INPUT, graph_id);
}

MLIRNodeArgIndex MLIRNodeArgIndex::initializer(unsigned int index,
                                               GraphId graph_id) {
  return MLIRNodeArgIndex(index, Type::INITIALIZER, graph_id);
}

MLIRNodeArgIndex MLIRNodeArgIndex::node_output(unsigned int index,
                                               GraphId graph_id) {
  return MLIRNodeArgIndex(index, Type::NODE_OUTPUT, graph_id);
}

MLIRNodeArgIndex MLIRNodeArgIndex::graph_output(unsigned int index,
                                                GraphId graph_id) {
  return MLIRNodeArgIndex(index, Type::GRAPH_OUTPUT, graph_id);
}

MLIRNodeArgIndex
MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(const void* ptr) {
  auto ret = MLIRNodeArgIndex::invalid();
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

MLIRNodeArgIndex MLIRNodeArgIndex::from_uint64(uint64_t v) {
  auto ret = MLIRNodeArgIndex::invalid();
  ret.value_ = v;
  return ret;
}

// Getter methods
unsigned int MLIRNodeArgIndex::get_index() const { return fields_.index_; }

MLIRNodeArgIndex::Type MLIRNodeArgIndex::get_type() const {
  return static_cast<Type>(fields_.type_);
}

GraphId MLIRNodeArgIndex::get_graph_id() const {
  return GraphId::from_raw(fields_.graph_id_);
}

// Validity and type checking methods
bool MLIRNodeArgIndex::is_valid() const { return get_type() != Type::INVALID; }

bool MLIRNodeArgIndex::is_constant() const {
  if (!is_valid()) {
    return false;
  }
  // Get the underlying MLIRNodeArg and delegate to its isConstantValue method
  const MLIRNodeArg& nodeArg = get_node_arg();
  return nodeArg.isConstantValue();
}

// Operator overloads
bool MLIRNodeArgIndex::operator==(const MLIRNodeArgIndex& other) const {
  return value_ == other.value_;
}

bool MLIRNodeArgIndex::operator!=(const MLIRNodeArgIndex& other) const {
  return value_ != other.value_;
}

std::size_t MLIRNodeArgIndex::hash() const {
  return std::hash<uint64_t>()(value_);
}

// Methods that require Graph infrastructure - stub implementations for now

bool MLIRNodeArgIndex::exists() const {
  // TODO: Implement when Graph class is available
  return is_valid();
}

std::optional<llvm::SmallVector<int64_t>>
MLIRNodeArgIndex::get_shape_i64() const {
  return get_node_arg().getShape();
}

void MLIRNodeArgIndex::set_shape_i64(const llvm::SmallVector<int64_t>& shape) {
  const_cast<MLIRNodeArg&>(get_node_arg()).setShape(shape);
}

std::vector<std::string>* MLIRNodeArgIndex::get_denotation_unsafe() const {
  // TODO: Implement when Graph class is available
  LOG(WARNING) << "get_denotation_unsafe not implemented in mlir_impl";
  return nullptr;
}

void MLIRNodeArgIndex::set_denotation(
    const std::vector<std::string>& /*denotation*/) {
  // TODO: Implement when Graph class is available
  LOG(WARNING) << "set_denotation not implemented in mlir_impl";
}

int MLIRNodeArgIndex::get_element_type() const {
  return get_node_arg().getElementType();
}

void MLIRNodeArgIndex::set_element_type(int type) {
  const_cast<MLIRNodeArg&>(get_node_arg()).setElementType(type);
}

int MLIRNodeArgIndex::external_location(std::string& external_file,
                                        size_t& offset, size_t& size,
                                        size_t& checksum) const {
  auto& arg = get_node_arg();

  // Stage 1: pre-materialization — ExternalRef lives in TensorDesc
  if (auto* ext = arg.getExternalRef()) {
    external_file = ext->location;
    offset = ext->offset;
    size = ext->size;
    checksum = 0;
    return 0;
  }

  // Stage 2: post-materialization — info embedded in onnx.Constant attributes
  auto& val = arg.getValue();
  if (val) {
    if (auto defining_op = val.getDefiningOp()) {
      if (defining_op->getName().getStringRef() != "onnx.Constant")
        return -1;
      auto loc_attr = defining_op->getAttrOfType<mlir::StringAttr>("location");
      if (!loc_attr)
        return -1;
      external_file = loc_attr.getValue().str();
      if (auto off = defining_op->getAttrOfType<mlir::IntegerAttr>("offset"))
        offset = static_cast<size_t>(off.getInt());
      if (auto sz = defining_op->getAttrOfType<mlir::IntegerAttr>("size"))
        size = static_cast<size_t>(sz.getInt());
      checksum = 0;
      return 0;
    }
  }

  return -1;
}

const std::string& MLIRNodeArgIndex::get_name() const {
  return get_node_arg().getName();
}

std::string MLIRNodeArgIndex::to_string() const {
  std::string type_str;
  switch (get_type()) {
  case Type::INVALID:
    type_str = "INVALID";
    break;
  case Type::GRAPH_INPUT:
    type_str = "GRAPH_INPUT";
    break;
  case Type::INITIALIZER:
    type_str = "INITIALIZER";
    break;
  case Type::NODE_OUTPUT:
    type_str = "NODE_OUTPUT";
    break;
  case Type::GRAPH_OUTPUT:
    type_str = "GRAPH_OUTPUT";
    break;
  }

  auto graph_id = get_graph_id();
  return "[" + type_str + ":" + std::to_string(get_index()) + ":g" +
         (graph_id.is_staging() ? "S" : "M") +
         std::to_string(graph_id.get_index()) + "]";
}

const void* MLIRNodeArgIndex::to_morphizen_core_node_arg_ptr() const {
  // TODO: Implement proper conversion
  if (!exists()) {
    return nullptr;
  }
  // Convert the value back to a pointer
  // This reverses the mapping done in from_morphizen_core_node_arg_ptr
  return reinterpret_cast<const void*>(static_cast<uintptr_t>(value_));
}

uint64_t MLIRNodeArgIndex::to_uint64() const { return value_; }
const MLIRNodeArg& MLIRNodeArgIndex::get_node_arg() const {
  auto node_arg = get_graph().get_node_arg(*this);
  CHECK(node_arg != nullptr)
      << "MLIRNodeArgIndex does not reference a valid MLIRNodeArg";
  return *node_arg;
}

//
const MLIRNodeArg& MLIRNodeArgIndex::get_const_data_as_tensor() const {
  // Get the underlying MLIRNodeArg and retrieve its constant data as a tensor
  // NodeArg  (data ) -> tensor
  // NodeArg -> tensor  [X]
  // maybe need check ths node_arg is tensor or not .
  CHECK(is_constant()) << "NodeArg is not constant";
  return get_node_arg();
}
const MLIRGraph& MLIRNodeArgIndex::get_graph() const {
  auto graph = GraphId::from_raw(fields_.graph_id_).get_graph();
  CHECK(graph != nullptr)
      << "MLIRNodeArgIndex does not reference a valid MLIRGraph";
  return *graph;
}

mlir::Operation* MLIRNodeArgIndex::get_producer_node() const {
  // Get the node argument name and delegate to graph's producer_node method
  if (!is_valid()) {
    return nullptr;
  }
  const std::string& node_arg_name = get_name();
  return get_graph().producer_node(node_arg_name);
}

} // namespace mlir_impl
} // namespace morphizen
