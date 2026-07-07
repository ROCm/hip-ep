/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-graph-id.hpp"
#include "./mlir-graph-store.hpp"
#include <sstream>
#include <stdexcept>

namespace morphizen {
namespace mlir_impl {

GraphId::GraphId(uint32_t value) : value_(value) {}

GraphId GraphId::create_main_graph(uint32_t index) {
  auto ret = GraphId::from_raw(0);
  ret.fields_.index_ = index & 0x7FFFFFFF; // Ensure index is within 31 bits
  ret.fields_.is_staging_ = 0;
  return ret;
}

GraphId GraphId::create_staging_graph(uint32_t index) {
  auto ret = GraphId::from_raw(0);
  ret.fields_.index_ = index & 0x7FFFFFFF; // Ensure index is within 31 bits
  ret.fields_.is_staging_ = 1;
  return ret;
}

std::string GraphId::to_string() const {
  std::ostringstream oss;
  oss << "GraphId(";
  if (is_staging()) {
    oss << "staging=true, ";
  } else {
    oss << "staging=false, ";
  }
  oss << "index=" << get_index() << ")";
  return oss.str();
}

GraphId GraphId::from_raw(uint32_t value) { return GraphId(value); }

bool GraphId::is_staging() const { return fields_.is_staging_; }

MLIRGraph *GraphId::get_graph() const {
  return GraphStore::get_graph_by_id(get_index());
}

uint32_t GraphId::get_index() const { return fields_.index_; }

uint32_t GraphId::get_raw() const { return value_; }

bool GraphId::operator==(const GraphId &other) const {
  return value_ == other.value_;
}

bool GraphId::operator!=(const GraphId &other) const {
  return value_ != other.value_;
}

bool GraphId::operator<(const GraphId &other) const {
  return value_ < other.value_;
}

bool GraphId::operator<=(const GraphId &other) const {
  return value_ <= other.value_;
}

bool GraphId::operator>(const GraphId &other) const {
  return value_ > other.value_;
}

bool GraphId::operator>=(const GraphId &other) const {
  return value_ >= other.value_;
}

} // namespace mlir_impl
} // namespace morphizen

// Hash specialization implementation
namespace std {
size_t hash<morphizen::mlir_impl::GraphId>::operator()(
    const morphizen::mlir_impl::GraphId &graph_id) const {
  return hash<uint32_t>()(graph_id.get_raw());
}
} // namespace std
