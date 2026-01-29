/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_graph_input.hpp"
#include <sstream>

#include "morphizen/graph.hpp"

#include "./pattern_log.hpp"
#include "morphizen/pattern.pb.h"

namespace vaip_core {
PatternGraphInput::PatternGraphInput(int id) : Pattern(id) {}
PatternGraphInput::~PatternGraphInput() {}

std::string PatternGraphInput::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("GraphInput");
  ret += std::string(")");
  return ret;
}

std::string PatternGraphInput::virtualize_label() const {
  std::ostringstream str;
  str << "[" << this->get_id() << "] GraphInput";
  return str.str();
}

BinderBuilderPtr
PatternGraphInput::match_uncached(const onnxruntime::Graph& graph,
                                  const NodeInput& node_input,
                                  const BinderBuilder& binder) const {
  auto ret = BinderBuilderPtr();
  if (node_input.node == nullptr) {
    auto inputs = graph_get_inputs(graph);
    auto it = std::find(inputs.begin(), inputs.end(), node_input.node_arg);
    if (it != inputs.end()) {
      ret = binder.add(get_id(), node_input);
    }
  }
  if (ret == nullptr) {
    MATCH_FAILED << "not a graph input: "
                 << (node_input.node != nullptr
                         ? node_as_string(*node_input.node)
                         : node_arg_as_string(*node_input.node_arg));
  } else {
    MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", graph input matched."
              << (node_input.node != nullptr
                      ? node_as_string(*node_input.node)
                      : node_arg_as_string(*node_input.node_arg));
  }
  return ret;
}
void PatternGraphInput::dump_to_proto_imp(RootPatternProto& /*pattern_proto*/,
                                          PatternProto& this_proto) const {
  this_proto.mutable_graph_input();
}
} // namespace vaip_core
