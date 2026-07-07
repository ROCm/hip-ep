/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_graph_input.hpp"
#include <sstream>

#include "morphizen/graph.hpp"

#include "./pattern_log.hpp"

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4946)
#endif

#include "morphizen/pattern.pb.h"

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

namespace morphizen {
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
                         ? morphizen_cxx::NodeConstRef::from_node(
                               graph, *node_input.node)
                               .to_string()
                         : morphizen_cxx::NodeArgConstRef::from_node_arg(
                               graph, *node_input.node_arg)
                               .to_string());
  } else {
    MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", graph input matched."
              << (node_input.node != nullptr
                      ? morphizen_cxx::NodeConstRef::from_node(graph,
                                                               *node_input.node)
                            .to_string()
                      : morphizen_cxx::NodeArgConstRef::from_node_arg(
                            graph, *node_input.node_arg)
                            .to_string());
  }
  return ret;
}
void PatternGraphInput::dump_to_proto_imp(RootPatternProto& /*pattern_proto*/,
                                          PatternProto& this_proto) const {
  this_proto.mutable_graph_input();
}
} // namespace morphizen
