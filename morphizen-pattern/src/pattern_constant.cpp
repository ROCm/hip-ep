/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_constant.hpp"
#include <sstream>

#include "./pattern_log.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4946)
#endif

#include "morphizen/pattern.pb.h"

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

namespace morphizen {
PatternConstant::PatternConstant(int id) : Pattern(id) {}
PatternConstant::~PatternConstant() {}
std::string PatternConstant::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("Constant");
  ret += std::string(")");
  return ret;
}

std::string PatternConstant::virtualize_label() const {
  std::ostringstream str;
  str << "[" << this->get_id() << "] Constant";
  return str.str();
}

BinderBuilderPtr
PatternConstant::match_uncached(const onnxruntime::Graph& graph,
                                const NodeInput& node_input,
                                const BinderBuilder& binder) const {
  auto ret = BinderBuilderPtr();
  if (node_input.node != nullptr) {
    auto node_ref =
        morphizen_cxx::NodeConstRef::from_node(graph, *node_input.node);
    if (node_ref.op_type() == "Constant") {
      ret = binder.add(this->get_id(), node_input);
    }
  } else {
    bool is_constant = node_arg_is_constant(graph, *node_input.node_arg);
    if (is_constant) {
      ret = binder.add(this->get_id(), node_input);
    }
  }
  if (ret == nullptr) {
    MATCH_FAILED << "not a constant: "
                 << (node_input.node != nullptr
                         ? morphizen_cxx::NodeConstRef::from_node(
                               graph, *node_input.node)
                               .to_string()
                         : morphizen_cxx::NodeArgConstRef::from_node_arg(
                               graph, *node_input.node_arg)
                               .to_string());
  } else {
    MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", constant matched: "
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
void PatternConstant::dump_to_proto_imp(RootPatternProto& /*pattern_proto*/,
                                        PatternProto& this_proto) const {
  this_proto.mutable_constant();
}
} // namespace morphizen
