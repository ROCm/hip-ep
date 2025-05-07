/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_wildcard.hpp"
#include "./pattern_log.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include "morphizen/pattern.pb.h"
namespace vaip_core {
PatternWildcard::PatternWildcard(int id) : Pattern(id) {}
PatternWildcard::~PatternWildcard() {}
std::string PatternWildcard::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("*");
  ret += std::string(")");
  return ret;
}

BinderBuilderPtr
PatternWildcard::match_uncached(const onnxruntime::Graph& /*graph*/,
                                const NodeInput& node_input,
                                const BinderBuilder& binder) const {
  MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", wildcard matched: "
            << (node_input.node != nullptr
                    ? node_as_string(*node_input.node)
                    : node_arg_as_string(*node_input.node_arg));
  return binder.add(this->get_id(), node_input);
}
void PatternWildcard::dump_to_proto_imp(RootPatternProto& /*pattern_proto*/,
                                        PatternProto& this_proto) const {
  this_proto.mutable_wildcard();
}
} // namespace vaip_core
