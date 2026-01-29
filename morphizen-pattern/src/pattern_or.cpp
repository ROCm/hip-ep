/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_or.hpp"

#include "./pattern_log.hpp"

namespace morphizen {
PatternOr::PatternOr(int id, std::vector<std::shared_ptr<Pattern>> args)
    : Pattern(id), or_patterns_(std::move(args)) {}
PatternOr::~PatternOr() {}
std::string PatternOr::debug_string() const {
  auto ret =
      std::string("#") + std::to_string(this->get_id()) + std::string("(");
  if (!or_patterns_.empty()) {
    ret += "OR ";
    ret += or_patterns_[0]->debug_string();
    for (auto i = 1u; i < or_patterns_.size(); i++) {
      ret += ", " + or_patterns_[i]->debug_string();
    }
  }
  ret += std::string(")");
  return ret;
}

BinderBuilderPtr PatternOr::match_uncached(const onnxruntime::Graph& graph,
                                           const NodeInput& node_input,
                                           const BinderBuilder& binder) const {

  auto ret = BinderBuilderPtr();
  auto index = 0u;
  auto size = or_patterns_.size();
  for (auto& p : or_patterns_) {
    ret = p->match_cached(graph, node_input, binder);
    if (ret) {
      MY_LOG(1) << "MATCH OK. ID=" << get_id() << " " << index << "/" << size
                << " OK "
                << ", node=" << node_input_as_string(graph, node_input);
      return ret->add(this->get_id(), node_input);
    } else {
      MATCH_FAILED << " " << index << "/" << size
                   << ", node=" << node_input_as_string(graph, node_input);
    }
    index = index + 1;
  }
  MATCH_FAILED << " ALL FAIL "
               << ", node=" << node_input_as_string(graph, node_input);
  return nullptr;
}

} // namespace morphizen
