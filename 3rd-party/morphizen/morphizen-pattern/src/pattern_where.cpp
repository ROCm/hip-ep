/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_where.hpp"

namespace morphizen {
PatternWhere::PatternWhere(
    std::unique_ptr<Pattern> pattern,
    std::function<bool(const NodeInput &)> condition_on_node_input)
    : Pattern(pattern->get_id()), pattern_(std::move(pattern)),
      condition_on_node_input_(condition_on_node_input) {}

PatternWhere::~PatternWhere() {}

std::string PatternWhere::debug_string() const {
  auto ret = std::string("[Where] #") + std::to_string(this->get_id()) +
             std::string(" {");
  ret += pattern_->debug_string();
  ret += std::string(" Where function");
  // TODO ret add function point address
  ret += std::string("}");
  return ret;
}

BinderBuilderPtr
PatternWhere::match_uncached(const onnxruntime::Graph &graph,
                             const NodeInput &node_input,
                             const BinderBuilder &binder) const {
  if (condition_on_node_input_(node_input)) {
    return pattern_->match_cached(graph, node_input, binder);
  }
  return nullptr;
}
} // namespace morphizen
