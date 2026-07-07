/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <regex>

#include "morphizen/pattern.hpp"
namespace morphizen {

class PatternWhere : public Pattern {
public:
  explicit PatternWhere(
      std::unique_ptr<Pattern> pattern,
      std::function<bool(const NodeInput &)> condition_on_node_input);

  ~PatternWhere();

private:
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph &graph, const NodeInput &node_input,
                 const BinderBuilder &binder_builder) const override final;
  virtual std::string debug_string() const override final;

private:
  std::unique_ptr<Pattern> pattern_;
  std::function<bool(const NodeInput &)> condition_on_node_input_;
};
} // namespace morphizen
