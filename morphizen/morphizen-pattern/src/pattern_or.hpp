/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <regex>

#include "morphizen/pattern.hpp"
namespace morphizen {

class PatternOr : public Pattern {
public:
  explicit PatternOr(int id, std::vector<std::shared_ptr<Pattern>> args);
  ~PatternOr();

private:
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph &graph, const NodeInput &node_input,
                 const BinderBuilder &cached_binder) const override final;
  virtual std::string debug_string() const override;

private:
  std::vector<std::shared_ptr<Pattern>> or_patterns_;
};
} // namespace morphizen
