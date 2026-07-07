/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "morphizen/pattern.hpp"

#include <optional>

namespace morphizen {

class PatternNodeOutputArg : public Pattern {
public:
  explicit PatternNodeOutputArg(int id, std::shared_ptr<Pattern> node_pattern,
                                size_t output_arg_index);

  ~PatternNodeOutputArg();

public:
  BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& cached_binder) const override final;

  std::string debug_string() const override;

  void dump_to_proto_imp(RootPatternProto& pattern_proto,
                         PatternProto& this_proto) const override final;

private:
  std::shared_ptr<Pattern> node_pattern_;
  size_t output_arg_index_;
};
} // namespace morphizen
