/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "morphizen/pattern.hpp"

namespace morphizen {

class PatternGraphOutput : public Pattern {
public:
  explicit PatternGraphOutput(int id, std::shared_ptr<Pattern> arg);
  explicit PatternGraphOutput(int id, std::shared_ptr<Pattern> arg,
                              size_t graph_output_index);
  explicit PatternGraphOutput(int id, std::shared_ptr<Pattern> arg,
                              const std::string& graph_output_name);

  ~PatternGraphOutput();

public:
  BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& cached_binder) const override final;

  std::string debug_string() const override;

  void dump_to_proto_imp(RootPatternProto& pattern_proto,
                         PatternProto& this_proto) const override final;

private:
  std::shared_ptr<Pattern> arg_;
  std::optional<size_t> graph_output_index_;
  std::optional<std::string> graph_output_name_;
};
} // namespace morphizen
