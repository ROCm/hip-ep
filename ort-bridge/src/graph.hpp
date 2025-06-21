/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include <gsl/gsl>
#include <map>
#include <string>
#include <vector>

namespace morphizen {

struct Graph : public ApiPtrs,
               Ort::detail::Base<Ort::detail::Unowned<const OrtGraph>> {
  using B = Ort::detail::Base<Ort::detail::Unowned<const OrtGraph>>;
  Graph(const ApiPtrs& api_ptrs, const OrtGraph& graph);
  const char* name() const;
  int64_t ir_version() const;

  // RAII-managed array access (preferred for long-lived access)
  OrtArraySpan<const OrtNode* const> nodes_managed() const;
  OrtArraySpan<const OrtValueInfo* const> inputs_managed() const;
  OrtArraySpan<const OrtValueInfo* const> outputs_managed() const;
  OrtArraySpan<const OrtValueInfo* const> initializers_managed() const;

  // Convenience methods that immediately copy to vector (for short-lived
  // access)
  std::vector<const OrtNode*> nodes() const;
  std::vector<const OrtValueInfo*> inputs() const;
  std::vector<const OrtValueInfo*> outputs() const;
  std::vector<const OrtValueInfo*> initializers() const;

  // Get the opset requirements for this graph
  std::map<std::string, int> guess_opset() const;
};

} // namespace morphizen
