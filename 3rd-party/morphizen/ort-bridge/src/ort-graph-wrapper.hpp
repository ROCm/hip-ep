/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <gsl/gsl>
#include <map>
#include <string>
#include <vector>

namespace morphizen {

/**
 * @brief Wrapper around ORT's incomplete Graph implementation
 *
 * This class provides a workaround for ONNX Runtime's incomplete graph API.
 * It patches missing functionality and provides a consistent interface to
 * access graph properties from ORT's internal graph representation.
 *
 * Note: This is a temporary solution until ORT provides proper graph access.
 */
struct OrtGraphWrapper
    : public ApiPtrs,
      Ort::detail::Base<Ort::detail::Unowned<const OrtGraph>> {
  using B = Ort::detail::Base<Ort::detail::Unowned<const OrtGraph>>;

  OrtGraphWrapper(const ApiPtrs &api_ptrs, const OrtGraph &graph);
  const OrtGraph &get() const;

  bool is_subgraph() const;
  const char *name() const;
  int64_t ir_version() const;

  // Convenience methods that immediately copy to vector (for short-lived
  // access)
  Ort::ModelMetadata get_model_metadata() const;
  std::vector<const OrtNode *> nodes() const;
  std::vector<const OrtValueInfo *> inputs() const;
  std::vector<const OrtValueInfo *> outputs() const;
  std::vector<const OrtValueInfo *> initializers() const;
  // Get the opset requirements for this graph
  std::map<std::string, int> guess_opset() const;
};

} // namespace morphizen
