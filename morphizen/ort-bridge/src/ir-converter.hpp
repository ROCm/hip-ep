/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include "./morphizen/morphizen-ort-api-ext.hpp"
namespace morphizen {
using GraphUniquePtr =
    std::unique_ptr<onnxruntime::Graph, void (*)(onnxruntime::Graph *)>;
using ModelUniquePtr =
    std::unique_ptr<onnxruntime::Model, void (*)(onnxruntime::Model *)>;
} // namespace morphizen

namespace morphizen {

struct IRConverterConfig {
  // Tensors larger than this threshold use the no-copy memory-address
  // external data mechanism (mirrors ORT kSmallTensorExternalDataThreshold).
  // Set to SIZE_MAX to disable the optimisation entirely.
  size_t external_data_threshold = 127;
  // Persistent/prebuilt artifacts embed initializer bytes. Bind those bytes
  // into cache identity only for paths that can reuse such an artifact.
  bool hash_initializer_data = false;
};

class IRConverter {

public:
  static ModelUniquePtr to_onnx_model(const ApiPtrs &api_ptrs,
                                      const OrtGraph &graph,
                                      const IRConverterConfig &config = {});
};
} // namespace morphizen
