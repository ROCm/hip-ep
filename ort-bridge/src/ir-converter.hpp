/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include "./morphizen/morphizen-ort-api-ext.hpp"
namespace morphizen {
using GraphUniquePtr =
    std::unique_ptr<onnxruntime::Graph, void (*)(onnxruntime::Graph*)>;
using ModelUniquePtr =
    std::unique_ptr<onnxruntime::Model, void (*)(onnxruntime::Model*)>;
} // namespace morphizen

namespace morphizen {
class IRConverter {

public:
  static ModelUniquePtr
  to_onnx_model(const ApiPtrs& api_ptrs,
                const OrtGraph& graph); // Instance method for converting graph
};
} // namespace morphizen
