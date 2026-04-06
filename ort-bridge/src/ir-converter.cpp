/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ir-converter-imp.hpp"
namespace morphizen {
ModelUniquePtr IRConverter::to_onnx_model(const ApiPtrs& api_ptrs,
                                          const OrtGraph& graph,
                                          const IRConverterConfig& config) {
  return IRConverterImp::to_onnx_model(api_ptrs, graph, config);
}
} // namespace morphizen
