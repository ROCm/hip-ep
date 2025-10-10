/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./mlir-deferred-value.hpp"

namespace mlir_impl {

MLIRDeferredValue::MLIRDeferredValue(const std::string& name,
                                     const std::vector<int>& shape,
                                     int element_type) {
  impl = MLIRDeferredValueImpl(name, shape, element_type);
}
} // namespace mlir_impl
