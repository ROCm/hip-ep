/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/Value.h"
#include <string>
#include <vector>

namespace morphizen {
namespace mlir_impl {
/**
 * @brief MLIRDeferredValue class
 *
 * This class represents a deferred value in MLIR, which is a specialized
 * type of value that can be used in various MLIR operations. It inherits from
 * mlir::Value.
 *
 * mlir::Value contains an implemetation pointer to the real `MLIRNodeArg` when
 * it is not meterialized, while `MLIRNodeArg` contains name, shape and element
 * type information.
 *
 * when this class is mertalized, the implementation pointer will point to a
 * OpResultImpl so that this class can be `llvm::dyn_cast`ed to
 * `mlir::OpResult` or `mlir::BlockArgument`.
 */
class alignas(8) MLIRDeferredValueImpl : public mlir::ValueImpl {

public:
  explicit MLIRDeferredValueImpl(const std::string& name,
                                 const std::vector<int>& shape,
                                 int element_type);
};
class MLIRDeferredValue : public mlir::Value {
public:
  using Value::Value;

  static bool classof(Value value) {
    return value.getImpl()->getKind() == 5; // 5 is used for this class.
  }

public:
  explicit MLIRDeferredValue(const std::string& name,
                             const std::vector<int>& shape, int element_type);
  MLIRDeferredValue() = delete;
  MLIRDeferredValue(const MLIRDeferredValue&) = delete;
  ~MLIRDeferredValue();

public:
  MLIRDeferredValue& operator=(const MLIRDeferredValue&) = delete;

public:
  MLIRNodeArg* getImpl() {
    return reinterpret_cast<MLIRNodeArg*>(typeAndKind.getPointer());
  }
};

} // namespace mlir_impl
} // namespace morphizen
