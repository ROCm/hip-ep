/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-constants.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include <glog/logging.h>

namespace morphizen {
namespace mlir_impl {
// In onnx-mlir , all Tensor element type if is signed integer 8bit use `i8`
// all attribute type if is signed Integer use `si64`
mlir::Type onnxElementTypeToMlirType(int element_type, mlir::OpBuilder& builder,
                                     const llvm::SmallVector<int64_t>* shape) {
  // First get the element type
  mlir::Type elementType;
  switch (element_type) {
  case 1: // FLOAT
    elementType = builder.getF32Type();
    break;
  case 2: // UINT8
    elementType = builder.getIntegerType(8, false);
    break;
  case 3: // INT8
    // builder.getIntegerType(8, true) -> si8
    // builder.getIntegerType(8, false) -> ui8
    // builder.getIntegerType(8) -> i8
    elementType = builder.getIntegerType(8);
    break;
  case 6: // INT32
    elementType = builder.getI32Type();
    break;
  case 7: // INT64
    elementType = builder.getI64Type();
    break;
  case 11: // DOUBLE
    elementType = builder.getF64Type();
    break;
  case 10: // FLOAT16
    elementType = builder.getF16Type();
    break;
  default:
    LOG(WARNING) << "Unsupported element type: " << element_type
                 << ", using F32";
    elementType = builder.getF32Type();
    break;
  }

  // If no shape is provided, return just the element type
  if (!shape) {
    return elementType;
  }

  // for scalar in onnx-mlir
  // onnx node attribute scalar use `RankedTensorType + empty shape`.
  // node output scalar use `UnrankedTensorType`

  // Create tensor type with shape
  if (shape->empty()) {
    // Create unranked tensor type
    return mlir::UnrankedTensorType::get(elementType);
  } else {
    // Create ranked tensor type with shape
    return mlir::RankedTensorType::get(*shape, elementType);
  }
}

} // namespace mlir_impl
} // namespace morphizen
