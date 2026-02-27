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
  case 1: // TensorProto_DataType_FLOAT
    elementType = builder.getF32Type();
    break;
  case 2: // TensorProto_DataType_UINT8
    elementType = builder.getIntegerType(8, false);
    break;
  case 3: // TensorProto_DataType_INT8
    // builder.getIntegerType(8, true) -> si8
    // builder.getIntegerType(8, false) -> ui8
    // builder.getIntegerType(8) -> i8
    elementType = builder.getIntegerType(8);
    break;
  case 4: // TensorProto_DataType_UINT16
    elementType = builder.getIntegerType(16, false);
    break;
  case 5: // TensorProto_DataType_INT16
    elementType = builder.getIntegerType(16);
    break;
  case 6: // TensorProto_DataType_INT32
    elementType = builder.getI32Type();
    break;
  case 7: // TensorProto_DataType_INT64
    elementType = builder.getI64Type();
    break;
  case 9: // TensorProto_DataType_BOOL
    // ONNX stores BOOL as 8-bit (1 byte): 0x00=False, 0x01=True
    // Use ui8 to match ONNX physical storage format
    elementType = builder.getIntegerType(8, false);
    break;
  case 10: // TensorProto_DataType_FLOAT16
    elementType = builder.getF16Type();
    break;
  case 11: // TensorProto_DataType_DOUBLE
    elementType = builder.getF64Type();
    break;
  default:
    // TensorProto_DataType_UNDEFINED = 0,
    // TensorProto_DataType_STRING = 8,
    // TensorProto_DataType_UINT32 = 12,
    // TensorProto_DataType_UINT64 = 13,
    // TensorProto_DataType_COMPLEX64 = 14,
    // TensorProto_DataType_COMPLEX128 = 15,
    // TensorProto_DataType_BFLOAT16 = 16,
    // TensorProto_DataType_FLOAT8E4M3FN = 17,
    // TensorProto_DataType_FLOAT8E4M3FNUZ = 18,
    // TensorProto_DataType_FLOAT8E5M2 = 19,
    // TensorProto_DataType_FLOAT8E5M2FNUZ = 20,
    // TensorProto_DataType_UINT4 = 21,
    // TensorProto_DataType_INT4 = 22
    LOG(WARNING) << "Unsupported element type: " << element_type
                 << ", using F32";
    elementType = builder.getF32Type();
    break;
  }

  // If no shape is provided, return just the element type
  if (!shape) {
    return elementType;
  }

  // Create tensor type with shape
  // Empty shape = scalar (rank-0 tensor), e.g. tensor<i32>
  // UnrankedTensorType (tensor<*xT>) cannot be lowered by dialect converters
  // because the rank is unknown, so we always use RankedTensorType.
  if (shape->empty()) {
    return mlir::RankedTensorType::get({}, elementType);
  } else {
    // Create ranked tensor type with shape
    return mlir::RankedTensorType::get(*shape, elementType);
  }
}

} // namespace mlir_impl
} // namespace morphizen
