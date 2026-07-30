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
mlir::Type onnxElementTypeToMlirElementType(int element_type,
                                            mlir::OpBuilder &builder) {
  switch (element_type) {
  case 1: // TensorProto_DataType_FLOAT
    return builder.getF32Type();
  case 2: // TensorProto_DataType_UINT8
    return builder.getIntegerType(8, false);
  case 3: // TensorProto_DataType_INT8
    // builder.getIntegerType(8, true) -> si8
    // builder.getIntegerType(8, false) -> ui8
    // builder.getIntegerType(8) -> i8
    return builder.getIntegerType(8);
  case 4: // TensorProto_DataType_UINT16
    return builder.getIntegerType(16, false);
  case 5: // TensorProto_DataType_INT16
    return builder.getIntegerType(16);
  case 6: // TensorProto_DataType_INT32
    return builder.getI32Type();
  case 7: // TensorProto_DataType_INT64
    return builder.getI64Type();
  case 9: // TensorProto_DataType_BOOL
    // ONNX stores BOOL as 8-bit (1 byte): 0x00=False, 0x01=True
    // Use ui8 to match ONNX physical storage format
    return builder.getIntegerType(8, false);
  case 10: // TensorProto_DataType_FLOAT16
    return builder.getF16Type();
  case 11: // TensorProto_DataType_DOUBLE
    return builder.getF64Type();
  case 21: // TensorProto_DataType_UINT4
    // No native 4-bit type in the EP: sub-byte quantized weights (e.g.
    // GatherBlockQuantized/MatMulNBits) ride as byte tensors, two nibbles per
    // byte. Carry the *signedness* on the byte element type so the downstream
    // dtype mapping (getHipdnnDataType: ui8 -> UINT8) selects the unsigned
    // dequant path (default zero-point = 2^(bits-1)). The packed-byte shape is
    // handled separately by the initializer legalization / runtime unpack.
    return builder.getIntegerType(8, false); // ui8
  case 22: // TensorProto_DataType_INT4
    // Signed 4-bit -> signless i8 (getHipdnnDataType: i8 -> INT8) so the signed
    // dequant path (default zero-point = 0) is selected. Mapping to F32 here
    // silently dropped the signedness and produced a +(2^(bits-1))*scale bias.
    return builder.getIntegerType(8); // signless i8
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
    //
    // NOTE: unmapped types silently fall back to F32 here. This is only safe
    // because every consumer that needs exact bytes re-validates the width
    // downstream: `create_tensor` (mlir-named-attribute.cpp) is the enforcement
    // point -- its DenseElementsAttr byte-size guard catches this F32 default
    // (4B/elem) against the real ONNX raw_data width and LOG(FATAL)s with a fix
    // hint rather than emitting a corrupted attribute. Add a real case above
    // when introducing support for any of these element types.
    LOG(WARNING) << "Unsupported element type: " << element_type
                 << ", using F32";
    return builder.getF32Type();
  }
}

mlir::Type onnxElementTypeToMlirType(int element_type, mlir::OpBuilder &builder,
                                     const llvm::SmallVector<int64_t> *shape) {
  auto elementType = onnxElementTypeToMlirElementType(element_type, builder);
  if (!shape) {
    return mlir::UnrankedTensorType::get(elementType);
  }
  if (shape->empty()) {
    return mlir::RankedTensorType::get({}, elementType);
  }
  return mlir::RankedTensorType::get(*shape, elementType);
}

} // namespace mlir_impl
} // namespace morphizen
