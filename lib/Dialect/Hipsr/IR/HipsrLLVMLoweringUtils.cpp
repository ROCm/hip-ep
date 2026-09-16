/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "hip/datatype_abi.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <algorithm>

using namespace mlir;

int64_t mlir::hipsr::getHipdnnDataType(Type elemType) {
  if (elemType.isF32()) {
    return HIPDNN_EP_DATATYPE_FLOAT;
  }
  if (elemType.isF16()) {
    return HIPDNN_EP_DATATYPE_HALF;
  }
  if (elemType.isBF16()) {
    return HIPDNN_EP_DATATYPE_BFLOAT16;
  }
  if (elemType.isInteger(32)) {
    return HIPDNN_EP_DATATYPE_INT32;
  }
  if (elemType.isInteger(64)) {
    return HIPDNN_EP_DATATYPE_INT64;
  }
  // A bool shares the unsigned-byte slot. The runtime gives every element its
  // own byte and reads a mask as 0 or 1, so an i1 tensor and a ui8 one have the
  // same representation.
  if (elemType.isUnsignedInteger(8) || elemType.isInteger(1)) {
    return HIPDNN_EP_DATATYPE_UINT8;
  }
  if (elemType.isSignedInteger(8) || elemType.isSignlessInteger(8)) {
    return HIPDNN_EP_DATATYPE_INT8;
  }
  if (elemType.isF64()) {
    return HIPDNN_EP_DATATYPE_DOUBLE;
  }
  if (elemType.isInteger(16)) {
    return HIPDNN_EP_DATATYPE_INT16;
  }
  return HIPDNN_EP_DATATYPE_UNSUPPORTED;
}

Value mlir::hipsr::extractContiguousMemRefPtr(
    Value memrefDesc, ConversionPatternRewriter &rewriter, Location loc) {
  return MemRefDescriptor(memrefDesc).alignedPtr(rewriter, loc);
}

llvm::SmallVector<Value>
mlir::hipsr::extractShape(MemRefType type, Value descriptor,
                          ConversionPatternRewriter &rewriter, Location loc,
                          Type i64Type) {
  MemRefDescriptor desc(descriptor);
  return llvm::map_to_vector(
      llvm::seq<int64_t>(type.getRank()), [&](int64_t dim) -> Value {
        if (type.isDynamicDim(dim)) {
          return desc.size(rewriter, loc, dim);
        }
        return LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            rewriter.getI64IntegerAttr(type.getDimSize(dim)));
      });
}

Value mlir::hipsr::emitHostI64Array(ValueRange values,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc) {
  Type i64Type = rewriter.getI64Type();
  Type hostPtrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  // LLVM rejects a zero-length array type, so a rank-0 shape still gets a slot.
  Value array = LLVM::AllocaOp::create(
      rewriter, loc, hostPtrType,
      LLVM::LLVMArrayType::get(i64Type, std::max<size_t>(values.size(), 1)),
      one,
      /*alignment=*/8);
  for (auto [index, value] : llvm::enumerate(values)) {
    Value element = LLVM::GEPOp::create(
        rewriter, loc, hostPtrType, i64Type, array,
        llvm::ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
    LLVM::StoreOp::create(rewriter, loc, value, element);
  }
  return array;
}

llvm::SmallVector<Value, 4>
mlir::hipsr::extractShape4D(MemRefType type, Value descriptor,
                            ConversionPatternRewriter &rewriter, Location loc,
                            Type i64Type) {
  llvm::SmallVector<Value, 4> dims;
  for (int64_t pad : llvm::seq<int64_t>(4 - type.getRank())) {
    dims.push_back(LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1)));
  }
  llvm::append_range(dims,
                     extractShape(type, descriptor, rewriter, loc, i64Type));
  return dims;
}
