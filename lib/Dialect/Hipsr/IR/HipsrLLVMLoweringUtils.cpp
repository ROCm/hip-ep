/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

int64_t mlir::hipsr::getHipdnnDataType(Type elemType) {
  if (elemType.isF32()) {
    return 0;
  }
  if (elemType.isF16()) {
    return 1;
  }
  if (elemType.isBF16()) {
    return 2;
  }
  if (elemType.isInteger(32)) {
    return 3;
  }
  if (elemType.isInteger(64)) {
    return 4;
  }
  if (elemType.isUnsignedInteger(8)) {
    return 7;
  }
  if (elemType.isSignedInteger(8) || elemType.isSignlessInteger(8)) {
    return 5;
  }
  if (elemType.isF64()) {
    return 6;
  }
  if (elemType.isInteger(16)) {
    return 8;
  }
  return -1;
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
