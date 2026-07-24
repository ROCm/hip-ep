/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/Sequence.h"

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
  Value ptr = MemRefDescriptor(memrefDesc).alignedPtr(rewriter, loc);
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  if (ptrTy.getAddressSpace() != 0) {
    ptr = LLVM::AddrSpaceCastOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0),
        ptr);
  }
  return ptr;
}

llvm::SmallVector<Value, 4>
mlir::hipsr::extractShape4D(MemRefType type, Value descriptor,
                            ConversionPatternRewriter &rewriter, Location loc,
                            Type i64Type) {
  auto createConst = [&](int64_t v) {
    return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                    rewriter.getI64IntegerAttr(v));
  };
  MemRefDescriptor desc(descriptor);
  int rank = type.getRank();
  llvm::SmallVector<Value, 4> dims;
  for (int i : llvm::seq(4 - rank)) {
    dims.push_back(createConst(1));
  }
  for (int i : llvm::seq(rank)) {
    if (type.isDynamicDim(i)) {
      dims.push_back(desc.size(rewriter, loc, i));
    } else {
      dims.push_back(createConst(type.getDimSize(i)));
    }
  }
  return dims;
}
