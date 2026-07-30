/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_LLVM_LOWERING_UTILS_H
#define HIPSR_LLVM_LOWERING_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace hipsr {

enum HipdnnTensorOp : int64_t {
  kTensorOpMul = 0,
  kTensorOpAdd = 1,
  kTensorOpMin = 2,
  kTensorOpMax = 3,
};

int64_t getHipdnnDataType(Type elemType);

Value extractContiguousMemRefPtr(Value memrefDesc,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc);

Value getMemRefDimSize(MemRefType type, unsigned dimIdx, Value descriptor,
                       ConversionPatternRewriter &rewriter, Location loc);

Value computeNumElements(MemRefType type, Value descriptor,
                         ConversionPatternRewriter &rewriter, Location loc);

llvm::SmallVector<Value, 4> extractShape4D(MemRefType type, Value descriptor,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc, Type i64Type);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_LLVM_LOWERING_UTILS_H
