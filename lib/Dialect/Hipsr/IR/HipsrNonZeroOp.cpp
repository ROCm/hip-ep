/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct NonZeroPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getInput() const { return in(0); }
};
} // namespace

namespace mlir {
namespace hipsr {

// The indices hold one row per input axis and, in the worst case, one column
// per input element. The count is one number.
LogicalResult populateNonZeroShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                         NonZeroOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  Type shapeType = shape::ShapeType::get(builder.getContext());
  NonZeroPlaceholderShapeArgs args{shapeBlock};
  Value inputShape = args.getInput();

  int64_t rank = cast<ShapedType>(op.getInput().getType()).getRank();
  Value rows = shape::ConstSizeOp::create(builder, loc, rank);
  Value capacity = shape::NumElementsOp::create(builder, loc, inputShape);
  Value indicesShape = shape::FromExtentsOp::create(builder, loc, shapeType,
                                                    ValueRange{rows, capacity});
  Value countShape = shape::ConstShapeOp::create(
      builder, loc, shapeType, builder.getIndexTensorAttr({1}));
  ShapeYieldOp::create(builder, loc, ValueRange{indicesShape, countShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

MutableOperandRange NonZeroOp::getDpsInitsMutable() {
  // The two inits are adjacent operands, and no generated accessor spans both.
  return MutableOperandRange(getOperation(),
                             getIndicesInitMutable().getOperandNumber(),
                             /*length=*/2);
}

LogicalResult NonZeroOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  auto indicesType = cast<ShapedType>(getIndicesInit().getType());
  auto countType = cast<ShapedType>(getCountInit().getType());

  if (!indicesType.getElementType().isInteger(64)) {
    return emitOpError("indices element type must be i64");
  }
  if (!countType.getElementType().isInteger(32)) {
    return emitOpError("count element type must be i32");
  }
  if (indicesType.getRank() != 2) {
    return emitOpError("indices must be rank-2: one row per input axis, one "
                       "column per position found");
  }
  // A dynamic row count fails this too: the input rank is known here.
  if (indicesType.getDimSize(0) != inputType.getRank()) {
    return emitOpError("indices must have one row per input axis; input rank "
                       "is ")
           << inputType.getRank();
  }
  if (countType.getRank() != 1 || countType.getDimSize(0) != 1) {
    return emitOpError("count must be a static single-element vector");
  }
  return success();
}

namespace {

constexpr const char *kWrapNonZero = "wrap_nonzero";

struct NonZeroLowering : ConvertOpToLLVMPattern<NonZeroOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NonZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto indicesType = dyn_cast<MemRefType>(op.getIndicesInit().getType());
    if (!inputType || !indicesType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Type i64Type = rewriter.getI64Type();
    llvm::SmallVector<Value> inputDims =
        extractShape(inputType, adaptor.getInput(), rewriter, loc, i64Type);

    Value numElements =
        inputDims.empty()
            ? LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1))
                  .getResult()
            : inputDims.front();
    for (Value dim : llvm::drop_begin(inputDims)) {
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dim);
    }

    llvm::SmallVector<Value> indicesDims = extractShape(
        indicesType, adaptor.getIndicesInit(), rewriter, loc, i64Type);
    Value capacity = indicesDims.back();

    Value dimsArray = emitHostI64Array(inputDims, rewriter, loc);

    using NonZeroCall = RuntimeFunc<i32, hostPtr, devicePtr, devicePtr,
                                    devicePtr, i64, i64, hostPtr, i64, i64>;
    auto nonZeroFunc =
        NonZeroCall::lookupOrCreateFn(rewriter, loc, module, kWrapNonZero);
    if (failed(nonZeroFunc)) {
      return failure();
    }
    if (failed(nonZeroFunc->call(
            adaptor.getCtx(), adaptor.getInput(), adaptor.getIndicesInit(),
            adaptor.getCountInit(), numElements, inputType.getRank(), dimsArray,
            capacity, dataType))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrNonZeroLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NonZeroLowering>(converter);
}
