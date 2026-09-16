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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct TransposePlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getInput() const { return in(0); }
};
} // namespace

MutableOperandRange TransposeOp::getDpsInitsMutable() {
  return getInitMutable();
}

LogicalResult TransposeOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  ArrayRef<int64_t> perm = getPerm();

  if (static_cast<int64_t>(perm.size()) != inputType.getRank()) {
    return emitOpError("perm must have one entry per input axis; expected ")
           << inputType.getRank() << ", got " << perm.size();
  }
  if (!isPermutationVector(perm)) {
    return emitOpError("perm must be a permutation of [0, rank)");
  }

  auto outputType = cast<ShapedType>(getInit().getType());
  if (failed(verifyCompatibleShape(applyPermutation(inputType.getShape(), perm),
                                   outputType.getShape()))) {
    return emitOpError("output shape must be the input shape permuted by perm");
  }
  return success();
}

namespace mlir {
namespace hipsr {

LogicalResult populateTransposeShapeRegion(OpBuilder &builder,
                                           Block &shapeBlock, TransposeOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  TransposePlaceholderShapeArgs args{shapeBlock};
  Value inputShape = args.getInput();
  SmallVector<Value> extents =
      llvm::map_to_vector(op.getPerm(), [&](int64_t axis) -> Value {
        return shape::GetExtentOp::create(builder, loc, inputShape, axis);
      });
  Value outputShape = shape::FromExtentsOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()), extents);
  ShapeYieldOp::create(builder, loc, ValueRange{outputShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapTranspose = "wrap_transpose";

// `wrap_transpose` reads the input shape and `perm` from host arrays of `rank`
// entries each, then moves `num_elements` elements of `element_size_bytes`.
struct TransposeLowering : ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }
    if (inputType.getRank() == 0) {
      return rewriter.notifyMatchFailure(op, "rank 0 has no axis to permute");
    }
    // The runtime copies whole elements and picks the kernel by their width.
    Type elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat() ||
        elementType.getIntOrFloatBitWidth() % 8 != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "element type must be whole bytes");
    }
    int64_t elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;

    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    SmallVector<Value> extents =
        extractShape(inputType, adaptor.getInput(), rewriter, loc, i64Type);
    Value numElements = extents.front();
    for (Value extent : llvm::drop_begin(extents)) {
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, extent);
    }
    Value inputShape = emitHostI64Array(extents, rewriter, loc);
    Value perm = emitHostI64Array(
        llvm::map_to_vector(op.getPerm(), createI64Const), rewriter, loc);

    using TransposeCall = RuntimeFunc<i32, hostPtr, devicePtr, devicePtr, i64,
                                      hostPtr, hostPtr, i64, i64>;
    auto transposeFunc =
        TransposeCall::lookupOrCreateFn(rewriter, loc, module, kWrapTranspose);
    if (failed(transposeFunc)) {
      return failure();
    }
    if (failed(transposeFunc->call(adaptor.getCtx(), adaptor.getInput(),
                                   adaptor.getInit(), inputType.getRank(),
                                   inputShape, perm, numElements,
                                   elementSizeBytes))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrTransposeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TransposeLowering>(converter);
}
