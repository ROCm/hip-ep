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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct EqualPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getLhs() const { return in(0); }
  Value getRhs() const { return in(1); }
};
} // namespace

MutableOperandRange EqualOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult EqualOp::verify() {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  // A comparison yields a bool whatever it compared, so the mask never takes
  // the operand element type.
  if (!outputType.getElementType().isInteger(1)) {
    return emitOpError("output element type must be i1");
  }

  SmallVector<int64_t> broadcastShape;
  if (!OpTrait::util::getBroadcastedShape(lhsType.getShape(),
                                          rhsType.getShape(), broadcastShape)) {
    return emitOpError("lhs and rhs shapes are not broadcast compatible");
  }
  if (failed(verifyCompatibleShape(broadcastShape, outputType.getShape()))) {
    return emitOpError("output shape must be the broadcast of lhs and rhs");
  }
  return success();
}

namespace mlir {
namespace hipsr {

LogicalResult populateEqualShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                       EqualOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  EqualPlaceholderShapeArgs args{shapeBlock};
  SmallVector<Value> operandShapes{args.getLhs(), args.getRhs()};
  Value broadcast = shape::BroadcastOp::create(
      builder, op.getLoc(), getBroadcastExtentTensorType(operandShapes),
      operandShapes, /*error=*/nullptr);
  ShapeYieldOp::create(builder, op.getLoc(), ValueRange{broadcast});
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapEqual = "wrap_equal";

// `wrap_equal` pads each shape to 4D (N, C, H, W) with leading ones and
// broadcasts the operands itself. Its `data_type` is the operand type it
// compares; the mask it writes is always one byte per element.
struct EqualLowering : ConvertOpToLLVMPattern<EqualOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type i64Type = rewriter.getI64Type();

    auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
    auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
    auto outType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!lhsType || !rhsType || !outType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }

    if (lhsType.getRank() > 4 || rhsType.getRank() > 4 ||
        outType.getRank() > 4) {
      return rewriter.notifyMatchFailure(
          op, "rank > 4 unsupported by the 4D broadcast descriptor API");
    }

    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    auto lhsDims =
        extractShape4D(lhsType, adaptor.getLhs(), rewriter, loc, i64Type);
    auto rhsDims =
        extractShape4D(rhsType, adaptor.getRhs(), rewriter, loc, i64Type);
    auto outDims =
        extractShape4D(outType, adaptor.getInit(), rewriter, loc, i64Type);

    using EqualCall =
        RuntimeFunc<i32, hostPtr, devicePtr, devicePtr, devicePtr, i64, i64,
                    i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>;
    auto equalFunc =
        EqualCall::lookupOrCreateFn(rewriter, loc, module, kWrapEqual);
    if (failed(equalFunc)) {
      return failure();
    }
    if (failed(equalFunc->call(adaptor.getCtx(), adaptor.getLhs(),
                               adaptor.getRhs(), adaptor.getInit(), lhsDims[0],
                               lhsDims[1], lhsDims[2], lhsDims[3], rhsDims[0],
                               rhsDims[1], rhsDims[2], rhsDims[3], outDims[0],
                               outDims[1], outDims[2], outDims[3], dataType))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrEqualLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<EqualLowering>(converter);
}
