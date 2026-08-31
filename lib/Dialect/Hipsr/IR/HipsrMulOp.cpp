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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct MulPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getLhs() const { return in(0); }
  Value getRhs() const { return in(1); }
};
} // namespace

MutableOperandRange MulOp::getDpsInitsMutable() { return getInitMutable(); }

namespace mlir {
namespace hipsr {

LogicalResult populateMulShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                     MulOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  MulPlaceholderShapeArgs args{shapeBlock};
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

constexpr const char *kWrapElementwise = "wrap_elementwise";

struct MulLowering : public ConvertOpToLLVMPattern<MulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
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
          op, "rank > 4 unsupported by the 4D shape-passing ABI");
    }

    auto lhsDims =
        extractShape4D(lhsType, adaptor.getLhs(), rewriter, loc, i64Type);
    auto rhsDims =
        extractShape4D(rhsType, adaptor.getRhs(), rewriter, loc, i64Type);
    auto outDims =
        extractShape4D(outType, adaptor.getInit(), rewriter, loc, i64Type);

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    using MulCall = RuntimeFunc<i32, hostPtr, slotIndex, devicePtr, devicePtr,
                                devicePtr, i64, i64, i64, i64, i64, i64, i64,
                                i64, i64, i64, i64, i64, i64, i64>;
    auto mulFunc =
        MulCall::lookupOrCreateFn(rewriter, loc, module, kWrapElementwise);
    if (failed(mulFunc)) {
      return failure();
    }
    if (failed(mulFunc->call(
            adaptor.getCtx(), SlotIndex{op.getOperation()}, adaptor.getLhs(),
            adaptor.getRhs(), adaptor.getInit(), lhsDims[0], lhsDims[1],
            lhsDims[2], lhsDims[3], rhsDims[0], rhsDims[1], rhsDims[2],
            rhsDims[3], outDims[0], outDims[1], outDims[2], outDims[3],
            dataType, static_cast<int64_t>(kTensorOpMul)))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrMulLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MulLowering>(converter);
}
