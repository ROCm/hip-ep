/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"
using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange AddOp::getDpsInitsMutable() { return getInitMutable(); }

namespace {

struct AddShapeArgs : ShapeRegionArgs<AddOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getLhs() const { return in(0); }
  Value getRhs() const { return in(1); }
};

PlaceholderType getAddPlaceholderType(AddOp) { return PlaceholderType::Normal; }

void populateAddShapeRegion(OpBuilder &builder, Block &block, AddOp op,
                            PlaceholderType placeholderType) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  AddShapeArgs args(placeholderType, block);
  auto shapeType = shape::ShapeType::get(builder.getContext());
  Value resultShape = builder.create<shape::BroadcastOp>(
      op.getLoc(), shapeType, ValueRange{args.getLhs(), args.getRhs()},
      /*error=*/nullptr);
  builder.create<ShapeYieldOp>(op.getLoc(), resultShape);
}

constexpr const char *kWrapMiopenOpTensor = "wrap_miopenOpTensor";

struct AddLowering : public ConvertOpToLLVMPattern<AddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
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
          op, "rank > 4 unsupported by MIOpen 4D descriptor API");
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

    using AddCall = RuntimeFunc<i32, hostPtr, slotIndex, devicePtr, devicePtr,
                                devicePtr, i64, i64, i64, i64, i64, i64, i64,
                                i64, i64, i64, i64, i64, i64, i64>;
    auto addFunc =
        AddCall::lookupOrCreateFn(rewriter, loc, module, kWrapMiopenOpTensor);
    if (failed(addFunc)) {
      return failure();
    }
    if (failed(addFunc->call(
            adaptor.getCtx(), SlotIndex{op.getOperation()}, adaptor.getLhs(),
            adaptor.getRhs(), adaptor.getInit(), lhsDims[0], lhsDims[1],
            lhsDims[2], lhsDims[3], rhsDims[0], rhsDims[1], rhsDims[2],
            rhsDims[3], outDims[0], outDims[1], outDims[2], outDims[3],
            dataType, static_cast<int64_t>(kTensorOpAdd)))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateAddShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns) {
  patterns.add<AddOp, getAddPlaceholderType, populateAddShapeRegion>();
}

void mlir::hipsr::populateHipsrAddLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<AddLowering>(converter);
}
