/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct CastShapeArgs : PlaceholderShapeRegionArgs {
  using PlaceholderShapeRegionArgs::PlaceholderShapeRegionArgs;
  Value getInput() const { return in(0); }
};
} // namespace

MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

void CastOp::populateShapeRegion(OpBuilder &, Block &) {}

namespace mlir {
namespace hipsr {

LogicalResult populateCastShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                      CastOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Value inputShape = CastShapeArgs{shapeBlock}.getInput();
  ShapeYield2Op::create(builder, op.getLoc(), ValueRange{inputShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapCast = "wrap_cast";

struct CastLowering : ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!inputType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }

    int64_t srcDataType = getHipdnnDataType(inputType.getElementType());
    int64_t dstDataType = getHipdnnDataType(outputType.getElementType());
    if (srcDataType < 0 || dstDataType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getInit());
    for (int64_t i : llvm::seq<int64_t>(0, outputType.getRank())) {
      Value dim = outputType.isDynamicDim(i)
                      ? outputDesc.size(rewriter, loc, i)
                      : createI64Const(outputType.getDimSize(i));
      numElements =
          LLVM::MulOp::create(rewriter, loc, numElements, dim).getResult();
    }

    using CastCall =
        RuntimeFunc<i32, hostPtr, devicePtr, devicePtr, i64, i64, i64>;
    auto castFunc =
        CastCall::lookupOrCreateFn(rewriter, loc, module, kWrapCast);
    if (failed(castFunc)) {
      return failure();
    }
    if (failed(castFunc->call(adaptor.getCtx(), adaptor.getInput(),
                              adaptor.getInit(), numElements, srcDataType,
                              dstDataType))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrCastLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CastLowering>(converter);
}
