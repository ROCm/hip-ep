/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct AddShapeArgs : ShapeRegionArgs<AddOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getLhs() const { return in(0); }
  Value getRhs() const { return in(1); }
};
} // namespace

MutableOperandRange AddOp::getDpsInitsMutable() { return getInitMutable(); }

void AddOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  AddShapeArgs args{shapeBlock};
  Value shLhs = builder.create<shape::ShapeOfOp>(loc, args.getLhs());
  Value shRhs = builder.create<shape::ShapeOfOp>(loc, args.getRhs());

  auto extentTensorTy = RankedTensorType::get(
      {ShapedType::kDynamic}, IndexType::get(builder.getContext()));
  Value bcast = builder.create<shape::BroadcastOp>(
      loc, extentTensorTy, ValueRange{shLhs, shRhs}, /*error=*/nullptr);

  int64_t outRank = cast<ShapedType>(getInit().getType()).getRank();
  SmallVector<Value> dims;
  dims.reserve(outRank);
  for (int64_t i : llvm::seq<int64_t>(0, outRank)) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    dims.push_back(builder.create<shape::GetExtentOp>(loc, bcast, idx));
  }

  Type elemTy = cast<ShapedType>(getInit().getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}

namespace {

constexpr const char *kWrapMiopenOpTensor = "wrap_miopenOpTensor";

struct AddLowering : public ConvertOpToLLVMPattern<AddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type hostPtrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type devicePtrType = LLVM::LLVMPointerType::get(ctx, 1);
    Type i32Type = rewriter.getI32Type();
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

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type, 19> paramTypes = {hostPtrType, i32Type, devicePtrType,
                                        devicePtrType, devicePtrType};
    paramTypes.append(14, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenOpTensor, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    Value opStateSlot = LLVM::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(-1));

    SmallVector<Value, 19> args = {
        adaptor.getCtx(), opStateSlot,
        extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getInit(), rewriter, loc)};
    args.append(lhsDims.begin(), lhsDims.end());
    args.append(rhsDims.begin(), rhsDims.end());
    args.append(outDims.begin(), outDims.end());
    args.push_back(createI64Const(dataType));
    args.push_back(createI64Const(kTensorOpAdd));

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrAddLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<AddLowering>(converter);
}
