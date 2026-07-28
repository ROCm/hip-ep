/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct CastShapeArgs : ShapeRegionArgs<CastOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getInput() const { return in(0); }
};
} // namespace

MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

void CastOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  Value input = CastShapeArgs{shapeBlock}.getInput();
  auto shapedTy = cast<ShapedType>(input.getType());
  Value shape = builder.create<shape::ShapeOfOp>(loc, input);
  SmallVector<Value> dims;
  dims.reserve(shapedTy.getRank());
  for (int64_t i : llvm::seq<int64_t>(0, shapedTy.getRank())) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    dims.push_back(builder.create<shape::GetExtentOp>(loc, shape, idx));
  }
  Type elemTy = cast<ShapedType>(getInit().getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}

namespace {

constexpr const char *kWrapCast = "wrap_cast";

struct CastLowering : ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type hostPtrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type devicePtrType = LLVM::LLVMPointerType::get(ctx, 1);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

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

    auto createI64Const = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
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

    SmallVector<Type, 6> paramTypes = {
        hostPtrType,   // state
        devicePtrType, // input
        devicePtrType, // output
        i64Type,       // num_elements
        i64Type,       // src_data_type
        i64Type        // dst_data_type
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCast, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 6> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getInit(), rewriter, loc),
        numElements,
        createI64Const(srcDataType),
        createI64Const(dstDataType)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrCastLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CastLowering>(converter);
}
