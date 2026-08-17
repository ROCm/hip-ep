/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct CheckedTileExtentOpLowering
    : public ConvertOpToLLVMPattern<CheckedTileExtentOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CheckedTileExtentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    auto i64Constant = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value one = i64Constant(1);
    Value zero = i64Constant(0);
    Value hostExtent =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, one, 8);
    Value hostElements =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, one, 8);
    LLVM::StoreOp::create(rewriter, loc, zero, hostExtent);
    LLVM::StoreOp::create(rewriter, loc, zero, hostElements);
    Value valid = LLVM::ZExtOp::create(rewriter, loc, i64Type,
                                       adaptor.getReadbackValid());
    Value expected = i64Constant(op.getExpectedExtent());

    SmallVector<Type> parameterTypes = {ptrType, ptrType, ptrType, i64Type,
                                        i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> function = LLVM::lookupOrCreateFn(
        rewriter, module, kHipCheckedTileExtent, parameterTypes, i32Type);
    if (failed(function))
      return failure();
    LLVM::CallOp call = LLVM::CallOp::create(
        rewriter, loc, *function,
        ValueRange{adaptor.getCtx(), hostExtent, hostElements, valid,
                   adaptor.getInputExtent(), adaptor.getRepeat(), expected,
                   adaptor.getPriorElements()});

    Value loadedExtent =
        LLVM::LoadOp::create(rewriter, loc, i64Type, hostExtent);
    Value loadedElements =
        LLVM::LoadOp::create(rewriter, loc, i64Type, hostElements);
    Value statusOk = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, call.getResult(),
        LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                 rewriter.getI32IntegerAttr(0)));
    Value safeExtent =
        LLVM::SelectOp::create(rewriter, loc, statusOk, loadedExtent, zero);
    Value safeElements =
        LLVM::SelectOp::create(rewriter, loc, statusOk, loadedElements, zero);
    rewriter.replaceOp(op, ValueRange{statusOk, safeExtent, safeElements});
    return success();
  }
};

// hip.tile(ctx, input, repeats, output)
//   -> wrap_tile(state, in_ptr, repeats_ptr, out_ptr,
//                in_shape_ptr, in_rank, out_shape_ptr, out_rank, data_type)
struct TileOpLowering : public ConvertOpToLLVMPattern<TileOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inType = cast<MemRefType>(op.getInput().getType());
    auto outType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataType = getHipdnnDataType(inType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value repeatsPtr =
        extractContiguousMemRefPtr(adaptor.getRepeats(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value inShape = emitShapeArray(inType, adaptor.getInput());
    Value outShape = emitShapeArray(outType, adaptor.getOutput());
    Value inRank = createI64Const(inType.getRank());
    Value outRank = createI64Const(outType.getRank());
    Value dataTypeVal = createI64Const(dataType);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, // state, in, repeats, out
                                       ptrType, i64Type, // in_shape, in_rank
                                       ptrType, i64Type, // out_shape, out_rank
                                       i64Type};         // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapTile, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, inPtr,   repeatsPtr,
                                  outPtr,   inShape, inRank,
                                  outShape, outRank, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTileLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<CheckedTileExtentOpLowering, TileOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
