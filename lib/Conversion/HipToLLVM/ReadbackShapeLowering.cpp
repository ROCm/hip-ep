/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.readback_shape(%ctx, %vector {count = N}) -> N x index
//   -> %host = llvm.alloca N x i64
//      llvm.call @hipdnn_ep_readback_shape_i64(..., %host, %vector, N)
//      N x llvm.load %host[i]
//
// The runtime performs one D2H copy and one stream synchronization for the
// entire vector, then clamps negative entries and records the recoverable error
// flag. Index lowers to i64 in the generated ABI.
struct ReadbackShapeOpLowering
    : public ConvertOpToLLVMPattern<ReadbackShapeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReadbackShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t count = op.getCount();
    if (count == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    auto vectorType = dyn_cast<MemRefType>(op.getVector().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "vector operand must be a memref");

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    Value vectorPtr = extractMemRefDataPtr(adaptor.getVector(), vectorType,
                                           typeConverter, rewriter, loc);
    if (!vectorPtr)
      return rewriter.notifyMatchFailure(op, "failed to compute vector ptr");

    Value countValue = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(count));
    Value hostArray =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, countValue,
                               /*alignment=*/8);

    SmallVector<Type, 4> parameterTypes = {ptrType, ptrType, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> function = LLVM::lookupOrCreateFn(
        rewriter, module, kHipReadbackShapeI64, parameterTypes, voidType);
    if (failed(function))
      return failure();
    LLVM::CallOp::create(
        rewriter, loc, *function,
        ValueRange{adaptor.getCtx(), hostArray, vectorPtr, countValue});

    SmallVector<Value> results;
    results.reserve(count);
    for (int64_t i : llvm::seq<int64_t>(0, count)) {
      Value index = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                       hostArray, index);
      results.push_back(LLVM::LoadOp::create(rewriter, loc, i64Type, slot));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void populateReadbackShapeLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<ReadbackShapeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
