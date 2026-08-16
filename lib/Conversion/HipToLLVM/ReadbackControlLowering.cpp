/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.readback_control(%ctx, %sources...) -> (i1, i64...)
//
// The generated stack arrays describe all source groups to one runtime call.
// The runtime initializes and fills the host result array, queues every D2H
// copy, and performs exactly one stream synchronization.
struct ReadbackControlOpLowering
    : public ConvertOpToLLVMPattern<ReadbackControlOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReadbackControlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    SmallVector<Type> sourceTypes(op.getSources().getTypes());
    FailureOr<ReadbackControlLayout> layout =
        getReadbackControlLayout(sourceTypes);
    if (failed(layout))
      return rewriter.notifyMatchFailure(op, "invalid source grouping");

    auto i64Constant = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto i32Constant = [&](int32_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                      rewriter.getI32IntegerAttr(value));
    };

    Value one = i64Constant(1);
    Value sourceCount = i64Constant(op.getSources().size());
    Value totalCount = i64Constant(layout->totalCount);
    Value hostCount = i64Constant(std::max<int64_t>(layout->totalCount, 1));
    Value hostValues =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, hostCount,
                               /*alignment=*/8);
    Value sourcePointers =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, sourceCount,
                               /*alignment=*/8);
    Value sourceLengths =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, sourceCount,
                               /*alignment=*/8);
    Value elementBytes =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, sourceCount,
                               /*alignment=*/8);

    Value zeroI64 = i64Constant(0);
    for (int64_t i : llvm::seq<int64_t>(layout->totalCount)) {
      Value index = i64Constant(i);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                       hostValues, index);
      LLVM::StoreOp::create(rewriter, loc, zeroI64, slot);
    }

    for (auto [i, source] : llvm::enumerate(op.getSources())) {
      auto memrefType = dyn_cast<MemRefType>(source.getType());
      if (!memrefType)
        return rewriter.notifyMatchFailure(op,
                                           "source operand must be a memref");
      Value sourcePtr = extractMemRefDataPtr(
          adaptor.getSources()[i], memrefType, typeConverter, rewriter, loc);
      if (!sourcePtr)
        return rewriter.notifyMatchFailure(op,
                                           "failed to compute source pointer");

      Value index = i64Constant(i);
      Value pointerSlot = LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType,
                                              sourcePointers, index);
      LLVM::StoreOp::create(rewriter, loc, sourcePtr, pointerSlot);

      Value lengthSlot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                             sourceLengths, index);
      LLVM::StoreOp::create(rewriter, loc,
                            i64Constant(layout->sourceLengths[i]), lengthSlot);

      Value widthSlot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                            elementBytes, index);
      int64_t bytes =
          cast<IntegerType>(memrefType.getElementType()).getWidth() / 8;
      LLVM::StoreOp::create(rewriter, loc, i64Constant(bytes), widthSlot);
    }

    SmallVector<Type> parameterTypes = {ptrType, ptrType, ptrType, ptrType,
                                        ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> function = LLVM::lookupOrCreateFn(
        rewriter, module, kHipReadbackControl, parameterTypes, i32Type);
    if (failed(function))
      return failure();
    LLVM::CallOp call = LLVM::CallOp::create(
        rewriter, loc, *function,
        ValueRange{adaptor.getCtx(), hostValues, sourcePointers, sourceLengths,
                   elementBytes, sourceCount, totalCount});

    Value status = call.getResult();
    Value valid = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                       status, i32Constant(0));

    SmallVector<Value> results;
    results.reserve(1 + layout->totalCount);
    results.push_back(valid);
    for (int64_t i : llvm::seq<int64_t>(layout->totalCount)) {
      Value index = i64Constant(i);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                       hostValues, index);
      results.push_back(LLVM::LoadOp::create(rewriter, loc, i64Type, slot));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void populateReadbackControlLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<ReadbackControlOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
