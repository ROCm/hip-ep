/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.non_max_suppression -> wrap_non_max_suppression runtime call
//===----------------------------------------------------------------------===//
//
// Runtime ABI:
//   wrap_non_max_suppression(state, boxes, scores, selected, count_ptr,
//                            B, C, N, capacity, max_output_boxes_per_class,
//                            iou_threshold (f32), score_threshold (f32),
//                            has_score_threshold, center_point_box)
//   -> i32
//
// B and N come from boxes, C from scores, and capacity from the leading
// extent of `selected`, so every extent may be dynamic.

struct NonMaxSuppressionOpLowering
    : public ConvertOpToLLVMPattern<NonMaxSuppressionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NonMaxSuppressionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto boxesType = dyn_cast<MemRefType>(op.getBoxes().getType());
    auto scoresType = dyn_cast<MemRefType>(op.getScores().getType());
    auto selectedType = dyn_cast<MemRefType>(op.getSelected().getType());
    auto countType = dyn_cast<MemRefType>(op.getCount().getType());
    if (!boxesType || !scoresType || !selectedType || !countType)
      return rewriter.notifyMatchFailure(op, "expected memref operands");
    if (boxesType.getRank() != 3 || scoresType.getRank() != 3 ||
        selectedType.getRank() != 2 || countType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "unexpected operand ranks");

    Value statePtr = adaptor.getCtx();
    Value boxesPtr =
        extractContiguousMemRefPtr(adaptor.getBoxes(), rewriter, loc);
    Value scoresPtr =
        extractContiguousMemRefPtr(adaptor.getScores(), rewriter, loc);
    Value selectedPtr =
        extractContiguousMemRefPtr(adaptor.getSelected(), rewriter, loc);
    Value countPtr =
        extractContiguousMemRefPtr(adaptor.getCount(), rewriter, loc);

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto createF32 = [&](APFloat v) -> Value {
      return LLVM::ConstantOp::create(
          rewriter, loc, f32Type, rewriter.getF32FloatAttr(v.convertToFloat()));
    };

    Value B = getMemRefDimSize(boxesType, 0, adaptor.getBoxes(), rewriter, loc);
    Value N = getMemRefDimSize(boxesType, 1, adaptor.getBoxes(), rewriter, loc);
    Value C =
        getMemRefDimSize(scoresType, 1, adaptor.getScores(), rewriter, loc);
    Value capacity =
        getMemRefDimSize(selectedType, 0, adaptor.getSelected(), rewriter, loc);

    SmallVector<Type, 14> paramTypes;
    SmallVector<Value, 14> args;
    auto addArg = [&](Type t, Value v) {
      paramTypes.push_back(t);
      args.push_back(v);
    };

    addArg(ptrType, statePtr);
    addArg(ptrType, boxesPtr);
    addArg(ptrType, scoresPtr);
    addArg(ptrType, selectedPtr);
    addArg(ptrType, countPtr);
    addArg(i64Type, B);
    addArg(i64Type, C);
    addArg(i64Type, N);
    addArg(i64Type, capacity);
    addArg(i64Type, createI64(op.getMaxOutputBoxesPerClass()));
    addArg(f32Type, createF32(op.getIouThreshold()));
    addArg(f32Type, createF32(op.getScoreThreshold()));
    addArg(i64Type, createI64(op.getHasScoreThreshold()));
    addArg(i64Type, createI64(op.getCenterPointBox()));

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNonMaxSuppression, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNonMaxSuppressionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NonMaxSuppressionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
