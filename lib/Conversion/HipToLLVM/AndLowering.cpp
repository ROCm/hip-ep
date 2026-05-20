/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

/// hip.and -> call @wrap_and(state, lhs_ptr, rhs_ptr, out_ptr, n_elements,
///                           dtype)
///
/// Element-wise logical AND on boolean (i1 stored as i8) tensors. Mirrors
/// the EqualOpLowering shape: extract contiguous data pointers, compute
/// num_elements honoring dynamic dims, and forward the input element
/// type as the runtime dtype so wrap_and can dispatch correctly. Inputs
/// and output share the same element type for AND (no type promotion).
struct AndOpLowering : public ConvertOpToLLVMPattern<AndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    // Boolean (i1) inputs are stored as i8 on device but i1 is not part of the
    // HIPDNN_EP_DATATYPE_* enum. Mirror the NotOpLowering fallback: pass a
    // sentinel value (0) since wrap_and's runtime path doesn't dispatch on
    // dtype for boolean AND. See UnaryElementwiseLowering.cpp for the same
    // pattern used by hip.not.
    if (dataType < 0 && lhsType.getElementType().isInteger(1))
      dataType = 0;
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapAnd, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    lhsPtr,
                                  rhsPtr,      outputPtr,
                                  numElements, createI64Const(dataType)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateAndLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.add<AndOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
