/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

/// hip.and(ctx, lhs, rhs, output)
///   -> wrap_and(state, lhs, rhs, output,
///               lhs_n..lhs_w, rhs_n..rhs_w, out_n..out_w, dtype)
///
/// Element-wise logical AND on boolean (i1 stored as i8) tensors. Full 4D
/// shapes are passed (rank <= 4, left-padded with 1) so the runtime can
/// materialise ONNX multidirectional broadcast via hip_expand before the flat
/// hip_elementwise_and kernel -- mirrors DivOpLowering. Inputs and output
/// share the same element type for AND (no type promotion).
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

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    if (lhsType.getRank() > 4 || rhsType.getRank() > 4 ||
        outputType.getRank() > 4)
      return rewriter.notifyMatchFailure(
          op, "rank > 4 unsupported by 4D broadcast descriptor API");

    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    // Boolean (i1) inputs are stored as i8 on device but i1 is not part of the
    // HIPDNN_EP_DATATYPE_* enum. Pass a sentinel value (0) since wrap_and's
    // runtime path treats the operands as 1-byte bool and does not dispatch on
    // dtype. See UnaryElementwiseLowering.cpp for the same pattern (hip.not).
    if (dataType < 0 && lhsType.getElementType().isInteger(1))
      dataType = 0;
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto lhsDims =
        extractShape4D(lhsType, adaptor.getLhs(), rewriter, loc, i64Type);
    auto rhsDims =
        extractShape4D(rhsType, adaptor.getRhs(), rewriter, loc, i64Type);
    auto outDims =
        extractShape4D(outputType, adaptor.getOutput(), rewriter, loc, i64Type);

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // 17 params: state + 3 data ptrs + 12 shape dims + data_type
    SmallVector<Type, 17> paramTypes(4, ptrType);
    paramTypes.append(13, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapAnd, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc)};
    args.append(lhsDims.begin(), lhsDims.end());
    args.append(rhsDims.begin(), rhsDims.end());
    args.append(outDims.begin(), outDims.end());
    args.push_back(createI64Const(dataType));

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
