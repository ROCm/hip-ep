/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

/// hip.or(ctx, lhs, rhs, output)
///   -> wrap_or(state, lhs, rhs, output,
///              lhs_n..lhs_w, rhs_n..rhs_w, out_n..out_w, dtype)
///
/// Element-wise logical OR on boolean (i1 stored as i8) tensors. Full 4D
/// shapes are passed (rank <= 4, left-padded with 1) so the runtime can
/// materialise ONNX multidirectional broadcast via hip_expand before the flat
/// hip_elementwise_or kernel -- mirrors DivOpLowering.
struct OrOpLowering : public ConvertOpToLLVMPattern<OrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
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
    // i1 has no HIPDNN dtype slot; pass a sentinel (0). wrap_or treats the
    // operands as 1-byte bool and does not dispatch on dtype.
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
        LLVM::lookupOrCreateFn(rewriter, module, kWrapOr, paramTypes, i32Type);
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

void populateOrLoweringPatterns(const LLVMTypeConverter &converter,
                                RewritePatternSet &patterns) {
  patterns.add<OrOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
