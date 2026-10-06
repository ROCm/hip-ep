/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.equal(ctx, lhs, rhs, output)
//   -> wrap_equal(state, lhs, rhs, output,
//                 lhs_n..lhs_w, rhs_n..rhs_w, out_n..out_w, data_type)
//
// Output is bool (1 byte/element); data_type identifies the comparison
// operand type. Full 4D shapes are passed (rank <= 4, left-padded with 1) so
// the runtime handles same-shape / scalar directly and materialises any other
// ONNX multidirectional broadcast via hip_expand -- mirrors DivOpLowering.
struct EqualOpLowering : public ConvertOpToLLVMPattern<EqualOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto shapes = extractBroadcastShapes4D(
        lhsType, adaptor.getLhs(), rhsType, adaptor.getRhs(), outputType,
        adaptor.getOutput(), rewriter, loc, i64Type);
    if (failed(shapes))
      return rewriter.notifyMatchFailure(
          op, "broadcast does not fold into the 4D broadcast descriptor API");
    auto &lhsDims = shapes->lhs;
    auto &rhsDims = shapes->rhs;
    auto &outDims = shapes->out;

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // 17 params: state + 3 data ptrs + 12 shape dims + data_type
    SmallVector<Type, 17> paramTypes(4, ptrType);
    paramTypes.append(13, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapEqual, paramTypes, i32Type);
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

void populateEqualLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<EqualOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
