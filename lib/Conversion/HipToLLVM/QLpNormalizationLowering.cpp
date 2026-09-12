/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qlpnormalization(%ctx) ins(%x) outs(%y)
//   -> wrap_qlpnormalization(state, x, y, numel, N, dtype,
//                            input_scale, input_zp, output_scale, output_zp,
//                            axis, p)
struct QLpNormalizationOpLowering
    : public ConvertOpToLLVMPattern<QLpNormalizationOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QLpNormalizationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    if (inputType.getElementType() != outputType.getElementType())
      return op.emitError(
          "hip.qlpnormalization: input and output element types must match");
    if (!inputType.getElementType().isUnsignedInteger(16))
      return op.emitError("hip.qlpnormalization: activation must be ui16");
    if (op.getP() != 2)
      return op.emitError("hip.qlpnormalization: only p=2 is implemented");
    if (op.getOutputScale().convertToFloat() == 0.0f)
      return op.emitError(
          "hip.qlpnormalization: output_scale must be non-zero");

    int64_t rank = inputType.getRank();
    int64_t normAxis = op.getAxis();
    if (normAxis < 0)
      normAxis += rank;
    if (rank == 0 || normAxis < 0 || normAxis >= rank)
      return op.emitError("hip.qlpnormalization: axis out of range");

    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value numElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    Value normNumElements = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
    for (int64_t dimIdx = normAxis; dimIdx < rank; ++dimIdx)
      normNumElements = LLVM::MulOp::create(
          rewriter, loc, normNumElements,
          getMemRefDimSize(inputType, static_cast<unsigned>(dimIdx),
                           adaptor.getInput(), rewriter, loc));

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value dtype = createI64(getHipdnnDataType(inputType.getElementType()));
    Value inputScale = LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                                op.getInputScaleAttr());
    Value outputScale = LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                                 op.getOutputScaleAttr());

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                    i64Type, i64Type, f32Type, i64Type,
                                    f32Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapQLpNormalization,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {adaptor.getCtx(),
                               inputPtr,
                               outputPtr,
                               numElements,
                               normNumElements,
                               dtype,
                               inputScale,
                               createI64(op.getInputZp()),
                               outputScale,
                               createI64(op.getOutputZp()),
                               createI64(op.getAxis()),
                               createI64(op.getP())};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQLpNormalizationLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<QLpNormalizationOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
