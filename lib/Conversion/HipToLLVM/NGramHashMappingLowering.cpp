/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct NGramHashMappingOpLowering
    : public ConvertOpToLLVMPattern<NGramHashMappingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NGramHashMappingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    auto idsType = cast<MemRefType>(op.getInputIds().getType());
    ValueRange outputs = adaptor.getOutputs();
    Value presentDesc = outputs.size() > 1 ? outputs[1] : Value();

    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInputIds(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getMultipliers(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getVocabSizes(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getPastIds(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getHeadOffsets(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getEosTokenId(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getSegmentIds(), rewriter, loc),
        extractContiguousMemRefPtr(outputs[0], rewriter, loc),
        extractOptionalMemRefPtr(presentDesc, rewriter, loc),
        getMemRefDimSize(idsType, 0, adaptor.getInputIds(), rewriter, loc),
        getMemRefDimSize(idsType, 1, adaptor.getInputIds(), rewriter, loc),
        createI64Const(op.getMaxNgramSize()),
        createI64Const(op.getNHeadPerNgram()),
        createI64Const(op.getPadId()),
        createI64Const(op.getResetOnEos()),
        createI64Const(idsType.getElementType().getIntOrFloatBitWidth() / 8)};

    SmallVector<Type, 17> paramTypes(10, ptrType);
    paramTypes.append(7, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNGramHashMapping, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNGramHashMappingLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NGramHashMappingOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
