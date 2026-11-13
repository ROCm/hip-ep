/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TriluLowering.cpp - hip.trilu -> wrap_trilu -----------------------===//

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.trilu(ctx, input, output) {k, upper}
//   -> wrap_trilu(state, input, output, input_elements, num_elements, rows,
//                 cols, k, upper, data_type)
//
// A one-element input (rank-0 splat) is broadcast over the output.
struct TriluOpLowering : public ConvertOpToLLVMPattern<TriluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TriluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    if (outputType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "trilu requires rank >= 2");

    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0 || (dataType > 2 && dataType != 6)) {
      return rewriter.notifyMatchFailure(
          op, "trilu supports f32, f16, bf16, and f64");
    }

    MemRefDescriptor outputDesc(adaptor.getOutput());
    auto dimSize = [&](int64_t dimIdx) -> Value {
      if (outputType.isDynamicDim(dimIdx))
        return outputDesc.size(rewriter, loc, dimIdx);
      return createI64Const(outputType.getDimSize(dimIdx));
    };

    Value numElements = createI64Const(1);
    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank()))
      numElements =
          LLVM::MulOp::create(rewriter, loc, numElements, dimSize(dimIdx));
    Value rows = dimSize(outputType.getRank() - 2);
    Value cols = dimSize(outputType.getRank() - 1);
    Value inputElements =
        computeNumElements(cast<MemRefType>(op.getInput().getType()),
                           adaptor.getInput(), rewriter, loc);

    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapTrilu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {statePtr,
                                   inputPtr,
                                   outputPtr,
                                   inputElements,
                                   numElements,
                                   rows,
                                   cols,
                                   createI64Const(op.getK()),
                                   createI64Const(op.getUpper()),
                                   createI64Const(dataType)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTriluLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<TriluOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
