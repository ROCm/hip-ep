/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.fast_gelu(ctx, input, [bias], output)
//   -> wrap_fast_gelu(state, input, bias_or_null, output, num_elements,
//                     bias_len, data_type)
struct FastGeluOpLowering : public ConvertOpToLLVMPattern<FastGeluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FastGeluOp op, OpAdaptor adaptor,
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
    Value biasPtr = extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());
    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize = outputType.isDynamicDim(dimIdx)
                          ? outputDesc.size(rewriter, loc, dimIdx)
                          : createI64Const(outputType.getDimSize(dimIdx));
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    Value biasLen = createI64Const(0);
    if (adaptor.getBias()) {
      auto biasType = cast<MemRefType>(op.getBias().getType());
      int64_t biasDimIdx = biasType.getRank() - 1;
      MemRefDescriptor biasDesc(adaptor.getBias());
      biasLen = biasType.isDynamicDim(biasDimIdx)
                    ? biasDesc.size(rewriter, loc, biasDimIdx)
                    : createI64Const(biasType.getDimSize(biasDimIdx));
    }

    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0 || (dataType > 2 && dataType != 6)) {
      return rewriter.notifyMatchFailure(
          op,
          "unsupported element type for FastGelu (expected f32/f16/bf16/f64)");
    }

    Value dataTypeVal = createI64Const(dataType);

    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapFastGelu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,    inputPtr, biasPtr,    outputPtr,
                                  numElements, biasLen,  dataTypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateFastGeluLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<FastGeluOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
