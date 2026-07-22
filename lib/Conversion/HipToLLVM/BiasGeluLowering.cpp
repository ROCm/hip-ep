/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.bias_gelu(ctx, data, bias, output)
//   -> wrap_bias_gelu(state, data, bias, output, num_elements, bias_len,
//                     data_type)
struct BiasGeluOpLowering : public ConvertOpToLLVMPattern<BiasGeluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BiasGeluOp op, OpAdaptor adaptor,
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
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value biasPtr =
        extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    auto biasType = cast<MemRefType>(op.getBias().getType());

    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());
    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize = outputType.isDynamicDim(dimIdx)
                          ? outputDesc.size(rewriter, loc, dimIdx)
                          : createI64Const(outputType.getDimSize(dimIdx));
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    int64_t biasDimIdx = biasType.getRank() - 1;
    MemRefDescriptor biasDesc(adaptor.getBias());
    Value biasLen = biasType.isDynamicDim(biasDimIdx)
                        ? biasDesc.size(rewriter, loc, biasDimIdx)
                        : createI64Const(biasType.getDimSize(biasDimIdx));

    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0 || (dataType > 2 && dataType != 6)) {
      return rewriter.notifyMatchFailure(
          op,
          "unsupported element type for BiasGelu (expected f32/f16/bf16/f64)");
    }

    Value dataTypeVal = createI64Const(dataType);

    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapBiasGelu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,    dataPtr, biasPtr,    outputPtr,
                                  numElements, biasLen, dataTypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateBiasGeluLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<BiasGeluOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
