/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.miopen.rms_norm(%handle) ins(%input, %weight) outs(%output)
//   -> hip_miopen_rms_norm(handle, input, weight, output, N, D)
// Rank-generic: N = product of all dims except last, D = last dim.
// For 3D [B,S,D]: N = B*S, D = D.
struct RmsNormOpLowering : public ConvertOpToLLVMPattern<RmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value scalePtr =
        extractContiguousMemRefPtr(adaptor.getScale(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute num_elements with dynamic shape support
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto scaleType = cast<MemRefType>(op.getScale().getType());
    Value scaleNumElements =
        computeNumElements(scaleType, adaptor.getScale(), rewriter, loc);

    // Compute element_size_bytes based on element type
    Type elementType = inputType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    Value elementSizeBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Extract attributes
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getAxis()));
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());
    Value stashTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getStashType()));

    // Runtime function signature (10 params)
    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, input, scale, output
        i64Type, i64Type, i64Type, // input_num_elements, scale_num_elements,
                                   // element_size_bytes
        i64Type, f32Type, i64Type  // axis, epsilon, stash_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapMiopenT5LayerNormForward,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               inputPtr,
                               scalePtr,
                               outputPtr,
                               inputNumElements,
                               scaleNumElements,
                               elementSizeBytesVal,
                               axisVal,
                               epsilonVal,
                               stashTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.skip_rms_norm lowering with dynamic shape support
struct SkipRmsNormOpLowering : public ConvertOpToLLVMPattern<SkipRmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SkipRmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value skipPtr =
        extractContiguousMemRefPtr(adaptor.getSkip(), rewriter, loc);
    Value gammaPtr =
        extractContiguousMemRefPtr(adaptor.getGamma(), rewriter, loc);
    Value biasPtr = getMemRefPtrOrNull(adaptor.getBias());
    // DPS outputs: outputs[0]=output, outputs[1]=input_skip_bias_sum (optional)
    auto outputs = adaptor.getOutputs();
    Value outputPtr = extractContiguousMemRefPtr(outputs[0], rewriter, loc);
    Value skipOutputPtr =
        outputs.size() > 1
            ? extractContiguousMemRefPtr(outputs[1], rewriter, loc)
            : nullPtr;

    // Compute num_elements for input and gamma
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto gammaType = cast<MemRefType>(op.getGamma().getType());
    Value gammaNumElements =
        computeNumElements(gammaType, adaptor.getGamma(), rewriter, loc);

    // Compute element_size_bytes based on element type
    Type elementType = inputType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    Value elementSizeBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Extract epsilon attribute
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    // Runtime function signature (11 params)
    SmallVector<Type> paramTypes = {
        ptrType, // state
        ptrType, // input
        ptrType, // skip
        ptrType, // gamma
        ptrType, // bias (may be nullptr)
        ptrType, // output
        ptrType, // input_skip_bias_sum (may be nullptr)
        i64Type, // input_num_elements
        i64Type, // gamma_num_elements
        i64Type, // element_size_bytes
        f32Type  // epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapSkipSimplifiedLayerNorm,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,         inputPtr,
                               skipPtr,          gammaPtr,
                               biasPtr,          outputPtr,
                               skipOutputPtr,    inputNumElements,
                               gammaNumElements, elementSizeBytesVal,
                               epsilonVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNormLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<RmsNormOpLowering, SkipRmsNormOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
