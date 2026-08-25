/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// HIP -> LLVM lowering for the Norm operator family.
//
// Each Norm op gets its own lowering struct here. Helpers shared across all
// Norm variants (e.g. nullable memref pointer extraction) live at the top of
// the anonymous namespace so that future Norm ops (BatchNorm, GroupNorm,
// InstanceNorm, ...) can plug in without duplicating boilerplate.
//===----------------------------------------------------------------------===//

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Returns the aligned data pointer for an optional memref operand, or a null
// pointer of address space 0 when the operand is absent. Local equivalent of
// extractOptionalMemRefPtr that explicitly takes the rewriter, kept here so
// the Norm lowerings can share it without leaking into the public utils.
inline Value getNullableMemRefPtr(Value memref,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  if (!memref)
    return LLVM::ZeroOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0));
  return extractContiguousMemRefPtr(memref, rewriter, loc);
}

// hip.rms_norm(%ctx) ins(%input, %scale) outs(%output)
//   -> wrap_rms_norm(state, input, scale, output,
//        input_num_elements, scale_num_elements, element_size_bytes,
//        axis, epsilon, stash_type)
// Rank-generic: the runtime derives the row count from
// input_num_elements / scale_num_elements, so a 3D [B,S,D] input normalizes
// B*S rows of width D.
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
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, output
        i64Type, i64Type, i64Type, // input_num_elements, scale_num_elements,
                                   // element_size_bytes
        i64Type, f32Type, i64Type  // axis, epsilon, stash_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRmsNorm, paramTypes, rewriter.getI32Type());
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

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value skipPtr =
        extractContiguousMemRefPtr(adaptor.getSkip(), rewriter, loc);
    Value gammaPtr =
        extractContiguousMemRefPtr(adaptor.getGamma(), rewriter, loc);
    Value biasPtr = getNullableMemRefPtr(adaptor.getBias(), rewriter, loc);
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

// hip.layer_norm(%ctx) ins(%input, %scale, [%bias])
//                       outs(%output, [%mean, [%inv_std]])
//   -> wrap_layer_normalization(state,
//        input, scale, bias_or_null,
//        output, mean_or_null, inv_std_or_null,
//        input_num_elements, scale_num_elements, element_size_bytes,
//        axis, epsilon, stash_type)
//
// bias / mean / inv_std are optional — when absent, a null pointer (address
// space 0) is forwarded so the runtime can early-out for that step.
struct LayerNormOpLowering : public ConvertOpToLLVMPattern<LayerNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    // Required pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value scalePtr =
        extractContiguousMemRefPtr(adaptor.getScale(), rewriter, loc);
    Value biasPtr = getNullableMemRefPtr(adaptor.getBias(), rewriter, loc);

    // DPS outputs: [output] | [output, mean] | [output, mean, inv_std]
    auto outputs = adaptor.getOutputs();
    if (outputs.empty())
      return rewriter.notifyMatchFailure(
          op, "hip.layer_norm requires at least one output buffer");

    Value outputPtr = extractContiguousMemRefPtr(outputs[0], rewriter, loc);
    Value meanPtr = outputs.size() > 1
                        ? extractContiguousMemRefPtr(outputs[1], rewriter, loc)
                        : LLVM::ZeroOp::create(rewriter, loc, ptrType);
    Value invStdPtr =
        outputs.size() > 2
            ? extractContiguousMemRefPtr(outputs[2], rewriter, loc)
            : LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Compute num_elements for input and scale, with dynamic-shape support.
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto scaleType = cast<MemRefType>(op.getScale().getType());
    Value scaleNumElements =
        computeNumElements(scaleType, adaptor.getScale(), rewriter, loc);

    // element_size_bytes derived from input element type.
    Type elementType = inputType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    Value elementSizeBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Attribute lowering.
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getAxis()));
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());
    Value stashTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getStashType()));

    // Runtime function signature (13 params):
    //   state, input, scale, bias, output, mean, inv_std,
    //   input_num_elements, scale_num_elements, element_size_bytes,
    //   axis, epsilon, stash_type
    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, bias (may be null)
        ptrType, ptrType, ptrType, // output, mean (may be null), inv_std
                                   // (may be null)
        i64Type, i64Type, i64Type, // input_num, scale_num, element_size_bytes
        i64Type, f32Type, i64Type  // axis, epsilon, stash_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapLayerNormalization,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,         inputPtr,
                               scalePtr,         biasPtr,
                               outputPtr,        meanPtr,
                               invStdPtr,        inputNumElements,
                               scaleNumElements, elementSizeBytesVal,
                               axisVal,          epsilonVal,
                               stashTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.instance_norm(%ctx) ins(%input, %scale, %bias) outs(%output)
//   -> wrap_instance_normalization(state, input, scale, bias, output,
//        n, c, spatial, data_type, epsilon)
struct InstanceNormOpLowering : public ConvertOpToLLVMPattern<InstanceNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(InstanceNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value scalePtr =
        extractContiguousMemRefPtr(adaptor.getScale(), rewriter, loc);
    Value biasPtr =
        extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    if (inputType.getRank() < 3)
      return rewriter.notifyMatchFailure(
          op, "hip.instance_norm requires input rank >= 3");

    Value n = getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value c = getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    Value spatial = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(1));
    for (int64_t dimIdx = 2, rank = inputType.getRank(); dimIdx < rank;
         ++dimIdx) {
      spatial = LLVM::MulOp::create(
          rewriter, loc,
          getMemRefDimSize(inputType, static_cast<unsigned>(dimIdx),
                           adaptor.getInput(), rewriter, loc),
          spatial);
    }

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    Value dataTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dataType));
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, bias
        ptrType,                   // output
        i64Type, i64Type, i64Type, // n, c, spatial
        i64Type, f32Type           // data_type, epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapInstanceNormalization,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,    inputPtr,  scalePtr, biasPtr,
                               outputPtr,   n,         c,        spatial,
                               dataTypeVal, epsilonVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNormLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<RmsNormOpLowering, SkipRmsNormOpLowering, LayerNormOpLowering,
               InstanceNormOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
