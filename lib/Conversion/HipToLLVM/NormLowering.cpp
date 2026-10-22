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
//        input_num_elements, scale_num_elements, norm_num_elements,
//        element_size_bytes, axis, epsilon, stash_type)
// Rank-generic. norm_num_elements is the ONNX reduction width -- the product
// of the input dims from `axis` on -- so a 3D [B,S,D] input with axis = -1
// normalizes B*S rows of width D. It has to be computed here, the only place
// that sees both the attribute and the input shape. Taking it from
// scale_num_elements instead is wrong for any norm whose scale spans more than
// the reduced axes, such as a grouped scale [G,D] against an [N,G,D] input:
// that reduces over G*D rather than D.
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

    // Reduction width = product of the input dims from `axis` on.
    int64_t rank = inputType.getRank();
    int64_t normAxis = op.getAxis();
    if (normAxis < 0)
      normAxis += rank;
    if (normAxis < 0 || normAxis >= rank)
      return rewriter.notifyMatchFailure(op, "rms_norm.axis_out_of_range");
    Value normNumElements = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
    for (int64_t dimIdx = normAxis; dimIdx < rank; ++dimIdx)
      normNumElements = LLVM::MulOp::create(
          rewriter, loc, normNumElements,
          getMemRefDimSize(inputType, static_cast<unsigned>(dimIdx),
                           adaptor.getInput(), rewriter, loc));

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

    // Runtime function signature (11 params)
    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, output
        i64Type, i64Type,          // input_num_elements, scale_num_elements
        i64Type, i64Type,          // norm_num_elements, element_size_bytes
        i64Type, f32Type, i64Type  // axis, epsilon, stash_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRmsNorm, paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,         inputPtr,
                               scalePtr,         outputPtr,
                               inputNumElements, scaleNumElements,
                               normNumElements,  elementSizeBytesVal,
                               axisVal,          epsilonVal,
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

// hip.batch_norm(%ctx) ins(%input, %scale, %bias, %mean, %variance)
//   outs(%output)
//   -> wrap_batch_normalization(state, input, scale, bias, mean, variance,
//        output, n, c, spatial, data_type, epsilon)
struct BatchNormOpLowering : public ConvertOpToLLVMPattern<BatchNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BatchNormOp op, OpAdaptor adaptor,
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
    Value meanPtr =
        extractContiguousMemRefPtr(adaptor.getMean(), rewriter, loc);
    Value variancePtr =
        extractContiguousMemRefPtr(adaptor.getVariance(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    if (inputType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "hip.batch_norm requires input rank >= 2");

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
        ptrType, // state
        ptrType, ptrType, ptrType,
        ptrType, ptrType,          // input, scale, bias, mean, var
        ptrType,                   // output
        i64Type, i64Type, i64Type, // n, c, spatial
        i64Type, f32Type           // data_type, epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapBatchNormalization,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr, inputPtr,    scalePtr,    biasPtr,
                               meanPtr,  variancePtr, outputPtr,   n,
                               c,        spatial,     dataTypeVal, epsilonVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.group_norm -> wrap_group_norm(state, input, scale, bias, output,
//   n, c, spatial, groups, channels_last, activation, data_type, epsilon)
struct GroupNormOpLowering : public ConvertOpToLLVMPattern<GroupNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GroupNormOp op, OpAdaptor adaptor,
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
          op, "hip.group_norm requires input rank >= 3");

    int64_t channelAxis = op.getChannelsLast() ? inputType.getRank() - 1 : 1;
    Value n = getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value c = getMemRefDimSize(inputType, static_cast<unsigned>(channelAxis),
                               adaptor.getInput(), rewriter, loc);
    Value spatial = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(1));
    for (int64_t dimIdx = 1, rank = inputType.getRank(); dimIdx < rank;
         ++dimIdx) {
      if (dimIdx == channelAxis)
        continue;
      spatial = LLVM::MulOp::create(
          rewriter, loc,
          getMemRefDimSize(inputType, static_cast<unsigned>(dimIdx),
                           adaptor.getInput(), rewriter, loc),
          spatial);
    }

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value dataTypeVal = i64Const(dataType);
    Value groupsVal = i64Const(op.getGroups());
    Value channelsLastVal = i64Const(op.getChannelsLast());
    Value activationVal = i64Const(op.getActivation());
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, bias
        ptrType,                   // output
        i64Type, i64Type, i64Type, // n, c, spatial
        i64Type, i64Type, i64Type, // groups, channels_last, activation
        i64Type, f32Type           // data_type, epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGroupNorm, paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               inputPtr,
                               scalePtr,
                               biasPtr,
                               outputPtr,
                               n,
                               c,
                               spatial,
                               groupsVal,
                               channelsLastVal,
                               activationVal,
                               dataTypeVal,
                               epsilonVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.gated_rms_norm -> wrap_gated_rms_norm(state, input, scale, gate, output,
//   input_num_elements, norm_size, element_size_bytes, activation, epsilon)
// activation: 0 = sigmoid, 1 = silu.
struct GatedRmsNormOpLowering : public ConvertOpToLLVMPattern<GatedRmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatedRmsNormOp op, OpAdaptor adaptor,
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
    Value gatePtr =
        extractContiguousMemRefPtr(adaptor.getGate(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto scaleType = cast<MemRefType>(op.getScale().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);
    Value normSize =
        computeNumElements(scaleType, adaptor.getScale(), rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value elementSize =
        i64Const(inputType.getElementType().getIntOrFloatBitWidth() / 8);
    Value activation = i64Const(op.getActivation() == "silu" ? 1 : 0);
    Value epsilon =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, gate
        ptrType,                   // output
        i64Type, i64Type, i64Type, // input_num_elements, norm_size, elem size
        i64Type, f32Type           // activation, epsilon
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGatedRmsNorm, paramTypes, rewriter.getI32Type());
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value> args = {
        statePtr,         inputPtr, scalePtr,    gatePtr,    outputPtr,
        inputNumElements, normSize, elementSize, activation, epsilon};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.engram_gate -> wrap_engram_gate(state, key, query, value, key_scale,
//   query_scale, conv_scale_or_null, output, normed_or_null,
//   key_num_elements, hc, hidden, element_size_bytes, epsilon)
struct EngramGateOpLowering : public ConvertOpToLLVMPattern<EngramGateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(EngramGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto ptr = [&](Value memref) {
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };
    auto outputs = adaptor.getOutputs();
    Value normedPtr = outputs.size() > 1
                          ? ptr(outputs[1])
                          : LLVM::ZeroOp::create(rewriter, loc, ptrType);
    Value convScalePtr =
        getNullableMemRefPtr(adaptor.getConvNormScale(), rewriter, loc);

    auto keyType = cast<MemRefType>(op.getKey().getType());
    Value keyNumElements =
        computeNumElements(keyType, adaptor.getKey(), rewriter, loc);
    Value hc = getMemRefDimSize(keyType, 2, adaptor.getKey(), rewriter, loc);
    Value hidden =
        getMemRefDimSize(keyType, 3, adaptor.getKey(), rewriter, loc);
    Value elementSize = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(
            keyType.getElementType().getIntOrFloatBitWidth() / 8));
    Value epsilon =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    SmallVector<Type> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // key, query, value
        ptrType, ptrType, ptrType, // key_scale, query_scale, conv_scale
        ptrType, ptrType,          // output, normed
        i64Type, i64Type, i64Type, // key_num_elements, hc, hidden
        i64Type, f32Type           // element_size_bytes, epsilon
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapEngramGate, paramTypes, rewriter.getI32Type());
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value> args = {adaptor.getCtx(),
                               ptr(adaptor.getKey()),
                               ptr(adaptor.getQuery()),
                               ptr(adaptor.getValue()),
                               ptr(adaptor.getKeyNormScale()),
                               ptr(adaptor.getQueryNormScale()),
                               convScalePtr,
                               ptr(outputs[0]),
                               normedPtr,
                               keyNumElements,
                               hc,
                               hidden,
                               elementSize,
                               epsilon};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};


// hip.hyper_connection_pre_mix -> wrap_hyper_connection_pre_mix
struct HyperConnectionPreMixOpLowering
    : public ConvertOpToLLVMPattern<HyperConnectionPreMixOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HyperConnectionPreMixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();
    auto streamsType = cast<MemRefType>(op.getStreams().getType());
    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value elementSize =
        i64Const(streamsType.getElementType().getIntOrFloatBitWidth() / 8);
    Value featureGate = i64Const(op.getGateLayout() == "feature" ? 1 : 0);
    Value gateSigmoid = i64Const(op.getGateSigmoid());
    Value scale = LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                           op.getReductionScaleAttr());

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                    i64Type, i64Type, i64Type, i64Type,
                                    i64Type, i64Type, f32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapHyperConnectionPreMix,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();
    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getStreams(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPreMix(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        computeNumElements(streamsType, adaptor.getStreams(), rewriter, loc),
        i64Const(op.getBranches()),
        i64Const(op.getHidden()),
        featureGate,
        gateSigmoid,
        elementSize,
        scale};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.hyper_connection_post_mix -> wrap_hyper_connection_post_mix
struct HyperConnectionPostMixOpLowering
    : public ConvertOpToLLVMPattern<HyperConnectionPostMixOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HyperConnectionPostMixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    auto streamsType = cast<MemRefType>(op.getStreams().getType());
    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                    ptrType, ptrType, i64Type, i64Type,
                                    i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapHyperConnectionPostMix,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();
    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getStreams(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBlockOutput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPostMix(), rewriter, loc),
        getNullableMemRefPtr(adaptor.getStreamMix(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        computeNumElements(streamsType, adaptor.getStreams(), rewriter, loc),
        i64Const(op.getBranches()),
        i64Const(op.getHidden()),
        i64Const(op.getGateLayout() == "feature" ? 1 : 0),
        i64Const(streamsType.getElementType().getIntOrFloatBitWidth() / 8)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct HyperConnectionPostMixNormOpLowering
    : public ConvertOpToLLVMPattern<HyperConnectionPostMixNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HyperConnectionPostMixNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();
    auto streamsType = cast<MemRefType>(op.getStreams().getType());
    auto outputs = adaptor.getOutputs();
    if (outputs.size() != 2)
      return failure();
    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, f32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHyperConnectionPostMixNorm, paramTypes,
        rewriter.getI32Type());
    if (failed(funcOp))
      return failure();
    Value epsilon = LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                             op.getEpsilonAttr());
    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getStreams(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBlockOutput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getGate(), rewriter, loc),
        getNullableMemRefPtr(adaptor.getStreamMix(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getNormScale(), rewriter, loc),
        extractContiguousMemRefPtr(outputs[0], rewriter, loc),
        extractContiguousMemRefPtr(outputs[1], rewriter, loc),
        computeNumElements(streamsType, adaptor.getStreams(), rewriter, loc),
        i64Const(op.getBranches()),
        i64Const(op.getHidden()),
        i64Const(op.getGateLayout() == "feature" ? 1 : 0),
        i64Const(op.getGateActivation()),
        i64Const(streamsType.getElementType().getIntOrFloatBitWidth() / 8),
        epsilon};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNormLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<RmsNormOpLowering, SkipRmsNormOpLowering, LayerNormOpLowering,
               InstanceNormOpLowering, BatchNormOpLowering, GroupNormOpLowering,
               GatedRmsNormOpLowering, EngramGateOpLowering,
               HyperConnectionPreMixOpLowering, HyperConnectionPostMixOpLowering,
               HyperConnectionPostMixNormOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
