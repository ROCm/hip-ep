/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// QMoEAmd Lowering
//===----------------------------------------------------------------------===//

struct QMoEAmdOpLowering : public ConvertOpToLLVMPattern<QMoEAmdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QMoEAmdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    Value statePtr = adaptor.getHandle();
    Value hiddenStatesPtr =
        extractContiguousMemRefPtr(adaptor.getHiddenStates(), rewriter, loc);
    Value fc1ExpertsWeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc1ExpertsWeights(), rewriter, loc);
    Value fc1ExpertsScalesPtr = extractContiguousMemRefPtr(
        adaptor.getFc1ExpertsScales(), rewriter, loc);
    Value fc2ExpertsWeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc2ExpertsWeights(), rewriter, loc);
    Value fc2ExpertsScalesPtr = extractContiguousMemRefPtr(
        adaptor.getFc2ExpertsScales(), rewriter, loc);
    Value fc1LatentWeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc1LatentWeights(), rewriter, loc);
    Value fc1LatentScalesPtr =
        extractContiguousMemRefPtr(adaptor.getFc1LatentScales(), rewriter, loc);
    Value fc2LatentWeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc2LatentWeights(), rewriter, loc);
    Value fc2LatentScalesPtr =
        extractContiguousMemRefPtr(adaptor.getFc2LatentScales(), rewriter, loc);
    Value sharedFc1WeightsPtr = extractContiguousMemRefPtr(
        adaptor.getSharedFc1Weights(), rewriter, loc);
    Value sharedFc1ScalesPtr =
        extractContiguousMemRefPtr(adaptor.getSharedFc1Scales(), rewriter, loc);
    Value sharedFc2WeightsPtr = extractContiguousMemRefPtr(
        adaptor.getSharedFc2Weights(), rewriter, loc);
    Value sharedFc2ScalesPtr =
        extractContiguousMemRefPtr(adaptor.getSharedFc2Scales(), rewriter, loc);
    Value routerWeightPtr =
        extractContiguousMemRefPtr(adaptor.getRouterWeight(), rewriter, loc);
    Value correctionBiasPtr =
        extractContiguousMemRefPtr(adaptor.getCorrectionBias(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto hiddenStatesType = cast<MemRefType>(op.getHiddenStates().getType());
    auto fc1ExpertsWeightsType =
        cast<MemRefType>(op.getFc1ExpertsWeights().getType());
    auto fc1LatentWeightsType =
        cast<MemRefType>(op.getFc1LatentWeights().getType());
    auto sharedFc1WeightsType =
        cast<MemRefType>(op.getSharedFc1Weights().getType());
    int64_t elemSize =
        hiddenStatesType.getElementType().getIntOrFloatBitWidth() / 8;

    // hidden_states shape: [batch, seq, ..., hidden] -- numTokens = product
    // of all dims except the last (hidden), supporting dynamic batch/seq.
    int64_t inputRank = hiddenStatesType.getRank();
    Value numTokensVal = createI64Const(1);
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      numTokensVal = LLVM::MulOp::create(
          rewriter, loc, numTokensVal,
          getMemRefDimSize(hiddenStatesType, i, adaptor.getHiddenStates(),
                           rewriter, loc));
    }
    Value hiddenSizeVal =
        getMemRefDimSize(hiddenStatesType, inputRank - 1,
                         adaptor.getHiddenStates(), rewriter, loc);

    // num_experts / moe_intermediate_size come from fc1_experts_weights'
    // leading two dims: [num_experts, moe_intermediate_size, latent/pack].
    Value numExpertsVal =
        getMemRefDimSize(fc1ExpertsWeightsType, 0,
                         adaptor.getFc1ExpertsWeights(), rewriter, loc);
    Value moeInterSizeVal =
        getMemRefDimSize(fc1ExpertsWeightsType, 1,
                         adaptor.getFc1ExpertsWeights(), rewriter, loc);
    // latent_size is fc1_latent_weights' leading dim: [latent, hidden/pack].
    Value latentSizeVal = getMemRefDimSize(
        fc1LatentWeightsType, 0, adaptor.getFc1LatentWeights(), rewriter, loc);
    // shared_intermediate_size is shared_fc1_weights' leading dim:
    // [shared_inter, hidden/pack].
    Value sharedInterSizeVal = getMemRefDimSize(
        sharedFc1WeightsType, 0, adaptor.getSharedFc1Weights(), rewriter, loc);

    // Translate both modes to their runtime identifiers without judging
    // whether a kernel implements them: an unrecognized name becomes UNKNOWN
    // and wrap_qmoe_amd rejects it. Mapping an unrecognized name onto relu2 /
    // sigmoid instead would compute the wrong function for a graph that asked
    // for something else.
    int64_t activationTypeEnum = kQMoEAmdActivationUnknown;
    if (op.getActivationType() == "relu2")
      activationTypeEnum = kQMoEAmdActivationRelu2;
    int64_t routingTypeEnum = kQMoEAmdRoutingUnknown;
    if (op.getRoutingType() == "sigmoid")
      routingTypeEnum = kQMoEAmdRoutingSigmoid;

    Value kVal = createI64Const(op.getK());
    Value expertWeightBitsVal = createI64Const(op.getExpertWeightBits());
    Value blockSizeVal = createI64Const(op.getBlockSize());
    Value normalizeVal = createI64Const(op.getNormalizeRoutingWeights());
    Value useCorrectionBiasVal = createI64Const(op.getUseCorrectionBias());
    Value routedScalingFactorVal =
        createF32Const(op.getRoutedScalingFactor().convertToFloat());
    Value activationTypeVal = createI64Const(activationTypeEnum);
    Value routingTypeVal = createI64Const(routingTypeEnum);
    Value elemSizeVal = createI64Const(elemSize);

    SmallVector<Type, 32> paramTypes = {
        ptrType, // state
        ptrType, // hidden_states
        ptrType, // fc1_experts_weights
        ptrType, // fc1_experts_scales
        ptrType, // fc2_experts_weights
        ptrType, // fc2_experts_scales
        ptrType, // fc1_latent_weights
        ptrType, // fc1_latent_scales
        ptrType, // fc2_latent_weights
        ptrType, // fc2_latent_scales
        ptrType, // shared_fc1_weights
        ptrType, // shared_fc1_scales
        ptrType, // shared_fc2_weights
        ptrType, // shared_fc2_scales
        ptrType, // router_weight
        ptrType, // correction_bias
        ptrType, // output
        i64Type, // num_tokens
        i64Type, // hidden_size
        i64Type, // latent_size
        i64Type, // moe_intermediate_size
        i64Type, // shared_intermediate_size
        i64Type, // num_experts
        i64Type, // k
        i64Type, // expert_weight_bits
        i64Type, // block_size
        i64Type, // normalize_routing_weights
        i64Type, // use_correction_bias
        f32Type, // routed_scaling_factor
        i64Type, // activation_type
        i64Type, // routing_type
        i64Type  // elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQMoEAmd, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 32> args = {statePtr,
                                   hiddenStatesPtr,
                                   fc1ExpertsWeightsPtr,
                                   fc1ExpertsScalesPtr,
                                   fc2ExpertsWeightsPtr,
                                   fc2ExpertsScalesPtr,
                                   fc1LatentWeightsPtr,
                                   fc1LatentScalesPtr,
                                   fc2LatentWeightsPtr,
                                   fc2LatentScalesPtr,
                                   sharedFc1WeightsPtr,
                                   sharedFc1ScalesPtr,
                                   sharedFc2WeightsPtr,
                                   sharedFc2ScalesPtr,
                                   routerWeightPtr,
                                   correctionBiasPtr,
                                   outputPtr,
                                   numTokensVal,
                                   hiddenSizeVal,
                                   latentSizeVal,
                                   moeInterSizeVal,
                                   sharedInterSizeVal,
                                   numExpertsVal,
                                   kVal,
                                   expertWeightBitsVal,
                                   blockSizeVal,
                                   normalizeVal,
                                   useCorrectionBiasVal,
                                   routedScalingFactorVal,
                                   activationTypeVal,
                                   routingTypeVal,
                                   elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQMoEAmdLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<QMoEAmdOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
