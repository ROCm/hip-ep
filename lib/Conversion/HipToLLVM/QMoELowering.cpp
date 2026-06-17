/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// QMoE Lowering
//===----------------------------------------------------------------------===//

struct QMoEOpLowering : public ConvertOpToLLVMPattern<QMoEOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QMoEOp op, OpAdaptor adaptor,
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
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value routerPtr =
        extractContiguousMemRefPtr(adaptor.getRouterProbs(), rewriter, loc);
    Value fc1WeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc1ExpertsWeights(), rewriter, loc);
    Value fc1ScalesPtr =
        extractContiguousMemRefPtr(adaptor.getFc1Scales(), rewriter, loc);
    Value fc1BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc1ExpertsBias(), rewriter, loc);
    Value fc2WeightsPtr = extractContiguousMemRefPtr(
        adaptor.getFc2ExpertsWeights(), rewriter, loc);
    Value fc2ScalesPtr =
        extractContiguousMemRefPtr(adaptor.getFc2Scales(), rewriter, loc);
    Value fc2BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc2ExpertsBias(), rewriter, loc);
    Value fc3WeightsPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ExpertsWeights(), rewriter, loc);
    Value fc3ScalesPtr =
        extractOptionalMemRefPtr(adaptor.getFc3Scales(), rewriter, loc);
    Value fc3BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ExpertsBias(), rewriter, loc);
    Value fc1ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc1ZeroPoints(), rewriter, loc);
    Value fc2ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc2ZeroPoints(), rewriter, loc);
    Value fc3ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ZeroPoints(), rewriter, loc);
    Value routerWeightsPtr =
        extractOptionalMemRefPtr(adaptor.getRouterWeights(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto routerType = cast<MemRefType>(op.getRouterProbs().getType());
    auto fc1Type = cast<MemRefType>(op.getFc1ExpertsWeights().getType());
    int64_t elemSize = inputType.getElementType().getIntOrFloatBitWidth() / 8;

    // input shape: [batch, seq, ..., hidden] — numTokens = product of all
    // dims except the last (hidden), supporting dynamic batch/seq dimensions.
    int64_t inputRank = inputType.getRank();
    Value numTokensVal = createI64Const(1);
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      numTokensVal = LLVM::MulOp::create(
          rewriter, loc, numTokensVal,
          getMemRefDimSize(inputType, i, adaptor.getInput(), rewriter, loc));
    }
    Value hiddenSizeVal = getMemRefDimSize(inputType, inputRank - 1,
                                           adaptor.getInput(), rewriter, loc);
    Value numExpertsVal =
        getMemRefDimSize(routerType, routerType.getRank() - 1,
                         adaptor.getRouterProbs(), rewriter, loc);

    int64_t swigluFusion = op.getSwigluFusion();
    int64_t fusionSize = (swigluFusion > 0) ? 2 : 1;
    Value interSizeVal = getMemRefDimSize(
        fc1Type, 1, adaptor.getFc1ExpertsWeights(), rewriter, loc);
    if (fusionSize > 1) {
      interSizeVal = LLVM::SDivOp::create(rewriter, loc, interSizeVal,
                                          createI64Const(fusionSize));
    }

    StringRef activationType = op.getActivationType();
    int64_t activationTypeEnum = 0;
    if (activationType == "relu") {
      activationTypeEnum = 0;
    } else if (activationType == "gelu") {
      activationTypeEnum = 1;
    } else if (activationType == "silu") {
      activationTypeEnum = 2;
    } else if (activationType == "swiglu") {
      activationTypeEnum = 3;
    } else if (activationType == "identity") {
      activationTypeEnum = 4;
    }

    Value kVal = createI64Const(op.getK());
    Value expertWeightBitsVal = createI64Const(op.getExpertWeightBits());
    Value blockSizeVal = createI64Const(op.getBlockSize());
    Value swigluFusionVal = createI64Const(swigluFusion);
    Value activationTypeVal = createI64Const(activationTypeEnum);
    Value activationAlphaVal =
        createF32Const(op.getActivationAlpha().convertToFloat());
    Value activationBetaVal =
        createF32Const(op.getActivationBeta().convertToFloat());
    Value swigluLimitVal = createF32Const(op.getSwigluLimit().convertToFloat());
    Value normalizeVal = createI64Const(op.getNormalizeRoutingWeights());
    Value elemSizeVal = createI64Const(elemSize);

    SmallVector<Type, 31> paramTypes = {
        ptrType, // state
        ptrType, // input
        ptrType, // router_probs
        ptrType, // router_weights (nullable)
        ptrType, // fc1_weights
        ptrType, // fc1_scales
        ptrType, // fc1_bias (nullable)
        ptrType, // fc2_weights
        ptrType, // fc2_scales
        ptrType, // fc2_bias (nullable)
        ptrType, // fc3_weights (nullable)
        ptrType, // fc3_scales (nullable)
        ptrType, // fc3_bias (nullable)
        ptrType, // fc1_zero_points (nullable)
        ptrType, // fc2_zero_points (nullable)
        ptrType, // fc3_zero_points (nullable)
        ptrType, // output
        i64Type, // num_tokens
        i64Type, // hidden_size
        i64Type, // inter_size
        i64Type, // num_experts
        i64Type, // k
        i64Type, // expert_weight_bits
        i64Type, // block_size
        i64Type, // swiglu_fusion
        i64Type, // activation_type
        f32Type, // activation_alpha
        f32Type, // activation_beta
        f32Type, // swiglu_limit
        i64Type, // normalize_routing_weights
        i64Type  // elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQMoE, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 31> args = {statePtr,
                                   inputPtr,
                                   routerPtr,
                                   routerWeightsPtr,
                                   fc1WeightsPtr,
                                   fc1ScalesPtr,
                                   fc1BiasPtr,
                                   fc2WeightsPtr,
                                   fc2ScalesPtr,
                                   fc2BiasPtr,
                                   fc3WeightsPtr,
                                   fc3ScalesPtr,
                                   fc3BiasPtr,
                                   fc1ZpPtr,
                                   fc2ZpPtr,
                                   fc3ZpPtr,
                                   outputPtr,
                                   numTokensVal,
                                   hiddenSizeVal,
                                   interSizeVal,
                                   numExpertsVal,
                                   kVal,
                                   expertWeightBitsVal,
                                   blockSizeVal,
                                   swigluFusionVal,
                                   activationTypeVal,
                                   activationAlphaVal,
                                   activationBetaVal,
                                   swigluLimitVal,
                                   normalizeVal,
                                   elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQMoELoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<QMoEOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
