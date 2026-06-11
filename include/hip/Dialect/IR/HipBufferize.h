/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_BUFFERIZE_H
#define HIP_BUFFERIZE_H

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace hip {

template <typename OpTy>
struct HipDstBufferizableModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          HipDstBufferizableModel<OpTy>, OpTy> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);

    SmallVector<Value> newOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (isa<TensorType>(operand.get().getType())) {
        FailureOr<Value> buffer =
            getBuffer(rewriter, operand.get(), options, state);
        if (failed(buffer))
          return failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(operand.get());
      }
    }

    OperationState newState(op->getLoc(), op->getName().getStringRef());
    newState.addOperands(newOperands);
    newState.addAttributes(op->getAttrs());
    newState.propertiesAttr = op->getPropertiesAsAttribute();
    rewriter.create(newState);

    SmallVector<Value> replacements;
    for (OpResult result : op->getResults()) {
      if (!isa<TensorType>(result.getType())) {
        replacements.push_back(result);
        continue;
      }
      OpOperand *initOperand = dstOp.getTiedOpOperand(result);
      FailureOr<Value> initBuffer =
          getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(initBuffer))
        return failure();
      replacements.push_back(*initBuffer);
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

inline void
registerHipBufferizableOpInterfaceModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipDialect *) {
    ConvOp::attachInterface<HipDstBufferizableModel<ConvOp>>(*ctx);
    MatmulOp::attachInterface<HipDstBufferizableModel<MatmulOp>>(*ctx);
    RmsNormOp::attachInterface<HipDstBufferizableModel<RmsNormOp>>(*ctx);
    SkipRmsNormOp::attachInterface<HipDstBufferizableModel<SkipRmsNormOp>>(
        *ctx);
    RopeOp::attachInterface<HipDstBufferizableModel<RopeOp>>(*ctx);
    MiopenAddOp::attachInterface<HipDstBufferizableModel<MiopenAddOp>>(*ctx);
    AddOp::attachInterface<HipDstBufferizableModel<AddOp>>(*ctx);
    MulOp::attachInterface<HipDstBufferizableModel<MulOp>>(*ctx);
    MiopenSoftmaxOp::attachInterface<HipDstBufferizableModel<MiopenSoftmaxOp>>(
        *ctx);
    TransposeOp::attachInterface<HipDstBufferizableModel<TransposeOp>>(*ctx);
    GatherOp::attachInterface<HipDstBufferizableModel<GatherOp>>(*ctx);
    RangeOp::attachInterface<HipDstBufferizableModel<RangeOp>>(*ctx);
    SiluOp::attachInterface<HipDstBufferizableModel<SiluOp>>(*ctx);
    GqaOp::attachInterface<HipDstBufferizableModel<GqaOp>>(*ctx);
    CastOp::attachInterface<HipDstBufferizableModel<CastOp>>(*ctx);
    SigmoidOp::attachInterface<HipDstBufferizableModel<SigmoidOp>>(*ctx);
    SoftplusOp::attachInterface<HipDstBufferizableModel<SoftplusOp>>(*ctx);
    GeluOp::attachInterface<HipDstBufferizableModel<GeluOp>>(*ctx);
    ResizeOp::attachInterface<HipDstBufferizableModel<ResizeOp>>(*ctx);
    GlobalPoolOp::attachInterface<HipDstBufferizableModel<GlobalPoolOp>>(*ctx);
    ReciprocalOp::attachInterface<HipDstBufferizableModel<ReciprocalOp>>(*ctx);
    SqrtOp::attachInterface<HipDstBufferizableModel<SqrtOp>>(*ctx);
    PoolOp::attachInterface<HipDstBufferizableModel<PoolOp>>(*ctx);
    SubOp::attachInterface<HipDstBufferizableModel<SubOp>>(*ctx);
    ReduceSumOp::attachInterface<HipDstBufferizableModel<ReduceSumOp>>(*ctx);
    ReduceMaxOp::attachInterface<HipDstBufferizableModel<ReduceMaxOp>>(*ctx);
    MatMulNBitsOp::attachInterface<HipDstBufferizableModel<MatMulNBitsOp>>(
        *ctx);
    QMoEOp::attachInterface<HipDstBufferizableModel<QMoEOp>>(*ctx);
    CausalConvWithStateOp::attachInterface<
        HipDstBufferizableModel<CausalConvWithStateOp>>(*ctx);
    HipDNNGraphOp::attachInterface<HipDstBufferizableModel<HipDNNGraphOp>>(
        *ctx);
    GemmOp::attachInterface<HipDstBufferizableModel<GemmOp>>(*ctx);
    WhereOp::attachInterface<HipDstBufferizableModel<WhereOp>>(*ctx);
    LinearAttentionOp::attachInterface<
        HipDstBufferizableModel<LinearAttentionOp>>(*ctx);
    LayerNormOp::attachInterface<HipDstBufferizableModel<LayerNormOp>>(*ctx);
    MinOp::attachInterface<HipDstBufferizableModel<MinOp>>(*ctx);
    NegOp::attachInterface<HipDstBufferizableModel<NegOp>>(*ctx);
    EqualOp::attachInterface<HipDstBufferizableModel<EqualOp>>(*ctx);
    DivOp::attachInterface<HipDstBufferizableModel<DivOp>>(*ctx);
    NotOp::attachInterface<HipDstBufferizableModel<NotOp>>(*ctx);
    AndOp::attachInterface<HipDstBufferizableModel<AndOp>>(*ctx);
    CosOp::attachInterface<HipDstBufferizableModel<CosOp>>(*ctx);
    SinOp::attachInterface<HipDstBufferizableModel<SinOp>>(*ctx);
    CumSumOp::attachInterface<HipDstBufferizableModel<CumSumOp>>(*ctx);
    PadOp::attachInterface<HipDstBufferizableModel<PadOp>>(*ctx);
    TileOp::attachInterface<HipDstBufferizableModel<TileOp>>(*ctx);
    ExpandOp::attachInterface<HipDstBufferizableModel<ExpandOp>>(*ctx);
    ReduceProdOp::attachInterface<HipDstBufferizableModel<ReduceProdOp>>(*ctx);
    LessOp::attachInterface<HipDstBufferizableModel<LessOp>>(*ctx);
    GatherNDOp::attachInterface<HipDstBufferizableModel<GatherNDOp>>(*ctx);
    SignOp::attachInterface<HipDstBufferizableModel<SignOp>>(*ctx);
    ModOp::attachInterface<HipDstBufferizableModel<ModOp>>(*ctx);
    SliceOp::attachInterface<HipDstBufferizableModel<SliceOp>>(*ctx);
    ScatterNDOp::attachInterface<HipDstBufferizableModel<ScatterNDOp>>(*ctx);
    MultiHeadAttentionOp::attachInterface<
        HipDstBufferizableModel<MultiHeadAttentionOp>>(*ctx);
    NonZeroOp::attachInterface<HipDstBufferizableModel<NonZeroOp>>(*ctx);
    SizeOp::attachInterface<HipDstBufferizableModel<SizeOp>>(*ctx);
    LoopOp::attachInterface<HipDstBufferizableModel<LoopOp>>(*ctx);
  });
}

} // namespace hip
} // namespace mlir

#endif // HIP_BUFFERIZE_H
