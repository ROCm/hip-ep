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

// Bufferization model for the non-DPS readback ops (hip.readback_dim,
// hip.readback_scalar). Each reads its `scalar` tensor operand (a device
// buffer) and produces a NON-tensor result (index for readback_dim, the
// operand's element type for readback_scalar). So bufferization only rewrites
// the tensor operand to its memref buffer; the result type passes through
// unchanged and nothing aliases the operand. `op->getResult(0).getType()`
// already carries the right result type for either op.
template <typename OpTy>
struct HipReadbackBufferizableModel
    : public bufferization::BufferizableOpInterface::ExternalModel<
          HipReadbackBufferizableModel<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto readback = cast<OpTy>(op);
    FailureOr<Value> scalarBuf =
        getBuffer(rewriter, readback.getScalar(), options, state);
    if (failed(scalarBuf))
      return failure();
    auto newOp =
        OpTy::create(rewriter, op->getLoc(), op->getResult(0).getType(),
                     readback.getCtx(), *scalarBuf);
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 newOp.getResult());
    return success();
  }
};

inline void
registerHipBufferizableOpInterfaceModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipDialect *) {
    ReadbackDimOp::attachInterface<HipReadbackBufferizableModel<ReadbackDimOp>>(
        *ctx);
    ReadbackScalarOp::attachInterface<
        HipReadbackBufferizableModel<ReadbackScalarOp>>(*ctx);
    ConvOp::attachInterface<HipDstBufferizableModel<ConvOp>>(*ctx);
    ConvTransposeOp::attachInterface<HipDstBufferizableModel<ConvTransposeOp>>(
        *ctx);
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
    GatherElementsOp::attachInterface<
        HipDstBufferizableModel<GatherElementsOp>>(*ctx);
    TopKOp::attachInterface<HipDstBufferizableModel<TopKOp>>(*ctx);
    ScatterElementsOp::attachInterface<
        HipDstBufferizableModel<ScatterElementsOp>>(*ctx);
    CompressOp::attachInterface<HipDstBufferizableModel<CompressOp>>(*ctx);
    OneHotOp::attachInterface<HipDstBufferizableModel<OneHotOp>>(*ctx);
    RangeOp::attachInterface<HipDstBufferizableModel<RangeOp>>(*ctx);
    SiluOp::attachInterface<HipDstBufferizableModel<SiluOp>>(*ctx);
    GqaOp::attachInterface<HipDstBufferizableModel<GqaOp>>(*ctx);
    CastOp::attachInterface<HipDstBufferizableModel<CastOp>>(*ctx);
    SigmoidOp::attachInterface<HipDstBufferizableModel<SigmoidOp>>(*ctx);
    TanhOp::attachInterface<HipDstBufferizableModel<TanhOp>>(*ctx);
    SoftplusOp::attachInterface<HipDstBufferizableModel<SoftplusOp>>(*ctx);
    GeluOp::attachInterface<HipDstBufferizableModel<GeluOp>>(*ctx);
    BiasGeluOp::attachInterface<HipDstBufferizableModel<BiasGeluOp>>(*ctx);
    FastGeluOp::attachInterface<HipDstBufferizableModel<FastGeluOp>>(*ctx);
    LeakyReluOp::attachInterface<HipDstBufferizableModel<LeakyReluOp>>(*ctx);
    ResizeOp::attachInterface<HipDstBufferizableModel<ResizeOp>>(*ctx);
    GridSampleOp::attachInterface<HipDstBufferizableModel<GridSampleOp>>(*ctx);
    GlobalPoolOp::attachInterface<HipDstBufferizableModel<GlobalPoolOp>>(*ctx);
    ReciprocalOp::attachInterface<HipDstBufferizableModel<ReciprocalOp>>(*ctx);
    SqrtOp::attachInterface<HipDstBufferizableModel<SqrtOp>>(*ctx);
    PoolOp::attachInterface<HipDstBufferizableModel<PoolOp>>(*ctx);
    SubOp::attachInterface<HipDstBufferizableModel<SubOp>>(*ctx);
    ReduceSumOp::attachInterface<HipDstBufferizableModel<ReduceSumOp>>(*ctx);
    ReduceMaxOp::attachInterface<HipDstBufferizableModel<ReduceMaxOp>>(*ctx);
    ReduceMinOp::attachInterface<HipDstBufferizableModel<ReduceMinOp>>(*ctx);
    ReduceMeanOp::attachInterface<HipDstBufferizableModel<ReduceMeanOp>>(*ctx);
    ReduceL2Op::attachInterface<HipDstBufferizableModel<ReduceL2Op>>(*ctx);
    MatMulNBitsOp::attachInterface<HipDstBufferizableModel<MatMulNBitsOp>>(
        *ctx);
    QMoEOp::attachInterface<HipDstBufferizableModel<QMoEOp>>(*ctx);
    GatherBlockQuantizedOp::attachInterface<
        HipDstBufferizableModel<GatherBlockQuantizedOp>>(*ctx);
    CausalConvWithStateOp::attachInterface<
        HipDstBufferizableModel<CausalConvWithStateOp>>(*ctx);
    HipDNNGraphOp::attachInterface<HipDstBufferizableModel<HipDNNGraphOp>>(
        *ctx);
    GemmOp::attachInterface<HipDstBufferizableModel<GemmOp>>(*ctx);
    WhereOp::attachInterface<HipDstBufferizableModel<WhereOp>>(*ctx);
    LinearAttentionOp::attachInterface<
        HipDstBufferizableModel<LinearAttentionOp>>(*ctx);
    LayerNormOp::attachInterface<HipDstBufferizableModel<LayerNormOp>>(*ctx);
    InstanceNormOp::attachInterface<HipDstBufferizableModel<InstanceNormOp>>(
        *ctx);
    MinOp::attachInterface<HipDstBufferizableModel<MinOp>>(*ctx);
    MaxOp::attachInterface<HipDstBufferizableModel<MaxOp>>(*ctx);
    AbsOp::attachInterface<HipDstBufferizableModel<AbsOp>>(*ctx);
    NegOp::attachInterface<HipDstBufferizableModel<NegOp>>(*ctx);
    EqualOp::attachInterface<HipDstBufferizableModel<EqualOp>>(*ctx);
    DivOp::attachInterface<HipDstBufferizableModel<DivOp>>(*ctx);
    NotOp::attachInterface<HipDstBufferizableModel<NotOp>>(*ctx);
    OrOp::attachInterface<HipDstBufferizableModel<OrOp>>(*ctx);
    AndOp::attachInterface<HipDstBufferizableModel<AndOp>>(*ctx);
    CosOp::attachInterface<HipDstBufferizableModel<CosOp>>(*ctx);
    ErfOp::attachInterface<HipDstBufferizableModel<ErfOp>>(*ctx);
    SinOp::attachInterface<HipDstBufferizableModel<SinOp>>(*ctx);
    CeilOp::attachInterface<HipDstBufferizableModel<CeilOp>>(*ctx);
    RoundOp::attachInterface<HipDstBufferizableModel<RoundOp>>(*ctx);
    AtanOp::attachInterface<HipDstBufferizableModel<AtanOp>>(*ctx);
    FloorOp::attachInterface<HipDstBufferizableModel<FloorOp>>(*ctx);
    ExpOp::attachInterface<HipDstBufferizableModel<ExpOp>>(*ctx);
    LogOp::attachInterface<HipDstBufferizableModel<LogOp>>(*ctx);
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
    // hip.if is a DPS control-flow op (getDpsInitsMutable, results alias
    // o_init) just like hip.loop. Without this model one-shot-bufferize aborts
    // with "op was not bufferized: hip.if" for any graph containing onnx.If,
    // which is silent CPU fallback at the EP. The hip.if->llvm LIT test starts
    // from the post-bufferize memref form, so the gap is invisible there.
    IfOp::attachInterface<HipDstBufferizableModel<IfOp>>(*ctx);
  });
}

} // namespace hip
} // namespace mlir

#endif // HIP_BUFFERIZE_H
