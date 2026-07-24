/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_BUFFERIZE_H
#define HIP_BUFFERIZE_H

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/Sequence.h"

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

// Bufferization model for hip.transfer_to_host: a tensor->tensor op that copies
// `src` (a device tensor) into HOST memory. It bufferizes to a host STACK
// `memref.alloca` (space-less, since alloca must live in the default LLVM stack
// address space) relabeled to `#hip.mem<host>`, then an async D2H copy + a
// stream sync (the host reads the result next). Stack-only, so use it for SMALL
// data (shape/param vectors, scalars); the alloca frees itself at function
// scope, needs no dealloc, and is skipped by hip-pool-allocs and
// hip-materialize-host-scalars.
//
//   Before:  %h = hip.transfer_to_host(%ctx, %pads : tensor<8xi64>)
//                   -> tensor<8xi64>
//   After:   %slot = memref.alloca() : memref<8xi64>
//            %dst  = memref.memory_space_cast %slot
//                      : memref<8xi64> to memref<8xi64, #hip.mem<host>>
//            hip.memcpy_d2h_async(%ctx, %dst, %srcBuf)
//            hip.stream_sync(%ctx)            // uses of %h replaced by %dst
struct HipTransferToHostBufferizableModel
    : public bufferization::BufferizableOpInterface::ExternalModel<
          HipTransferToHostBufferizableModel, TransferToHostOp> {

  bool bufferizesToAllocation(Operation *, Value) const { return true; }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &) const {
    // Only the `src` tensor operand is read; `ctx` is not a tensor.
    return opOperand.get() == cast<TransferToHostOp>(op).getSrc();
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    // Result is a fresh allocation, never an alias of an operand.
    return {};
  }

  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value /*value*/,
                const bufferization::BufferizationOptions & /*options*/,
                const bufferization::BufferizationState & /*state*/,
                SmallVector<Value> & /*invocationStack*/) const {
    auto transfer = cast<TransferToHostOp>(op);
    auto tensorTy = cast<TensorType>(transfer.getResult().getType());
    // Result buffer is HOST memory with a static identity layout (so the pad
    // lowering sees a dense buffer). The op is host-only, so the space is fixed
    // here, not read from an attribute.
    auto hostSpace =
        MemorySpaceAttr::get(op->getContext(), MemorySpaceKind::Host);
    BaseMemRefType bt = bufferization::getMemRefTypeWithStaticIdentityLayout(
        tensorTy, hostSpace);
    return cast<bufferization::BufferLikeType>(bt);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto transfer = cast<TransferToHostOp>(op);
    Location loc = op->getLoc();

    FailureOr<Value> srcBuf =
        getBuffer(rewriter, transfer.getSrc(), options, state);
    if (failed(srcBuf))
      return failure();

    // Get the destination buffer type from the framework (it calls our
    // getBufferType() hook above: host space, static identity layout).
    // Computing it again here could drift from that declared type.
    FailureOr<bufferization::BufferLikeType> dstTyOr =
        bufferization::getBufferType(transfer.getResult(), options, state);
    if (failed(dstTyOr))
      return failure();
    auto dstTy = cast<MemRefType>(*dstTyOr);

    // Dynamic dst dims mirror the src buffer's dims.
    SmallVector<Value> dynDims;
    for (int64_t i : llvm::seq<int64_t>(dstTy.getRank()))
      if (dstTy.isDynamicDim(i))
        dynDims.push_back(memref::DimOp::create(rewriter, loc, *srcBuf, i));

    // Host dst is ALWAYS a stack alloca (space-less, since alloca must live in
    // the default LLVM stack AS) lifted to #hip.mem<host>. See header comment.
    auto slotTy = MemRefType::get(dstTy.getShape(), dstTy.getElementType());
    Value slot = memref::AllocaOp::create(rewriter, loc, slotTy, dynDims);
    Value dst = memref::MemorySpaceCastOp::create(rewriter, loc, dstTy, slot);

    // Device -> host copy, then sync: the host reads `dst` next, so wait for
    // the async copy to complete before the op's uses observe the buffer.
    Value ctx = transfer.getCtx();
    MemcpyD2HAsyncOp::create(rewriter, loc, ctx, dst, *srcBuf);
    StreamSyncOp::create(rewriter, loc, ctx);

    bufferization::replaceOpWithBufferizedValues(rewriter, op, dst);
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
    TransferToHostOp::attachInterface<HipTransferToHostBufferizableModel>(*ctx);
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
    RangeOp::attachInterface<HipDstBufferizableModel<RangeOp>>(*ctx);
    SiluOp::attachInterface<HipDstBufferizableModel<SiluOp>>(*ctx);
    GqaOp::attachInterface<HipDstBufferizableModel<GqaOp>>(*ctx);
    CastOp::attachInterface<HipDstBufferizableModel<CastOp>>(*ctx);
    SigmoidOp::attachInterface<HipDstBufferizableModel<SigmoidOp>>(*ctx);
    TanhOp::attachInterface<HipDstBufferizableModel<TanhOp>>(*ctx);
    SoftplusOp::attachInterface<HipDstBufferizableModel<SoftplusOp>>(*ctx);
    GeluOp::attachInterface<HipDstBufferizableModel<GeluOp>>(*ctx);
    LeakyReluOp::attachInterface<HipDstBufferizableModel<LeakyReluOp>>(*ctx);
    ResizeOp::attachInterface<HipDstBufferizableModel<ResizeOp>>(*ctx);
    GlobalPoolOp::attachInterface<HipDstBufferizableModel<GlobalPoolOp>>(*ctx);
    ReciprocalOp::attachInterface<HipDstBufferizableModel<ReciprocalOp>>(*ctx);
    SqrtOp::attachInterface<HipDstBufferizableModel<SqrtOp>>(*ctx);
    PoolOp::attachInterface<HipDstBufferizableModel<PoolOp>>(*ctx);
    SubOp::attachInterface<HipDstBufferizableModel<SubOp>>(*ctx);
    ReduceSumOp::attachInterface<HipDstBufferizableModel<ReduceSumOp>>(*ctx);
    ReduceMaxOp::attachInterface<HipDstBufferizableModel<ReduceMaxOp>>(*ctx);
    ReduceMeanOp::attachInterface<HipDstBufferizableModel<ReduceMeanOp>>(*ctx);
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
    MinOp::attachInterface<HipDstBufferizableModel<MinOp>>(*ctx);
    MaxOp::attachInterface<HipDstBufferizableModel<MaxOp>>(*ctx);
    NegOp::attachInterface<HipDstBufferizableModel<NegOp>>(*ctx);
    EqualOp::attachInterface<HipDstBufferizableModel<EqualOp>>(*ctx);
    DivOp::attachInterface<HipDstBufferizableModel<DivOp>>(*ctx);
    NotOp::attachInterface<HipDstBufferizableModel<NotOp>>(*ctx);
    AndOp::attachInterface<HipDstBufferizableModel<AndOp>>(*ctx);
    CosOp::attachInterface<HipDstBufferizableModel<CosOp>>(*ctx);
    SinOp::attachInterface<HipDstBufferizableModel<SinOp>>(*ctx);
    ExpOp::attachInterface<HipDstBufferizableModel<ExpOp>>(*ctx);
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
