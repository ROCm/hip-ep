/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- BufferizableOpInterfaceImpl.cpp - hipsr bufferization models -------===//
//
// BufferizableOpInterface models for the hipsr dialect. A new model belongs in
// the anonymous namespace below and is attached in the single registration
// function at the bottom.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/BufferizableOpInterfaceImpl.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::hipsr;

namespace mlir {
namespace hipsr {
namespace {

// hipsr.preserve_shape only records that `$shape` describes `$data`. It has no
// results and no init operand, so it is not DPS and cannot reuse
// DstBufferizableOpInterfaceExternalModel. It touches no memory, so nothing
// needs copying and a later DPS op can write the buffer in place.
struct PreserveShapeBufferizableModel
    : public BufferizableOpInterface::ExternalModel<
          PreserveShapeBufferizableModel, PreserveShapeOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto preserveOp = cast<PreserveShapeOp>(op);
    FailureOr<Value> dataBuf =
        getBuffer(rewriter, preserveOp.getData(), options, state);
    if (failed(dataBuf)) {
      return failure();
    }
    replaceOpWithNewBufferizedOp<PreserveShapeOp>(
        rewriter, op, preserveOp.getShape(), *dataBuf);
    return success();
  }
};

// Bufferizes any hipsr destination-passing op, ported from upstream linalg's
// bufferizeDestinationStyleOpInterface. A hipsr DPS op carries no region, so
// the bufferized op is built in one step.
LogicalResult bufferizeDpsOp(RewriterBase &rewriter,
                             DestinationStyleOpInterface op,
                             const BufferizationOptions &options,
                             const BufferizationState &state) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  if (op.hasPureBufferSemantics()) {
    return success();
  }
  if (!op.hasPureTensorSemantics()) {
    return op->emitOpError("does not have pure tensor semantics");
  }

  // A DPS scalar has no buffer, so it is forwarded as it is. `!hipsr.context`
  // is the one every hipsr DPS op carries.
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (OpOperand *input : op.getDpsInputOperands()) {
    if (op.isScalar(input)) {
      newOperands.push_back(input->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, input->get(), options, state);
    if (failed(buffer)) {
      return failure();
    }
    newOperands.push_back(*buffer);
  }

  SmallVector<Value> initBuffers;
  for (OpResult result : op->getOpResults()) {
    OpOperand *init = op.getDpsInitOperand(result.getResultNumber());
    FailureOr<Value> buffer = getBuffer(rewriter, init->get(), options, state);
    if (failed(buffer)) {
      return failure();
    }
    initBuffers.push_back(*buffer);
  }
  llvm::append_range(newOperands, initBuffers);

  // Reading the buffers above may have inserted allocations.
  rewriter.setInsertionPoint(op);

  // Every Hipsr_DpsOp result is a tensor tied to an init, so the bufferized op
  // writes through its init buffers and keeps no result. Each tensor result is
  // therefore replaced by the buffer of its init instead of by a result of the
  // new op, which is what rules out replaceOpWithNewBufferizedOp here.
  OperationState newState(op->getLoc(), op->getName(), newOperands, TypeRange{},
                          op->getAttrs());
  rewriter.create(newState);
  replaceOpWithBufferizedValues(rewriter, op, initBuffers);
  return success();
}

// Binds the shared destination-passing rewrite to one op, the way upstream's
// LinalgOpInterface does: the DPS base model answers the analysis queries and
// bufferize() delegates to bufferizeDpsOp.
template <typename OpTy>
struct DpsBufferizableModel
    : public DstBufferizableOpInterfaceExternalModel<DpsBufferizableModel<OpTy>,
                                                     OpTy> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    return bufferizeDpsOp(rewriter, cast<DestinationStyleOpInterface>(op),
                          options, state);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir

void mlir::hipsr::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipsrDialect *) {
    PreserveShapeOp::attachInterface<PreserveShapeBufferizableModel>(*ctx);
    CastOp::attachInterface<DpsBufferizableModel<CastOp>>(*ctx);
  });
}
