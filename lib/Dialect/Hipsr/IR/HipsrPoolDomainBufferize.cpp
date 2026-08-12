/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrBufferize.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

PoolDomainYieldOp getYieldOp(PoolDomainOp domainOp) {
  return cast<PoolDomainYieldOp>(domainOp.getBody().front().getTerminator());
}

BlockArgument getEntryArgument(PoolDomainOp domainOp, unsigned operandIndex) {
  return domainOp.getBody().front().getArgument(operandIndex);
}

// Rewrite `to_buffer(bridge)` into the buffer the bridge wraps, then drop the
// bridges left without users. A bridge stays when its user asked for a
// different buffer type, since removing it there would need a cast.
void foldBridges(RewriterBase &rewriter,
                 ArrayRef<bufferization::ToTensorOp> bridges) {
  for (bufferization::ToTensorOp bridge : bridges) {
    Value buffer = bridge.getBuffer();
    SmallVector<bufferization::ToBufferOp> redundantOps;
    for (Operation *user : bridge.getResult().getUsers()) {
      auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(user);
      if (toBufferOp && toBufferOp.getResult().getType() == buffer.getType())
        redundantOps.push_back(toBufferOp);
    }
    for (bufferization::ToBufferOp toBufferOp : redundantOps)
      rewriter.replaceOp(toBufferOp, buffer);
    if (bridge.getResult().use_empty())
      rewriter.eraseOp(bridge);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// PoolDomainOpBufferization
//===----------------------------------------------------------------------===//

// Only the body will reads the entry block argument if it is a tensor.
bool PoolDomainOpBufferization::bufferizesToMemoryRead(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &state) const {
  BlockArgument argument =
      getEntryArgument(cast<PoolDomainOp>(op), opOperand.getOperandNumber());
  if (!isa<TensorType>(argument.getType()))
    return false;
  return state.isValueRead(argument);
}

bool PoolDomainOpBufferization::bufferizesToMemoryWrite(
    Operation *, OpOperand &, const bufferization::AnalysisState &) const {
  return false;
}

bool PoolDomainOpBufferization::isWritable(
    Operation *, Value value, const bufferization::AnalysisState &) const {
  return !isa<BlockArgument>(value);
}

bufferization::AliasingValueList PoolDomainOpBufferization::getAliasingValues(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &) const {
  return {
      {getEntryArgument(cast<PoolDomainOp>(op), opOperand.getOperandNumber()),
       bufferization::BufferRelation::Equivalent}};
}

bufferization::AliasingOpOperandList
PoolDomainOpBufferization::getAliasingOpOperands(
    Operation *op, Value value, const bufferization::AnalysisState &) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return {{&op->getOpOperand(blockArg.getArgNumber()),
             bufferization::BufferRelation::Equivalent}};

  auto opResult = cast<OpResult>(value);
  return {{&getYieldOp(cast<PoolDomainOp>(op))
                ->getOpOperand(opResult.getResultNumber()),
           bufferization::BufferRelation::Equivalent}};
}

FailureOr<bufferization::BufferLikeType>
PoolDomainOpBufferization::getBufferType(
    Operation *op, Value value,
    const bufferization::BufferizationOptions &options,
    const bufferization::BufferizationState &state,
    SmallVector<Value> &invocationStack) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return bufferization::getBufferType(op->getOperand(blockArg.getArgNumber()),
                                        options, state, invocationStack);

  // A result's buffer is the yielded value's, which the default reaches through
  // the equivalent alias getAliasingOpOperands reports.
  return bufferization::detail::defaultGetBufferType(value, options, state,
                                                     invocationStack);
}

LogicalResult PoolDomainOpBufferization::bufferize(
    Operation *op, RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) const {
  auto domainOp = cast<PoolDomainOp>(op);
  Block *oldBody = &domainOp.getBody().front();

  SmallVector<Value> bufferizedOperands;
  for (Value operand : domainOp.getOperands()) {
    if (isa<TensorType>(operand.getType())) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, operand, options, state);
      if (failed(buffer))
        return failure();
      bufferizedOperands.push_back(*buffer);
    } else {
      bufferizedOperands.push_back(operand);
    }
  }

  auto newOp = PoolDomainOp::create(
      rewriter, op->getLoc(), TypeRange(getYieldOp(domainOp)->getOperands()),
      bufferizedOperands, domainOp.getDomainId());

  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  for (BlockArgument oldArgument : oldBody->getArguments()) {
    argumentTypes.push_back(
        bufferizedOperands[oldArgument.getArgNumber()].getType());
    argumentLocs.push_back(oldArgument.getLoc());
  }
  Block *newBody = rewriter.createBlock(&newOp.getBody(), newOp.getBody().end(),
                                        argumentTypes, argumentLocs);

  rewriter.setInsertionPointToStart(newBody);
  SmallVector<Value> argumentReplacements;
  SmallVector<bufferization::ToTensorOp> bridges;
  for (BlockArgument oldArgument : oldBody->getArguments()) {
    Value newArgument = newBody->getArgument(oldArgument.getArgNumber());
    if (!isa<TensorType>(oldArgument.getType())) {
      argumentReplacements.push_back(newArgument);
      continue;
    }
    auto bridge = bufferization::ToTensorOp::create(
        rewriter, oldArgument.getLoc(), oldArgument.getType(), newArgument);
    bridges.push_back(bridge);
    argumentReplacements.push_back(bridge.getResult());
  }
  rewriter.mergeBlocks(oldBody, newBody, argumentReplacements);
  foldBridges(rewriter, bridges);

  bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                               newOp->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// PoolDomainYieldOpBufferization
//===----------------------------------------------------------------------===//

bufferization::AliasingValueList
PoolDomainYieldOpBufferization::getAliasingValues(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &) const {
  return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
           bufferization::BufferRelation::Equivalent}};
}

LogicalResult PoolDomainYieldOpBufferization::bufferize(
    Operation *op, RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) const {
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    if (isa<TensorType>(operand.getType())) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, operand, options, state);
      if (failed(buffer))
        return failure();
      newOperands.push_back(*buffer);
    } else {
      newOperands.push_back(operand);
    }
  }

  bufferization::replaceOpWithNewBufferizedOp<PoolDomainYieldOp>(rewriter, op,
                                                                 newOperands);
  return success();
}
