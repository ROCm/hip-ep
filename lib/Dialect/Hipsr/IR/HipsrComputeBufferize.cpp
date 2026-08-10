/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Bufferization of hipsr.compute and hipsr.compute_yield. HipsrBufferize.h
// declares the two models and explains the three-layer aliasing they set up;
// the bodies live here so that header stays a registration header.
//

#include "hip/Dialect/Hipsr/IR/HipsrBufferize.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

// SingleBlockImplicitTerminator pins the body's terminator to this op.
ComputeYieldOp getYieldOp(ComputeOp computeOp) {
  return cast<ComputeYieldOp>(computeOp.getBody().front().getTerminator());
}

// The entry block argument the operand at `operandIndex` is forwarded to. ctx
// takes index 0 of both lists, so operands and block arguments line up without
// an offset.
BlockArgument getEntryArgument(ComputeOp computeOp, unsigned operandIndex) {
  return computeOp.getBody().front().getArgument(operandIndex);
}

} // namespace

//===----------------------------------------------------------------------===//
// ComputeOpBufferization
//===----------------------------------------------------------------------===//

bool ComputeOpBufferization::bufferizesToMemoryRead(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &state) const {
  return state.isValueRead(
      getEntryArgument(cast<ComputeOp>(op), opOperand.getOperandNumber()));
}

bool ComputeOpBufferization::bufferizesToMemoryWrite(
    Operation *, OpOperand &opOperand,
    const bufferization::AnalysisState &) const {
  return isHipsrDestinationOperand(opOperand);
}

bool ComputeOpBufferization::isWritable(
    Operation *op, Value value, const bufferization::AnalysisState &) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return isHipsrDestinationOperand(op->getOpOperand(blockArg.getArgNumber()));
  return true;
}

bufferization::AliasingValueList ComputeOpBufferization::getAliasingValues(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &) const {
  return {{getEntryArgument(cast<ComputeOp>(op), opOperand.getOperandNumber()),
           bufferization::BufferRelation::Equivalent}};
}

bufferization::AliasingOpOperandList
ComputeOpBufferization::getAliasingOpOperands(
    Operation *op, Value value, const bufferization::AnalysisState &) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return {{&op->getOpOperand(blockArg.getArgNumber()),
             bufferization::BufferRelation::Equivalent}};

  auto opResult = cast<OpResult>(value);
  return {{&getYieldOp(cast<ComputeOp>(op))
                ->getOpOperand(opResult.getResultNumber()),
           bufferization::BufferRelation::Equivalent}};
}

FailureOr<bufferization::BufferLikeType> ComputeOpBufferization::getBufferType(
    Operation *op, Value value,
    const bufferization::BufferizationOptions &options,
    const bufferization::BufferizationState &state,
    SmallVector<Value> &invocationStack) const {
  // block argument's buffer is the operand's buffer.
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return bufferization::getBufferType(op->getOperand(blockArg.getArgNumber()),
                                        options, state, invocationStack);

  // A result's buffer is the yielded value's, which the default reaches through
  // the equivalent alias getAliasingOpOperands reports.
  return bufferization::detail::defaultGetBufferType(value, options, state,
                                                     invocationStack);
}

LogicalResult ComputeOpBufferization::bufferize(
    Operation *op, RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) const {
  auto computeOp = cast<ComputeOp>(op);
  Block *oldBody = &computeOp.getBody().front();

  Value ctx = computeOp.getCtx();

  SmallVector<Value> bufferizedInputs;
  for (Value input : computeOp.getInputs()) {
    if (isa<TensorType>(input.getType())) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, input, options, state);
      if (failed(buffer))
        return failure();
      bufferizedInputs.push_back(*buffer);
    } else {
      bufferizedInputs.push_back(input);
    }
  }

  SmallVector<Value> bufferizedOutputs;
  for (Value output : computeOp.getOutputs()) {
    if (isa<TensorType>(output.getType())) {
      FailureOr<Value> buffer =
          bufferization::getBuffer(rewriter, output, options, state);
      if (failed(buffer))
        return failure();
      bufferizedOutputs.push_back(*buffer);
    } else {
      bufferizedOutputs.push_back(output);
    }
  }

  // Create new op with memref operands
  auto newOp = ComputeOp::create(
      rewriter, op->getLoc(), TypeRange(getYieldOp(computeOp)->getOperands()),
      ctx, bufferizedInputs, bufferizedOutputs);

  // ctx + ins + outs, aligned with bb0 block arguments.
  SmallVector<Value> newOperands;
  newOperands.push_back(ctx);
  newOperands.append(bufferizedInputs.begin(), bufferizedInputs.end());
  newOperands.append(bufferizedOutputs.begin(), bufferizedOutputs.end());

  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  for (BlockArgument oldArgument : oldBody->getArguments()) {
    argumentTypes.push_back(newOperands[oldArgument.getArgNumber()].getType());
    argumentLocs.push_back(oldArgument.getLoc());
  }
  Block *newBody = rewriter.createBlock(&newOp.getBody(), newOp.getBody().end(),
                                        argumentTypes, argumentLocs);

  rewriter.setInsertionPointToStart(newBody);
  SmallVector<Value> argumentReplacements;
  for (BlockArgument oldArgument : oldBody->getArguments()) {
    Value newArgument = newBody->getArgument(oldArgument.getArgNumber());
    if (!isa<TensorType>(oldArgument.getType())) {
      argumentReplacements.push_back(newArgument);
      continue;
    }
    argumentReplacements.push_back(
        bufferization::ToTensorOp::create(rewriter, oldArgument.getLoc(),
                                          oldArgument.getType(), newArgument)
            .getResult());
  }
  rewriter.mergeBlocks(oldBody, newBody, argumentReplacements);

  bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                               newOp->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ComputeYieldOpBufferization
//===----------------------------------------------------------------------===//

bufferization::AliasingValueList
ComputeYieldOpBufferization::getAliasingValues(
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &) const {
  // Layer 3: a yielded value and the result in the same position are one
  // buffer. The RegionBranchOpInterface exit edge already checks that the two
  // lists have the same length.
  return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
           bufferization::BufferRelation::Equivalent}};
}

LogicalResult ComputeYieldOpBufferization::bufferize(
    Operation *op, RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options,
    bufferization::BufferizationState &state) const {
  // Nothing to reconcile on the way out: the parent's result type is defined as
  // the type of the buffer yielded here.
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    if (!isa<TensorType>(operand.getType())) {
      newOperands.push_back(operand);
      continue;
    }
    FailureOr<Value> buffer =
        bufferization::getBuffer(rewriter, operand, options, state);
    if (failed(buffer))
      return failure();
    newOperands.push_back(*buffer);
  }

  bufferization::replaceOpWithNewBufferizedOp<ComputeYieldOp>(rewriter, op,
                                                              newOperands);
  return success();
}
