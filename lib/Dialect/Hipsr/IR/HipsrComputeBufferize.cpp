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
#include "llvm/ADT/DenseSet.h"
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

OpResult getResultHeldIn(ComputeOp computeOp, OpOperand &opOperand) {
  OperandRange outputs = computeOp.getOutputs();
  if (outputs.empty())
    return {};

  unsigned begin = outputs.getBeginOperandIndex();
  unsigned number = opOperand.getOperandNumber();
  if (number < begin || number - begin >= computeOp->getNumResults())
    return {};
  return computeOp->getResult(number - begin);
}

OpOperand *getDestinationOf(ComputeOp computeOp, unsigned resultIndex) {
  OperandRange outputs = computeOp.getOutputs();
  if (resultIndex >= outputs.size())
    return nullptr;
  return &computeOp->getOpOperand(outputs.getBeginOperandIndex() + resultIndex);
}

bool isValueWritten(Value value, const bufferization::AnalysisState &state) {
  SmallVector<OpOperand *> worklist;
  DenseSet<OpOperand *> visited;
  for (OpOperand &use : value.getUses())
    worklist.push_back(&use);

  while (!worklist.empty()) {
    OpOperand *use = worklist.pop_back_val();
    if (!visited.insert(use).second)
      continue;
    if (state.bufferizesToMemoryWrite(*use))
      return true;
    if (state.bufferizesToAliasOnly(*use)) {
      for (const bufferization::AliasingValue &alias :
           state.getAliasingValues(*use)) {
        for (OpOperand &aliasUse : alias.value.getUses())
          worklist.push_back(&aliasUse);
      }
    }
  }
  return false;
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
    Operation *op, OpOperand &opOperand,
    const bufferization::AnalysisState &state) const {
  if (!isHipsrDestinationOperand(opOperand))
    return false;
  return isValueWritten(
      getEntryArgument(cast<ComputeOp>(op), opOperand.getOperandNumber()),
      state);
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
  auto computeOp = cast<ComputeOp>(op);

  bufferization::AliasingValueList aliases;
  aliases.addAlias({getEntryArgument(computeOp, opOperand.getOperandNumber()),
                    bufferization::BufferRelation::Equivalent});
 
  if (OpResult result = getResultHeldIn(computeOp, opOperand))
    aliases.addAlias({result, bufferization::BufferRelation::Equivalent});
  return aliases;
}

bufferization::AliasingOpOperandList
ComputeOpBufferization::getAliasingOpOperands(
    Operation *op, Value value, const bufferization::AnalysisState &) const {
  auto computeOp = cast<ComputeOp>(op);
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return {{&op->getOpOperand(blockArg.getArgNumber()),
             bufferization::BufferRelation::Equivalent}};

  unsigned resultIndex = cast<OpResult>(value).getResultNumber();
  bufferization::AliasingOpOperandList aliases;
  aliases.addAlias({&getYieldOp(computeOp)->getOpOperand(resultIndex),
                    bufferization::BufferRelation::Equivalent});
  if (OpOperand *destination = getDestinationOf(computeOp, resultIndex))
    aliases.addAlias({destination, bufferization::BufferRelation::Equivalent});
  return aliases;
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

  // The outs operand supplies the memory a result is held in, the yield the
  // type that memory is viewed through, so the type comes from the yield. The
  // default would pick whichever equivalent alias comes first instead.
  unsigned resultIndex = cast<OpResult>(value).getResultNumber();
  return bufferization::getBufferType(
      getYieldOp(cast<ComputeOp>(op))->getOperand(resultIndex), options, state,
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
    if (isa<TensorType>(oldArgument.getType())) {
      argumentReplacements.push_back(
        bufferization::ToTensorOp::create(rewriter, oldArgument.getLoc(),
                                          oldArgument.getType(), newArgument)
            .getResult());
    }else{
      argumentReplacements.push_back(newArgument);
    }

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
  return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
           bufferization::BufferRelation::Equivalent}};
}

LogicalResult ComputeYieldOpBufferization::bufferize(
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
    }else{
      newOperands.push_back(operand);
    }
  }

  bufferization::replaceOpWithNewBufferizedOp<ComputeYieldOp>(rewriter, op,
                                                              newOperands);
  return success();
}
