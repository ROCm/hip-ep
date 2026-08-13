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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::hipsr;

namespace mlir {
namespace hipsr {
namespace {

template <typename OpTy>
BlockArgument getEntryArgument(OpTy op, unsigned operandIndex) {
  return op.getBody().front().getArgument(operandIndex);
}

//===----------------------------------------------------------------------===//
// PreserveShapeOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ComputeOp / ComputeYieldOp
//===----------------------------------------------------------------------===//

ComputeYieldOp getYieldOp(ComputeOp computeOp) {
  return cast<ComputeYieldOp>(computeOp.getBody().front().getTerminator());
}

// get result by output operand
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

// get output operand by result index
OpOperand *getDestinationOf(ComputeOp computeOp, unsigned resultIndex) {
  OperandRange outputs = computeOp.getOutputs();
  if (resultIndex >= outputs.size())
    return nullptr;
  return &computeOp->getOpOperand(outputs.getBeginOperandIndex() + resultIndex);
}

// Returns true if the buffer underlying `value` or any of its aliases is written.
bool isValueWritten(Value value, const AnalysisState &state) {
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
      for (const AliasingValue &alias : state.getAliasingValues(*use)) {
        for (OpOperand &aliasUse : alias.value.getUses())
          worklist.push_back(&aliasUse);
      }
    }
  }
  return false;
}

struct ComputeOpBufferization
    : public BufferizableOpInterface::ExternalModel<ComputeOpBufferization,
                                                    ComputeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return state.isValueRead(
        getEntryArgument(cast<ComputeOp>(op), opOperand.getOperandNumber()));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    if (!isHipsrDestinationOperand(opOperand))
      return false;
    return isValueWritten(
        getEntryArgument(cast<ComputeOp>(op), opOperand.getOperandNumber()),
        state);
  }

  bool isWritable(Operation *op, Value value, const AnalysisState &) const {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return isHipsrDestinationOperand(
          op->getOpOperand(blockArg.getArgNumber()));
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    auto computeOp = cast<ComputeOp>(op);

    AliasingValueList aliases;
    aliases.addAlias({getEntryArgument(computeOp, opOperand.getOperandNumber()),
                      BufferRelation::Equivalent});

    if (OpResult result = getResultHeldIn(computeOp, opOperand))
      aliases.addAlias({result, BufferRelation::Equivalent});
    return aliases;
  }

  AliasingOpOperandList getAliasingOpOperands(Operation *op, Value value,
                                              const AnalysisState &) const {
    // block argument <-> input operand
    auto computeOp = cast<ComputeOp>(op);
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return {{&op->getOpOperand(blockArg.getArgNumber()),
               BufferRelation::Equivalent}};

    // op result <-> yield op result
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    AliasingOpOperandList aliases;
    aliases.addAlias({&getYieldOp(computeOp)->getOpOperand(resultIndex),
                      BufferRelation::Equivalent});

    // op result <-> output operand
    if (OpOperand *destination = getDestinationOf(computeOp, resultIndex))
      aliases.addAlias({destination, BufferRelation::Equivalent});
    return aliases;
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    // A block argument's buffer is the operand's buffer.
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return bufferization::getBufferType(
          op->getOperand(blockArg.getArgNumber()), options, state,
          invocationStack);

    // The outs operand supplies the memory a result is held in, the yield the
    // type that memory is viewed through, so the type comes from the yield. The
    // default would pick whichever equivalent alias comes first instead.
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    return bufferization::getBufferType(
        getYieldOp(cast<ComputeOp>(op))->getOperand(resultIndex), options,
        state, invocationStack);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto computeOp = cast<ComputeOp>(op);
    Block *oldBody = &computeOp.getBody().front();

    Value ctx = computeOp.getCtx();

    SmallVector<Value> bufferizedInputs;
    for (Value input : computeOp.getInputs()) {
      if (isa<TensorType>(input.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, input, options, state);
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
        FailureOr<Value> buffer = getBuffer(rewriter, output, options, state);
        if (failed(buffer))
          return failure();
        bufferizedOutputs.push_back(*buffer);
      } else {
        bufferizedOutputs.push_back(output);
      }
    }

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
      argumentTypes.push_back(
          newOperands[oldArgument.getArgNumber()].getType());
      argumentLocs.push_back(oldArgument.getLoc());
    }
    Block *newBody = rewriter.createBlock(
        &newOp.getBody(), newOp.getBody().end(), argumentTypes, argumentLocs);

    rewriter.setInsertionPointToStart(newBody);
    SmallVector<Value> argumentReplacements;
    for (BlockArgument oldArgument : oldBody->getArguments()) {
      Value newArgument = newBody->getArgument(oldArgument.getArgNumber());
      if (isa<TensorType>(oldArgument.getType())) {
        argumentReplacements.push_back(
            ToTensorOp::create(rewriter, oldArgument.getLoc(),
                               oldArgument.getType(), newArgument)
                .getResult());
      } else {
        argumentReplacements.push_back(newArgument);
      }
    }
    rewriter.mergeBlocks(oldBody, newBody, argumentReplacements);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }
};

struct ComputeYieldOpBufferization
    : public BufferizableOpInterface::ExternalModel<ComputeYieldOpBufferization,
                                                    ComputeYieldOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  // A result's buffer is the buffer of the operand yielded here
  // so this operand must be bufferized in-place.
  bool mustBufferizeInPlace(Operation *, OpOperand &,
                            const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
        if (failed(buffer))
          return failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(operand);
      }
    }

    replaceOpWithNewBufferizedOp<ComputeYieldOp>(rewriter, op, newOperands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PoolDomainOp / PoolDomainYieldOp
//===----------------------------------------------------------------------===//

PoolDomainYieldOp getYieldOp(PoolDomainOp domainOp) {
  return cast<PoolDomainYieldOp>(domainOp.getBody().front().getTerminator());
}

// Rewrite `to_buffer(bridge)` into the buffer the bridge wraps, then drop the
// bridges left without users. A bridge stays when its user asked for a
// different buffer type, since removing it there would need a cast.
void foldBridges(RewriterBase &rewriter, ArrayRef<ToTensorOp> bridges) {
  for (ToTensorOp bridge : bridges) {
    Value buffer = bridge.getBuffer();
    SmallVector<ToBufferOp> redundantOps;
    for (Operation *user : bridge.getResult().getUsers()) {
      auto toBufferOp = dyn_cast<ToBufferOp>(user);
      if (toBufferOp && toBufferOp.getResult().getType() == buffer.getType())
        redundantOps.push_back(toBufferOp);
    }
    for (ToBufferOp toBufferOp : redundantOps)
      rewriter.replaceOp(toBufferOp, buffer);
    if (bridge.getResult().use_empty())
      rewriter.eraseOp(bridge);
  }
}

struct PoolDomainOpBufferization
    : public BufferizableOpInterface::ExternalModel<PoolDomainOpBufferization,
                                                    PoolDomainOp> {
  // Only the body reads the entry block argument, and only if it is a tensor.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    BlockArgument argument =
        getEntryArgument(cast<PoolDomainOp>(op), opOperand.getOperandNumber());
    if (!isa<TensorType>(argument.getType()))
      return false;
    return state.isValueRead(argument);
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  bool isWritable(Operation *, Value value, const AnalysisState &) const {
    return !isa<BlockArgument>(value);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    return {
        {getEntryArgument(cast<PoolDomainOp>(op), opOperand.getOperandNumber()),
         BufferRelation::Equivalent}};
  }

  AliasingOpOperandList getAliasingOpOperands(Operation *op, Value value,
                                              const AnalysisState &) const {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return {{&op->getOpOperand(blockArg.getArgNumber()),
               BufferRelation::Equivalent}};

    auto opResult = cast<OpResult>(value);
    return {{&getYieldOp(cast<PoolDomainOp>(op))
                  ->getOpOperand(opResult.getResultNumber()),
             BufferRelation::Equivalent}};
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return bufferization::getBufferType(
          op->getOperand(blockArg.getArgNumber()), options, state,
          invocationStack);

    // A result's buffer is the yielded value's, which the default reaches
    // through the equivalent alias getAliasingOpOperands reports.
    return bufferization::detail::defaultGetBufferType(value, options, state,
                                                       invocationStack);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto domainOp = cast<PoolDomainOp>(op);
    Block *oldBody = &domainOp.getBody().front();

    SmallVector<Value> bufferizedOperands;
    for (Value operand : domainOp.getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
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
    Block *newBody = rewriter.createBlock(
        &newOp.getBody(), newOp.getBody().end(), argumentTypes, argumentLocs);

    rewriter.setInsertionPointToStart(newBody);
    SmallVector<Value> argumentReplacements;
    for (BlockArgument oldArgument : oldBody->getArguments()) {
      Value newArgument = newBody->getArgument(oldArgument.getArgNumber());
      if (isa<TensorType>(oldArgument.getType())) {
        argumentReplacements.push_back(
            ToTensorOp::create(rewriter, oldArgument.getLoc(),
                                oldArgument.getType(), newArgument)
                .getResult());
      } else {
        argumentReplacements.push_back(newArgument);
      }
    }
    rewriter.mergeBlocks(oldBody, newBody, argumentReplacements);

    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }
};

struct PoolDomainYieldOpBufferization
    : public BufferizableOpInterface::ExternalModel<
          PoolDomainYieldOpBufferization, PoolDomainYieldOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  // A result's buffer is the buffer of the operand yielded here, so yielding
  // out of place would make that result a buffer allocated inside the body.
  bool mustBufferizeInPlace(Operation *, OpOperand &,
                            const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
        if (failed(buffer))
          return failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(operand);
      }
    }

    replaceOpWithNewBufferizedOp<PoolDomainYieldOp>(rewriter, op, newOperands);
    return success();
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir

void mlir::hipsr::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipsrDialect *) {
    PreserveShapeOp::attachInterface<PreserveShapeBufferizableModel>(*ctx);
    ComputeOp::attachInterface<ComputeOpBufferization>(*ctx);
    ComputeYieldOp::attachInterface<ComputeYieldOpBufferization>(*ctx);
    PoolDomainOp::attachInterface<PoolDomainOpBufferization>(*ctx);
    PoolDomainYieldOp::attachInterface<PoolDomainYieldOpBufferization>(*ctx);
  });
}
