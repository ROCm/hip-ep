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
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

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

// Returns true if the buffer underlying `value` or any of its aliases is
// written.
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
    auto computeOp = cast<ComputeOp>(op);
    if (!isHipsrDestinationOperand(opOperand)) {
      return false;
    }
    BlockArgument blockArg =
        getEntryArgument(computeOp, opOperand.getOperandNumber());
    // Check each use directly, without tracing through aliases
    for (OpOperand &use : blockArg.getUses()) {
      if (state.bufferizesToMemoryWrite(use)) {
        return true;
      }
    }
    return false;
  }

  bool isWritable(Operation *op, Value value, const AnalysisState &) const {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      return isHipsrDestinationOperand(
          op->getOpOperand(blockArg.getArgNumber()));
    }
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &) const {
    auto computeOp = cast<ComputeOp>(op);

    AliasingValueList aliases;
    aliases.addAlias({getEntryArgument(computeOp, opOperand.getOperandNumber()),
                      BufferRelation::Equivalent});

    if (OpResult result = getResultHeldIn(computeOp, opOperand)) {
      aliases.addAlias({result, BufferRelation::Equivalent});
    }
    return aliases;
  }

  AliasingOpOperandList getAliasingOpOperands(Operation *op, Value value,
                                              const AnalysisState &) const {
    // block argument <-> input operand
    auto computeOp = cast<ComputeOp>(op);
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      return {{&op->getOpOperand(blockArg.getArgNumber()),
               BufferRelation::Equivalent}};
    }

    // op result <-> yield op result
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    AliasingOpOperandList aliases;
    aliases.addAlias({&getYieldOp(computeOp)->getOpOperand(resultIndex),
                      BufferRelation::Equivalent});

    // op result <-> output operand
    if (OpOperand *destination = getDestinationOf(computeOp, resultIndex)) {
      aliases.addAlias({destination, BufferRelation::Equivalent});
    }
    return aliases;
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &analysisState,
                                 const BufferizationState &) const {
    auto computeOp = cast<ComputeOp>(op);

    // Check each output operand to see if it aliases any input operand.
    // If so, save the mapping as an attribute for use during bufferization.
    // We use aliasing (not equivalence) because out-of-place decisions break
    // equivalence but preserve aliasing relationships.
    SmallVector<int32_t> aliasingMap;
    for (auto [outIdx, output] : llvm::enumerate(computeOp.getOutputs())) {
      int32_t aliasedInputIdx = -1; // -1 means no aliasing

      if (isa<TensorType>(output.getType())) {
        // Find which input (if any) this output aliases
        for (auto [inIdx, input] : llvm::enumerate(computeOp.getInputs())) {
          if (!isa<TensorType>(input.getType())) {
            continue;
          }

          if (analysisState.areAliasingBufferizedValues(output, input)) {
            aliasedInputIdx = static_cast<int32_t>(inIdx);
            break;
          }
        }
      }

      aliasingMap.push_back(aliasedInputIdx);
    }

    // Save the mapping as an array attribute
    if (!aliasingMap.empty()) {
      auto arrayAttr = rewriter.getI32ArrayAttr(aliasingMap);
      op->setAttr("__output_alias_to_input", arrayAttr);
    }

    return success();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    // A block argument's buffer is the operand's buffer.
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      return bufferization::getBufferType(
          op->getOperand(blockArg.getArgNumber()), options, state,
          invocationStack);
    }

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

    // Get the aliasing mapping saved by resolveConflicts
    ArrayAttr aliasingAttr =
        op->getAttrOfType<ArrayAttr>("__output_alias_to_input");

    SmallVector<Value> bufferizedOutputs;
    for (auto [idx, output] : llvm::enumerate(computeOp.getOutputs())) {
      if (isa<TensorType>(output.getType())) {
        // Check if this output aliases an input
        int32_t aliasedInputIdx = -1;
        if (aliasingAttr && idx < aliasingAttr.size()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(aliasingAttr[idx])) {
            aliasedInputIdx = intAttr.getInt();
          }
        }

        if (aliasedInputIdx >= 0 &&
            static_cast<size_t>(aliasedInputIdx) < bufferizedInputs.size()) {
          // This output aliases an input - reuse that input's buffer
          bufferizedOutputs.push_back(bufferizedInputs[aliasedInputIdx]);
        } else {
          // No aliasing or invalid index - get normal buffer
          FailureOr<Value> buffer = getBuffer(rewriter, output, options, state);
          if (failed(buffer)) {
            return failure();
          }
          bufferizedOutputs.push_back(*buffer);
        }
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

//===----------------------------------------------------------------------===//
// Destination-passing ops
//===----------------------------------------------------------------------===//

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
    ComputeOp::attachInterface<ComputeOpBufferization>(*ctx);
    ComputeYieldOp::attachInterface<ComputeYieldOpBufferization>(*ctx);
    PoolDomainOp::attachInterface<PoolDomainOpBufferization>(*ctx);
    PoolDomainYieldOp::attachInterface<PoolDomainYieldOpBufferization>(*ctx);
    CastOp::attachInterface<DpsBufferizableModel<CastOp>>(*ctx);
    MulOp::attachInterface<DpsBufferizableModel<MulOp>>(*ctx);
  });
}
