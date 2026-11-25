/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ProveStateDonation.cpp - Decide in-place KV cache reuse ----------===//
//
// A state result may be written into its own input's allocation only when
// nothing can observe the old bytes afterwards. This proves that, per state,
// for the split compiler to tell the backend which outputs it may hand the
// input's memory.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Ggml/Transforms/StateDonation.h"

#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlProvenance.h"
#include "hip/Dialect/Ggml/IR/GgmlStatePairs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir {
namespace ggml {

namespace {

/// True when a value's recorded strides are the dense ones its extents imply,
/// for every request rather than merely the one captured.
bool isDense(Value value) {
  return deriveContiguity(value) == Contiguity::Dense;
}

/// Proves, for one state, that its result may reuse its input's allocation.
class DonationProver {
public:
  DonationProver(func::FuncOp function) : function(function) {}

  /// True when donating is safe. `reason` names the first requirement that did
  /// not hold, so a state that loses reuse can say why.
  bool isEligible(Value stateInput, Value stateOutput, std::string &reason);

private:
  bool no(std::string &reason, const Twine &text) {
    reason = text.str();
    return false;
  }

  /// True when nothing after `writer` reads the storage except through the
  /// value the write produced.
  bool noLaterReadOfOldBytes(Operation *writer, Value storage,
                             std::string &reason);

  func::FuncOp function;
};

bool DonationProver::noLaterReadOfOldBytes(Operation *writer, Value storage,
                                           std::string &reason) {
  // Replacing in place changes the old bytes immediately, so every later read
  // of this storage must be data-dependent on the replacement. An older alias
  // read afterwards would see something the out-of-place graph never produces.
  DenseSet<Value> derived;
  derived.insert(writer->getResult(0));
  // Rows gathered from the state before the write are a copy in the graph,
  // but the lowering reads a one-row gather straight from the state (a
  // recurrent state of one sequence), so a later read of them is a read of
  // the old bytes too.
  DenseSet<Value> gathered;

  bool seenWriter = false;
  WalkResult walked = function.walk([&](Operation *op) {
    if (op == writer) {
      seenWriter = true;
      return WalkResult::advance();
    }
    if (!seenWriter) {
      if (auto rows = dyn_cast<GetRowsOp>(op))
        if (getStorageRoot(rows.getData()) == storage)
          gathered.insert(rows.getResult());
      return WalkResult::advance();
    }

    SmallVector<Value> sameStorage;
    for (Value operand : op->getOperands()) {
      Value root = getStorageRoot(operand);
      if (!root) {
        reason = "a later operand has unresolved or cyclic storage";
        return WalkResult::interrupt();
      }
      if (gathered.contains(root)) {
        reason = "a later operation reads rows gathered from the state's old "
                 "bytes";
        return WalkResult::interrupt();
      }
      if (root == storage)
        sameStorage.push_back(operand);
    }
    for (Value operand : sameStorage) {
      if (!derived.contains(operand)) {
        reason = "a later operation reads the state's old bytes";
        return WalkResult::interrupt();
      }
    }

    if (getWrittenStorage(op) == storage) {
      if (sameStorage.empty()) {
        reason = "a later operation writes the state without reading the "
                 "replacement";
        return WalkResult::interrupt();
      }
      if (op->getNumResults() == 1)
        derived.insert(op->getResult(0));
    }
    return WalkResult::advance();
  });
  return !walked.wasInterrupted();
}

bool DonationProver::isEligible(Value stateInput, Value stateOutput,
                                std::string &reason) {
  // The public input owns the whole persistent buffer; the public result may be
  // an alias, but it must describe that same buffer byte for byte.
  if (!isStorageRoot(stateInput) || getStorageOffset(stateInput) != 0 ||
      !isDense(stateInput)) {
    return no(reason, "the state input is not the whole buffer's root");
  }
  if (getStorageRoot(stateOutput) != stateInput ||
      getStorageOffset(stateOutput) != 0 || !isDense(stateOutput)) {
    return no(reason, "the state result does not cover the whole buffer");
  }

  // The layout is the type, so equal types mean the result holds the same
  // elements in the same arrangement starting at the same byte. How far the
  // buffer reaches is not asked separately: the result was just required to
  // view this input, so its span is this input's.
  if (stateInput.getType() != stateOutput.getType())
    return no(reason, "the state result does not have its input's layout");

  auto writer = stateOutput.getDefiningOp<SetRowsOp>();
  if (!writer)
    return no(reason, "the state result is not produced by ggml.set_rows");
  // (Or on that input as it is: llama.cpp writes a cache of several streams
  // through a reshape to its rows, which here is its own layout.)
  Value destination = writer.getDestination();
  if (auto same = destination.getDefiningOp<ReshapeOp>();
      same && same.getInput() == stateInput &&
      destination.getType() == stateInput.getType())
    destination = stateInput;
  if (destination != stateInput)
    return no(reason, "the write does not land on the state's own input");
  if (getWrittenStorage(writer) != stateInput)
    return no(reason, "the write does not replace this state's storage");

  // The result must view the buffer being donated rather than merely sit in a
  // buffer of the same shape. It is not the root -- the input is -- so the
  // value it views has to be that input.
  if (getAliasRoot(stateOutput) != stateInput)
    return no(reason, "the state result does not view its input's buffer");

  // Nothing further is asked about the write covering the whole buffer. The
  // input is that buffer's root at offset zero, the result sits at offset zero
  // in the same buffer, and the two agree on extents, strides, and byte
  // length -- so the result is the buffer, byte for byte.

  // The bytes being written must come from somewhere else, or the replacement
  // would read what it is overwriting.
  Builder builder(function.getContext());
  if (haveSameStorage(writer.getUpdates(), stateInput) ||
      haveSameStorage(writer.getIndices(), stateInput)) {
    return no(reason, "the write's own operands live in the state's storage");
  }
  auto updatesType = dyn_cast<RankedTensorType>(writer.getUpdates().getType());
  auto indicesType = dyn_cast<RankedTensorType>(writer.getIndices().getType());
  if (!indicesType || indicesType.getElementType() != builder.getI64Type())
    return no(reason, "the write's indices are not i64");
  if (!updatesType || !(updatesType.getElementType().isF16() ||
                        updatesType.getElementType().isF32())) {
    return no(reason, "the write's updates are not f16 or f32");
  }

  // One public result per storage, or two callers would claim the same buffer.
  auto terminator =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  unsigned exposed = 0;
  for (Value result : terminator.getOperands()) {
    if (haveSameStorage(result, stateInput))
      ++exposed;
  }
  if (exposed != 1)
    return no(reason, "this state's storage leaves through " + Twine(exposed) +
                          " results");

  return noLaterReadOfOldBytes(writer, stateInput, reason);
}

} // namespace

FailureOr<SmallVector<StateDonation>> proveStateDonation(ModuleOp module) {
  SmallVector<StateDonation> eligible;
  auto function = module.lookupSymbol<func::FuncOp>("main_graph");
  if (!function)
    return eligible;
  std::string reason;
  auto pairs = readStatePairs(function, reason);
  if (failed(pairs))
    return function.emitOpError(reason);
  if (pairs->empty())
    return eligible;
  if (!function.getBody().hasOneBlock())
    return function.emitOpError("state donation requires a single-block body");

  Block &entry = function.getBody().front();
  auto terminator = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!terminator || terminator.getNumOperands() != function.getNumResults())
    return function.emitOpError("does not end in a return");

  DonationProver prover(function);
  for (const StatePair &pair : *pairs) {
    StateDonation donation;
    donation.inPlace = prover.isEligible(
        entry.getArgument(pair.inputIndex),
        terminator.getOperand(pair.outputIndex), donation.reason);
    eligible.push_back(std::move(donation));
  }
  return eligible;
}

} // namespace ggml
} // namespace mlir
