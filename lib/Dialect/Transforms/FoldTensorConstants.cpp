/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "hip-fold-tensor-constants"

STATISTIC(NumOpsFolded, "Number of constant-input tensor ops folded");

namespace mlir::hip {

#define GEN_PASS_DEF_FOLDTENSORCONSTANTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// The externalizer diagnoses duplicate explicit serialization orders. Folding
/// may erase one of the duplicate carriers and hide that producer bug, so the
/// pass refuses to touch a module where the diagnostic would fire.
bool hasUniqueExplicitSerializationOrders(ModuleOp module) {
  llvm::DenseSet<int64_t> orders;
  bool unique = true;
  module.walk([&](ConstantOp constant) {
    IntegerAttr order = constant.getSerializationOrderAttr();
    if (order && !orders.insert(order.getInt()).second)
      unique = false;
  });
  return unique;
}

/// Returns the data inputs of `op` if it is an evaluable element-wise boolean
/// operation whose inputs and single result all share one static i1 tensor
/// type (which excludes broadcasting forms), std::nullopt otherwise.
std::optional<SmallVector<Value, 2>> getEvaluableInputs(Operation *op) {
  SmallVector<Value, 2> inputs;
  if (auto notOp = dyn_cast<NotOp>(op))
    inputs = {notOp.getX()};
  else if (auto andOp = dyn_cast<AndOp>(op))
    inputs = {andOp.getLhs(), andOp.getRhs()};
  else if (auto orOp = dyn_cast<OrOp>(op))
    inputs = {orOp.getLhs(), orOp.getRhs()};
  else
    return std::nullopt;

  if (op->getNumResults() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isInteger(1))
    return std::nullopt;
  for (Value input : inputs)
    if (input.getType() != type)
      return std::nullopt;
  return inputs;
}

/// Returns the dense payload of `value` when it is produced by an inline
/// `hip.constant` of exactly `type`, nullptr otherwise. File- and
/// memory-backed carriers are not read in this phase.
DenseElementsAttr getInlinePayload(Value value, RankedTensorType type) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant || constant.getSourceKind() != ConstantOp::SourceKind::Inline)
    return nullptr;
  auto dense = dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
  if (!dense || dense.getType() != type)
    return nullptr;
  return dense;
}

/// A result may only be replaced by a constant when every use reads it:
/// each user must be a destination-passing-style op in the same block using
/// it as a DPS input. A DPS init use is refused because bufferization would
/// turn the replacement constant into the op's destination buffer, and a
/// terminator use is refused because graph outputs stay runtime-allocated.
bool allUsesAreFoldableReads(Operation *op) {
  for (OpOperand &use : op->getResult(0).getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getBlock() != op->getBlock())
      return false;
    auto dps = dyn_cast<DestinationStyleOpInterface>(owner);
    if (!dps || dps.isDpsInit(&use))
      return false;
  }
  return true;
}

template <typename Fn>
DenseElementsAttr mapUnary(DenseElementsAttr input, RankedTensorType type,
                           Fn fn) {
  if (input.isSplat())
    return DenseElementsAttr::get(type, fn(input.getSplatValue<bool>()));
  SmallVector<bool> folded;
  folded.reserve(type.getNumElements());
  for (bool value : input.getValues<bool>())
    folded.push_back(fn(value));
  return DenseElementsAttr::get(type, folded);
}

template <typename Fn>
DenseElementsAttr mapBinary(DenseElementsAttr lhs, DenseElementsAttr rhs,
                            RankedTensorType type, Fn fn) {
  if (lhs.isSplat() && rhs.isSplat())
    return DenseElementsAttr::get(
        type, fn(lhs.getSplatValue<bool>(), rhs.getSplatValue<bool>()));
  SmallVector<bool> folded;
  folded.reserve(type.getNumElements());
  for (auto [lhsValue, rhsValue] :
       llvm::zip_equal(lhs.getValues<bool>(), rhs.getValues<bool>()))
    folded.push_back(fn(lhsValue, rhsValue));
  return DenseElementsAttr::get(type, folded);
}

/// Evaluates `op` over the dense payloads of its data inputs. All payloads
/// have already been checked to match the result type exactly.
DenseElementsAttr evaluate(Operation *op, ArrayRef<DenseElementsAttr> payloads,
                           RankedTensorType type) {
  if (isa<NotOp>(op))
    return mapUnary(payloads[0], type, [](bool value) { return !value; });
  if (isa<AndOp>(op))
    return mapBinary(payloads[0], payloads[1], type,
                     [](bool lhs, bool rhs) { return lhs && rhs; });
  if (isa<OrOp>(op))
    return mapBinary(payloads[0], payloads[1], type,
                     [](bool lhs, bool rhs) { return lhs || rhs; });
  return nullptr;
}

struct FoldTensorConstantsPass final
    : public impl::FoldTensorConstantsPassBase<FoldTensorConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!hasUniqueExplicitSerializationOrders(module))
      return;

    IRRewriter rewriter(&getContext());
    // Carriers whose payloads were consumed by a fold. Erased afterwards only
    // if dead, so pre-existing dead carriers and carriers with other users are
    // left untouched.
    SetVector<Operation *> consumedCarriers;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isDeclaration() || !function.getBody().hasOneBlock())
        continue;

      // Walking in block order folds producer-before-consumer chains in one
      // sweep: a folded result is an inline constant by the time a later
      // candidate reads it.
      for (Operation &op :
           llvm::make_early_inc_range(function.getBody().front())) {
        std::optional<SmallVector<Value, 2>> inputs = getEvaluableInputs(&op);
        if (!inputs)
          continue;
        // A dead result is DCE work, not folding work.
        if (op.getResult(0).use_empty())
          continue;
        if (!allUsesAreFoldableReads(&op))
          continue;

        auto type = cast<RankedTensorType>(op.getResult(0).getType());
        SmallVector<DenseElementsAttr, 2> payloads;
        for (Value input : *inputs) {
          if (DenseElementsAttr payload = getInlinePayload(input, type))
            payloads.push_back(payload);
          else
            break;
        }
        if (payloads.size() != inputs->size())
          continue;

        DenseElementsAttr folded = evaluate(&op, payloads, type);
        if (!folded)
          continue;

        for (Value input : *inputs)
          consumedCarriers.insert(input.getDefiningOp());

        // The replacement carrier gets no serialization_order: unordered
        // carriers follow ordered ones in stable module walk order.
        rewriter.setInsertionPoint(&op);
        Value replacement =
            ConstantOp::create(rewriter, op.getLoc(), type, folded);
        rewriter.replaceOp(&op, replacement);
        ++NumOpsFolded;
      }
    }

    // Erase consumed carriers that became dead so the externalizer does not
    // assign storage to them. Dead `tensor.empty` inits are left for the
    // canonicalizer that already runs before bufferization.
    for (Operation *carrier : consumedCarriers)
      if (carrier->use_empty())
        rewriter.eraseOp(carrier);
  }
};

} // namespace
} // namespace mlir::hip
