/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
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
    return DenseElementsAttr::get(type, fn(input.getSplatValue<APInt>()));
  SmallVector<APInt> folded;
  folded.reserve(type.getNumElements());
  for (const APInt &value : input.getValues<APInt>())
    folded.push_back(fn(value));
  return DenseElementsAttr::get(type, folded);
}

template <typename Fn>
DenseElementsAttr mapBinary(DenseElementsAttr lhs, DenseElementsAttr rhs,
                            RankedTensorType type, Fn fn) {
  if (lhs.isSplat() && rhs.isSplat())
    return DenseElementsAttr::get(
        type, fn(lhs.getSplatValue<APInt>(), rhs.getSplatValue<APInt>()));
  SmallVector<APInt> folded;
  folded.reserve(type.getNumElements());
  for (auto [lhsValue, rhsValue] :
       llvm::zip_equal(lhs.getValues<APInt>(), rhs.getValues<APInt>()))
    folded.push_back(fn(lhsValue, rhsValue));
  return DenseElementsAttr::get(type, folded);
}

/// Lazily resolves SSA values to compile-time payloads. A value resolves when
/// it is an inline hip.constant, a static same-count tensor.expand_shape of a
/// resolvable value, or an evaluable HIP op over resolvable inputs. Results
/// (including failures) are memoized; every op a successful resolution walked
/// through is tracked so the post-fold cleanup can retire it once dead.
///
/// Resolution never mutates IR: an op whose own uses fail the replacement
/// audit can still be evaluated through (partial evaluation); the op then
/// simply stays behind for its other users.
class PayloadResolver {
public:
  PayloadResolver(int64_t maxElements, int64_t evalBudget)
      : maxElements(maxElements), evalBudget(evalBudget) {}

  DenseElementsAttr resolve(Value value) { return resolveImpl(value, 0); }

  /// Ops a successful resolution read or evaluated, in discovery order.
  const llvm::SetVector<Operation *> &trackedOps() const { return tracked; }

  /// Drops the memo entry for a value that is about to be erased, so a later
  /// allocation reusing its address cannot alias a stale payload.
  void forget(Value value) { memo.erase(value); }

  /// Evaluates one HIP op over resolved inputs; null when the op kind has no
  /// evaluator, a contract is violated, or a budget is exhausted. Charges the
  /// evaluation budget before computing so failed attempts count too.
  DenseElementsAttr evaluateOp(Operation *op, int depth) {
    if (op->getNumResults() != 1)
      return nullptr;
    auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getNumElements() > maxElements)
      return nullptr;
    evalBudget -= type.getNumElements();
    if (evalBudget < 0)
      return nullptr;

    if (isa<NotOp, AndOp, OrOp>(op))
      return evaluateBoolElementwise(op, type, depth);
    return nullptr;
  }

private:
  DenseElementsAttr resolveImpl(Value value, int depth) {
    if (depth > kMaxDepth)
      return nullptr;
    if (auto it = memo.find(value); it != memo.end())
      return it->second;

    DenseElementsAttr result = nullptr;
    Operation *def = value.getDefiningOp();
    if (!def) {
      memo[value] = nullptr;
      return nullptr;
    }

    if (auto constant = dyn_cast<ConstantOp>(def)) {
      if (constant.getSourceKind() == ConstantOp::SourceKind::Inline) {
        auto dense =
            dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
        if (dense && dense.getType() == value.getType())
          result = dense;
      }
    } else if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      // A verified expand_shape preserves linear element order, so reshaping
      // the payload matches its semantics. Dynamic forms bail.
      auto srcType = dyn_cast<RankedTensorType>(expand.getSrc().getType());
      auto dstType = dyn_cast<RankedTensorType>(expand.getType());
      if (srcType && dstType && srcType.hasStaticShape() &&
          dstType.hasStaticShape() &&
          srcType.getElementType() == dstType.getElementType() &&
          srcType.getNumElements() == dstType.getNumElements())
        if (DenseElementsAttr src = resolveImpl(expand.getSrc(), depth + 1))
          result = src.reshape(dstType);
    } else {
      result = evaluateOp(def, depth + 1);
    }

    if (result)
      tracked.insert(def);
    memo[value] = result;
    return result;
  }

  /// i1-only element-wise group: hip.not / hip.and / hip.or. Every data input
  /// must share the op's exact static i1 result type (excludes broadcasting
  /// forms) and resolve to a payload.
  DenseElementsAttr evaluateBoolElementwise(Operation *op,
                                            RankedTensorType type, int depth) {
    if (!type.getElementType().isInteger(1))
      return nullptr;
    SmallVector<Value, 2> inputs;
    if (auto notOp = dyn_cast<NotOp>(op))
      inputs = {notOp.getX()};
    else if (auto andOp = dyn_cast<AndOp>(op))
      inputs = {andOp.getLhs(), andOp.getRhs()};
    else if (auto orOp = dyn_cast<OrOp>(op))
      inputs = {orOp.getLhs(), orOp.getRhs()};
    SmallVector<DenseElementsAttr, 2> payloads;
    for (Value input : inputs) {
      if (input.getType() != type)
        return nullptr;
      DenseElementsAttr payload = resolveImpl(input, depth + 1);
      if (!payload)
        return nullptr;
      payloads.push_back(payload);
    }
    if (isa<NotOp>(op))
      return mapUnary(payloads[0], type,
                      [](const APInt &value) { return ~value; });
    if (isa<AndOp>(op))
      return mapBinary(
          payloads[0], payloads[1], type,
          [](const APInt &lhs, const APInt &rhs) { return lhs & rhs; });
    return mapBinary(
        payloads[0], payloads[1], type,
        [](const APInt &lhs, const APInt &rhs) { return lhs | rhs; });
  }

  static constexpr int kMaxDepth = 128;
  int64_t maxElements;
  int64_t evalBudget;
  llvm::DenseMap<Value, DenseElementsAttr> memo;
  llvm::SetVector<Operation *> tracked;
};

/// Kinds the sweep may replace with a carrier. Grows with the evaluators.
bool isReplaceableKind(Operation *op) { return isa<NotOp, AndOp, OrOp>(op); }

struct FoldTensorConstantsPass final
    : public impl::FoldTensorConstantsPassBase<FoldTensorConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!hasUniqueExplicitSerializationOrders(module))
      return;

    // Cumulative evaluation budget across the whole module, charged per
    // evaluated element (failed attempts included) so wide constant DAGs
    // cannot exhaust compile-time memory.
    constexpr int64_t kEvalBudget = int64_t(1) << 22;
    IRRewriter rewriter(&getContext());

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isDeclaration() || !function.getBody().hasOneBlock())
        continue;
      Block &block = function.getBody().front();
      PayloadResolver resolver(maxFoldElements, kEvalBudget);

      // Walking in block order folds producer-before-consumer chains in one
      // sweep: a folded result is an inline constant by the time a later
      // candidate reads it, and unfolded intermediates are still evaluated
      // through by the resolver.
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (!isReplaceableKind(&op))
          continue;
        // A dead result is DCE work, not folding work.
        if (op.getResult(0).use_empty())
          continue;
        if (!allUsesAreFoldableReads(&op))
          continue;
        DenseElementsAttr folded = resolver.evaluateOp(&op, 0);
        if (!folded)
          continue;

        // The replacement carrier gets no serialization_order: unordered
        // carriers follow ordered ones in stable module walk order.
        auto type = cast<RankedTensorType>(op.getResult(0).getType());
        resolver.forget(op.getResult(0));
        rewriter.setInsertionPoint(&op);
        Value replacement =
            ConstantOp::create(rewriter, op.getLoc(), type, folded);
        rewriter.replaceOp(&op, replacement);
        ++NumOpsFolded;
      }

      // Retire the chain the folds consumed: visit live ops in reverse block
      // order (so consumers die before their producers) and erase tracked
      // ops that became dead. Membership is only ever queried with live op
      // pointers. Dead tensor.empty inits are not tracked and stay for the
      // canonicalizer that already runs before bufferization.
      const llvm::SetVector<Operation *> &tracked = resolver.trackedOps();
      for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
        if (tracked.contains(&op) && op.use_empty() && isOpTriviallyDead(&op))
          rewriter.eraseOp(&op);
    }
  }
};

} // namespace
} // namespace mlir::hip
