/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "QLinearPlan.h"
#include "RocMlirElementwise.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

#include <optional>

namespace mlir::hip {

#define GEN_PASS_DEF_FUSEROCMLIRELEMENTWISEPASS
#define GEN_PASS_DEF_INLINEROCMLIRELEMENTWISEPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

void inlineRocMlirElementwise(RocMlirElementwiseOp op, RewriterBase &rewriter) {
  Block &body = op.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  SmallVector<Value> results(yield.getValues());
  SmallVector<Value> arguments = {op.getCtx()};
  llvm::append_range(arguments, op.getInputs());
  rewriter.inlineBlockBefore(&body, op, arguments);
  rewriter.eraseOp(yield);
  rewriter.replaceOp(op, results);
}

namespace {

enum class Role { Compute, View, Dequantize, Quantize };

bool isKernelIOValue(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return false;
  auto intType = dyn_cast<IntegerType>(type.getElementType());
  return !intType || intType.getWidth() <= 32;
}

// The operands an op reads as data: not the context and not a DPS init.
SmallVector<Value> getDataOperands(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps)
    return SmallVector<Value>(op->getOperands());
  SmallVector<Value> inputs;
  for (Value input : dps.getDpsInputs())
    if (!isa<ContextType>(input.getType()))
      inputs.push_back(input);
  return inputs;
}

// The anchor pass reads the integer operands of these dequantizes directly
// (planDequantizedConv), so they stay outside a module.
bool feedsPlannedDequantizedConv(Operation *op) {
  return isa<DequantizeLinearOp>(op) &&
         llvm::any_of(op->getUsers(), [](Operation *user) {
           return isa<ConvOp>(user) && isPlannedQLinear(user);
         });
}

// R1, R2: what a module may hold. A DPS op writes a tensor.empty init, which
// the module clones. A broadcast, a slice or a dynamic reshape gets no role
// and stays outside, as a module input. So does a rank-0 value, such as the
// dequantize of a scalar constant: the anchor pass recomputes it in the
// kernel, and as a module of its own it would take its readers away from the
// module of the anchor.
std::optional<Role> getRole(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return std::nullopt;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() == 0)
    return std::nullopt;
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    bool emptyInits = llvm::all_of(dps.getDpsInits(), [](Value init) {
      return init.getDefiningOp<tensor::EmptyOp>() != nullptr;
    });
    if (!emptyInits)
      return std::nullopt;
  }
  if (isa<TransposeOp>(op))
    return isTosaExpressiblePointwiseOp(op) ? std::optional(Role::View)
                                            : std::nullopt;
  if (isTosaLowerableReshape(op))
    return Role::View;
  if (isa<DequantizeLinearOp, QuantizeLinearOp>(op)) {
    if (!isTosaLowerableQdq(op) || feedsPlannedDequantizedConv(op))
      return std::nullopt;
    return isa<DequantizeLinearOp>(op) ? Role::Dequantize : Role::Quantize;
  }
  // isFusablePointwise also admits ops that HipToTosa lowers only inside an
  // anchor kernel, such as a Where on a ui8 mask.
  if (isTosaExpressiblePointwiseOp(op) || isFusablePointwise(op))
    return Role::Compute;
  return std::nullopt;
}

// The ops of one module, grown from its first op in block order.
struct Module {
  SetVector<Operation *> ops;
  // The earliest op in block order; a pulled dequantize can precede the op
  // the module grew from.
  Operation *earliest = nullptr;

  void insert(Operation *op) {
    ops.insert(op);
    if (!earliest || op->isBeforeInBlock(earliest))
      earliest = op;
  }
  bool readsOutside(Operation *op) const {
    return llvm::any_of(op->getUsers(),
                        [&](Operation *user) { return !ops.contains(user); });
  }
};

class ModulePlanner {
public:
  explicit ModulePlanner(Block &block) : block(block) {}

  SmallVector<Module> plan() {
    SmallVector<Module> modules;
    for (Operation &op : block) {
      if (assigned.contains(&op))
        continue;
      std::optional<Role> role = getRole(&op);
      if (!role || !llvm::all_of(getDataOperands(&op), isKernelIOValue))
        continue;
      Module module;
      module.insert(&op);
      grow(module);
      trim(module);
      if (!hasComputation(module))
        continue;
      assigned.insert_range(module.ops);
      modules.push_back(std::move(module));
    }
    return modules;
  }

private:
  void grow(Module &module) {
    bool grew = true;
    while (grew) {
      grew = false;
      SmallVector<Operation *> candidates;
      for (Operation *op : module.ops) {
        for (Operation *user : op->getUsers())
          if (user->getBlock() == &block && !module.ops.contains(user) &&
              !assigned.contains(user))
            candidates.push_back(user);
        for (Value operand : getDataOperands(op))
          if (Operation *def = operand.getDefiningOp();
              def && def->getBlock() == &block && !module.ops.contains(def) &&
              !assigned.contains(def) && isa<DequantizeLinearOp>(def))
            candidates.push_back(def);
      }
      llvm::sort(candidates, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      candidates.erase(llvm::unique(candidates), candidates.end());
      for (Operation *candidate : candidates) {
        if (module.ops.contains(candidate) ||
            !(canJoin(candidate, module) || canPull(candidate, module)))
          continue;
        module.insert(candidate);
        grew = true;
      }
    }
  }

  bool canJoin(Operation *op, const Module &module) const {
    std::optional<Role> role = getRole(op);
    // R5: a dequantize only starts a module.
    if (!role || *role == Role::Dequantize)
      return false;
    auto resultType = cast<ShapedType>(op->getResult(0).getType());
    for (Value operand : getDataOperands(op)) {
      Operation *def = operand.getDefiningOp();
      if (def && module.ops.contains(def)) {
        // R5: a quantize ends a module.
        if (isa<QuantizeLinearOp>(def))
          return false;
        // R3: rocMLIR cannot invert a broadcast of an epilogue value.
        if (*role == Role::Compute &&
            cast<ShapedType>(operand.getType()).getShape() !=
                resultType.getShape())
          return false;
        continue;
      }
      // R6: what the kernel loads.
      if (!isKernelIOValue(operand))
        return false;
      // R8: the module would read its own result through outside ops.
      if (dependsOnChain(operand, module.earliest, module.ops))
        return false;
    }
    return true;
  }

  // R5: a dequantize that only the module reads joins it from above, as the
  // other side of a residual add does, so the kernel loads its 8-bit input.
  bool canPull(Operation *op, const Module &module) const {
    if (getRole(op) != Role::Dequantize ||
        !llvm::all_of(op->getUsers(), [&](Operation *user) {
          return module.ops.contains(user);
        }))
      return false;
    return llvm::all_of(getDataOperands(op), [&](Value operand) {
      return isKernelIOValue(operand) &&
             !dependsOnChain(operand, module.earliest, module.ops);
    });
  }

  // R4, R6: a value read outside is a module result, which needs a DPS init
  // and a type the kernel stores. Drop each op that breaks this, with
  // whatever in the module reads it, until none is left.
  void trim(Module &module) const {
    bool dropped = true;
    while (dropped) {
      dropped = false;
      for (Operation *op : module.ops) {
        if (!module.readsOutside(op))
          continue;
        auto type = cast<RankedTensorType>(op->getResult(0).getType());
        bool storable = isa<DestinationStyleOpInterface>(op) &&
                        isKernelIOValue(op->getResult(0)) && type.getRank() > 0;
        if (storable)
          continue;
        dropWithReaders(module, op);
        dropped = true;
        break;
      }
    }
  }

  static void dropWithReaders(Module &module, Operation *root) {
    SmallVector<Operation *> worklist = {root};
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!module.ops.remove(op))
        continue;
      llvm::append_range(worklist, op->getUsers());
    }
  }

  // R7: a module of views alone saves no kernel.
  bool hasComputation(const Module &module) const {
    return llvm::any_of(
        module.ops, [&](Operation *op) { return getRole(op) != Role::View; });
  }

  Block &block;
  DenseSet<Operation *> assigned;
};

// The readers of a module result that sit above the last module op, with
// whatever reads them in turn, in block order. They move below the new op.
// Returns nullopt if one feeds the module, which R8 rules out.
std::optional<SmallVector<Operation *>>
readersToMove(ArrayRef<Operation *> ops, const DenseSet<Operation *> &inModule,
              ArrayRef<Value> results) {
  DenseSet<Value> escaped(results.begin(), results.end());
  auto readsEscaped = [&](Operation *op) {
    WalkResult walk = op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (escaped.contains(operand))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return walk.wasInterrupted();
  };
  SmallVector<Operation *> moved;
  if (ops.size() == 1)
    return moved;
  for (Operation &op : llvm::make_range(std::next(ops.front()->getIterator()),
                                        ops.back()->getIterator())) {
    if (inModule.contains(&op) || !readsEscaped(&op))
      continue;
    moved.push_back(&op);
    escaped.insert(op.result_begin(), op.result_end());
  }
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      if (!inModule.contains(operand.getDefiningOp()) &&
          escaped.contains(operand))
        return std::nullopt;
  return moved;
}

} // namespace

RocMlirElementwiseOp groupIntoRocMlirElementwise(ArrayRef<Operation *> opList,
                                                 RewriterBase &rewriter) {
  SmallVector<Operation *> ops(opList);
  if (ops.empty())
    return {};
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  DenseSet<Operation *> inModule(ops.begin(), ops.end());

  Value context;
  SetVector<Value> inputs;
  SmallVector<Value> results;
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      if (isa<ContextType>(operand.getType())) {
        if (context && context != operand)
          return {};
        context = operand;
      }
    }
    for (Value operand : getDataOperands(op))
      if (!inModule.contains(operand.getDefiningOp()))
        inputs.insert(operand);
    if (llvm::any_of(op->getUsers(),
                     [&](Operation *user) { return !inModule.contains(user); }))
      results.push_back(op->getResult(0));
  }
  if (!context || results.empty())
    return {};
  std::optional<SmallVector<Operation *>> moved =
      readersToMove(ops, inModule, results);
  if (!moved)
    return {};

  Operation *last = ops.back();
  rewriter.setInsertionPointAfter(last);
  auto resultTypes =
      llvm::map_to_vector(results, [](Value v) { return v.getType(); });
  auto group = RocMlirElementwiseOp::create(
      rewriter, last->getLoc(), resultTypes, context, inputs.getArrayRef());
  Operation *after = group;
  for (Operation *op : *moved) {
    rewriter.moveOpAfter(op, after);
    after = op;
  }

  SmallVector<Type> argumentTypes = {context.getType()};
  SmallVector<Location> argumentLocs = {context.getLoc()};
  for (Value input : inputs) {
    argumentTypes.push_back(input.getType());
    argumentLocs.push_back(input.getLoc());
  }
  Block *body =
      rewriter.createBlock(&group.getBody(), {}, argumentTypes, argumentLocs);
  IRMapping mapping;
  mapping.map(context, body->getArgument(0));
  for (auto [input, argument] :
       llvm::zip(inputs, body->getArguments().drop_front()))
    mapping.map(input, argument);
  SmallVector<Operation *> inits;
  for (Operation *op : ops) {
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
      for (Value init : dps.getDpsInits()) {
        if (mapping.contains(init))
          continue;
        Operation *empty = init.getDefiningOp();
        rewriter.clone(*empty, mapping);
        inits.push_back(empty);
      }
    }
    rewriter.clone(*op, mapping);
  }
  YieldOp::create(
      rewriter, last->getLoc(),
      llvm::map_to_vector(results, [&](Value v) { return mapping.lookup(v); }));

  for (auto [value, replacement] : llvm::zip(results, group.getResults()))
    rewriter.replaceUsesWithIf(value, replacement, [&](OpOperand &use) {
      return !inModule.contains(use.getOwner());
    });
  for (Operation *op : llvm::reverse(ops))
    rewriter.eraseOp(op);
  for (Operation *empty : inits)
    if (empty->use_empty())
      rewriter.eraseOp(empty);
  return group;
}

namespace {

void materialize(const Module &module, RewriterBase &rewriter) {
  groupIntoRocMlirElementwise(
      SmallVector<Operation *>(module.ops.begin(), module.ops.end()), rewriter);
}

struct FuseRocMlirElementwisePass
    : public impl::FuseRocMlirElementwisePassBase<FuseRocMlirElementwisePass> {
  using impl::FuseRocMlirElementwisePassBase<
      FuseRocMlirElementwisePass>::FuseRocMlirElementwisePassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph" || funcOp.isExternal())
      return;
    SmallVector<Module> modules =
        ModulePlanner(funcOp.getBody().front()).plan();
    IRRewriter rewriter(&getContext());
    for (const Module &module : modules)
      materialize(module, rewriter);
  }
};

struct InlineRocMlirElementwisePass
    : public impl::InlineRocMlirElementwisePassBase<
          InlineRocMlirElementwisePass> {
  void runOnOperation() override {
    SmallVector<RocMlirElementwiseOp> groups;
    getOperation().walk([&](RocMlirElementwiseOp op) { groups.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (RocMlirElementwiseOp op : groups) {
      rewriter.setInsertionPoint(op);
      inlineRocMlirElementwise(op, rewriter);
    }
  }
};

} // namespace
} // namespace mlir::hip
