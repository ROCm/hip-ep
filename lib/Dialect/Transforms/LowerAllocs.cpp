/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LowerAllocs.cpp - memref.alloc -> hip.alloc + hip.free -------------===//
//
// Replaces memref.alloc with hip.alloc (device allocation via hipMalloc) and
// inserts hip.free for buffers not returned from the function.  Bridges
// one-shot-bufferize output to HIP-specific lowering.
//
// Alias-aware free placement uses BufferViewFlowAnalysis to transitively
// follow view-like ops so that hip.free is placed after all consumers of
// any derived view, preventing use-after-free.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-lower-allocs"

STATISTIC(NumAllocsLowered, "Number of memref.alloc lowered to hip.alloc");
STATISTIC(NumFreesInserted, "Number of hip.free ops inserted");
STATISTIC(NumDeallocsConverted,
          "Number of memref.dealloc converted to hip.free");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOWERALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Walk \p val back through view-like alias ops and return the root
/// hip::AllocOp if the chain originates from one, or nullptr otherwise.
static AllocOp traceToHipAlloc(Value val) {
  while (val) {
    if (auto alloc = val.getDefiningOp<AllocOp>())
      return alloc;
    auto viewLike = dyn_cast_or_null<ViewLikeOpInterface>(val.getDefiningOp());
    if (!viewLike)
      return nullptr;
    val = viewLike.getViewSource();
  }
  return nullptr;
}

/// Results of \p user that still name \p value's storage. View-like ops and
/// arith.select both forward a memref; region branches do not. A value passed
/// to scf.yield / scf.condition is a transfer, planned with the successor.
static void appendForwardedMemRefs(Operation *user, Value value,
                                   SmallVectorImpl<Value> &forwarded) {
  if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
    if (view.getViewSource() != value)
      return;
    for (Value result : user->getResults())
      if (isa<BaseMemRefType>(result.getType()))
        forwarded.push_back(result);
    return;
  }
  auto select = dyn_cast<arith::SelectOp>(user);
  if (!select || !isa<BaseMemRefType>(select.getType()))
    return;
  if (select.getTrueValue() == value || select.getFalseValue() == value)
    forwarded.push_back(select.getResult());
}

/// Users of \p root and of views or selects derived from it.
static void collectLocalUsers(Value root, SmallVectorImpl<Operation *> &users) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> seen{root};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      users.push_back(user);
      SmallVector<Value> forwarded;
      appendForwardedMemRefs(user, value, forwarded);
      for (Value result : forwarded)
        if (seen.insert(result).second)
          worklist.push_back(result);
    }
  }
}

/// Where to free a buffer owned by one block.
struct LocalFreePlan {
  Block *block = nullptr;
  /// Free after this op; at the block start when null.
  Operation *after = nullptr;
  /// The buffer leaves the block, so its new holder frees it.
  bool transfers = false;
  /// scf.if ops in the block that may yield the buffer. Free it only when
  /// none of their results is the buffer.
  SmallVector<Operation *> conditionalTargets;
  /// Allocations that must stay live. Free only when \c root names a
  /// different allocation than every value here.
  SmallVector<Value> preserved;
};

static void extendLocalFreePlan(Value value, Block &block,
                                LocalFreePlan &plan) {
  if (Operation *def = value.getDefiningOp())
    if (Operation *ancestor = block.findAncestorOpInBlock(*def))
      if (!plan.after || plan.after->isBeforeInBlock(ancestor))
        plan.after = ancestor;
  SmallVector<Operation *> users;
  collectLocalUsers(value, users);
  for (Operation *user : users) {
    Operation *ancestor = block.findAncestorOpInBlock(*user);
    if (!ancestor || isa<func::ReturnOp>(user) ||
        (ancestor == user && user->hasTrait<OpTrait::IsTerminator>())) {
      plan.transfers = true;
      continue;
    }
    if (!plan.after || plan.after->isBeforeInBlock(ancestor))
      plan.after = ancestor;
  }
}

static LocalFreePlan planLocalFree(Value root, Block &block) {
  LocalFreePlan plan;
  plan.block = &block;
  if (Operation *def = root.getDefiningOp())
    plan.after = block.findAncestorOpInBlock(*def);
  SmallVector<Operation *> users;
  collectLocalUsers(root, users);
  for (Operation *user : users) {
    Operation *ancestor = block.findAncestorOpInBlock(*user);
    if (!ancestor || isa<func::ReturnOp>(user)) {
      plan.transfers = true;
      continue;
    }
    if (isa<RegionBranchTerminatorOpInterface>(user)) {
      Operation *parent = user->getParentOp();
      if (ancestor == user || parent != ancestor || !isa<scf::IfOp>(parent)) {
        plan.transfers = true;
        continue;
      }
      if (!llvm::is_contained(plan.conditionalTargets, parent))
        plan.conditionalTargets.push_back(parent);
    }
    if (!plan.after || plan.after->isBeforeInBlock(ancestor))
      plan.after = ancestor;
  }
  return plan;
}

/// Memref values that take over a buffer from region branches, paired with
/// the values they may receive.
static void collectCarriedValues(
    Operation *op,
    SmallVectorImpl<std::pair<Value, SmallVector<Value>>> &carried) {
  auto isMemRef = [](Value value) {
    return isa<BaseMemRefType>(value.getType());
  };
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    scf::ConditionOp condition = whileOp.getConditionOp();
    scf::YieldOp yield = whileOp.getYieldOp();
    for (auto [index, arg] : llvm::enumerate(whileOp.getBeforeArguments()))
      if (isMemRef(arg))
        carried.push_back(
            {arg, {whileOp.getInits()[index], yield.getOperand(index)}});
    for (auto [index, arg] : llvm::enumerate(whileOp.getAfterArguments()))
      if (isMemRef(arg))
        carried.push_back({arg, {condition.getArgs()[index]}});
    for (auto [index, result] : llvm::enumerate(whileOp.getResults()))
      if (isMemRef(result))
        carried.push_back({result, {condition.getArgs()[index]}});
    return;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isMemRef(result))
        continue;
      SmallVector<Value> sources{ifOp.thenYield().getOperand(index)};
      if (!ifOp.getElseRegion().empty())
        sources.push_back(ifOp.elseYield().getOperand(index));
      carried.push_back({result, std::move(sources)});
    }
  }
}

/// Index of the allocated base pointer, the address hip.free releases.
/// A memref.view with a nonzero byte shift has a different aligned pointer
/// but the same allocated pointer as its source.
static Value allocatedPointerIndex(OpBuilder &builder, Location loc,
                                   Value memref) {
  if (!isa<MemRefType>(memref.getType()))
    return memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, memref);
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  return memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, metadata.getBaseBuffer());
}

static bool isTransferredAlloc(Value value,
                               const DenseSet<Operation *> &transferred) {
  auto alloc = value.getDefiningOp<AllocOp>();
  return alloc && transferred.contains(alloc);
}

/// \p value can be named at a free inserted in \p block just after \p after
/// (\p after is null when the free is at the block start).
static bool availableAt(Value value, Block &block, Operation *after) {
  Operation *def = nullptr;
  Block *defBlock = nullptr;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    defBlock = arg.getOwner();
  } else {
    def = value.getDefiningOp();
    if (!def)
      return false;
    defBlock = def->getBlock();
  }
  if (defBlock == &block) {
    if (!def)
      return true;
    if (!after)
      return false;
    return def == after || def->isBeforeInBlock(after);
  }
  for (Operation *parent = block.getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getBlock() == defBlock)
      return true;
  return false;
}

static bool definedInside(Value value, Operation *regionOp) {
  Operation *def = isa<BlockArgument>(value)
                       ? cast<BlockArgument>(value).getOwner()->getParentOp()
                       : value.getDefiningOp();
  return def && regionOp->isAncestor(def);
}

/// Keep allocations that this free does not own. \p regionOp is the scf.while
/// or scf.if that produced \p carried. Origins are the reverse buffer-view
/// chain, so selects, views, and region arguments are included.
static void preserveOutsideOrigins(Value carried, Operation *regionOp,
                                   BufferViewFlowAnalysis &aliases,
                                   const DenseSet<Operation *> &transferred,
                                   LocalFreePlan &plan) {
  for (Value origin : aliases.resolveReverse(carried)) {
    if (origin == carried || !isa<BaseMemRefType>(origin.getType()))
      continue;
    if (definedInside(origin, regionOp))
      continue;
    if (!origin.getDefiningOp<AllocOp>() &&
        !aliases.mayBeTerminalBuffer(origin))
      continue;
    if (isTransferredAlloc(origin, transferred))
      continue;
    if (!availableAt(origin, *plan.block, plan.after))
      continue;
    if (!llvm::is_contained(plan.preserved, origin))
      plan.preserved.push_back(origin);
  }
}

static void mergePlans(LocalFreePlan &dst, const LocalFreePlan &src) {
  dst.transfers |= src.transfers;
  if (src.after && dst.block == src.block &&
      (!dst.after || dst.after->isBeforeInBlock(src.after)))
    dst.after = src.after;
  for (Operation *target : src.conditionalTargets)
    if (!llvm::is_contained(dst.conditionalTargets, target))
      dst.conditionalTargets.push_back(target);
  for (Value value : src.preserved)
    if (!llvm::is_contained(dst.preserved, value))
      dst.preserved.push_back(value);
}

/// Two scf.if results may name one allocation when their reverse alias sets
/// meet (the same yielded memref, or a shared alloc / argument behind a view
/// or select).
static bool ifResultsMayAlias(Value lhs, Value rhs,
                              BufferViewFlowAnalysis &aliases) {
  auto left = dyn_cast<OpResult>(lhs);
  auto right = dyn_cast<OpResult>(rhs);
  if (!left || !right || left.getOwner() != right.getOwner())
    return false;
  if (!isa<scf::IfOp>(left.getOwner()))
    return false;
  auto leftOrigins = aliases.resolveReverse(lhs);
  for (Value origin : aliases.resolveReverse(rhs))
    if (leftOrigins.contains(origin))
      return true;
  return false;
}

static void emitLocalFree(OpBuilder &builder, Value ctx, Value root,
                          const LocalFreePlan &plan) {
  if (plan.transfers)
    return;
  if (plan.after)
    builder.setInsertionPointAfter(plan.after);
  else
    builder.setInsertionPointToStart(plan.block);
  Location loc = root.getLoc();
  SmallVector<Value> exclude;
  for (Operation *target : plan.conditionalTargets)
    for (Value result : target->getResults())
      if (isa<BaseMemRefType>(result.getType()) && result != root)
        exclude.push_back(result);
  for (Value preserved : plan.preserved)
    if (preserved != root && !llvm::is_contained(exclude, preserved))
      exclude.push_back(preserved);
  if (exclude.empty()) {
    FreeOp::create(builder, loc, ctx, root);
    ++NumFreesInserted;
    return;
  }
  Value pointer = allocatedPointerIndex(builder, loc, root);
  Value blocked = arith::ConstantIntOp::create(builder, loc, 0, 1);
  for (Value other : exclude) {
    Value otherPointer = allocatedPointerIndex(builder, loc, other);
    Value same = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       pointer, otherPointer);
    blocked = arith::OrIOp::create(builder, loc, blocked, same);
  }
  Value release = arith::XOrIOp::create(
      builder, loc, blocked, arith::ConstantIntOp::create(builder, loc, 1, 1));
  auto ifOp =
      scf::IfOp::create(builder, loc, release, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  FreeOp::create(builder, loc, ctx, root);
  ++NumFreesInserted;
}

struct LowerAllocsPass : public impl::LowerAllocsPassBase<LowerAllocsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect, memref::MemRefDialect, arith::ArithDialect,
                    scf::SCFDialect>();
    arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override;
};

void LowerAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;
  // TODO: Generalize to multi-block functions using MLIR's Liveness analysis.
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-lower-allocs requires single-block functions; "
                     "findLastAliasedUser uses isBeforeInBlock which does "
                     "not generalize to multi-block control flow");
    return signalPassFailure();
  }

  // Get !hip.context from function argument 0.
  Value ctx;
  if (!funcOp.getArguments().empty() &&
      isa<ContextType>(funcOp.getArgument(0).getType()))
    ctx = funcOp.getArgument(0);
  if (!ctx) {
    funcOp.emitError("no !hip.context argument found; "
                     "hip-add-context-arg must run first");
    return signalPassFailure();
  }

  OpBuilder builder(funcOp.getContext());

  // ---- Phase 1: Replace each memref.alloc with hip.alloc ----------------
  SmallVector<memref::AllocOp> allocs;
  funcOp.walk([&](memref::AllocOp op) { allocs.push_back(op); });

  SmallVector<AllocOp> hipAllocs;
  for (memref::AllocOp allocOp : allocs) {
    builder.setInsertionPoint(allocOp);
    auto hipAlloc =
        AllocOp::create(builder, allocOp.getLoc(), allocOp.getType(), ctx,
                        allocOp.getDynamicSizes());
    allocOp.replaceAllUsesWith(hipAlloc.getResult());
    allocOp.erase();
    hipAllocs.push_back(hipAlloc);
    ++NumAllocsLowered;
  }

  // ---- Phase 2: Build alias analysis on the *modified* IR ---------------
  //
  // CRITICAL: BufferViewFlowAnalysis must be constructed AFTER the
  // memref.alloc -> hip.alloc replacement above.  The analysis records
  // forward alias edges (source -> view) for ViewLikeOpInterface ops
  // (memref.view, memref.subview, memref.cast, etc.).  resolve(value)
  // then returns all downstream aliases of that value.
  //
  // If the analysis were built on the original IR (with memref.alloc),
  // then queried with the NEW hip.alloc SSA values, resolve() would
  // return only {hipAlloc} -- missing all downstream views.  This causes:
  //   - isAliasInSet: fails to detect that a returned memref.view aliases
  //     the pool, so hip.free is incorrectly inserted for returned buffers.
  //   - findLastAliasedUser: finds only the immediate SSA user (e.g. the
  //     memref.view op) rather than the last transitive consumer, placing
  //     hip.free before ops that still read/write through the view.
  //
  // Both are use-after-free bugs triggered in the pool-allocs -> lower-allocs
  // pipeline, where the single pool hip.alloc has memref.view aliases that
  // are returned from the function.
  BufferViewFlowAnalysis aliasAnalysis(funcOp);

  // Collect returned values AFTER all replacements so the set contains
  // the new hip.alloc results (not the erased memref.alloc results).
  DenseSet<Value> returnedValues;
  funcOp.walk([&](func::ReturnOp ret) {
    for (Value v : ret.getOperands())
      returnedValues.insert(v);
  });

  Block &entry = funcOp.getBody().front();

  // ---- Phase 3: Convert memref.dealloc -> hip.free ----------------------
  //
  // Explicit deallocs that trace back to a hip.alloc are converted first.
  // traceToHipAlloc walks through ViewLikeOpInterface ops to find the root.
  DenseSet<Value> deallocated;
  SmallVector<memref::DeallocOp> deallocs;
  funcOp.walk([&](memref::DeallocOp op) { deallocs.push_back(op); });
  for (memref::DeallocOp deallocOp : deallocs) {
    Value memref = deallocOp.getMemref();
    AllocOp rootAlloc = traceToHipAlloc(memref);
    if (!rootAlloc)
      continue;

    builder.setInsertionPoint(deallocOp);
    FreeOp::create(builder, deallocOp.getLoc(), ctx, memref);
    deallocated.insert(rootAlloc.getResult());
    deallocOp.erase();
    ++NumDeallocsConverted;
  }

  // ---- Phase 4: Insert hip.free for allocs without explicit dealloc -----
  //
  // Uses BufferViewFlowAnalysis to find all downstream aliases (views,
  // subviews, casts) of each hip.alloc.  hip.free is placed after the
  // last user of any alias, ensuring all transitive consumers have
  // completed before the memory is released.
  //
  // An alloc inside a loop or branch region is planned against its own
  // block instead; see planLocalFree.
  DenseSet<Operation *> transferredAllocs;
  SmallVector<std::pair<AllocOp, LocalFreePlan>> nestedPlans;
  for (AllocOp hipAlloc : hipAllocs) {
    if (isAliasInSet(hipAlloc.getResult(), aliasAnalysis, returnedValues))
      continue;
    if (deallocated.contains(hipAlloc.getResult()))
      continue;

    if (hipAlloc->getBlock() != &entry) {
      LocalFreePlan plan =
          planLocalFree(hipAlloc.getResult(), *hipAlloc->getBlock());
      if (plan.transfers || !plan.conditionalTargets.empty())
        transferredAllocs.insert(hipAlloc);
      nestedPlans.push_back({hipAlloc, std::move(plan)});
      continue;
    }

    Operation *lastUser =
        findLastAliasedUser(hipAlloc.getResult(), aliasAnalysis, entry);
    builder.setInsertionPointAfter(lastUser);
    FreeOp::create(builder, hipAlloc.getLoc(), ctx, hipAlloc.getResult());
    ++NumFreesInserted;
    LLVM_DEBUG(llvm::dbgs()
               << "  Inserted hip.free after " << *lastUser << "\n");
  }

  // ---- Phase 5: Free loop-carried and branch-merged buffers -------------
  //
  // A nested alloc that leaves its block through scf.yield / scf.condition
  // hands its buffer to a region argument or an op result. Free those values
  // after their last use in their block. The initializer of a while before
  // argument stays with the enclosing block: the free fires only when the
  // carried pointer is a replacement buffer. Entry allocs are freed in phase
  // 4; a func arg or other terminal is not ours.
  SmallVector<std::pair<Value, LocalFreePlan>> carriedPlans;
  funcOp.walk([&](Operation *op) {
    SmallVector<std::pair<Value, SmallVector<Value>>> carried;
    collectCarriedValues(op, carried);
    for (auto &[value, sources] : carried) {
      if (isAliasInSet(value, aliasAnalysis, returnedValues))
        continue;
      Block *block = isa<BlockArgument>(value)
                         ? cast<BlockArgument>(value).getOwner()
                         : value.getDefiningOp()->getBlock();
      scf::WhileOp whileOp;
      auto beforeArg = dyn_cast<BlockArgument>(value);
      bool isBeforeArg = false;
      if (beforeArg) {
        whileOp = dyn_cast<scf::WhileOp>(beforeArg.getOwner()->getParentOp());
        isBeforeArg =
            whileOp && beforeArg.getOwner() == whileOp.getBeforeBody();
      }
      if (isBeforeArg) {
        unsigned index = beforeArg.getArgNumber();
        Value init = whileOp.getInits()[index];
        Value backedge = whileOp.getYieldOp().getOperand(index);
        // Drop the initializer's origins so a replacement buffer allocated
        // inside the loop still counts as owned by the backedge.
        auto introduced = aliasAnalysis.resolveReverse(backedge);
        for (Value origin : aliasAnalysis.resolveReverse(init))
          introduced.erase(origin);
        bool ownsReplacement = llvm::any_of(introduced, [&](Value origin) {
          return isTransferredAlloc(origin, transferredAllocs);
        });
        if (!ownsReplacement)
          continue;
        LocalFreePlan plan = planLocalFree(value, *block);
        if (init != value && !llvm::is_contained(plan.preserved, init))
          plan.preserved.push_back(init);
        carriedPlans.push_back({value, std::move(plan)});
        continue;
      }
      bool ownsTransferred = llvm::any_of(sources, [&](Value source) {
        return llvm::any_of(
            aliasAnalysis.resolveReverse(source), [&](Value origin) {
              return isTransferredAlloc(origin, transferredAllocs);
            });
      });
      if (!ownsTransferred)
        continue;
      LocalFreePlan plan = planLocalFree(value, *block);
      // A branch result may be one of the branch sources still used in this
      // block after the branch.
      if (isa<OpResult>(value))
        for (Value source : sources)
          extendLocalFreePlan(source, *block, plan);
      carriedPlans.push_back({value, std::move(plan)});
    }
  });

  bool merged = true;
  while (merged) {
    merged = false;
    for (auto &left : carriedPlans) {
      for (auto &right : carriedPlans) {
        if (&left == &right ||
            !ifResultsMayAlias(left.first, right.first, aliasAnalysis))
          continue;
        LocalFreePlan before = left.second;
        mergePlans(left.second, right.second);
        mergePlans(right.second, before);
        if (left.second.transfers != before.transfers ||
            left.second.after != before.after ||
            left.second.preserved.size() != before.preserved.size() ||
            left.second.conditionalTargets.size() !=
                before.conditionalTargets.size())
          merged = true;
      }
    }
  }
  for (auto &[value, plan] : carriedPlans) {
    Operation *regionOp =
        isa<BlockArgument>(value)
            ? cast<BlockArgument>(value).getOwner()->getParentOp()
            : value.getDefiningOp();
    preserveOutsideOrigins(value, regionOp, aliasAnalysis, transferredAllocs,
                           plan);
  }

  for (auto &[hipAlloc, plan] : nestedPlans)
    emitLocalFree(builder, ctx, hipAlloc.getResult(), plan);
  for (auto &[value, plan] : carriedPlans)
    emitLocalFree(builder, ctx, value, plan);
}

} // namespace
} // namespace hip
} // namespace mlir
