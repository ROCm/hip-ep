/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferShapesPass.cpp - Static shape refinement for HIP DPS ops ------===//
//
// Module-level pass: drive `ReifyRankedShapedTypeOpInterface` on every
// HIP-dialect op that implements it, narrow `?` dims in the op's result
// type, rebuild the `tensor.empty` producer of any refined DPS init, and
// emit a `tensor.cast` on every non-DPS-init use to preserve the
// consumer's signature. The cast is a per-op propagation barrier:
// downstream consumers refine locally against the OLD operand type,
// no whole-chain narrowing. Upstream interface ops (e.g. `tensor::EmptyOp`)
// are intentionally skipped — they have their own folders.
//
// The pass runs in two phases:
//   Phase 1: per-func reify walk. Each `func.func` (main_graph + every
//            outlined `hip.loop` body func) has its reify-impl ops
//            collected in post-order and refined via `refineFuncBody`.
//   Phase 2: hip.loop signature catch-up. When Phase 1 narrowed an
//            upstream producer feeding a `hip.loop`'s `$v_init` operand,
//            the v_init operand type updates in place (DPS-init uses
//            skip cast emission), but the loop's own result types and
//            its body func's signature go stale. `refineLoopSignatures`
//            walks every `hip.loop`, syncs (a) body func arg types at
//            the v_carry slots, (b) body func return types at the same
//            slots (so `func.return` operand types match), (c) the
//            loop op's own result types. If body args changed, the
//            body func is re-walked once so body op result types
//            catch up with the now-tighter entry block arg types.
//
// See `docs/design/hip-shape-inference.md` for rationale, layout, and the
// recipe for wiring a new op.
//
// Before:
//   %e = tensor.empty(%c2) : tensor<?x4x8xf16>
//   %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4x4xf16>, tensor<4x8xf16>)
//                         outs(%e : tensor<?x4x8xf16>) -> tensor<?x4x8xf16>
// After:
//   %e = tensor.empty() : tensor<2x4x8xf16>
//   %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4x4xf16>, tensor<4x8xf16>)
//                         outs(%e : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-infer-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

STATISTIC(NumResultsRefined,
          "Number of op result types refined by --hip-infer-shapes");
STATISTIC(NumProducersRefined,
          "Number of tensor.empty producers rebuilt with a more-static shape");
STATISTIC(NumLoopSignaturesRefined,
          "Number of hip.loop op + body func signatures synced from refined "
          "v_init operand types");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INFERSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Compose a refined shape vector from the current type's shape and the
/// reified per-dim `OpFoldResult`s. For each dim, the existing static
/// value is preserved; if the dim is currently `kDynamic` and the reified
/// `OpFoldResult` is a constant integer, the constant replaces it.
/// Returns true iff at least one dim moved from `kDynamic` to static.
///
/// Precondition: the complete reification has passed
/// `validateReifiedResultShapes`, including rank and static-extent checks.
static bool composeRefinedShape(ArrayRef<int64_t> cur,
                                ArrayRef<OpFoldResult> reif,
                                SmallVectorImpl<int64_t> &out) {
  bool refined = false;
  out.assign(cur.begin(), cur.end());
  for (size_t d : llvm::seq<size_t>(0, cur.size())) {
    std::optional<int64_t> reifCst = getConstantIntValue(reif[d]);
    if (!ShapedType::isDynamic(cur[d]))
      continue;
    if (reifCst) {
      out[d] = *reifCst;
      refined = true;
    }
  }
  return refined;
}

/// Validate every result slot before the pass mutates a result or DPS init.
/// A malformed successful reifier is an op-author/compiler defect, so diagnose
/// it and fail in release builds rather than relying on assertions.
static LogicalResult
validateReifiedResultShapes(Operation *op,
                            const ReifiedRankedShapedTypeDims &reified) {
  if (reified.size() != op->getNumResults())
    return op->emitOpError()
           << "--hip-infer-shapes: successful reification returned "
           << reified.size() << " shape vector(s) for " << op->getNumResults()
           << " result(s); expected exactly one vector per result";

  for (auto [resultIdx, dims] : llvm::enumerate(reified)) {
    Type resultType = op->getResult(resultIdx).getType();
    auto shapedType = dyn_cast<ShapedType>(resultType);
    if (!shapedType) {
      if (!dims.empty())
        return op->emitOpError()
               << "--hip-infer-shapes: successful reification described "
                  "non-shaped result #"
               << resultIdx;
      continue;
    }
    if (!shapedType.hasRank())
      return op->emitOpError()
             << "--hip-infer-shapes: successful reification described "
                "unranked result #"
             << resultIdx;

    ArrayRef<int64_t> currentShape = shapedType.getShape();
    if (dims.size() != currentShape.size())
      return op->emitOpError()
             << "--hip-infer-shapes: successful reification returned "
             << dims.size() << " dimension(s) for result #" << resultIdx
             << " of rank " << currentShape.size();

    for (auto [dimIdx, dim] : llvm::enumerate(dims)) {
      if (!dim)
        continue;
      std::optional<int64_t> constant = getConstantIntValue(dim);
      if (!constant)
        continue;
      if (*constant < 0)
        return op->emitOpError()
               << "--hip-infer-shapes: successful reification returned "
                  "negative extent "
               << *constant << " for result #" << resultIdx << ", dimension #"
               << dimIdx;
      int64_t current = currentShape[dimIdx];
      if (!ShapedType::isDynamic(current) && *constant != current)
        return op->emitOpError()
               << "--hip-infer-shapes: successful reification returned extent "
               << *constant << " for result #" << resultIdx << ", dimension #"
               << dimIdx << ", contradicting existing static extent "
               << current;
    }
  }
  return success();
}

/// Rebuild `emptyOp` with the refined shape. Dynamic-dim operands whose
/// corresponding dim became static are dropped.
///
/// Before: %init = tensor.empty(%d0, %d1) : tensor<?x4x?xf16>; newShape=[2,4,8]
/// After : %init = tensor.empty() : tensor<2x4x8xf16>
static LogicalResult refineTensorEmptyProducer(RewriterBase &rewriter,
                                               tensor::EmptyOp emptyOp,
                                               ArrayRef<int64_t> newShape) {
  auto curType = emptyOp.getType();
  if (curType.getShape().size() != newShape.size())
    return failure();

  // For each old-dynamic dim, advance the operand cursor and keep the
  // operand iff the dim is still dynamic. (composeRefinedShape only
  // narrows, so we never see a static->dynamic transition.)
  SmallVector<Value> newDyn;
  unsigned operandIdx = 0;
  for (size_t d : llvm::seq<size_t>(0, newShape.size())) {
    if (!curType.isDynamicDim(d))
      continue;
    if (ShapedType::isDynamic(newShape[d]))
      newDyn.push_back(emptyOp.getDynamicSizes()[operandIdx]);
    ++operandIdx;
  }

  // `clone(shape)` preserves the encoding attr; `RankedTensorType::get`
  // would drop it.
  auto newType = curType.clone(newShape);
  rewriter.setInsertionPoint(emptyOp);
  auto newEmpty =
      tensor::EmptyOp::create(rewriter, emptyOp.getLoc(), newType, newDyn);
  rewriter.replaceOp(emptyOp, newEmpty.getResult());
  ++NumProducersRefined;
  return success();
}

/// Snapshot every non-DPS-init use of `result` that needs a `tensor.cast`
/// barrier when `result` is in-place mutated to `newType`. Two skip
/// rules:
///
///   1. DPS-init uses (`result_type == outs_operand_type` is the DPS
///      contract; casting on that edge would re-trigger the same
///      producer-refinement gate on the consumer).
///   2. `func.return` uses whose parent func's declared return type at
///      that operand slot already matches `newType` — typically because
///      the caller (`syncLoopResultsAndInsertCasts`) updated the body
///      func's signature in lockstep with the inner hip.loop's result
///      type. Inserting a cast there would leave the func.return
///      operand at `oldType` and mismatch the now-tighter declared
///      return type.
///
/// Returns the snapshot for the caller to rewire after mutating
/// `result`'s type.
static SmallVector<OpOperand *> snapshotNonDpsInitUses(Value result,
                                                       Type newType) {
  SmallVector<OpOperand *> usesToCast;
  for (OpOperand &use : result.getUses()) {
    if (auto userDps = dyn_cast<DestinationStyleOpInterface>(use.getOwner())) {
      if (userDps.isDpsInit(&use))
        continue;
    }
    if (auto ret = dyn_cast<func::ReturnOp>(use.getOwner())) {
      if (auto parentFunc = ret->getParentOfType<func::FuncOp>()) {
        unsigned slot = use.getOperandNumber();
        ArrayRef<Type> declared = parentFunc.getFunctionType().getResults();
        if (slot < declared.size() && declared[slot] == newType)
          continue;
      }
    }
    usesToCast.push_back(&use);
  }
  return usesToCast;
}

/// Insert a single `tensor.cast` from `result.getType()` (the new,
/// post-mutation type) to `oldType` immediately after `op`, then rewire
/// every snapshotted use to consume the cast. No-op when `usesToCast` is
/// empty.
static void rewireUsesThroughCast(RewriterBase &rewriter, Operation *op,
                                  Value result, Type oldType,
                                  ArrayRef<OpOperand *> usesToCast) {
  if (usesToCast.empty())
    return;
  rewriter.setInsertionPointAfter(op);
  auto cast = tensor::CastOp::create(rewriter, op->getLoc(), oldType, result);
  for (OpOperand *use : usesToCast)
    use->set(cast.getResult());
}

/// Refine a single result of a single op. Returns success() iff at least
/// one dim was narrowed; failure() means "nothing to do" or "couldn't
/// safely refine here".
static LogicalResult refineOneResult(RewriterBase &rewriter,
                                     ReifyRankedShapedTypeOpInterface reifyOp,
                                     unsigned resultIdx,
                                     ArrayRef<OpFoldResult> reifiedDims) {
  Operation *op = reifyOp.getOperation();
  auto curType = dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!curType)
    return failure();

  SmallVector<int64_t> newShape;
  if (!composeRefinedShape(curType.getShape(), reifiedDims, newShape))
    return failure();

  // DPS contract is `result_type == outs_operand_type`; only tensor.empty
  // outs producers can be rebuilt zero-cost today, so we skip the rest.
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (dpsOp && resultIdx < dpsOp.getDpsInits().size()) {
    Value outsOperand = dpsOp.getDpsInits()[resultIdx];
    auto emptyProducer = outsOperand.getDefiningOp<tensor::EmptyOp>();
    if (!emptyProducer) {
      LLVM_DEBUG(DBGS() << "skip " << op->getName() << " result #" << resultIdx
                        << ": outs producer is not tensor.empty\n");
      return failure();
    }
    // Refining a shared `tensor.empty` would retype every sibling
    // consumer's outs operand without retyping their result, breaking
    // the DPS contract (outs.type == result.type) on the unrefined
    // sibling. No converter aliases empties today; the guard removes
    // that as a correctness precondition for this pass.
    if (!emptyProducer->hasOneUse()) {
      LLVM_DEBUG(DBGS() << "skip " << op->getName() << " result #" << resultIdx
                        << ": outs producer is a shared tensor.empty\n");
      return failure();
    }
    if (failed(refineTensorEmptyProducer(rewriter, emptyProducer, newShape)))
      return failure();
  }

  // In-place type mutation; can't replaceOp here because we still need the
  // existing OpOperand pointers to insert casts on the use edges below.
  Value result = op->getResult(resultIdx);
  Type oldType = result.getType();
  Type newTypeVal = curType.clone(newShape);
  SmallVector<OpOperand *> usesToCast =
      snapshotNonDpsInitUses(result, newTypeVal);
  result.setType(newTypeVal);
  ++NumResultsRefined;
  rewireUsesThroughCast(rewriter, op, result, oldType, usesToCast);
  return success();
}

/// Phase 1 worker: collect every `ReifyRankedShapedTypeOpInterface` op
/// within `funcOp`'s body region in post-order, then refine each of
/// their results. Post-order visits producers before consumers — the
/// safe order for in-place result-type narrowing followed by cast
/// insertion.
///
/// HIP-dialect ops only. Upstream ops with the same interface (e.g.
/// `tensor.empty`) carry operand-shape invariants this pass would
/// desync, and they have their own canonicalizers anyway.
static LogicalResult refineFuncBody(func::FuncOp funcOp, IRRewriter &rewriter) {
  HipDialect *hipDialect = funcOp->getContext()->getLoadedDialect<HipDialect>();
  SmallVector<ReifyRankedShapedTypeOpInterface> ops;
  funcOp.walk([&](ReifyRankedShapedTypeOpInterface reifyOp) {
    Operation *op = reifyOp.getOperation();
    if (op->getDialect() != hipDialect)
      return;
    // memref-mode DPS ops have no tensor results; their shapes are pinned
    // at bufferization time.
    if (llvm::none_of(op->getResults(), [](Value v) {
          return isa<RankedTensorType>(v.getType());
        }))
      return;
    ops.push_back(reifyOp);
  });

  for (ReifyRankedShapedTypeOpInterface reifyOp : ops) {
    Operation *op = reifyOp.getOperation();
    rewriter.setInsertionPoint(op);

    ReifiedRankedShapedTypeDims reified;
    if (failed(reifyOp.reifyResultShapes(rewriter, reified)))
      continue;
    if (failed(validateReifiedResultShapes(op, reified)))
      return failure();

    for (auto [resultIdx, dims] : llvm::enumerate(reified))
      (void)refineOneResult(rewriter, reifyOp, resultIdx, dims);
  }
  return success();
}

/// Sync `bodyFunc`'s `FunctionType` + entry-block-arg types at the
/// v_carry slots from `loopOp`'s `$v_init` operand types. The outlined
/// body func's argument layout is fixed by `OnnxLoopOutlinePass`:
///
///   [0]      ctx        (!hip.context)
///   [1]      iter       (rank-0 i64 tensor)
///   [2]      cond_in    (rank-0 i1 tensor)
///   [3..3+N) v_carry    (loop-carried values; types must match v_init)
///   [3+N..]  captures   (outer-graph values referenced inside the body)
///
/// The body func's declared return types follow the yield sliced by
/// `cond_is_passthrough`: when set, the yield's operand 0 (cond_out) is
/// elided and v_carry returns occupy slots [0..N); when unset they
/// occupy [1..1+N) and the cond_out type sits at slot 0.
///
/// Both arg slots and the corresponding return slots are kept in sync
/// with v_init types — keeping just one side would leave `func.return`
/// type-incompatible with the FunctionType result list and fail
/// verification.
///
/// Returns true iff any signature type was rewritten (arg or result).
static bool syncBodyFuncSignatureFromVInit(func::FuncOp bodyFunc,
                                           hip::LoopOp loopOp) {
  static constexpr unsigned kArgVCarryStart = 3;
  Operation::operand_range vInit = loopOp.getVInit();
  size_t N = vInit.size();
  unsigned resultVCarryStart = loopOp.getCondIsPassthrough() ? 0u : 1u;

  Block &entry = bodyFunc.getBody().front();
  FunctionType oldType = bodyFunc.getFunctionType();
  SmallVector<Type> newInputs(oldType.getInputs().begin(),
                              oldType.getInputs().end());
  SmallVector<Type> newResults(oldType.getResults().begin(),
                               oldType.getResults().end());

  bool changed = false;
  for (auto [i, v] : llvm::enumerate(vInit)) {
    Type vt = v.getType();

    unsigned argSlot = kArgVCarryStart + i;
    if (argSlot < newInputs.size() && newInputs[argSlot] != vt) {
      newInputs[argSlot] = vt;
      entry.getArgument(argSlot).setType(vt);
      changed = true;
    }

    unsigned resSlot = resultVCarryStart + i;
    if (resSlot < newResults.size() && newResults[resSlot] != vt) {
      newResults[resSlot] = vt;
      changed = true;
    }
  }
  if (changed)
    bodyFunc.setType(
        FunctionType::get(bodyFunc.getContext(), newInputs, newResults));
  return changed;
}

/// Sync `loopOp`'s result types from its `$v_init` operand types and
/// emit a `tensor.cast` on every non-DPS-init use of any result whose
/// type changed (so consumer signatures are preserved exactly as
/// `refineOneResult` does for ordinary DPS-op result narrowing).
///
/// hip.loop's verifier requires `result_type[i] == v_init[i].type`
/// (HipOps.td); when Phase 1 has narrowed a v_init producer's result
/// type and the narrowing propagated into the v_init operand (DPS-init
/// uses skip cast emission), the loop op's own result types are stale
/// until this sync runs.
static bool syncLoopResultsAndInsertCasts(IRRewriter &rewriter,
                                          hip::LoopOp loopOp) {
  Operation::operand_range vInit = loopOp.getVInit();
  bool changed = false;
  for (auto [i, v] : llvm::enumerate(vInit)) {
    if (i >= loopOp->getNumResults())
      break;
    Value result = loopOp->getResult(i);
    Type newType = v.getType();
    if (result.getType() == newType)
      continue;
    Type oldType = result.getType();
    SmallVector<OpOperand *> usesToCast =
        snapshotNonDpsInitUses(result, newType);
    result.setType(newType);
    rewireUsesThroughCast(rewriter, loopOp.getOperation(), result, oldType,
                          usesToCast);
    changed = true;
  }
  return changed;
}

/// Phase 2 worker: walk every `hip.loop` in `module` until fixed point,
/// syncing its result types and its outlined body func's signature
/// from `$v_init`, and re-walking the body func once per change so
/// body op result types catch up with the now-tighter entry block args.
///
/// `module.walk` is depth-agnostic, so the helper is safe on hand-written
/// test cases nesting one `hip.loop` inside another's body func (the
/// production outline pass forbids nested `onnx.Loop` upstream, so this
/// is purely a robustness contract).
///
/// Why iterate. Body op refinement is monotone (`?` dims only narrow),
/// but the order in which sibling `hip.loop` ops show up in
/// `module.walk` depends on document order — nested cases where the
/// outer loop's body func contains the inner loop won't converge in a
/// single pass if the inner is visited first (its v_init operand is a
/// body block arg whose type doesn't tighten until the outer pass).
/// The loop terminates fast in practice (≤ 2 iters in production, ≤ a
/// handful for synthetic deep nesting); a hard cap guards against any
/// bug-induced non-monotone behavior.
///
/// Before:
///   %0 = hip.matmul ... -> tensor<128xf32>      // narrowed by Phase 1
///   %r = hip.loop(...) iter_args(%0 : tensor<128xf32>) -> tensor<?xf32>
///        body @loop_body
///   func.func @loop_body(%ctx, %iter, %cond, %carry: tensor<?xf32>, ...)
///                       -> (i1, tensor<?xf32>) { ... }
///
/// After:
///   %0 = hip.matmul ... -> tensor<128xf32>
///   %r = hip.loop(...) iter_args(%0 : tensor<128xf32>) -> tensor<128xf32>
///        body @loop_body
///   // tensor.cast from tensor<128xf32> back to tensor<?xf32> on every
///   // non-DPS-init use of %r so consumer signatures are preserved.
///   func.func @loop_body(%ctx, %iter, %cond, %carry: tensor<128xf32>, ...)
///                       -> (i1, tensor<128xf32>) { ... }
static LogicalResult refineLoopSignatures(ModuleOp module,
                                          IRRewriter &rewriter) {
  // Hard cap. Each iteration narrows a finite type lattice across a
  // finite set of hip.loop ops — bounded; the cap is a paranoia guard
  // against a buggy reify impl that returns a wider type than the
  // current one (which would non-monotonically loop).
  static constexpr unsigned kMaxIters = 16;
  for (unsigned iter = 0; iter < kMaxIters; ++iter) {
    bool changed = false;
    WalkResult walkResult = module.walk([&](hip::LoopOp loopOp) -> WalkResult {
      auto bodyFunc =
          module.lookupSymbol<func::FuncOp>(loopOp.getBodyFuncAttr());
      if (!bodyFunc || bodyFunc.getBody().empty())
        return WalkResult::advance();

      bool bodyChanged = syncBodyFuncSignatureFromVInit(bodyFunc, loopOp);
      bool resultsChanged = syncLoopResultsAndInsertCasts(rewriter, loopOp);
      if (bodyChanged || resultsChanged) {
        ++NumLoopSignaturesRefined;
        changed = true;
      }

      // Refining body args makes prior cloned-from-ONNX body op operand
      // types disagree with the now-tighter entry block arg types; a
      // re-walk of this body func via `refineFuncBody` lets each
      // body op's `reifyResultShapes` catch up.
      if (bodyChanged && failed(refineFuncBody(bodyFunc, rewriter)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
    if (!changed)
      return success();
  }
  module.emitWarning() << "--hip-infer-shapes: hip.loop signature refinement "
                          "did not converge after "
                       << kMaxIters << " iterations; bailing";
  return success();
}

struct InferShapesPass : public impl::InferShapesPassBase<InferShapesPass> {
  // Restated here (also in TableGen) to match other HIP passes' style.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, tensor::TensorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void InferShapesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(&getContext());

  // Phase 1: per-func reify walk. Refines all hip-dialect Hip_DpsOp
  // result types in-place; cast insertion preserves consumer
  // signatures except on DPS-init uses (which propagate refinement
  // directly — `hip.loop`'s `$v_init` operands are DPS-init by
  // `LoopOp::getDpsInitsMutable`, see HipDialect.cpp).
  WalkResult walkResult = module.walk([&](func::FuncOp funcOp) -> WalkResult {
    return succeeded(refineFuncBody(funcOp, rewriter))
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // Phase 2: hip.loop signature catch-up. Sync each loop's result
  // types and its body func's signature from the (potentially
  // Phase-1-refined) v_init operand types, then re-walk the body func
  // so body op result types catch up with the tighter entry-block
  // arg types.
  if (failed(refineLoopSignatures(module, rewriter)))
    signalPassFailure();
}

} // namespace
} // namespace hip
} // namespace mlir
