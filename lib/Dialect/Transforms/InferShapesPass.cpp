/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferShapesPass.cpp - Static shape refinement for HIP DPS ops ------===//
//
// Module-level pass that drives `ReifyRankedShapedTypeOpInterface` on every
// HIP-dialect op that implements it, to refine `?` (kDynamic) dims in
// result types into concrete integer dims using the per-op
// `inferContractionShape`-style helpers as the source of truth. Upstream
// ops that happen to carry the interface (e.g. `tensor::EmptyOp`) are
// intentionally skipped: they have their own folders / invariants between
// operand SSA values and result shape that an in-place result-type narrow
// here would desync, and they are not the target of this pass.
//
// DPS-aware
// ---------
// HIP DPS ops require `result_type == outs_operand_type`; refining the op's
// result type without matching the outs operand's producer would produce a
// verifier error.  This pass handles that by also rewriting the producer
// when it is a `tensor.empty` (the only zero-cost refinement we can do
// today: tensor.empty is a pure constructor that supports any rank / dyn-dim
// arrangement, so we rebuild it with the new shape and drop any dynamic-dim
// operands that became static).  Producers we don't know how to refine are
// skipped, leaving the op's result type unchanged at that result index.
//
// Walk order and chained DPS ops
// ------------------------------
// Pre-collect candidates with `module.walk` (post-order). The walk order
// gives us producers before consumers in a straight-line function. We
// process candidates in that order and for each refined result we emit
// exactly one `tensor.cast` that bridges the new (more-static) type back
// to the original type for every non-DPS-init use; DPS-init uses are
// skipped because for a DPS consumer `result_type == outs_operand_type`
// is the spec contract -- if we cast on that edge, refining the consumer
// would fail the "outs producer is tensor.empty" gate downstream.
//
// IMPORTANT: the cast is *the* propagation barrier, not a propagation
// channel. A consumer reading the cast as an `ins` operand sees the OLD
// type, so its own reify runs against the OLD operand type. Refinement
// of the consumer's result therefore depends only on what the consumer's
// own static reify can deduce from operand types it can still see -- it
// does NOT transitively inherit the producer's narrowing across an `ins`
// edge. Refinement is per-op, not whole-chain. This is intentional:
// a transitive narrowing scheme would require either retyping the cast
// (and re-checking every downstream signature) or skipping the cast on
// trust (and breaking ops whose verify pins ins types). Per-op refinement
// is the local, terminating, verifier-safe choice. See
// `refine_chained_matmul` in test/lit/Dialect/hip-infer-shapes.mlir for
// the canonical example.
//
// What this pass does NOT do
// --------------------------
//   * No control-flow analysis.  Result types inside `scf.if` / `scf.while`
//     bodies are not propagated across the region boundary.  Block argument
//     types stay as written.
//   * No element-type refinement.  Only the shape (dim sizes) is touched.
//   * No verifier replay.  Callers should run `mlir-opt --verify-diagnostics`
//     once after this pass to confirm the refined IR still verifies.
//
// Example (matmul, dynamic batch becoming static)
// -----------------------------------------------
//
// Before:
//
//   %empty = tensor.empty(%c2) : tensor<?x4x8xf16>
//   %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4x4xf16>,
//                                       tensor<4x8xf16>)
//                          outs(%empty : tensor<?x4x8xf16>)
//                          -> tensor<?x4x8xf16>
//
// After:
//
//   %empty = tensor.empty() : tensor<2x4x8xf16>
//   %y = hip.matmul(%ctx) ins(%a, %b : tensor<2x4x4xf16>,
//                                       tensor<4x8xf16>)
//                          outs(%empty : tensor<2x4x8xf16>)
//                          -> tensor<2x4x8xf16>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

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
/// Precondition: `cur.size() == reif.size()`. The
/// `ReifyRankedShapedTypeOpInterface` contract guarantees one
/// `OpFoldResult` per dim of the reified result; a mismatch signals a
/// programmer error in the op's `reifyResultShapes` impl.
static bool composeRefinedShape(ArrayRef<int64_t> cur,
                                ArrayRef<OpFoldResult> reif,
                                SmallVectorImpl<int64_t> &out) {
  assert(cur.size() == reif.size() &&
         "rank mismatch between current type and reified shape");
  bool refined = false;
  out.assign(cur.begin(), cur.end());
  for (size_t d : llvm::seq<size_t>(0, cur.size())) {
    if (!ShapedType::isDynamic(cur[d]))
      continue;
    if (std::optional<int64_t> s = getConstantIntValue(reif[d])) {
      out[d] = *s;
      refined = true;
    }
  }
  return refined;
}

/// If `emptyOp`'s rank matches `newShape`, rebuild it with the refined
/// shape and replace all uses; dynamic-dim operands whose corresponding
/// dim has become static are dropped from the operand list.
///
/// Before: `%init = tensor.empty(%d0, %d1) : tensor<?x4x?xf16>`
///         and newShape = [2, 4, 8]
/// After : `%init = tensor.empty() : tensor<2x4x8xf16>`
///
/// Returns failure() and leaves the producer untouched on rank mismatch.
static LogicalResult refineTensorEmptyProducer(RewriterBase &rewriter,
                                               tensor::EmptyOp emptyOp,
                                               ArrayRef<int64_t> newShape) {
  auto curType = emptyOp.getType();
  if (curType.getShape().size() != newShape.size())
    return failure();

  // Rebuild the dyn-dim operand list: keep an operand iff the dim was
  // dynamic before AND remains dynamic after. We never see a "now-dynamic
  // formerly-static" case because composeRefinedShape only narrows.
  SmallVector<Value> newDyn;
  unsigned operandIdx = 0;
  for (size_t d : llvm::seq<size_t>(0, newShape.size())) {
    bool curDynamic = curType.isDynamicDim(d);
    bool newDynamic = ShapedType::isDynamic(newShape[d]);
    if (curDynamic && !newDynamic) {
      ++operandIdx;
      continue;
    }
    if (curDynamic && newDynamic) {
      newDyn.push_back(emptyOp.getDynamicSizes()[operandIdx++]);
      continue;
    }
  }

  // `clone(shape)` preserves the encoding attr; plain
  // `RankedTensorType::get(shape, elemType)` would drop it.
  auto newType = curType.clone(newShape);
  rewriter.setInsertionPoint(emptyOp);
  auto newEmpty =
      tensor::EmptyOp::create(rewriter, emptyOp.getLoc(), newType, newDyn);
  rewriter.replaceOp(emptyOp, newEmpty.getResult());
  ++NumProducersRefined;
  return success();
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

  // For DPS ops, the spec contract is `result_type == outs_operand_type`.
  // The only producer we can rebuild zero-cost today is `tensor.empty`;
  // everything else (function args, results of other DPS ops, etc.) is
  // left alone -- skipping is the verifier-safe choice.
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (dpsOp && resultIdx < dpsOp.getDpsInits().size()) {
    Value outsOperand = dpsOp.getDpsInits()[resultIdx];
    auto emptyProducer = outsOperand.getDefiningOp<tensor::EmptyOp>();
    if (!emptyProducer) {
      LLVM_DEBUG(DBGS() << "skip " << op->getName() << " result #" << resultIdx
                        << ": outs producer is not tensor.empty\n");
      return failure();
    }
    if (failed(refineTensorEmptyProducer(rewriter, emptyProducer, newShape)))
      return failure();
  }

  // In-place type narrow on the op's result. We don't `rewriter.replaceOp`
  // here because that would invalidate every existing use's `OpOperand`
  // pointer before we get to insert the cast. Type-only mutation followed
  // by selective cast insertion on the use edges keeps the op (and its
  // operand list) in place — there's no need for clone+replace because
  // we are not changing operands.
  Value result = op->getResult(resultIdx);
  Type oldType = result.getType();
  result.setType(curType.clone(newShape));
  ++NumResultsRefined;

  // Snapshot uses before mutation; iterating `result.getUses()` while
  // simultaneously rewriting use edges is undefined.
  SmallVector<OpOperand *> usesToCast;
  for (OpOperand &use : result.getUses()) {
    // DPS-init uses on the consumer are intentionally NOT cast: for a
    // DPS consumer, `result_type == outs_operand_type` is the spec, so
    // a cast on that edge would force us to also retype the consumer's
    // result, fighting the same gate this pass enforces in the
    // producer-refinement step above.
    if (auto userDps = dyn_cast<DestinationStyleOpInterface>(use.getOwner())) {
      if (userDps.isDpsInit(&use))
        continue;
    }
    usesToCast.push_back(&use);
  }
  if (!usesToCast.empty()) {
    rewriter.setInsertionPointAfter(op);
    auto cast = tensor::CastOp::create(rewriter, op->getLoc(), oldType, result);
    for (OpOperand *use : usesToCast)
      use->set(cast.getResult());
  }
  return success();
}

struct InferShapesPass : public impl::InferShapesPassBase<InferShapesPass> {
  // The TableGen `let dependentDialects = [...]` covers loading at pass-
  // manager init, but stating it explicitly here too matches the style
  // used elsewhere in this dialect (PromoteStridedHipOperands.cpp,
  // PoolAllocs.cpp) and makes it grep-discoverable from the .cpp.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void InferShapesPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Collect candidates first; mutating the IR mid-walk can re-trigger the
  // walk on rebuilt ops or invalidate iterators. The walk is post-order
  // by default, so consumers come after producers in `ops`; we then iterate
  // forward, which gives producers-before-consumers (the safe order).
  //
  // Restrict to HIP-dialect ops. Upstream ops that also implement
  // `ReifyRankedShapedTypeOpInterface` (e.g. `tensor::EmptyOp`,
  // `tensor::ExtractSliceOp`, `tensor::PadOp`) carry their own per-op
  // invariants between operand SSA values and the result shape that
  // an in-place result-type narrow here can desync — concretely, a
  // `tensor.empty(%c12)` whose constant index operand makes the dim
  // statically reifiable would have its result narrowed to fully static
  // while the now-stale operand stayed in the operand list, tripping the
  // op verifier with `incorrect number of dynamic sizes`. Refining
  // upstream tensor ops is also out of charter for this pass — they are
  // already canonicalised by their own folders before bufferize. Keep
  // the contract tight: HIP DPS ops only.
  SmallVector<ReifyRankedShapedTypeOpInterface> ops;
  module.walk([&](ReifyRankedShapedTypeOpInterface reifyOp) {
    Operation *op = reifyOp.getOperation();
    if (op->getDialect() != op->getContext()->getLoadedDialect<HipDialect>())
      return;
    // memref-mode DPS ops have no tensor results: skip. Their shapes are
    // pinned at bufferization time.
    if (llvm::none_of(op->getResults(), [](Value v) {
          return isa<RankedTensorType>(v.getType());
        }))
      return;
    ops.push_back(reifyOp);
  });

  IRRewriter rewriter(&getContext());
  for (ReifyRankedShapedTypeOpInterface reifyOp : ops) {
    Operation *op = reifyOp.getOperation();
    rewriter.setInsertionPoint(op);

    ReifiedRankedShapedTypeDims reified;
    if (failed(reifyOp.reifyResultShapes(rewriter, reified)))
      continue;
    if (reified.size() != op->getNumResults())
      continue;

    for (auto [resultIdx, dims] : llvm::enumerate(reified))
      (void)refineOneResult(rewriter, reifyOp, resultIdx, dims);
  }
}

} // namespace
} // namespace hip
} // namespace mlir
