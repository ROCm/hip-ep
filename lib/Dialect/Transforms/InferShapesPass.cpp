/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferShapesPass.cpp - Static shape refinement for HIP DPS ops ------===//
//
// Module-level pass: drive `ReifyRankedShapedTypeOpInterface` on every
// HIP-dialect op that implements it, narrow `?` dims in the op's result
// type, rebuild the `tensor.empty` producer of any refined DPS init, and
// emit a `tensor.cast` barrier on every non-DPS-init use to preserve the
// consumer's signature. Refinement is per-op, not whole-chain (the cast
// is the barrier, not a propagation channel). Upstream interface ops
// (e.g. `tensor::EmptyOp`) are intentionally skipped — they have their
// own folders.
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

/// Build a refined `tensor.empty` for `emptyOp` and attach it to the
/// `outs` operand of `dpsOp` at `resultIdx`. Per-edge mutation: only the
/// DPS-init operand is rewired; other users of `emptyOp` (if any) keep
/// their original less-static type. The old `emptyOp` is left in place;
/// canonicalization DCEs it if no live uses remain.
///
/// `replaceOp` would rewire every user of `emptyOp` to the more-static
/// new value, leaking the static type past the per-op refinement
/// boundary that `refineOneResult` enforces with `tensor.cast` on the
/// result side. The cast barrier is for non-DPS-init uses of the OP
/// RESULT; producer-side leaks need a symmetric per-edge primitive.
///
/// Before:
///   %e = tensor.empty(%d) : tensor<?xf16>
///   %y = hip.foo outs(%e : tensor<?xf16>) : tensor<?xf16>
///   <other_use>(%e)
/// After (newShape narrows to tensor<8xf16> for hip.foo's outs):
///   %e = tensor.empty(%d) : tensor<?xf16>          // unchanged
///   <other_use>(%e)                                // unchanged
///   %e2 = tensor.empty() : tensor<8xf16>
///   %y = hip.foo outs(%e2 : tensor<8xf16>) : tensor<8xf16>
static LogicalResult
refineTensorEmptyProducer(RewriterBase &rewriter,
                          DestinationStyleOpInterface dpsOp, unsigned resultIdx,
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
  // Per-edge rewire: only this DPS-init operand sees the new value.
  dpsOp.getDpsInitsMutable()[resultIdx].set(newEmpty.getResult());
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
    if (failed(refineTensorEmptyProducer(rewriter, dpsOp, resultIdx,
                                         emptyProducer, newShape)))
      return failure();
  }

  // In-place type mutation; can't replaceOp here because we still need the
  // existing OpOperand pointers to insert casts on the use edges below.
  Value result = op->getResult(resultIdx);
  Type oldType = result.getType();
  result.setType(curType.clone(newShape));
  ++NumResultsRefined;

  // Snapshot uses before mutation (rewriting while iterating is UB). DPS-init
  // uses skip the cast: result_type == outs_operand_type is the DPS contract,
  // and casting on that edge would re-trigger the same producer-refinement
  // gate on the consumer.
  SmallVector<OpOperand *> usesToCast;
  for (OpOperand &use : result.getUses()) {
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
  // Restated here (also in TableGen) to match other HIP passes' style.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void InferShapesPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Collect first; mutating mid-walk would invalidate iterators. Post-order
  // visits producers before consumers, the safe order for in-place result-
  // type narrowing followed by cast insertion.
  //
  // HIP-dialect ops only. Upstream ops with the same interface (e.g.
  // tensor.empty) carry operand-shape invariants this pass would desync,
  // and they have their own canonicalizers anyway.
  SmallVector<ReifyRankedShapedTypeOpInterface> ops;
  module.walk([&](ReifyRankedShapedTypeOpInterface reifyOp) {
    Operation *op = reifyOp.getOperation();
    if (op->getDialect() != op->getContext()->getLoadedDialect<HipDialect>())
      return;
    // memref-mode DPS ops have no tensor results; their shapes are pinned
    // at bufferization time.
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
