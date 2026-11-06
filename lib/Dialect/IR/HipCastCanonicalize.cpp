/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Fold a cast round trip through ops that only move elements.
//
// A widening float cast is exact, and an op that only copies or selects
// elements gives the same values on either side of it. So a narrowing cast
// back to the source type undoes the widening one, and the ops between them
// can run on the narrow type:
//
//   cast_f16(f(cast_f32(x))) == f(x)
//
// Before:
//   %w = hip.cast(%ctx) ins(%x : tensor<1x64x72x84xf16>)
//                       outs(%e0 : tensor<1x64x72x84xf32>)
//   %r = hip.resize(%ctx) ins(%w) outs(%e1 : tensor<1x64x144x168xf32>)
//        {mode = 0 : i64}
//   %y = hip.cast(%ctx) ins(%r) outs(%e2 : tensor<1x64x144x168xf16>)
// After:
//   %y = hip.resize(%ctx) ins(%x) outs(%e : tensor<1x64x144x168xf16>)
//        {mode = 0 : i64}
//
// The pattern runs in the tensor domain, before hip-fuse-rocmlir, which would
// otherwise absorb the widening cast into a producer kernel's epilogue.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

bool isStaticTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.hasStaticShape();
}

Type getElementType(Value value) {
  return cast<RankedTensorType>(value.getType()).getElementType();
}

// Every value of `from` is exactly representable in `to`.
bool isExactFloatWidening(Type from, Type to) {
  if (to.isF32())
    return from.isF16() || from.isBF16();
  if (to.isF64())
    return from.isF16() || from.isBF16() || from.isF32();
  return false;
}

// A tensor-form cast with a static shape, or null.
CastOp getTensorCast(Operation *op) {
  auto castOp = dyn_cast_if_present<CastOp>(op);
  if (!castOp || castOp->getNumResults() != 1 ||
      !isStaticTensor(castOp.getInput().getType()) ||
      !isStaticTensor(castOp->getResult(0).getType()))
    return {};
  return castOp;
}

struct FoldCastRoundTripThroughElementMoves : public OpRewritePattern<CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp outer,
                                PatternRewriter &rewriter) const override {
    if (!getTensorCast(outer))
      return rewriter.notifyMatchFailure(outer, "not a static tensor cast");

    // Walk back from the outer cast's input to the widening cast, collecting
    // the element-move ops between them, last first. Each value on the way
    // has one use, so the chain dies with the fold.
    SmallVector<ElementMoveOpInterface> chain;
    Value value = outer.getInput();
    CastOp inner;
    while (true) {
      if (!value.hasOneUse())
        return rewriter.notifyMatchFailure(outer, "value has other readers");
      Operation *def = value.getDefiningOp();
      inner = getTensorCast(def);
      if (inner)
        break;
      auto move = dyn_cast_if_present<ElementMoveOpInterface>(def);
      if (!move || !move.movesElements() || def->getNumResults() != 1 ||
          !isStaticTensor(def->getResult(0).getType()))
        return rewriter.notifyMatchFailure(outer, "not an element-move chain");
      SmallVector<OpOperand *> moved = move.getMovedOperands();
      if (moved.size() != 1)
        return rewriter.notifyMatchFailure(def, "moves more than one operand");
      chain.push_back(move);
      value = moved.front()->get();
    }
    if (chain.empty())
      return rewriter.notifyMatchFailure(outer, "no element-move op between");

    Value source = inner.getInput();
    Type narrow = getElementType(source);
    if (!isExactFloatWidening(narrow, getElementType(inner->getResult(0))))
      return rewriter.notifyMatchFailure(inner, "not an exact float widening");
    if (getElementType(outer->getResult(0)) != narrow)
      return rewriter.notifyMatchFailure(outer, "does not undo the widening");

    // Rebuild each op in front of the original, so its other operands still
    // dominate it.
    Value current = source;
    for (ElementMoveOpInterface move : llvm::reverse(chain)) {
      rewriter.setInsertionPoint(move);
      current = move.cloneWithElementType(rewriter, current, narrow);
    }
    rewriter.replaceOp(outer, current);
    for (ElementMoveOpInterface move : chain)
      rewriter.eraseOp(move);
    rewriter.eraseOp(inner);
    return success();
  }
};

} // namespace

void CastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<FoldCastRoundTripThroughElementMoves>(context);
}
