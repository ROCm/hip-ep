/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Collapse chained Transposes.
//
// A permutation composed with its own inverse is a copy, and nothing in the IR
// says so -- each Transpose verifies, and the pair survives every pass that
// reasons about ops one at a time:
//
//   %m = hip.transpose(%x) perm = [0, 2, 1, 3]   // [B,H,S,D] -> [B,S,H,D]
//   %y = hip.transpose(%m) perm = [0, 2, 1, 3]   // back to [B,H,S,D]
//
// %y is %x. Exporters emit this when an internal layout normalization brackets
// a region whose ops turn out to need no permutation at all; a backend that
// folds layout attributes into its kernels never pays for it, so it can reach a
// consumer that does. On a 32-layer decoder carrying a [1,32,4096,96] f16 KV
// cache it is 64 such pairs, and each one reads and writes 25 MB per token to
// reproduce its input exactly.
//
// Two patterns, because the general rewrite cannot always fire:
//
//   * Composition needs the first Transpose to die, or a chain whose
//     intermediate has another reader trades one permutation for two.
//   * Collapsing to the identity does not: it replaces the second Transpose
//     with a value that already exists, so the first op's other readers are no
//     reason to keep permuting on this path.
//
// Both run in the tensor domain, on the canonicalizer Pipelines.cpp schedules
// before bufferization -- early enough that the folded buffers are never
// allocated. Afterwards a Transpose has no results, the chain is not
// expressible, and the guards below fail closed.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

/// The `perm` attribute as integers. Fails on anything the verifier would have
/// rejected, so a malformed op is left alone rather than folded on a guess.
bool getPermutation(TransposeOp transpose, SmallVectorImpl<int64_t> &perm) {
  for (Attribute entry : transpose.getPerm()) {
    auto value = dyn_cast<IntegerAttr>(entry);
    if (!value)
      return false;
    perm.push_back(value.getInt());
  }
  return !perm.empty();
}

bool isIdentity(ArrayRef<int64_t> perm) {
  return llvm::all_of(llvm::seq<int64_t>(0, perm.size()),
                      [&](int64_t i) { return perm[i] == i; });
}

/// Compose `secondPerm` after `firstPerm`.
///
/// `perm` reads "output dim i is input dim perm[i]", so for
/// `m = T(x, firstPerm)` and `y = T(m, secondPerm)`:
///
///   y.shape[i] = m.shape[secondPerm[i]] = x.shape[firstPerm[secondPerm[i]]]
///
/// which makes the single equivalent permutation `firstPerm[secondPerm[i]]`.
/// Getting the order backwards still yields a permutation, and still verifies,
/// so the bounds check below is the only thing standing between a typo and
/// silently reinterpreted data.
bool composePermutations(ArrayRef<int64_t> firstPerm,
                         ArrayRef<int64_t> secondPerm,
                         SmallVectorImpl<int64_t> &composed) {
  if (firstPerm.size() != secondPerm.size())
    return false;
  for (int64_t index : secondPerm) {
    if (index < 0 || index >= static_cast<int64_t>(firstPerm.size()))
      return false;
    composed.push_back(firstPerm[index]);
  }
  return true;
}

/// The Transpose that runs before `second`, only if it feeds `second` its data.
/// One reached through the destination operand instead would make the rewrite
/// drop a write.
TransposeOp getFirstTranspose(TransposeOp second) {
  auto first = second.getInput().getDefiningOp<TransposeOp>();
  if (!first || first->getNumResults() != 1)
    return nullptr;
  if (second.getInput() != first->getResult(0))
    return nullptr;
  return first;
}

/// `T(T(x, p), q)` -> `T(x, p o q)`, or `x` when the two undo each other.
struct FoldTransposeChain : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp second,
                                PatternRewriter &rewriter) const override {
    if (second->getNumResults() != 1)
      return rewriter.notifyMatchFailure(second, "not in tensor form");

    TransposeOp first = getFirstTranspose(second);
    if (!first)
      return rewriter.notifyMatchFailure(second,
                                         "input is not a transpose result");

    SmallVector<int64_t> firstPerm, secondPerm, composed;
    if (!getPermutation(first, firstPerm) ||
        !getPermutation(second, secondPerm) ||
        !composePermutations(firstPerm, secondPerm, composed))
      return rewriter.notifyMatchFailure(second, "unreadable permutations");

    if (isIdentity(composed)) {
      Value source = first.getInput();
      // The composition says the shapes agree. Confirm it against the types
      // rather than trust it: a mismatch here means the permutations did not
      // describe what the tensors actually are, and substituting the source
      // would hand the consumer a differently shaped buffer.
      if (source.getType() != second->getResult(0).getType())
        return rewriter.notifyMatchFailure(
            second, "identity permutation over mismatched types");
      rewriter.replaceOp(second, source);
      return success();
    }

    // Rewriting to one Transpose is only a win if the intermediate stops being
    // materialized; otherwise this adds a third permutation beside the two it
    // meant to replace.
    if (!first->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(first,
                                         "intermediate has other readers");

    // The destination this op already owns has the result type and is defined
    // ahead of it, so it dominates its new use.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        second, second->getResultTypes(), second.getCtx(), first.getInput(),
        second.getOutput(), rewriter.getI64ArrayAttr(composed));
    return success();
  }
};

/// A Transpose that permutes nothing.
struct FoldIdentityTranspose : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    if (transpose->getNumResults() != 1)
      return rewriter.notifyMatchFailure(transpose, "not in tensor form");

    SmallVector<int64_t> perm;
    if (!getPermutation(transpose, perm))
      return rewriter.notifyMatchFailure(transpose, "unreadable permutation");
    if (!isIdentity(perm))
      return rewriter.notifyMatchFailure(transpose, "permutes its input");
    if (transpose.getInput().getType() != transpose->getResult(0).getType())
      return rewriter.notifyMatchFailure(
          transpose, "identity permutation over mismatched types");

    rewriter.replaceOp(transpose, transpose.getInput());
    return success();
  }
};

} // namespace

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<FoldTransposeChain, FoldIdentityTranspose>(context);
}
