/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Absorb the Transpose pair around a channels-first causal convolution.
//
// ONNX Conv is channels-first, so an exporter whose surrounding graph is
// channels-last has to bracket the convolution:
//
//   %t0 = hip.transpose(%x)  perm = [0, 2, 1]     // [B,L,C] -> [B,C,L]
//   %y, %s = hip.causal_conv_with_state(%t0, ...)
//   %t1 = hip.transpose(%y)  perm = [0, 2, 1]     // [B,C,L] -> [B,L,C]
//
// Both transposes are pure data movement in service of a layout the kernel does
// not actually require. On Qwen3.6-35B-A3B each one moves 65 MB per layer and
// the pair together costs more than the convolution between them, across all 30
// linear-attention layers. This pattern rewrites the triple to a single
// convolution with `channels_last` set, which reads and writes [B,L,C]
// directly.
//
// Only input and output are permuted. The carry state is [B, C, k-1] under both
// layouts, so past_state and present_state are forwarded untouched -- the fold
// cannot invalidate a state buffer carried in from a previous step.
//
// The rewrite runs in the tensor domain, on the canonicalizer pass that
// Pipelines.cpp schedules before bufferization. Afterwards there are no results
// to trace a use-chain through, and the guards below fail closed.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

/// The only permutation this fold understands: swap the trailing two axes of a
/// rank-3 tensor, which is what converts between [B,L,C] and [B,C,L].
bool isLastTwoAxisSwap(TransposeOp transpose) {
  ArrayAttr perm = transpose.getPerm();
  if (perm.size() != 3)
    return false;
  const int64_t expected[3] = {0, 2, 1};
  for (int64_t i : llvm::seq<int64_t>(0, 3)) {
    auto value = dyn_cast<IntegerAttr>(perm[i]);
    if (!value || value.getInt() != expected[i])
      return false;
  }
  return true;
}

/// Whether the channels-last custom kernel will actually accept this shape.
///
/// Only that kernel can read (B, L, C), so folding a shape it would refuse
/// turns a working convolution into a hard runtime failure. Mirrors the
/// envelope in wrap_causal_conv_with_state, and reads the kernel width the way
/// CausalConvWithStateLowering does: the product of the weight dims from 2 on.
///
/// The width cap is deliberately well below what the kernel can do. Past k=8
/// the channels-last kernel keeps its sliding window in a per-lane LDS ring, so
/// its real limit is a function of the device's LDS budget -- around k=500 on a
/// 64 KB workgroup -- which is not knowable here. 128 stays inside that even on
/// a 32 KB device while being far past any width a causal convolution actually
/// uses, which keeps the invariant above true without a compile-time guess at
/// the hardware.
bool fastPathAcceptsShape(CausalConvWithStateOp conv) {
  auto weightType = dyn_cast<RankedTensorType>(conv.getWeight().getType());
  if (!weightType || weightType.getRank() < 3)
    return false;
  int64_t kernelSize = 1;
  for (int64_t i : llvm::seq<int64_t>(2, weightType.getRank())) {
    if (weightType.isDynamicDim(i))
      return false; // unknowable here, so leave the Transposes in place
    kernelSize *= weightType.getDimSize(i);
  }
  if (kernelSize < 1 || kernelSize > 128)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(conv.getInput().getType());
  if (!inputType)
    return false;
  unsigned bytes = inputType.getElementType().getIntOrFloatBitWidth() / 8;
  return bytes == 2 || bytes == 4;
}

/// tensor.empty for a DPS init, taking any dynamic extent from `source`. The
/// caller guarantees `source` has the same shape as `type`, so extents align
/// positionally.
Value createEmptyLike(OpBuilder &builder, Location loc, RankedTensorType type,
                      Value source) {
  SmallVector<Value> dynSizes;
  for (int64_t dim : llvm::seq<int64_t>(0, type.getRank()))
    if (type.isDynamicDim(dim))
      dynSizes.push_back(tensor::DimOp::create(builder, loc, source, dim));
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType(), dynSizes);
}

struct FoldTransposePairIntoChannelsLast
    : public OpRewritePattern<CausalConvWithStateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CausalConvWithStateOp conv,
                                PatternRewriter &rewriter) const override {
    if (conv.getChannelsLast())
      return rewriter.notifyMatchFailure(conv, "already channels-last");

    // The channels-last kernel is 1D only, and only the 1D op has a single
    // sequence axis for the permutation to be about.
    if (conv.getNdim() != 1)
      return rewriter.notifyMatchFailure(conv, "channels_last needs ndim=1");

    // Tensor domain only: after bufferization the op has no results and the
    // producer/consumer chain below is not expressible.
    if (conv->getNumResults() != 2)
      return rewriter.notifyMatchFailure(conv, "not in tensor form");

    if (!fastPathAcceptsShape(conv))
      return rewriter.notifyMatchFailure(
          conv, "outside the channels-last kernel's supported envelope");

    auto leading = conv.getInput().getDefiningOp<TransposeOp>();
    if (!leading || !isLastTwoAxisSwap(leading))
      return rewriter.notifyMatchFailure(conv,
                                         "input is not a [0,2,1] transpose");

    // The leading transpose must die with the fold. If its result is read
    // anywhere else the transpose still runs, and rewriting the convolution
    // then buys a layout change without removing the traffic that motivated it.
    if (!leading->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(leading,
                                         "transpose result has other readers");

    // Same on the way out: the convolution's output must feed exactly one
    // transpose back to channels-last, and feed it as data rather than as that
    // transpose's own destination buffer.
    Value convOutput = conv->getResult(0);
    if (!convOutput.hasOneUse())
      return rewriter.notifyMatchFailure(conv,
                                         "output has more than one reader");
    auto trailing = dyn_cast<TransposeOp>(*convOutput.getUsers().begin());
    if (!trailing || !isLastTwoAxisSwap(trailing))
      return rewriter.notifyMatchFailure(
          conv, "output is not consumed by a [0,2,1] transpose");
    if (trailing.getInput() != convOutput)
      return rewriter.notifyMatchFailure(
          trailing, "convolution output is the transpose destination");

    Value nlcInput = leading.getInput();
    auto nlcInputType = dyn_cast<RankedTensorType>(nlcInput.getType());
    auto nlcOutputType =
        dyn_cast<RankedTensorType>(trailing->getResult(0).getType());
    if (!nlcInputType || !nlcOutputType)
      return rewriter.notifyMatchFailure(conv, "unranked operands");

    // Both transposes undo each other, so the convolution's channels-last input
    // and output must agree in shape. Anything else means the perms did not
    // compose the way the checks above assume, and the fold would silently
    // reinterpret the buffer.
    if (nlcInputType.getShape() != nlcOutputType.getShape() ||
        nlcInputType.getElementType() != nlcOutputType.getElementType())
      return rewriter.notifyMatchFailure(
          conv, "transposes do not compose to the identity");

    // A fresh init rather than the trailing transpose's: that one is defined
    // after this convolution, so reusing it here would not dominate its use.
    Value outputInit =
        createEmptyLike(rewriter, conv.getLoc(), nlcOutputType, nlcInput);

    auto folded = CausalConvWithStateOp::create(
        rewriter, conv.getLoc(),
        TypeRange{nlcOutputType, conv->getResult(1).getType()}, conv.getCtx(),
        nlcInput, conv.getWeight(), conv.getBias(), conv.getPastState(),
        outputInit, conv.getPresentState(), conv.getActivation(),
        conv.getNdim(), /*channels_last=*/true);

    rewriter.replaceOp(trailing, folded->getResult(0));
    rewriter.replaceOp(conv, {folded->getResult(0), folded->getResult(1)});
    // Guarded above as single-use, and that use was the convolution.
    if (leading->getResult(0).use_empty())
      rewriter.eraseOp(leading);
    return success();
  }
};

} // namespace

void CausalConvWithStateOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldTransposePairIntoChannelsLast>(context);
}
