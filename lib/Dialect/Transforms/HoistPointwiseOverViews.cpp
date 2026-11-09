/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "KernelOutline.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

namespace mlir::hip {
#define GEN_PASS_DEF_HOISTPOINTWISEOVERVIEWSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// The extra elements a hoisted op may compute on a root larger than its own
// operand, as a fraction of that operand. In the epilogue the op costs a few
// instructions per element, against a kernel of its own that reads a strided
// view; a crop after a small ConvTranspose (8x8 -> 18x18 grid, 16x16 kept)
// drops about a fifth of the grid.
constexpr int64_t kMaxExtraNumerator = 1;
constexpr int64_t kMaxExtraDenominator = 2;

// The one tensor input of `dps` that is not a constant scalar, when all the
// others are. The op then maps each element of that input on its own, so it
// commutes with any op that only moves elements.
OpOperand *getDataOperand(DestinationStyleOpInterface dps) {
  OpOperand *data = nullptr;
  for (OpOperand *input : dps.getDpsInputOperands()) {
    if (!isa<RankedTensorType>(input->get().getType()) ||
        isConstantScalarExpr(input->get()))
      continue;
    if (data)
      return nullptr;
    data = input;
  }
  return data;
}

// Before:
//   %c = hip.conv ... : tensor<1x32x5x5xf16>
//   %s = tensor.expand_shape %c ... into tensor<1x2x2x8x5x5xf16>
//   %t = hip.transpose(%ctx) ins(%s) {perm = [0, 3, 4, 1, 5, 2]}
//   %g = tensor.collapse_shape %t ... into tensor<1x8x10x10xf16>
//   %r = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
//   %y = hip.leaky_relu(%ctx) ins(%r) : tensor<1x8x8x8xf16>
// After:
//   %c = hip.conv ... : tensor<1x32x5x5xf16>
//   %l = hip.leaky_relu(%ctx) ins(%c) : tensor<1x32x5x5xf16>
//   %s = tensor.expand_shape %l ...
//   %t = hip.transpose(%ctx) ins(%s) ...
//   %g = tensor.collapse_shape %t ...
//   %y = tensor.extract_slice %g[0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
//
// The anchor fusion takes the conv's epilogue only up to the first view it
// cannot absorb, here the crop, so before the rewrite the leaky_relu is a
// kernel of its own reading a strided view. Every view in the chain must have
// one user, since the original chain would otherwise stay live. A root in
// `hoisted` is an op already moved onto an anchor, so a second pointwise op
// follows the first into the epilogue.
Operation *hoist(RewriterBase &rewriter, Operation *op,
                 const DenseSet<Operation *> &hoisted) {
  if (!isFusablePointwise(op) || op->getNumResults() != 1)
    return nullptr;
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!dps || dps.getNumDpsInits() != 1 || !resultType ||
      !resultType.hasStaticShape())
    return nullptr;
  OpOperand *data = getDataOperand(dps);
  if (!data || cast<RankedTensorType>(data->get().getType()).getShape() !=
                   resultType.getShape())
    return nullptr;

  // From the op's input back to the anchor's result.
  SmallVector<ElementMoveOpInterface> chain;
  Value root = data->get();
  while (auto move = root.getDefiningOp<ElementMoveOpInterface>()) {
    auto type = dyn_cast<RankedTensorType>(root.getType());
    if (!move.movesElements() || move->getNumResults() != 1 ||
        !root.hasOneUse() || !type || !type.hasStaticShape())
      break;
    SmallVector<OpOperand *> moved = move.getMovedOperands();
    if (moved.size() != 1)
      break;
    chain.push_back(move);
    root = moved.front()->get();
  }
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  Operation *anchor = root.getDefiningOp();
  if (chain.empty() || !anchor || !root.hasOneUse() ||
      !(isAnchorOp(anchor) || hoisted.contains(anchor)) || !rootType ||
      !rootType.hasStaticShape() ||
      rootType.getNumElements() * kMaxExtraDenominator >
          resultType.getNumElements() *
              (kMaxExtraDenominator + kMaxExtraNumerator))
    return nullptr;

  SmallVector<Value> deadInits{dps.getDpsInits().front()};
  for (ElementMoveOpInterface move : chain)
    if (auto moveDps = dyn_cast<DestinationStyleOpInterface>(*move))
      llvm::append_range(deadInits, moveDps.getDpsInits());

  rewriter.setInsertionPoint(op);
  Type elementType = resultType.getElementType();
  Value init = tensor::EmptyOp::create(rewriter, op->getLoc(),
                                       rootType.getShape(), elementType);
  Operation *hoistedOp = rewriter.clone(*op);
  hoistedOp->setOperand(data->getOperandNumber(), root);
  hoistedOp->setOperand(dps.getDpsInitsMutable()[0].getOperandNumber(), init);
  hoistedOp->getResult(0).setType(rootType.clone(elementType));
  Value current = hoistedOp->getResult(0);
  for (ElementMoveOpInterface move : llvm::reverse(chain))
    current = move.cloneWithElementType(rewriter, current, elementType);

  rewriter.replaceOp(op, current);
  for (ElementMoveOpInterface move : chain)
    rewriter.eraseOp(move);
  for (Value dead : deadInits)
    if (auto empty = dead.getDefiningOp<tensor::EmptyOp>();
        empty && empty->use_empty())
      rewriter.eraseOp(empty);
  return hoistedOp;
}

struct HoistPointwiseOverViewsPass
    : public impl::HoistPointwiseOverViewsPassBase<
          HoistPointwiseOverViewsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    impl::HoistPointwiseOverViewsPassBase<
        HoistPointwiseOverViewsPass>::getDependentDialects(registry);
    registerElementMoveExternalModels(registry);
  }

  void runOnOperation() override {
    // Program order, so a second pointwise op after a hoisted one finds the
    // replayed chain under it.
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (isFusablePointwise(op))
        candidates.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    DenseSet<Operation *> hoisted;
    for (Operation *op : candidates)
      if (Operation *moved = hoist(rewriter, op, hoisted))
        hoisted.insert(moved);
  }
};

} // namespace
} // namespace mlir::hip
