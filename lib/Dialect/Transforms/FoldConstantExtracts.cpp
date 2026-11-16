/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FoldConstantExtracts.cpp - Fold compile-time tensor.extract --------===//
//
// A shape vector assembled from scalars and small constants
//
//   %v0 = tensor.from_elements %d : tensor<1xi64>
//   %v1 = tensor.insert_slice %v0 into %empty[0] [1] [1]
//   %v2 = tensor.insert_slice %cst into %v1[1] [2] [1]    // %cst = [0, 512]
//   %e  = tensor.extract %v2[%c2]
//
// is read back element by element when it sizes a tensor (ConstantOfShape,
// Expand, ...). Upstream folds stop at the insert_slice and hip.constant has
// no folder, so every element survives to runtime, where reading it costs a
// device-to-host copy and a stream sync per call. This pass resolves a
// constant-index extract through the insert_slice chain to the scalar it
// reads: an element of a tensor.from_elements, of an inline integer
// hip.constant, or of any other integer constant. Only that scalar is
// materialised; where the constant tensor itself is stored is still decided
// by hip-externalize-constants.
//
// Pipeline placement: immediately before `--hip-infer-shapes`, whose reify
// walk then sees these scalars as constants, and before
// `--hip-externalize-constants`, which may replace the inline hip.constant
// payloads read here.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "hip-fold-constant-extracts"

STATISTIC(NumExtractsFolded,
          "Number of constant-index tensor.extract ops resolved to a scalar "
          "through insert_slice / from_elements / constants");

namespace mlir::hip {

#define GEN_PASS_DEF_FOLDCONSTANTEXTRACTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// The scalar `op` reads, or null when it is not known at compile time. An
/// element of an integer constant is materialised before `op`.
static Value resolveExtract(tensor::ExtractOp op, RewriterBase &rewriter) {
  if (!op.getType().isIntOrIndex())
    return {};
  SmallVector<int64_t> idx;
  for (Value v : op.getIndices()) {
    std::optional<int64_t> c = getConstantIntValue(v);
    if (!c)
      return {};
    idx.push_back(*c);
  }

  // Row-major position of `idx` in `type`, or nullopt when out of bounds.
  auto linearize = [&](ShapedType type) -> std::optional<int64_t> {
    if (!type.hasStaticShape() || type.getRank() != (int64_t)idx.size())
      return std::nullopt;
    int64_t lin = 0;
    for (auto [i, extent] : llvm::enumerate(type.getShape())) {
      if (idx[i] < 0 || idx[i] >= extent)
        return std::nullopt;
      lin = lin * extent + idx[i];
    }
    return lin;
  };

  auto materialize = [&](DenseIntElementsAttr dense) -> Value {
    std::optional<int64_t> lin = linearize(dense.getType());
    if (!lin)
      return {};
    rewriter.setInsertionPoint(op);
    return arith::ConstantOp::create(rewriter, op.getLoc(),
                                     dense.getValues<IntegerAttr>()[*lin]);
  };

  // Every step moves to an operand of the current producer, which
  // dominates it, so the walk terminates.
  Value src = op.getTensor();
  while (auto ins = src.getDefiningOp<tensor::InsertSliceOp>()) {
    if (ins.getSourceType().getRank() != ins.getDestType().getRank())
      return {};
    ArrayRef<int64_t> offs = ins.getStaticOffsets();
    ArrayRef<int64_t> sizes = ins.getStaticSizes();
    ArrayRef<int64_t> strides = ins.getStaticStrides();
    bool inside = true;
    for (size_t i = 0; i < idx.size(); ++i) {
      if (ShapedType::isDynamic(offs[i]) || ShapedType::isDynamic(sizes[i]) ||
          strides[i] != 1)
        return {};
      if (idx[i] < offs[i] || idx[i] >= offs[i] + sizes[i])
        inside = false;
    }
    if (inside) {
      for (size_t i = 0; i < idx.size(); ++i)
        idx[i] -= offs[i];
      src = ins.getSource();
    } else {
      src = ins.getDest();
    }
  }

  if (auto fe = src.getDefiningOp<tensor::FromElementsOp>()) {
    std::optional<int64_t> lin = linearize(fe.getType());
    return lin ? fe.getElements()[*lin] : Value();
  }
  if (auto cst = src.getDefiningOp<hip::ConstantOp>()) {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(cst.getValueAttr());
    return dense ? materialize(dense) : Value();
  }
  DenseIntElementsAttr dense;
  if (matchPattern(src, m_Constant(&dense)))
    return materialize(dense);
  return {};
}

/// Erases `op` and every producer that erasing it leaves without uses. Code
/// that was already dead is left for the canonicalizer.
static void eraseWithDeadProducers(Operation *op, RewriterBase &rewriter) {
  llvm::SetVector<Operation *> worklist;
  worklist.insert(op);
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    if (!isOpTriviallyDead(cur))
      continue;
    for (Value operand : cur->getOperands())
      if (Operation *def = operand.getDefiningOp())
        worklist.insert(def);
    rewriter.eraseOp(cur);
  }
}

struct FoldConstantExtractsPass final
    : public impl::FoldConstantExtractsPassBase<FoldConstantExtractsPass> {
  using Base::Base;
  void runOnOperation() override;
};

void FoldConstantExtractsPass::runOnOperation() {
  SmallVector<tensor::ExtractOp> extracts;
  getOperation().walk([&](tensor::ExtractOp op) { extracts.push_back(op); });

  // Producers precede the extracts that read them in walk order, so nothing
  // erased below is visited later.
  IRRewriter rewriter(&getContext());
  for (tensor::ExtractOp op : extracts) {
    Value scalar = resolveExtract(op, rewriter);
    if (!scalar)
      continue;
    rewriter.replaceAllUsesWith(op.getResult(), scalar);
    eraseWithDeadProducers(op, rewriter);
    ++NumExtractsFolded;
  }
}

} // namespace
} // namespace mlir::hip
