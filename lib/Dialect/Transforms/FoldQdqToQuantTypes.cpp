/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-fold-qdq-to-quant-types: fold the Q/DQ around any quant-capable op into
// its element types, so the op is the same one a float model has, only with
// `!quant.uniform` tensors.
//
//   %x = hip.dequantize_linear(%a_int) ... ; %y = hip.dequantize_linear(%b_int)
//   %s = <op>(%x, %y) outs(%init : f32)       // single use
//   %q = hip.quantize_linear(%s)
// ->
//   %a = hip.storage_cast %a_int : to quant ; %b = hip.storage_cast %b_int ...
//   %s = <op>(%a, %b) outs(%init_q : quant)   // retyped in place
//   %q = hip.storage_cast %s : quant to int
//
// One pattern for every op implementing HipQuantCapableOpInterface (every hip
// DPS op outside the deny-list in HipQuantTypes.cpp); the op is retyped in
// place, so its attributes (transA/B, alpha/beta, ...) and optional operands
// are untouched. Folding is per operand: a float input or a result without a
// Q stays float, so the op may mix quantized and float types (e.g. a
// weight-only matmul). Kernel availability is not checked here.

#include "HipQdqUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipQuantTypes.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::hip {
#define GEN_PASS_DEF_FOLDQDQTOQUANTTYPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Per-tensor quant type of the Q or DQ `qdq`, or null if it has none or if
/// its integer side `storage` and float side `value` differ in shape
/// (hip.storage_cast needs equal shapes).
quant::UniformQuantizedType getFoldableQuantType(Operation *qdq, Value storage,
                                                 Value value) {
  auto type = ::hip::fusion_transform::getHipQdqQuantType(qdq);
  auto storageType = dyn_cast<RankedTensorType>(storage.getType());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !storageType || !valueType ||
      storageType.getShape() != valueType.getShape())
    return {};
  return type;
}

struct FoldQdqIntoQuantCapableOp
    : public OpInterfaceRewritePattern<HipQuantCapableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(HipQuantCapableOpInterface quantOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = quantOp;
    // Only tensor-mode ops that have not been legalized yet.
    if (op->getNumResults() != 1 || op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(op, "not a single-result tensor op");
    if (op->hasAttr("quant_types"))
      return rewriter.notifyMatchFailure(op, "already legalized");

    // Each input that is a per-tensor DQ folds; float inputs stay float, so
    // the op may mix quantized and float types. Integer inputs (indices) are
    // never DQ results. A DQ that is not per-tensor (per-axis, packed int4,
    // runtime scale) leaves the whole op to the Q-specific fusions.
    SmallVector<Value> operands = quantOp.getQuantOperands();
    if (operands.size() < 2)
      return rewriter.notifyMatchFailure(op, "no input");
    Value init = operands.back();
    SmallVector<std::pair<DequantizeLinearOp, quant::UniformQuantizedType>> dqs;
    for (Value input : ArrayRef(operands).drop_back()) {
      auto dq = input.getDefiningOp<DequantizeLinearOp>();
      if (!dq)
        continue;
      auto type = getFoldableQuantType(dq, dq.getInput(), input);
      if (!type)
        return rewriter.notifyMatchFailure(
            op, "DQ input is not per-tensor; left to the Q-specific fusions");
      dqs.emplace_back(dq, type);
    }

    // The output folds if the float result feeds exactly one per-tensor Q;
    // otherwise it stays float.
    Value result = op->getResult(0);
    auto resultType = cast<RankedTensorType>(result.getType());
    QuantizeLinearOp q;
    quant::UniformQuantizedType outQuant;
    if (isa<FloatType>(resultType.getElementType()) && result.hasOneUse()) {
      q = dyn_cast<QuantizeLinearOp>(*result.getUsers().begin());
      if (q && q.getInput() == result)
        outQuant = getFoldableQuantType(q, q->getResult(0), result);
    }
    if (dqs.empty() && !outQuant)
      return rewriter.notifyMatchFailure(op, "no foldable Q or DQ");

    // Build the replacements before touching the op.
    DenseMap<Value, Value> replacement;
    for (auto [dq, type] : dqs) {
      if (replacement.count(dq.getResult(0)))
        continue;
      auto storage = cast<RankedTensorType>(dq.getInput().getType());
      replacement[dq.getResult(0)] = StorageCastOp::create(
          rewriter, dq.getLoc(), storage.clone(type), dq.getInput());
    }
    Value newInit;
    RankedTensorType newResultType;
    if (outQuant) {
      newResultType = resultType.clone(outQuant);
      newInit = ::hip::fusion_transform::buildInitValue(rewriter, newResultType,
                                                        init);
    }

    // DPS: the result type must equal the init type, so both change.
    rewriter.modifyOpInPlace(op, [&] {
      for (OpOperand &use : op->getOpOperands()) {
        if (newInit && use.get() == init)
          use.set(newInit);
        else if (auto it = replacement.find(use.get()); it != replacement.end())
          use.set(it->second);
      }
      if (outQuant)
        result.setType(newResultType);
    });

    if (outQuant) {
      rewriter.setInsertionPointAfter(op);
      auto out = StorageCastOp::create(rewriter, q.getLoc(),
                                       q->getResult(0).getType(), result);
      rewriter.replaceOp(q, out.getResult());
    }
    return success();
  }
};

struct FoldQdqToQuantTypesPass
    : public impl::FoldQdqToQuantTypesPassBase<FoldQdqToQuantTypesPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldQdqIntoQuantCapableOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
