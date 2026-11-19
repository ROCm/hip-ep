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
// One pattern for every op implementing HipQuantCapableOpInterface; the op is
// retyped in place, so its attributes (transA/B, alpha/beta, ...) and optional
// operands are untouched. Kernel availability is not checked here.

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

struct FoldQdqIntoQuantCapableOp
    : public OpInterfaceRewritePattern<HipQuantCapableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(HipQuantCapableOpInterface quantOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = quantOp;
    // Only tensor-mode ops that have not been legalized yet.
    if (op->getNumResults() != 1 || op->hasAttr("quant_types"))
      return failure();

    // The result feeds exactly one Q with a per-tensor quant type.
    Value result = op->getResult(0);
    if (!result.hasOneUse())
      return failure();
    auto q = dyn_cast<QuantizeLinearOp>(*result.getUsers().begin());
    if (!q || q.getInput() != result)
      return failure();
    auto outQuant = ::hip::fusion_transform::getHipQdqQuantType(q);
    auto resultType = cast<RankedTensorType>(result.getType());
    // hip.storage_cast needs equal shapes, so a shape-refining Q is skipped.
    if (!outQuant ||
        cast<RankedTensorType>(q->getResult(0).getType()).getShape() !=
            resultType.getShape())
      return failure();

    // Every quantized input is a per-tensor DQ: all or none.
    SmallVector<Value> operands = quantOp.getQuantOperands();
    if (operands.size() < 2)
      return failure();
    Value init = operands.back();
    SmallVector<std::pair<DequantizeLinearOp, quant::UniformQuantizedType>> dqs;
    for (Value input : ArrayRef(operands).drop_back()) {
      auto dq = input.getDefiningOp<DequantizeLinearOp>();
      auto type = dq ? ::hip::fusion_transform::getHipQdqQuantType(dq)
                     : quant::UniformQuantizedType();
      auto dqInput = dq ? dyn_cast<RankedTensorType>(dq.getInput().getType())
                        : RankedTensorType();
      // The cast keeps the DQ input's shape, which must be the operand's.
      if (!type || !dqInput ||
          dqInput.getShape() !=
              cast<RankedTensorType>(input.getType()).getShape())
        return failure();
      dqs.emplace_back(dq, type);
    }

    // Build the replacements before touching the op.
    DenseMap<Value, Value> replacement;
    for (auto [dq, type] : dqs) {
      if (replacement.count(dq.getResult(0)))
        continue;
      auto storage = cast<RankedTensorType>(dq.getInput().getType());
      replacement[dq.getResult(0)] = StorageCastOp::create(
          rewriter, dq.getLoc(), storage.clone(type), dq.getInput());
    }
    auto newResultType = resultType.clone(outQuant);
    Value newInit =
        ::hip::fusion_transform::buildInitValue(rewriter, newResultType, init);

    // DPS: the result type must equal the init type, so both change.
    rewriter.modifyOpInPlace(op, [&] {
      for (OpOperand &use : op->getOpOperands()) {
        if (use.get() == init)
          use.set(newInit);
        else if (auto it = replacement.find(use.get()); it != replacement.end())
          use.set(it->second);
      }
      result.setType(newResultType);
    });

    rewriter.setInsertionPointAfter(op);
    auto out = StorageCastOp::create(rewriter, q.getLoc(),
                                     q->getResult(0).getType(), result);
    rewriter.replaceOp(q, out.getResult());
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
