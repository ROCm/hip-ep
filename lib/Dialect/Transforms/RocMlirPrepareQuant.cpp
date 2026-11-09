/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "DecomposeMatMulNBits.h"
#include "QLinearPlan.h"
#include "RebiasUnsignedQuantize.h"
#include "SinkQuantize.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::hip {

#define GEN_PASS_DEF_ROCMLIRPREPAREQUANTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// acc = quant(x, w') - quant(x, zw'), the integer sum of a QLinear op with
// asymmetric weights before its requantize epilogue. Each integer op becomes
// a rocMLIR kernel of its own; the subtraction joins one of their epilogues.
template <typename BuildIntegerOp>
Value emitAsymmetricAccumulator(RewriterBase &rewriter, Location loc, Value ctx,
                                RankedTensorType accType,
                                const AsymmetricQLinearPlan &plan,
                                BuildIntegerOp buildIntegerOp) {
  Value acc = buildIntegerOp(emitConstant(rewriter, loc, plan.weights));
  if (!plan.zeroPointFilter)
    return acc;
  Value windowSums =
      buildIntegerOp(emitConstant(rewriter, loc, plan.zeroPointFilter));
  return SubOp::create(rewriter, loc, TypeRange{accType}, ctx, acc, windowSums,
                       emitTensorEmpty(rewriter, loc, accType))
      .getResult(0);
}

// Rewrites each QLinear conv and matmul whose weights rocMLIR cannot take as
// they are -- ui8, or with a non-zero zero point -- into integer ops and the
// requantize epilogue. The others stay for hip-fuse-rocmlir-anchor, which
// decomposes them itself, or keep their HIP kernel.
void splitAsymmetricQLinear(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<QLinearConvOp, QLinearMatMulOp>(op))
      candidates.push_back(op);
  });
  for (Operation *op : candidates) {
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    if (auto conv = dyn_cast<QLinearConvOp>(op)) {
      std::optional<AsymmetricQLinearPlan> plan =
          planAsymmetricQLinearConv(conv);
      if (!plan)
        continue;
      auto accType = cast<RankedTensorType>(conv.getOutput().getType())
                         .clone(rewriter.getI32Type());
      Value acc = emitAsymmetricAccumulator(
          rewriter, loc, conv.getCtx(), accType, *plan, [&](Value weights) {
            return QuantConvOp::create(rewriter, loc, TypeRange{accType},
                                       conv.getCtx(), conv.getInput(), weights,
                                       emitTensorEmpty(rewriter, loc, accType),
                                       conv.getKernelShape(), conv.getStrides(),
                                       conv.getPads(), conv.getDilations(),
                                       conv.getGroupAttr())
                .getResult(0);
          });
      rewriter.replaceOp(
          conv, emitRequantizeEpilogue(rewriter, loc, conv.getCtx(), acc,
                                       plan->requantize, conv.getOutput()));
      continue;
    }
    auto matmul = cast<QLinearMatMulOp>(op);
    std::optional<AsymmetricQLinearPlan> plan =
        planAsymmetricQLinearMatMul(matmul);
    if (!plan)
      continue;
    auto accType = cast<RankedTensorType>(matmul.getOutput().getType())
                       .clone(rewriter.getI32Type());
    Value acc = emitAsymmetricAccumulator(
        rewriter, loc, matmul.getCtx(), accType, *plan, [&](Value b) {
          return QuantMatmulOp::create(rewriter, loc, TypeRange{accType},
                                       matmul.getCtx(), matmul.getA(), b,
                                       emitTensorEmpty(rewriter, loc, accType))
              .getResult(0);
        });
    rewriter.replaceOp(
        matmul, emitRequantizeEpilogue(rewriter, loc, matmul.getCtx(), acc,
                                       plan->requantize, matmul.getOutput()));
  }
}

struct RocMlirPrepareQuantPass
    : public impl::RocMlirPrepareQuantPassBase<RocMlirPrepareQuantPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    impl::RocMlirPrepareQuantPassBase<
        RocMlirPrepareQuantPass>::getDependentDialects(registry);
    registerElementMoveExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;
    decomposeMatMulNBits(funcOp);
    sinkQuantizeThroughElementMoves(funcOp);
    splitAsymmetricQLinear(funcOp);
    rebiasUnsignedQuantizedTrees(funcOp, isPlannedQLinear);
  }
};

} // namespace
} // namespace mlir::hip
