/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {
namespace {

/// Return true if this op can be compiled via the hipDNN graph API.
/// Requires: (1) supported op type and (2) all tensor shapes are static
/// (hipDNN compiles execution plans for concrete dimensions).
static bool isSupportedOp(Operation *op) {
  if (op->getName().getStringRef() != "onnx.Conv")
    return false;
  auto isStaticIfTensor = [](Type t) {
    if (auto ranked = dyn_cast<RankedTensorType>(t))
      return ranked.hasStaticShape();
    return true;
  };
  return llvm::all_of(op->getOperandTypes(), isStaticIfTensor) &&
         llvm::all_of(op->getResultTypes(), isStaticIfTensor);
}

//===----------------------------------------------------------------------===//
// OutlineOnnxToHipDNNPass
//===----------------------------------------------------------------------===//

struct OutlineOnnxToHipDNNPass
    : public PassWrapper<OutlineOnnxToHipDNNPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineOnnxToHipDNNPass)

  StringRef getArgument() const override { return "outline-onnx-to-hipdnn"; }

  StringRef getDescription() const override {
    return "Outline supported ONNX ops into hip.hipdnn_graph_outline regions "
           "for downstream hipDNN graph compilation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override;
};

void OutlineOnnxToHipDNNPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isDeclaration())
      continue;

    SmallVector<Operation *> supported_ops;
    func->walk([&](Operation *op) {
      if (isSupportedOp(op))
        supported_ops.push_back(op);
    });

    for (Operation *onnx_op : supported_ops) {
      OpBuilder builder(onnx_op);
      auto loc = onnx_op->getLoc();

      auto outlineOp = HipDNNGraphOutlineOp::create(
          builder, loc, onnx_op->getResultTypes(), onnx_op->getOperands());

      Block *block = builder.createBlock(&outlineOp.getBody());
      IRMapping mapping;
      for (auto [operand, type] :
           llvm::zip(onnx_op->getOperands(), onnx_op->getOperandTypes())) {
        mapping.map(operand, block->addArgument(type, loc));
      }

      builder.setInsertionPointToEnd(block);
      Operation *cloned = builder.clone(*onnx_op, mapping);
      YieldOp::create(builder, loc, cloned->getResults());

      onnx_op->replaceAllUsesWith(outlineOp->getResults());
      onnx_op->erase();
    }
  }
}

} // namespace

std::unique_ptr<Pass> createOutlineOnnxToHipDNNPass() {
  return std::make_unique<OutlineOnnxToHipDNNPass>();
}

} // namespace hip
} // namespace mlir
