//===- OutlineOnnxToHipDNN.cpp - ONNX -> hipDNN graph outliner -*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Why this pass exists
// --------------------
// The hipDNN graph API compiles execution plans for *complete* op subgraphs
// with statically known shapes; it cannot accept arbitrary ONNX IR directly.
// This pass walks the module and wraps every supported ONNX op (today: only
// `onnx.Conv` with fully static operand and result tensor types) inside a
// `hip.hipdnn_graph_outline` region.  Anything unsupported - dynamic shapes
// or op types not yet vetted with the runtime - is left untouched for the
// standard `--convert-onnx-to-hip` lowering to handle, so the pipeline
// remains complete regardless of how much hipDNN coverage exists.
//
// Why split outlining from compilation
// ------------------------------------
// `OutlineOnnxToHipDNNPass` is intentionally handle-free and CLI-testable
// (`hip-mlir-opt --outline-onnx-to-hipdnn`).  The companion
// `CompileHipDNNGraphsPass` consumes its output, calls into hipDNN with a
// live `hipdnnHandle_t`, and replaces successful regions with
// `hip.hipdnn_graph`. On compilation failure the un-outline routine in
// CompileHipDNNGraphs.cpp restores the original ONNX ops so the graph stays
// compilable end-to-end.
//
//===----------------------------------------------------------------------===//

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

#define GEN_PASS_DEF_OUTLINEONNXTOHIPDNNPASS
#include "hip/Conversion/Passes.h.inc"

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
    : public impl::OutlineOnnxToHipDNNPassBase<OutlineOnnxToHipDNNPass> {
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
} // namespace hip
} // namespace mlir
