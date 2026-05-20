/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SimplifyOnnx.cpp - Pre-lowering ONNX graph simplifications --------===//
//
// Implements the simplify-onnx pass. See SimplifyOnnxPass in
// include/hip/Dialect/Transforms/Passes.td for the contract.
//
// The pass is pure ONNX-dialect: no HIP-dialect dependency. It is positioned
// at the head of the pipeline (before hip-add-context-arg) so it operates in
// the original ONNX function index space, which is also what makes it
// straightforward to reuse from any other frontend.
//
// Adding a new simplifier
// -----------------------
// 1. Write a `static void simplify<Op>(mlir::func::FuncOp)` helper in the
//    anonymous namespace below. Its job is to rewrite the op locally and
//    drop only its own operand uses; the shared dead-function-argument
//    sweep at the bottom of runOnOperation handles arg-erasure.
// 2. Call it from `SimplifyOnnxPass::runOnOperation` between the
//    live-before-rewrite snapshot and the dead-arg sweep.
// 3. Add a LIT test under test/lit/Conversion/onnx-to-hip/ exercising
//    the new rewrite via `hip-mlir-opt --simplify-onnx`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "simplify-onnx"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SIMPLIFYONNXPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Per-op simplifiers
//===----------------------------------------------------------------------===//

// onnx.CastLike -> onnx.Cast.
//
// CastLike's second operand is a *type donor* whose data is never read; the
// target dtype is fully encoded in the result type. We rewrite to a plain
// onnx.Cast (the shared CastConversion pattern in convert-onnx-to-hip derives
// the target ONNX dtype enum from the result type, so no `to` attribute is
// required here). Identity casts -- where the input and result element types
// already match -- short-circuit by forwarding the input directly.
static void simplifyCastLike(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> castLikeOps;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.CastLike")
      castLikeOps.push_back(op);
  });

  for (mlir::Operation *op : castLikeOps) {
    if (op->getNumOperands() < 2 || op->getNumResults() < 1)
      continue;
    mlir::Value input = op->getOperand(0);
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!resultType)
      continue;

    // Identity short-circuit.
    if (inputType &&
        inputType.getElementType() == resultType.getElementType()) {
      op->getResult(0).replaceAllUsesWith(input);
      op->erase();
      continue;
    }

    mlir::OpBuilder builder(op);
    mlir::OperationState state(op->getLoc(), "onnx.Cast");
    state.addOperands({input});
    state.addTypes({resultType});
    if (auto nodeName = op->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      state.addAttribute("onnx_node_name", nodeName);
    mlir::Operation *castOp = builder.create(state);
    op->getResult(0).replaceAllUsesWith(castOp->getResult(0));
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Shared dead-function-argument sweep
//===----------------------------------------------------------------------===//

// Drop function arguments that were live BEFORE the simplifiers ran and are
// now use-empty (i.e. their last use was eliminated by one of the rewrites
// above). Arguments that were already dead in the input IR are deliberately
// preserved -- generateModuleMetadata's "captures the original signature"
// contract (asserted by test/lit/Pipeline/module-metadata.mlir) requires
// that.
//
// Because this pass runs BEFORE hip-add-context-arg, no `!hip.context`
// argument is ever present here -- the dead-arg sweep doesn't need a
// dialect-specific "skip arg N" guard.
static mlir::LogicalResult
dropArgsKilledBySimplifiers(mlir::func::FuncOp funcOp,
                            llvm::ArrayRef<bool> wasLiveBeforeRewrite) {
  llvm::BitVector argsToErase(funcOp.getNumArguments());
  for (unsigned i : llvm::seq<unsigned>(0u, funcOp.getNumArguments())) {
    if (wasLiveBeforeRewrite[i] && funcOp.getArgument(i).use_empty())
      argsToErase.set(i);
  }
  if (argsToErase.any() && mlir::failed(funcOp.eraseArguments(argsToErase)))
    return funcOp.emitError(
        "failed to drop dead function arguments after simplify-onnx");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SimplifyOnnxPass : public impl::SimplifyOnnxPassBase<SimplifyOnnxPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
      if (funcOp.isDeclaration())
        continue;

      // Snapshot which function arguments were live BEFORE any simplifier
      // ran, so we never silently drop an arg that was already dead in the
      // input IR.
      llvm::SmallVector<bool> wasLiveBeforeRewrite(funcOp.getNumArguments(),
                                                   false);
      for (unsigned i : llvm::seq<unsigned>(0u, funcOp.getNumArguments()))
        wasLiveBeforeRewrite[i] = !funcOp.getArgument(i).use_empty();

      // ===== Per-op simplifiers (add new ones here; one line per op) =====
      simplifyCastLike(funcOp);
      // simplifyEyeLike(funcOp);
      // simplifyRandomNormalLike(funcOp);

      if (mlir::failed(
              dropArgsKilledBySimplifiers(funcOp, wasLiveBeforeRewrite)))
        return signalPassFailure();
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
