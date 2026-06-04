/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxInferShapesPass.cpp - Refine onnx.* result types ---------------===//
//
// `--onnx-infer-shapes`: walk every `func.func` in the module, refine
// each `onnx.*` op result type from operand types via the rules library
// (`refineResultTypeFromOperands`), MEET the candidate with the
// existing type to preserve any static dim already in place
// (`meetRankedTypes`), and sync each function's declared return types
// from its terminator's operand types after the walk. SSA propagation
// cascades a refined op's result to its consumers' operands
// automatically, so a single forward walk per func is sufficient.
//
// Why a separate pass: ONNX protobuf shape inference does not recurse
// into `onnx.Loop` body subgraphs. The outliner clones body ops with
// rank-0 placeholder result types even when the v_carry entry args
// carry real ranks. Without a refinement step,
// `--convert-onnx-to-hip`'s rank-aware patterns silently bail and the
// pipeline aborts at one-shot-bufferize. This pass between outline and
// convert mirrors upstream `--tosa-infer-shapes` (LLVM) and onnx-mlir's
// `--shape-inference`.
//
// Pipeline placement (set in `Pipelines.cpp`):
//   simplify-onnx -> hip-add-context-arg -> onnx-loop-outline ->
//   *onnx-infer-shapes* -> convert-onnx-to-hip -> hip-infer-shapes
//
// Before:
//   func.func @loop_body(%c: tensor<?x?x?xf16>) -> tensor<f16> {
//     %add = "onnx.Add"(%c, %c)
//          : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<f16>
//     %tanh = "onnx.Tanh"(%add) : (tensor<f16>) -> tensor<f16>
//     return %tanh : tensor<f16>
//   }
// After:
//   func.func @loop_body(%c: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
//     %add = "onnx.Add"(%c, %c)
//          : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
//     %tanh = "onnx.Tanh"(%add)
//          : (tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
//     return %tanh : tensor<?x?x?xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "RefineOnnxResultType.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx-infer-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

STATISTIC(NumOnnxResultsRefined,
          "Number of onnx.* op result types refined by --onnx-infer-shapes");
STATISTIC(NumFuncSignaturesSynced,
          "Number of func.func declared return types synced from refined "
          "terminator operand types");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_ONNXINFERSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Rewrite `funcOp`'s declared return types from its terminator's
/// operand types. Required after refining body op result types: a
/// refined value flowing into `func.return` updates the operand type
/// (via SSA propagation), but the func's declared return type stays
/// at the pre-refinement type until we rewrite the FunctionType.
/// Without this sync, the func verifier rejects the IR at the next
/// pass boundary.
///
/// Returns true iff the function type was changed.
///
/// Modeled after onnx-mlir's `inferFunctionReturnShapes`
/// (`src/Dialect/ONNX/Transforms/ShapeInference.cpp`).
static bool syncFuncReturnTypes(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return false;
  Operation *terminator = funcOp.getBody().back().getTerminator();
  if (!terminator || !isa<func::ReturnOp>(terminator))
    return false;

  FunctionType oldType = funcOp.getFunctionType();
  if (terminator->getOperandTypes() == oldType.getResults())
    return false;
  funcOp.setType(
      oldType.clone(oldType.getInputs(), terminator->getOperandTypes()));
  return true;
}

/// Forward walk of `funcOp`: for every `onnx.*` op, refine each ranked-
/// tensor result type via the rules library + meet, in place. SSA
/// propagation cascades refinements to downstream consumers, so a
/// single walk converges on monotone input.
///
/// Returns true iff at least one result type was refined.
static bool refineOnnxOpsInFunc(func::FuncOp funcOp) {
  bool changed = false;
  funcOp.walk([&](Operation *op) {
    if (!op->getName().getStringRef().starts_with("onnx."))
      return;
    for (unsigned i : llvm::seq<unsigned>(0, op->getNumResults())) {
      auto current = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!current)
        continue;
      RankedTensorType candidate = refineResultTypeFromOperands(op, i);
      if (!candidate) {
        LLVM_DEBUG(DBGS() << "skip " << op->getName() << " result #" << i
                          << ": no rule\n");
        continue;
      }
      RankedTensorType merged = meetRankedTypes(current, candidate);
      if (!merged) {
        LLVM_DEBUG(DBGS() << "skip " << op->getName() << " result #" << i
                          << ": meet conflict (current=" << current
                          << ", candidate=" << candidate << ")\n");
        continue;
      }
      if (merged == current)
        continue;
      LLVM_DEBUG(DBGS() << "refine " << op->getName() << " result #" << i
                        << ": " << current << " -> " << merged << "\n");
      op->getResult(i).setType(merged);
      ++NumOnnxResultsRefined;
      changed = true;
    }
  });
  return changed;
}

struct OnnxInferShapesPass
    : public impl::OnnxInferShapesPassBase<OnnxInferShapesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    getOperation().walk([](func::FuncOp funcOp) {
      if (funcOp.isDeclaration())
        return;
      if (refineOnnxOpsInFunc(funcOp) && syncFuncReturnTypes(funcOp))
        ++NumFuncSignaturesSynced;
    });
  }
};

} // namespace

} // namespace hip
} // namespace mlir
