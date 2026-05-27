/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RefineLoopBodyTypes.cpp - Refine onnx.* result types in outlined ---===//
//                              hip.loop body functions
//
// Runs AFTER --onnx-loop-outline and BEFORE --convert-onnx-to-hip.
//
// PROBLEM. --onnx-loop-outline builds the outlined body func's signature
// from the hip.loop op's v_init operand types (the source of truth for
// runtime values flowing in across iterations). The op's BLOCK args from
// the original onnx.Loop region get cloned in via an IRMapping that maps
// them onto the new func entry args, so each cloned op's OPERAND types
// now reflect the refined v_init types. But each cloned op's RESULT
// types are copies of the original op's result types — annotated on the
// ORIGINAL onnx.Loop body, which some ONNX exporters set to a degenerate
// rank-0 sentinel type for accumulator iter_vars (e.g. tensor<f16> when
// the actual accumulator grows via Concat across iterations into
// tensor<?x?x?xf16>). Cloning leaves those stale.
//
// Symptom of the stale annotation surviving into ConvertOnnxToHip:
//   error: onnx.Concat survived convert-onnx-to-hip ...
//   (ConcatDecompose rejects rank-0 result; the structured Reshape /
//    expand-shape lowerings reject malformed operand-vs-result rank.)
//
// FIX. Walk every body func reachable from a hip.loop op. For each
// op in the body, re-derive the result type from current operand types
// using the shared inference helpers in OnnxResultTypeInference.h, plus
// a tiny rule registry built into this file. Set the result type
// unconditionally on change (we are NOT in tightening mode — the cloned
// types are stale, possibly rank-different, and the operand types are
// the source of truth). Iterate to a fixed point inside each body so
// chained refinements propagate (e.g. Concat → Reshape consumer →
// Reshape producer's consumer).
//
// After per-op refinement, propagate the refined yield/return operand
// types into:
//   (a) the body func's signature (result types), and
//   (b) every hip.loop op that names this body — rebuild the hip.loop
//       op with the new result types AND new v_init operand types
//       (v_init types must match result types per the op verifier).
//       v_init source values may need tensor.cast bridges if their type
//       is rank-different from the refined target.
//
// Before:
//
//   func.func @main_graph(...) {
//     ...
//     %v_init = ... : tensor<?x?x?xf16>          // refined source
//     %r = hip.loop(%ctx, %m) iter_args(%v_init : tensor<?x?x?xf16>)
//              body @main_graph_loop_body_n0 : tensor<f16>
//     // ^ note: hip.loop result is tensor<f16>, mismatch with v_init
//   }
//   func.func private @main_graph_loop_body_n0(
//       %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<ui8>,
//       %acc: tensor<?x?x?xf16>, ...) -> tensor<f16> {
//     %15 = hip.multi_head_attention(...) : tensor<?x?x?xf16>
//     %16 = "onnx.Concat"(%acc, %15) {axis=1}
//             : (tensor<?x?x?xf16>, tensor<?x?x?xf16>) -> tensor<f16>
//     return %16 : tensor<f16>
//   }
//
// After:
//
//   func.func @main_graph(...) {
//     %r = hip.loop(...) iter_args(%v_init : tensor<?x?x?xf16>)
//              body @main_graph_loop_body_n0 : tensor<?x?x?xf16>
//   }
//   func.func private @main_graph_loop_body_n0(...) -> tensor<?x?x?xf16> {
//     %15 = hip.multi_head_attention(...) : tensor<?x?x?xf16>
//     %16 = "onnx.Concat"(...) : (...) -> tensor<?x?x?xf16>
//     return %16 : tensor<?x?x?xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/Passes.h"

#include "OnnxResultTypeInference.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

namespace {

//===----------------------------------------------------------------------===//
// Per-op type re-inference
//===----------------------------------------------------------------------===//

/// Re-derive an op's result type from CURRENT operand types. Returns
/// nullptr if no rule applies (caller leaves the existing type alone).
///
/// Unlike `InferOnnxShapes::inferOpType`, this caller is NOT in tightening
/// mode — it overrides the existing result type unconditionally (the
/// cloned-body result type is stale and may be rank-different from
/// what the operand types now imply). Each helper here therefore returns
/// the BEST type it can derive from operands, regardless of the existing
/// result type.
static Type reinferOpResultType(Operation *op) {
  StringRef name = op->getName().getStringRef();

  // Single-result onnx.* ops only. Anything else (multi-result, non-onnx)
  // is left untouched.
  if (!name.starts_with("onnx.") || op->getNumResults() != 1)
    return nullptr;

  // Element-type-preserving unary same-shape (Tanh / Softmax / LayerNorm
  // / Sqrt / Gelu / Sigmoid / Neg / Erf / Identity / Relu / Cos / Sin /
  // Floor / Ceil / Abs / Exp / Log / Reciprocal): output type == input
  // type modulo element type from input.
  static constexpr llvm::StringLiteral kUnary[] = {
      "onnx.Tanh", "onnx.Softmax",  "onnx.LayerNormalization",
      "onnx.Sqrt", "onnx.Gelu",     "onnx.Sigmoid",
      "onnx.Neg",  "onnx.Erf",      "onnx.Identity",
      "onnx.Relu", "onnx.Cos",      "onnx.Sin",
      "onnx.Floor", "onnx.Ceil",    "onnx.Abs",
      "onnx.Exp",  "onnx.Log",      "onnx.Reciprocal",
  };
  if (llvm::is_contained(kUnary, name) && op->getNumOperands() >= 1) {
    auto t = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    return inferUnarySameShapeResultType(t);
  }

  // Element-type-preserving binary broadcast (Add / Sub / Mul / Div /
  // Pow / Min / Max / Mod): numpy-style right-aligned broadcast.
  static constexpr llvm::StringLiteral kBinaryBroadcast[] = {
      "onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div",
      "onnx.Pow", "onnx.Min", "onnx.Max", "onnx.Mod",
  };
  if (llvm::is_contained(kBinaryBroadcast, name) &&
      op->getNumOperands() == 2) {
    auto l = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto r = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    return inferBinaryBroadcastResultType(l, r);
  }

  if (name == "onnx.Concat") {
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr)
      return nullptr;
    return inferConcatResultType(op->getOperands(), axisAttr.getSInt());
  }

  if (name == "onnx.Transpose") {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return nullptr;
    llvm::SmallVector<int64_t> perm;
    if (auto permAttr = op->getAttrOfType<ArrayAttr>("perm")) {
      for (Attribute a : permAttr)
        if (auto i = dyn_cast<IntegerAttr>(a))
          perm.push_back(i.getInt());
    } else {
      // Default perm = reverse.
      for (int64_t i = inputType.getRank() - 1; i >= 0; --i)
        perm.push_back(i);
    }
    return inferTransposeResultType(inputType, perm);
  }

  if (name == "onnx.MatMul" && op->getNumOperands() == 2) {
    auto l = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto r = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    return inferMatMulResultType(l, r);
  }

  if (name == "onnx.Cast") {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return nullptr;
    auto toAttr = op->getAttrOfType<TypeAttr>("to");
    if (!toAttr)
      return nullptr;
    return inferCastResultType(inputType, toAttr.getValue());
  }

  if (name == "onnx.Reshape" && op->getNumOperands() == 2) {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return nullptr;
    // Preserve declared output rank (the shape operand carries values
    // we can't easily re-derive here; the caller relies on the original
    // annotation getting the rank right even if the dim sizes are
    // stale). When the original result type was rank-0 nonsense, the
    // helper returns a placeholder which we then take.
    auto curResult = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    int64_t outRank = curResult ? curResult.getRank() : 0;
    int64_t allowzero = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("allowzero"))
      allowzero = a.getSInt();
    return inferReshapeResultType(inputType, op->getOperand(1), outRank,
                                  allowzero);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Body-func refinement (fixed point)
//===----------------------------------------------------------------------===//

/// Walk `bodyFn` in forward (topological for SSACFG-only bodies) order,
/// re-inferring each onnx.* op's result type from its operands. Iterate
/// until quiescence (no result type changes in a round). Returns true
/// iff at least one op result type was updated.
static bool refineBodyOps(func::FuncOp bodyFn) {
  // Bound the fixed-point iteration. In practice 2 rounds is enough
  // (round 0 mutates, round 1 confirms quiescence) for the body shapes
  // we see today; the cap is a defensive guard against a pathological
  // pattern producing oscillating updates.
  constexpr int kMaxRounds = 8;
  bool anyChanged = false;
  for (int round = 0; round < kMaxRounds; ++round) {
    bool changedThisRound = false;
    bodyFn.walk([&](Operation *op) {
      Type newType = reinferOpResultType(op);
      if (!newType)
        return;
      Type oldType = op->getResult(0).getType();
      if (newType == oldType)
        return;
      op->getResult(0).setType(newType);
      changedThisRound = true;
    });
    if (!changedThisRound)
      break;
    anyChanged = true;
  }
  return anyChanged;
}

/// Update `bodyFn`'s function signature so its result types match the
/// (now-refined) return op operand types. Returns the new result types
/// (potentially unchanged), useful to the caller for propagating into
/// any enclosing `hip.loop` op.
static SmallVector<Type> refineBodySignature(func::FuncOp bodyFn) {
  // Find the single return terminator (onnx.Return or func.return).
  Operation *retOp = nullptr;
  bodyFn.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "onnx.Return" || isa<func::ReturnOp>(op)) {
      retOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!retOp)
    return llvm::to_vector(bodyFn.getFunctionType().getResults());

  FunctionType funcType = bodyFn.getFunctionType();
  SmallVector<Type> newResults =
      llvm::to_vector(funcType.getResults());
  if (retOp->getNumOperands() != newResults.size())
    return newResults;

  bool changed = false;
  for (auto [i, opnd] : llvm::enumerate(retOp->getOperands())) {
    if (newResults[i] != opnd.getType()) {
      newResults[i] = opnd.getType();
      changed = true;
    }
  }
  if (changed)
    bodyFn.setType(FunctionType::get(bodyFn.getContext(),
                                     funcType.getInputs(), newResults));
  return newResults;
}

//===----------------------------------------------------------------------===//
// hip.loop result + v_init propagation
//===----------------------------------------------------------------------===//

/// Rebuild every `hip.loop` op that names `bodyFn` so its result types
/// (and v_init operand types) match the refined body return types.
/// Inserts a `tensor.cast` on any v_init source whose declared type
/// differs from the refined target (cast is a pure type-assertion;
/// downstream bufferize keeps it zero-cost when shapes agree).
///
/// hip.loop's op verifier enforces v_init type == result type, so the
/// two updates must happen together.
static void propagateToCallers(ModuleOp module, func::FuncOp bodyFn,
                               ArrayRef<Type> refinedBodyResults) {
  StringRef bodyName = bodyFn.getName();
  SmallVector<LoopOp> callers;
  module.walk([&](LoopOp loopOp) {
    if (loopOp.getBodyFunc() == bodyName)
      callers.push_back(loopOp);
  });

  for (LoopOp oldLoop : callers) {
    unsigned numLoopCarried =
        static_cast<unsigned>(oldLoop.getNumLoopCarried());
    if (refinedBodyResults.size() < numLoopCarried) {
      // Body return shape doesn't fit hip.loop's loop-carried count —
      // unexpected; leave the op alone so the rest of the pipeline can
      // surface the structural error in a normal way.
      continue;
    }
    ArrayRef<Type> newResultTypes =
        refinedBodyResults.take_back(numLoopCarried);
    if (oldLoop.getResultTypes() == TypeRange(newResultTypes))
      continue;

    OpBuilder builder(oldLoop);
    Location loc = oldLoop.getLoc();

    // Build new v_init operands: cast when the existing type differs
    // from the refined target. tensor.cast lowers to a no-op when the
    // underlying memref descriptors are compatible after bufferize.
    SmallVector<Value> newVInit;
    newVInit.reserve(oldLoop.getVInit().size());
    for (auto [src, target] :
         llvm::zip(oldLoop.getVInit(), newResultTypes)) {
      if (src.getType() == target) {
        newVInit.push_back(src);
        continue;
      }
      if (isa<RankedTensorType>(src.getType()) &&
          isa<RankedTensorType>(target)) {
        Value casted =
            tensor::CastOp::create(builder, loc, target, src).getResult();
        newVInit.push_back(casted);
      } else {
        // Non-tensor type mismatch — leave alone (memref / non-ranked
        // cases are not produced by LoopOutline today; this branch is a
        // defensive fall-through).
        newVInit.push_back(src);
      }
    }

    auto newLoop = LoopOp::create(
        builder, loc,
        /*v_final=*/TypeRange(newResultTypes),
        /*ctx=*/oldLoop.getCtx(),
        /*max_trip_count=*/oldLoop.getMaxTripCount(),
        /*cond_init=*/oldLoop.getCondInit(),
        /*v_init=*/newVInit,
        /*captures=*/oldLoop.getCaptures(),
        /*body_func=*/oldLoop.getBodyFuncAttr(),
        /*num_loop_carried=*/builder.getI32IntegerAttr(numLoopCarried),
        /*cond_is_passthrough=*/
        oldLoop.getCondIsPassthrough() ? builder.getUnitAttr() : nullptr);

    // Move the insertion point AFTER the new loop so any bridging
    // tensor.cast we emit dominates downstream uses of the old loop's
    // results (which we're about to replaceAllUsesWith). Without this,
    // the bridge cast would be inserted before newLoop and reference
    // its own newRes operand from above the def site (`operand #0
    // does not dominate this use`).
    builder.setInsertionPointAfter(newLoop);

    // Bridge result-type rank changes for downstream users that haven't
    // been refined yet (they will be — but their cloned op result types
    // may still be stale in this round; tensor.cast keeps the IR
    // well-typed until they refine).
    for (auto [oldRes, newRes] :
         llvm::zip(oldLoop.getResults(), newLoop.getResults())) {
      if (oldRes.getType() == newRes.getType()) {
        oldRes.replaceAllUsesWith(newRes);
        continue;
      }
      Value bridged =
          tensor::CastOp::create(builder, loc, oldRes.getType(), newRes)
              .getResult();
      oldRes.replaceAllUsesWith(bridged);
    }
    oldLoop.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

class RefineLoopBodyTypesPass
    : public PassWrapper<RefineLoopBodyTypesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineLoopBodyTypesPass)

  StringRef getArgument() const final { return "refine-loop-body-types"; }
  StringRef getDescription() const final {
    return "Re-infer onnx.* result types in outlined hip.loop body "
           "functions and propagate refinements into the function "
           "signature and the enclosing hip.loop op";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<HipDialect, func::FuncDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect every body func reachable from a hip.loop op. A given func
    // may be referenced by multiple loops — once is enough; we refine
    // the body once and propagate to all callers.
    llvm::SmallSetVector<func::FuncOp, 8> bodyFns;
    module.walk([&](LoopOp loopOp) {
      auto fn = module.lookupSymbol<func::FuncOp>(loopOp.getBodyFunc());
      if (fn)
        bodyFns.insert(fn);
    });

    for (func::FuncOp bodyFn : bodyFns) {
      refineBodyOps(bodyFn);
      SmallVector<Type> refined = refineBodySignature(bodyFn);
      propagateToCallers(module, bodyFn, refined);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createRefineLoopBodyTypesPass() {
  return std::make_unique<RefineLoopBodyTypesPass>();
}

} // namespace hip
} // namespace mlir
