/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferLoopBodyShapesPass.cpp - Rank unranked tensors in loop bodies -===//
//
// Module pass that runs AFTER `onnx-loop-outline` and BEFORE
// `convert-onnx-to-hip`.
//
// Problem. The importer emits `tensor<*xT>` (unranked) for values whose
// shape it cannot infer. The canonical case is an outlined `hip.loop`
// body's loop-carried output -- e.g. an `onnx.Concat` that accumulates a
// KV-cache slice along the sequence axis. The ONNX-to-HIP converters
// require ranked result types (`ConcatConversion` bails on unranked), so
// an unranked body output (a) blocks conversion -- the op survives as
// `onnx.*` and later fails bufferization -- and (b) leaves the body
// `func.return` operand type-incompatible with the func signature.
//
// Fix. For every `hip.loop` body func, establish rank using two
// complementary sources of truth, then reconcile the signature:
//
//   1. Seed  -- copy the loop op's `$v_init` operand types onto the body's
//               loop-carried block args (already ranked post-outline; done
//               for parity with onnx-mlir `ONNXLoopOp::inferShapes`, which
//               likewise seeds body args from the loop inputs).
//   2. Infer -- forward-propagate rank onto unranked `onnx.*` results from
//               their (now ranked) operands, to a fixed point.
//   3. Backstop -- any loop-carried body output still unranked is set to the
//               matching `$v_init` type: the ONNX Loop spec mandates
//               body-output type == loop-carried-input type, so this is
//               authoritative even where no forward rule exists.
//   4. Reconcile -- rebuild the body func signature from its (seeded) args
//               and its (now ranked) terminator operands.
//
// Scope. This pass only ESTABLISHES rank so conversion can proceed; all
// `?`-dim narrowing on the resulting HIP-dialect ops remains the job of the
// post-conversion `--hip-infer-shapes`. See
// `docs/design/hip-shape-inference.md`.
//
// Before:
//   func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
//       -> tensor<*xf16> {
//     %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
//     %r = "onnx.Concat"(%acc, %a) {axis = 1} :
//            (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<*xf16>
//     return %r : tensor<*xf16>
//   }
// After:
//   func.func private @loop_body(%ctx, %i, %c, %acc: tensor<1x?x1152xf16>, ...)
//       -> tensor<1x?x1152xf16> {
//     %a = hip.multi_head_attention ... : tensor<?x?x?xf16>
//     %r = "onnx.Concat"(%acc, %a) {axis = 1} :
//            (tensor<1x?x1152xf16>, tensor<?x?x?xf16>) -> tensor<1x?x1152xf16>
//     return %r : tensor<1x?x1152xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-infer-loop-body-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

STATISTIC(NumOnnxResultsRanked,
          "Unranked onnx.* results rank-established by forward inference");
STATISTIC(NumLoopContractRanked,
          "Loop-carried body outputs rank-established from $v_init "
          "(loop-contract backstop)");
STATISTIC(NumBodyFuncsReconciled,
          "Loop-body func signatures reconciled from terminator operand types");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INFERLOOPBODYSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Forward shape rule for `onnx.Concat`: infer a ranked result type from the
/// operand types, or return null when the rule cannot apply (some operand is
/// unranked, ranks or element types disagree, or `axis` is missing / out of
/// range).
///
/// Follows the ONNX Concat spec (and onnx-mlir `ONNXConcatOpShapeHelper`):
/// every operand shares the result's rank and element type; a non-axis dim
/// takes any operand's static extent if one exists (else stays dynamic), and
/// the axis dim is the sum of operand extents (dynamic if any operand is
/// dynamic there). Example: `Concat((1x?x1152, ?x?x?), axis=1)` -> `1x?x1152`.
static Type inferConcatResult(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() != 1)
    return {};
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!axisAttr)
    return {};

  // All operands must be ranked tensors of one common rank and element type.
  SmallVector<RankedTensorType> operands;
  operands.reserve(op->getNumOperands());
  int64_t rank = -1;
  Type elementType;
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
      return {};
    if (rank < 0) {
      rank = t.getRank();
      elementType = t.getElementType();
    } else if (t.getRank() != rank || t.getElementType() != elementType) {
      return {};
    }
    operands.push_back(t);
  }
  if (rank <= 0) // rank-0 tensors have no axis to concatenate.
    return {};

  // ONNX `axis` is signed (`si64`); getSInt sign-extends correctly. Negative
  // axes count from the end.
  int64_t axis = axisAttr.getSInt();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  for (int64_t d : llvm::seq<int64_t>(0, rank)) {
    if (d != axis) {
      // Non-axis dim: adopt the first static extent any operand provides.
      for (RankedTensorType t : operands)
        if (!t.isDynamicDim(d)) {
          shape[d] = t.getDimSize(d);
          break;
        }
      continue;
    }
    // Axis dim: sum of operand extents, dynamic if any contributor is.
    int64_t sum = 0;
    bool allStatic = true;
    for (RankedTensorType t : operands) {
      if (t.isDynamicDim(d)) {
        allStatic = false;
        break;
      }
      sum += t.getDimSize(d);
    }
    shape[d] = allStatic ? sum : ShapedType::kDynamic;
  }
  return RankedTensorType::get(shape, elementType);
}

/// Dispatch a forward shape rule for `op`, but only when its single result is
/// currently unranked. Returns the inferred ranked type, or null if the
/// result is already ranked or no rule covers the op. Register new op rules
/// in the name switch below.
///
/// Why this is keyed on op name (and not the HIP dialect / an interface).
/// Rank must be established BEFORE `convert-onnx-to-hip`: an unranked result
/// blocks the converters (`ConcatConversion` bails on unranked), so the op
/// never reaches the HIP dialect to be refined post-conversion by
/// `--hip-infer-shapes`. At this stage the `onnx.*` ops are still
/// unregistered operations carried by a stub `onnx` dialect (`OnnxStubDialect`
/// in `InitAllPasses.h`, which `allowUnknownOperations` so unranked tensors
/// round-trip) -- this repo matches ONNX by name via the generic `Operation`
/// API rather than depending on onnx-mlir's registered op classes, so there is
/// no op class and no `ShapeInferenceOpInterface` to dispatch on. A name
/// switch is therefore the only handle available, and it mirrors how the
/// converter layer itself is organized (`RewritePattern("onnx.Concat", ...)`).
/// The op-agnostic loop-contract backstop in `inferLoopBodyShapes` is the
/// general safety net for the failure mode (the loop-carried output); these
/// forward rules are the enhancement tier that also ranks *interior* unranked
/// values.
///
/// How to make this generic. Once the build takes a proper dependency on a
/// registered ONNX dialect (onnx-mlir), every op implements
/// `ShapeInferenceOpInterface::inferShapes()`, and this whole switch collapses
/// into a single interface-driven walk -- the onnx-mlir `InferShapesPattern`
/// (`OpInterfaceRewritePattern<ShapeInferenceOpInterface>`) calls each op's
/// own `inferShapes`, so no per-op rule lives here anymore.
static Type inferUnrankedOnnxResult(Operation *op) {
  if (op->getNumResults() != 1 ||
      !isa<UnrankedTensorType>(op->getResult(0).getType()))
    return {};
  if (op->getName().getStringRef() == "onnx.Concat")
    return inferConcatResult(op);
  return {};
}

/// Forward-propagate rank onto unranked `onnx.*` results within `body` until
/// no further result can be ranked.
///
/// A single in-program-order walk suffices for a straight-line body (every
/// operand is defined by an earlier op or a block arg, so a producer is
/// always ranked before its consumer is visited). The bounded fixed point is
/// a guard for any future shape in which a consumer precedes its producer in
/// walk order; it is monotone (unranked -> ranked only) so it always
/// terminates well within the cap.
static void forwardInferUnrankedResults(func::FuncOp body) {
  static constexpr unsigned kMaxIters = 8;
  for (unsigned iter = 0; iter < kMaxIters; ++iter) {
    bool changed = false;
    body.walk([&](Operation *op) {
      if (Type inferred = inferUnrankedOnnxResult(op)) {
        LLVM_DEBUG(DBGS() << "rank " << op->getName() << ": "
                          << op->getResult(0).getType() << " -> " << inferred
                          << "\n");
        op->getResult(0).setType(inferred);
        ++NumOnnxResultsRanked;
        changed = true;
      }
    });
    if (!changed)
      return;
  }
  LLVM_DEBUG(DBGS() << body.getSymName()
                    << ": forward inference hit the iteration cap\n");
}

/// Establish rank inside one outlined `hip.loop` body func, then reconcile its
/// signature. The body's argument and return layouts are fixed by
/// `OnnxLoopOutlinePass`:
///
///   args:    [0] ctx, [1] iter, [2] cond, [3 .. 3+N) v_carry, [3+N ..]
///   captures returns: v_carry occupies [0 .. N) when `cond_is_passthrough`,
///   else
///            [1 .. 1+N) with cond_out at slot 0.
static void inferLoopBodyShapes(func::FuncOp body, hip::LoopOp loopOp) {
  if (body.getBody().empty())
    return;
  Block &entry = body.getBody().front();
  Operation::operand_range vInit = loopOp.getVInit();
  static constexpr unsigned kArgVCarryStart = 3;

  // 1. Seed loop-carried block args from the conservative joined loop result
  //    types. Never re-narrow from a more-static zero/one-trip seed.
  for (auto [i, v] : llvm::enumerate(vInit)) {
    (void)v;
    unsigned argSlot = kArgVCarryStart + i;
    if (argSlot < entry.getNumArguments())
      entry.getArgument(argSlot).setType(loopOp.getResult(i).getType());
  }

  // 2. Forward-infer unranked onnx.* results from their (seeded) operands.
  forwardInferUnrankedResults(body);

  // 3. Loop-contract backstop: a still-unranked loop-carried output must equal
  //    its v_init type per the ONNX Loop spec (covers ops with no forward
  //    rule).
  Operation *terminator = entry.getTerminator();
  unsigned resultVCarryStart = loopOp.getCondIsPassthrough() ? 1u : 2u;
  for (auto [i, v] : llvm::enumerate(vInit)) {
    unsigned slot = resultVCarryStart + i;
    if (slot >= terminator->getNumOperands())
      break;
    Value carried = terminator->getOperand(slot);
    if (auto cast = carried.getDefiningOp<tensor::CastOp>()) {
      Value source = cast.getSource();
      if (isa<UnrankedTensorType>(source.getType())) {
        Type contract = loopOp.getResult(i).getType();
        source.setType(contract);
        ++NumLoopContractRanked;
      }
    }
    if (isa<UnrankedTensorType>(carried.getType())) {
      Type contract = loopOp.getResult(i).getType();
      LLVM_DEBUG(DBGS() << "backstop return slot " << slot << ": "
                        << carried.getType() << " -> " << contract << "\n");
      carried.setType(contract);
      ++NumLoopContractRanked;
    }
  }

  // 4. Reconcile the signature so func.return matches the declared result
  //    types. Inputs follow the (seeded) entry block args; results follow the
  //    (now ranked) terminator operands.
  FunctionType reconciled = body.getFunctionType().clone(
      entry.getArgumentTypes(), terminator->getOperandTypes());
  if (reconciled != body.getFunctionType()) {
    body.setType(reconciled);
    ++NumBodyFuncsReconciled;
  }
}

struct InferLoopBodyShapesPass
    : public impl::InferLoopBodyShapesPassBase<InferLoopBodyShapesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, func::FuncDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](hip::LoopOp loopOp) {
      if (auto body =
              module.lookupSymbol<func::FuncOp>(loopOp.getBodyFuncAttr()))
        inferLoopBodyShapes(body, loopOp);
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
