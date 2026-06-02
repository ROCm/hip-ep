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

//===----------------------------------------------------------------------===//
// PrecisionFreeCast: avoid fp16 saturation feeding LayerNormalization.
//
// Mirrors ORT CPU's `InsertCastTransformer` / "PrecisionFreeCast" optimization
// at the ONNX graph level. The motivating bug is the residual Add inside the
// Qwen3.5 vision encoder (canonical case: block 22's `add_3477`): two fp16
// activations whose sum saturates to -Inf at one row position, fed through
// `onnx.Cast(fp16->fp32)` into `onnx.LayerNormalization`. The fp32 input keeps
// the -Inf, LN computes `var = E[x^2] - E[x]^2 = Inf - Inf = NaN`, and the
// affine normalize step poisons the entire row -- which then cascades through
// QKV/attention/MLP/residual into 100% NaN image_features.
//
// Promoting the Add to fp32 eliminates the saturation at the source: each
// fp16->fp32 input cast is exact (no precision loss for finite fp16 values),
// and fp32 has 16+ orders of magnitude headroom over fp16 so the sum cannot
// saturate.
//
// Before (the saturation pattern; %add saturates one row to -Inf):
// ```
//   %add = onnx.Add %a_f16, %b_f16 : (tensor<f16>, tensor<f16>) -> tensor<f16>
//   %up  = onnx.Cast %add          : tensor<f16> -> tensor<f32>
//   %ln  = onnx.LayerNormalization %up, %scale_f32, %bias_f32  // -> NaN
// ```
//
// After (single residual block, no other fp16 user of %add; finite end-to-end):
// ```
//   %a_f32 = onnx.Cast %a_f16      : tensor<f16> -> tensor<f32>
//   %b_f32 = onnx.Cast %b_f16      : tensor<f16> -> tensor<f32>
//   %add   = onnx.Add %a_f32, %b_f32                      // fp32, finite
//   %ln    = onnx.LayerNormalization %add, %scale_f32, %bias_f32   // finite
// ```
//
// After (residual chain — block N's Add result also feeds block N's MLP
// residual Add; promotion propagates because each outer-Add input scan peeks
// through any incoming Cast(fp32->fp16) bridge to use the fp32 source):
// ```
//   %a_f32 = onnx.Cast %a_f16      : tensor<f16> -> tensor<f32>
//   %b_f32 = onnx.Cast %b_f16      : tensor<f16> -> tensor<f32>
//   %add1  = onnx.Add %a_f32, %b_f32                     // fp32
//   %ln1   = onnx.LayerNormalization %add1, ...
//   ... MLP in fp32 ...
//   %mlp_f32 = ... : tensor<f32>
//   %add2  = onnx.Add %add1, %mlp_f32                    // promoted fp32
//   %ln2   = onnx.LayerNormalization %add2, ...
// ```
//
// Why not the kernel-level Inf filter (the previous fix in
// `3rd-party/custom_kernels/hip/layer_norm_kernel.hip`):
//   * The kernel filter only masks the SYMPTOM at LN-compute time and leaves
//     the per-row divergence permanent (canonical Qwen vision: `max|d|≈12.6`
//     vs CPU). The graph rewrite avoids the saturation in the first place,
//     so the result is bit-equivalent to ORT CPU.
//   * The kernel filter executes on every LN call (one extra `isfinite` check
//     per element); the graph rewrite has zero per-inference cost.
//   * The kernel filter does not catch saturation in any other op (e.g. an
//     fp16 `onnx.Mul` saturating before an `onnx.Cast(fp32) -> onnx.Sigmoid`).
//     The graph rewrite is the right layer to extend for those patterns.
static bool isLayerNormLikeOp(mlir::Operation *op) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == "onnx.LayerNormalization";
}

// Promotable producer of a Cast(fp16->fp32) anchor: an op whose result can be
// lifted to fp32 by casting its fp16 inputs and rebuilding it with fp32 result
// type. We require fp16 result so we don't no-op rewrite an already-fp32
// producer.
//
// The set is the union of:
//  * Element-wise reductions/adds where saturation can occur on the residual
//    path: Add, Sum.
//  * GEMM-class ops where saturation occurs on the dot-product accumulator
//    path: Gemm, MatMul. ORT CPU's MLAS uses fp32 internal accumulation for
//    fp16 GEMM and (effectively) keeps the fp32 result on the wire when the
//    downstream consumer is fp32 -- promoting at the ONNX level makes the
//    same effect explicit and reproducible across backends.
static bool isPromotableProducer(mlir::Operation *op) {
  if (!op || op->getNumResults() != 1)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  if (name != "onnx.Add" && name != "onnx.Sum" && name != "onnx.Gemm" &&
      name != "onnx.MatMul")
    return false;
  auto resTy =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  return resTy && resTy.getElementType().isF16();
}

// A user of the anchor Cast is "fp32-pure" if either (a) it's an LN-like op
// (the original motivating consumer), or (b) all of its results are fp32
// tensors -- meaning it does not fold the fp32 value back to fp16 before
// downstream computation. Examples: an already-promoted fp32 Add (from a
// previous PFC iteration on the residual chain), an `onnx.Cast(fp32, fp32)`
// (no-op), an `onnx.Mul(fp32)` etc.
//
// This is the gate that makes PFC v2 safe: if even one user is fp16-result,
// promoting the producer to fp32 would force a fp32->fp16 bridge cast right
// after the producer for that user, which re-introduces the saturation we
// were trying to eliminate.
static bool isFp32PureConsumer(mlir::Operation *user) {
  if (!user)
    return false;
  if (isLayerNormLikeOp(user))
    return true;
  if (user->getNumResults() == 0)
    return false;
  for (mlir::Value r : user->getResults()) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(r.getType());
    if (!t || !t.getElementType().isF32())
      return false;
  }
  return true;
}

// PrecisionFreeCast rewrite, generalized over all promotable producers
// (Add/Sum/Gemm/MatMul) and all fp32-pure consumers (LN or any fp32-result op).
//
// The key invariant: an `onnx.Cast(fp16, fp32)` whose producer is promotable
// AND whose consumers are all fp32-pure can be eliminated by promoting the
// producer to a fp32 result -- the fp16 value at the producer's original
// output is never observed by downstream compute.
//
// Implementation walk:
//   1. Collect every fp16->fp32 Cast whose users are all fp32-pure AND whose
//      defining op is a promotable producer.
//   2. For each anchor (top-down walk order):
//      a. For each producer input: emit a Cast(fp16->fp32) UNLESS the input
//         itself is the result of an `onnx.Cast(fp32->fp16)` -- in that case
//         use the fp32 source directly (peek-through eliminates the
//         round-trip).
//      b. Build a new fp32 result-type op of the same kind as the producer.
//         Copy ALL attributes (Gemm carries alpha/beta/transA/transB, etc.).
//      c. Replace the anchor Cast's uses with the new fp32 result; erase the
//         anchor Cast.
//      d. If the original producer still has remaining fp16 users, emit a
//         Cast(fp32->fp16) bridge for them. The bridge is the canonical
//         place where saturation can still occur in the fp16 path; if its
//         consumer is itself a promotable op feeding another Cast, the next
//         anchor iteration peeks through this bridge and chains the
//         promotion.
//
// Iteration: PFC v1 (Add anchor only) was single-pass because the producers
// it accepted were all rooted at the residual chain, walking top-down. PFC v2
// adds Gemm/MatMul anchors that are typically PRODUCERS for the residual
// Adds (i.e., the GEMM output is the Add's RHS). After PFC v1 promotes the
// Add and inserts `Cast(linear_85 fp16, fp32)` as a new anchor candidate,
// PFC v2 must run again to match it. We do this by looping until the anchor
// set stops growing (at most O(depth) iterations in practice; vision encoder
// has 32 blocks so we'd cap at 64 iterations to be safe).
static bool simplifyPrecisionFreeCastOnce(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> anchors;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "onnx.Cast")
      return;
    if (op->getNumOperands() < 1 || op->getNumResults() < 1)
      return;
    auto inTy =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outTy =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy)
      return;
    if (!inTy.getElementType().isF16() || !outTy.getElementType().isF32())
      return;
    if (op->getResult(0).use_empty())
      return;
    for (mlir::Operation *user : op->getResult(0).getUsers()) {
      if (!isFp32PureConsumer(user))
        return;
    }
    if (!isPromotableProducer(op->getOperand(0).getDefiningOp()))
      return;
    anchors.push_back(op);
  });

  if (anchors.empty())
    return false;

  for (mlir::Operation *cast : anchors) {
    mlir::Operation *prodOp = cast->getOperand(0).getDefiningOp();
    // A previous anchor's processing may have erased our prodOp via the
    // bridge-replacement path below (rare; only happens when two anchors
    // share the same producer). Skip stale anchors.
    if (!prodOp || !isPromotableProducer(prodOp))
      continue;

    mlir::OpBuilder builder(prodOp);
    mlir::Location loc = prodOp->getLoc();
    mlir::Type fp32Ty = builder.getF32Type();

    llvm::SmallVector<mlir::Value> newInputs;
    newInputs.reserve(prodOp->getNumOperands());
    for (mlir::Value in : prodOp->getOperands()) {
      auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(in.getType());
      if (!inTy || inTy.getElementType().isF32()) {
        newInputs.push_back(in);
        continue;
      }
      // Peek through onnx.Cast(fp32 -> fp16): use the fp32 source directly so
      // we don't materialize a fresh fp16->fp32 round-trip cast that would
      // re-saturate at the bridge created by an earlier anchor.
      if (mlir::Operation *defOp = in.getDefiningOp()) {
        if (defOp->getName().getStringRef() == "onnx.Cast" &&
            defOp->getNumOperands() == 1) {
          auto srcTy = mlir::dyn_cast<mlir::RankedTensorType>(
              defOp->getOperand(0).getType());
          if (srcTy && srcTy.getElementType().isF32()) {
            newInputs.push_back(defOp->getOperand(0));
            continue;
          }
        }
      }
      auto castedTy = inTy.clone(fp32Ty);
      mlir::OperationState castState(loc, "onnx.Cast");
      castState.addOperands(in);
      castState.addTypes(castedTy);
      newInputs.push_back(builder.create(castState)->getResult(0));
    }

    auto prodTy =
        mlir::cast<mlir::RankedTensorType>(prodOp->getResult(0).getType());
    auto newProdTy = prodTy.clone(fp32Ty);
    mlir::OperationState newProdState(loc, prodOp->getName().getStringRef());
    newProdState.addOperands(newInputs);
    newProdState.addTypes(newProdTy);
    // Copy ALL attributes: Gemm carries alpha/beta/transA/transB, MatMul has
    // none, Add/Sum may carry onnx_node_name. Generic copy is safest.
    for (mlir::NamedAttribute attr : prodOp->getAttrs())
      newProdState.addAttribute(attr.getName(), attr.getValue());
    mlir::Operation *newProd = builder.create(newProdState);

    cast->getResult(0).replaceAllUsesWith(newProd->getResult(0));
    cast->erase();

    if (!prodOp->getResult(0).use_empty()) {
      mlir::OperationState bridgeState(loc, "onnx.Cast");
      bridgeState.addOperands(newProd->getResult(0));
      bridgeState.addTypes(prodTy);
      mlir::Operation *bridge = builder.create(bridgeState);
      prodOp->getResult(0).replaceAllUsesWith(bridge->getResult(0));
    }
    prodOp->erase();
  }
  return true;
}

static void simplifyPrecisionFreeCast(mlir::func::FuncOp funcOp) {
  // Iterate until no new anchors. Cap is a safety bound -- in practice the
  // pass converges in 1-2 iterations for typical residual chains; deeper
  // chains (vision encoder with 32 blocks) may need a few more rounds as
  // each Add promotion exposes a Gemm anchor and vice versa.
  constexpr int kMaxIters = 64;
  for (int i = 0; i < kMaxIters; ++i) {
    if (!simplifyPrecisionFreeCastOnce(funcOp))
      break;
  }
}

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
      // PrecisionFreeCast must run AFTER simplifyCastLike so any CastLike that
      // would have anchored a promotion has already been canonicalized to
      // onnx.Cast.
      simplifyPrecisionFreeCast(funcOp);
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
