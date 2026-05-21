/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InferOnnxShapes.cpp - Forward shape inference for onnx.* ops -------===//
//
// Single forward MLIR pass that tightens every onnx.* op's output type to
// the maximum static info derivable from the (still-symbolic) graph inputs.
// Runs in a fixed-point loop with the other pre-lowering rewrites
// (FastGeluFusion, ProjectorOpsRewrites) at the head of
// `convert-onnx-to-hip`; downstream passes (OneShotBufferize, PoolAllocs,
// MaterializeHostScalars) then see refined types and the LLM-shape /
// ViT-shape distinction collapses.
//
// Why this matters
// ----------------
// HuggingFace ONNX exports leave intermediate `RankedTensorType`s loose
// (`<?x?x?x?>`) even when the shape operand uniquely determines them — see
// the canonical ViT attention reshape:
//
//     shape = onnx.Concat(onnx.Slice(onnx.Shape(x), [0,2]), [H], [D])
//     y     = onnx.Reshape(x : <?xSxH*D>, shape) : <?x?x?x?>
//
// MLIR's `getReassociationIndicesForReshape` can't pick a unique reassoc
// when the output is all-dynamic, so without refinement these Reshapes
// trip OneShotBufferize with `error: op was not bufferized` (silent CPU
// fallback at the EP level).
//
// Design
// ------
//   * Forward, single-direction walk (topological by SSA def-before-use).
//   * Per-op handler functions delegate to pure rules in
//     `OnnxResultTypeInference.{h,cpp}`. Handlers are thin wrappers that
//     apply `isStrictlyTighter` so refinement never widens, never changes
//     element type, never changes rank, and refuses any proposal that
//     conflicts with an existing static dim.
//   * After op-level refinement, `refineFunctionSignature` tightens the
//     enclosing func.func signature when a return-op operand type
//     improved.
//   * After the type walk, `traceOutputOrigins` SSA-walks backward from
//     each function output to find which input arg + dim provides each
//     output dim's runtime value. Results land in two thread-local
//     stashes (refined-shapes + origins) read by C ABI exports
//     `hip_get_last_compile_output_shapes` / `..._dim_origins` and
//     surfaced to the EP via DimSource entries in metadata.proto.
//
// What's still required alongside this pass
// -----------------------------------------
//   * `--hip-materialize-host-scalars` — even with full refinement, tiny
//     shape-arithmetic memrefs (rank-0 / 1xi64 / small i32) survive for
//     the truly-dynamic dim cases. The host-scratch redirect is
//     load-bearing on architectures whose `hipMalloc` returns true device
//     memory.
//   * Reshape same-rank-dyn decomposition in ReshapeConversion.cpp.
//     Refinement does NOT remove the need for it — batch + seqlen dims are
//     genuinely runtime-dynamic in LLM decode, so `<?x?xH*D> ↔ <?x?xD>`
//     reshapes arrive at lowering with the dyn dims intact. The same-rank
//     decomposition is the standard lowering for this shape pattern.
//
// What this pass deliberately does NOT do
// ---------------------------------------
//   * Mutate constants or fold sub-graphs. It only refines result TYPES.
//     Downstream constant folding stays in its dedicated pattern set.
//   * Handle every onnx op. Only the set needed by LLM + ViT today.
//     Unknown ops are left unchanged. Adding a new op = adding one rule
//     in OnnxResultTypeInference and one wrapper here.
//   * Run shape inference on `hip.*` ops. Those get their types set at
//     conversion time and don't have rank/shape ambiguity by then.
//
//===----------------------------------------------------------------------===//

#include "OnnxResultTypeInference.h"
#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-onnx-shapes"

STATISTIC(NumOpTypesRefined,
          "Number of onnx.* op result types refined by InferOnnxShapes");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INFERONNXSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

// File-scope (not anonymous-namespace) thread-local stash of last-computed
// per-output dim origins. `DimOriginInfo` is declared in `OnnxToHipUtils.h`
// so both the anonymous-namespace helper that writes this stash AND the
// public `getInferredOutputOrigins` accessor (which CompilerAPI.cpp
// forward-declares) see the same type. Each entry is a triple
// `(arg_idx, dim_idx, mult)` — at runtime the output dim's value is
// `round(inputs[arg_idx].shape[dim_idx] * mult)`. `mult == 1.0` is the
// identity passthrough case (most LLM dynshape outputs); `mult == 1/K`
// covers Reshape-induced spatial mergers like Qwen vision's patch merger.
static thread_local std::vector<std::vector<DimOriginInfo>>
    g_last_output_origins;

// Companion thread-local for refined output shapes (positive ints for
// static dims, -1 for genuinely dynamic). Populated by the same
// `inferOnnxShapes` call that fills the origins stash — both must
// happen WHILE the function is still a `func::FuncOp` (before HipToLLVM
// conversion turns it into `llvm.func`). Read by the public accessor
// `getLastRefinedOutputShapes`.
static thread_local std::vector<std::vector<int64_t>>
    g_last_refined_output_shapes;

namespace {

//===----------------------------------------------------------------------===//
// Per-op type inference handlers
//===----------------------------------------------------------------------===//
//
// Each handler is a thin wrapper around a pure rule in
// `OnnxResultTypeInference.{h,cpp}`. The handler is responsible for:
//   * Extracting operand types + attributes from `op`.
//   * Calling the matching rule.
//   * Applying `isStrictlyTighter` so refinement never widens, never
//     changes element type, never changes rank, and refuses any proposal
//     that conflicts with an existing static dim.
//   * Returning `nullptr` when no refinement is possible.

/// True if `proposed` is a strict refinement of `current` (same rank,
/// same element type, and every position either equal or proposed-static
/// where current-dynamic). False if no improvement OR if the proposal
/// would conflict.
static bool isStrictlyTighter(mlir::RankedTensorType current,
                              mlir::RankedTensorType proposed) {
  if (!proposed || current == proposed)
    return false;
  if (current.getRank() != proposed.getRank() ||
      current.getElementType() != proposed.getElementType())
    return false;
  bool tightens = false;
  for (int64_t i : llvm::seq<int64_t>(current.getRank())) {
    bool curDyn = current.isDynamicDim(i);
    bool propDyn = proposed.isDynamicDim(i);
    if (curDyn && !propDyn) {
      tightens = true;
      continue;
    }
    if (!curDyn && !propDyn) {
      if (current.getDimSize(i) != proposed.getDimSize(i))
        return false; // conflicting static dims — refuse
      continue;
    }
    if (!curDyn && propDyn) {
      // Proposal would widen — refuse.
      return false;
    }
    // both dynamic — no change at this position
  }
  return tightens;
}

/// Merge `cur` with the pure-helper `proposal` per-dim: keep cur's static
/// dims (defends against an unsound resolver), take proposal's static
/// dims when cur is dynamic. Element type and rank from `cur`. Used by
/// the per-op handlers below to ensure that the pure helper's proposal
/// never weakens what the IR already declares.
static mlir::RankedTensorType
mergeWithCurrent(mlir::RankedTensorType cur, mlir::RankedTensorType proposal) {
  if (!proposal)
    return cur;
  if (cur.getRank() != proposal.getRank())
    return cur;
  llvm::SmallVector<int64_t> dims;
  dims.reserve(cur.getRank());
  for (int64_t i : llvm::seq<int64_t>(cur.getRank())) {
    int64_t curDim =
        cur.isDynamicDim(i) ? mlir::ShapedType::kDynamic : cur.getDimSize(i);
    int64_t propDim = proposal.isDynamicDim(i) ? mlir::ShapedType::kDynamic
                                               : proposal.getDimSize(i);
    if (curDim != mlir::ShapedType::kDynamic)
      dims.push_back(curDim);
    else if (propDim != mlir::ShapedType::kDynamic)
      dims.push_back(propDim);
    else
      dims.push_back(mlir::ShapedType::kDynamic);
  }
  return mlir::RankedTensorType::get(dims, cur.getElementType());
}

//----------------------------------------------------------------------------//
// onnx.Reshape
//----------------------------------------------------------------------------//

static mlir::Type inferReshape(mlir::Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return nullptr;
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outputType || outputType.hasStaticShape())
    return nullptr;
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return nullptr;
  int64_t allowzero = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("allowzero"))
    allowzero = a.getSInt();
  auto proposal = inferReshapeResultType(inputType, op->getOperand(1),
                                         outputType.getRank(), allowzero);
  auto merged = mergeWithCurrent(outputType, proposal);
  return isStrictlyTighter(outputType, merged) ? merged : nullptr;
}

//----------------------------------------------------------------------------//
// onnx.Transpose: output[i] = input[perm[i]]. Default perm = reverse.
//----------------------------------------------------------------------------//

static mlir::Type inferTranspose(mlir::Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return nullptr;
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType || outputType.hasStaticShape())
    return nullptr;
  int64_t rank = inputType.getRank();
  if (outputType.getRank() != rank)
    return nullptr;
  llvm::SmallVector<int64_t> perm;
  perm.reserve(rank);
  if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
    if (static_cast<int64_t>(permAttr.size()) != rank)
      return nullptr;
    for (mlir::Attribute a : permAttr) {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
      if (!ia)
        return nullptr;
      perm.push_back(ia.getValue().getSExtValue());
    }
  } else {
    for (int64_t i : llvm::seq<int64_t>(rank))
      perm.push_back(rank - 1 - i);
  }
  auto proposal = inferTransposeResultType(inputType, perm);
  auto merged = mergeWithCurrent(outputType, proposal);
  return isStrictlyTighter(outputType, merged) ? merged : nullptr;
}

//----------------------------------------------------------------------------//
// onnx.Cast: output shape == input shape, different element type.
//----------------------------------------------------------------------------//

static mlir::Type inferCast(mlir::Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return nullptr;
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType || outputType.hasStaticShape())
    return nullptr;
  if (inputType.getRank() != outputType.getRank())
    return nullptr;
  auto proposal = inferCastResultType(inputType, outputType.getElementType());
  auto merged = mergeWithCurrent(outputType, proposal);
  return isStrictlyTighter(outputType, merged) ? merged : nullptr;
}

//----------------------------------------------------------------------------//
// Unary same-shape: result shape == input shape, element type preserved.
//----------------------------------------------------------------------------//

static mlir::Type inferUnarySameShape(mlir::Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1)
    return nullptr;
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType || outputType.hasStaticShape())
    return nullptr;
  if (inputType.getRank() != outputType.getRank())
    return nullptr;
  // Pure helper returns `inputType` verbatim. Re-stamp the element type
  // from the existing output (defensive against malformed IR where the
  // input/output element types disagree, which the merge would then
  // refuse via isStrictlyTighter's element-type check).
  auto proposal = mlir::RankedTensorType::get(
      inferUnarySameShapeResultType(inputType).getShape(),
      outputType.getElementType());
  auto merged = mergeWithCurrent(outputType, proposal);
  return isStrictlyTighter(outputType, merged) ? merged : nullptr;
}

//----------------------------------------------------------------------------//
// Binary broadcast: numpy-style right-aligned broadcast.
//----------------------------------------------------------------------------//

static mlir::Type inferBinaryBroadcast(mlir::Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return nullptr;
  auto lhs =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto rhs =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto out = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!lhs || !rhs || !out || out.hasStaticShape())
    return nullptr;
  if (out.getRank() != std::max(lhs.getRank(), rhs.getRank()))
    return nullptr;
  auto proposal = inferBinaryBroadcastResultType(lhs, rhs);
  auto merged = mergeWithCurrent(out, proposal);
  return isStrictlyTighter(out, merged) ? merged : nullptr;
}

//----------------------------------------------------------------------------//
// onnx.MatMul: batched matmul. out = [...broadcast outer, lhs[-2], rhs[-1]].
//----------------------------------------------------------------------------//

static mlir::Type inferMatMul(mlir::Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return nullptr;
  auto lhs =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto rhs =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto out = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!lhs || !rhs || !out || out.hasStaticShape())
    return nullptr;
  if (lhs.getRank() < 2 || rhs.getRank() < 2)
    return nullptr;
  if (out.getRank() != std::max(lhs.getRank(), rhs.getRank()))
    return nullptr;
  auto proposal = inferMatMulResultType(lhs, rhs);
  auto merged = mergeWithCurrent(out, proposal);
  return isStrictlyTighter(out, merged) ? merged : nullptr;
}

//===----------------------------------------------------------------------===//
// Master dispatch
//===----------------------------------------------------------------------===//

static mlir::Type inferOpType(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "onnx.Reshape")
    return inferReshape(op);
  if (name == "onnx.Transpose")
    return inferTranspose(op);
  if (name == "onnx.MatMul")
    return inferMatMul(op);
  if (name == "onnx.Cast")
    return inferCast(op);
  // Unary same-shape activations / norms.
  if (name == "onnx.Tanh" || name == "onnx.Softmax" ||
      name == "onnx.LayerNormalization" || name == "onnx.Sqrt" ||
      name == "onnx.Gelu" || name == "onnx.Sigmoid" || name == "onnx.Neg" ||
      name == "onnx.Erf")
    return inferUnarySameShape(op);
  // Binary broadcast elementwise.
  if (name == "onnx.Add" || name == "onnx.Sub" || name == "onnx.Mul" ||
      name == "onnx.Div" || name == "onnx.Pow")
    return inferBinaryBroadcast(op);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Function signature refinement (matches return-op operand types)
//===----------------------------------------------------------------------===//

/// After op-level refinement, the function's return op may carry operands
/// whose types are tighter than the declared function result types.
/// Update the function signature in place. `pass_main.cpp::build_metadata_json`
/// reads from the morphizen graph (not this MLIR function type) for the
/// EP-side proto, so DimSource is unaffected.
///
/// Returns true iff the function type was actually updated. Caller uses
/// this together with the per-op `setType` count to detect quiescence.
static bool refineFunctionSignature(mlir::func::FuncOp funcOp) {
  // Find the (single) return terminator. At this stage onnx returns are
  // still onnx.Return (lowerOnnxReturns runs later).
  mlir::Operation *retOp = nullptr;
  funcOp.walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "onnx.Return" || mlir::isa<mlir::func::ReturnOp>(op)) {
      retOp = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!retOp)
    return false;
  auto funcType = funcOp.getFunctionType();
  if (retOp->getNumOperands() != funcType.getNumResults())
    return false;
  llvm::SmallVector<mlir::Type> newResults;
  newResults.reserve(funcType.getNumResults());
  bool changed = false;
  for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
    mlir::Type curOpType = retOp->getOperand(i).getType();
    mlir::Type declared = funcType.getResult(i);
    auto curRt = mlir::dyn_cast<mlir::RankedTensorType>(curOpType);
    auto decRt = mlir::dyn_cast<mlir::RankedTensorType>(declared);
    if (curRt && decRt && isStrictlyTighter(decRt, curRt)) {
      newResults.push_back(curOpType);
      changed = true;
    } else {
      newResults.push_back(declared);
    }
  }
  if (!changed)
    return false;
  auto newFuncType = mlir::FunctionType::get(funcOp.getContext(),
                                             funcType.getInputs(), newResults);
  funcOp.setType(newFuncType);
  return true;
}

//===----------------------------------------------------------------------===//
// Per-output dynamic-dim origin tracing
//===----------------------------------------------------------------------===//
//
// After type refinement, each function output may still have dynamic dims.
// For DimSource at the EP↔ORT boundary to resolve those dims at runtime,
// we need to know which graph-input dim each output dim came from. The
// ONNX `dim_param` mechanism captures this at AUTHORING TIME — but real
// exports often use semantically equivalent names that don't match
// (Gemma-3 vision: input `num_images`, output `num_image_tokens`). To
// avoid requiring on-disk model normalization, we trace the SSA chain
// backward from each output dim and report the originating
// (graph_arg_index, dim_idx) pair to the EP, which can populate
// DimSource directly.
//
// Tracing rules (per op):
//   * Function block argument: terminal — return (arg_number, dim).
//   * Unary same-shape (Cast, Tanh, Softmax, LayerNorm, Sqrt, Gelu,
//     Sigmoid, Neg, Erf): trace into operand 0 at the same dim.
//   * Transpose: trace into operand 0 at dim `perm[dim]`.
//   * Binary broadcast (Add/Sub/Mul/Div/Pow): trace into the operand
//     whose right-aligned dim is the non-`1` (non-broadcast) one. If
//     both broadcast, give up (no unique origin).
//   * MatMul: M (output[-2]) from lhs[-2]; N (output[-1]) from rhs[-1];
//     outer batch dims follow the binary-broadcast rule.
//   * Reshape: dim 0 passthrough when both input and output have
//     dynamic dim 0 (the canonical "batch-flows-through" case). Other
//     dims: conservative no-trace (the dim got split/merged).
//   * Conv / AveragePool / MaxPool: dim 0 from input; spatial / channel
//     dims have no SSA input origin.
//   * Anything else: no trace.
//
// Cycle protection: SSA values have no cycles by construction, so a
// straightforward recursion is safe. Depth bound: graph depth (~hundreds
// for big LLMs).

/// One traced origin for a dynamic output dim. Encodes "this dim's runtime
/// value is `round(inputs[arg_idx].shape[dim_idx] * mult)`". `mult` is a
/// positive scalar composed across Reshape ops that change the dim size:
///   * Identity passthrough → mult = 1.0
///   * Divide-by-K (e.g. 2x2 spatial patch merger turning
///     `[num_patches, hidden]` into `[num_patches/4, 4*hidden]`) →
///     mult = 1.0 / K = 0.25 for K=4
///   * Future multiply-by-K (e.g. spatial upsamplers) → mult = K
/// `double` (not int): a single field carries both directions and any
/// integer ratio with no precision loss for shape values up to ~2^52.
struct DimOrigin {
  int64_t arg_idx;
  int64_t dim_idx;
  double mult = 1.0;
};

static std::optional<DimOrigin> traceDimOrigin(mlir::Value v, int64_t dim);

/// Binary-broadcast helper: returns origin from the operand whose aligned
/// dim is non-1 (i.e. the one that *provides* the output dim). When both
/// dims are size-1 the result is also size-1 with no meaningful origin.
static std::optional<DimOrigin> traceBroadcastDim(mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  int64_t outRank,
                                                  int64_t dim) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    return std::nullopt;
  int64_t lhsShift = outRank - lhsType.getRank();
  int64_t rhsShift = outRank - rhsType.getRank();
  int64_t lhsIdx = dim - lhsShift;
  int64_t rhsIdx = dim - rhsShift;
  bool lhsHas = lhsIdx >= 0 && lhsIdx < lhsType.getRank();
  bool rhsHas = rhsIdx >= 0 && rhsIdx < rhsType.getRank();
  bool lhsIsOne = lhsHas && !lhsType.isDynamicDim(lhsIdx) &&
                  lhsType.getDimSize(lhsIdx) == 1;
  bool rhsIsOne = rhsHas && !rhsType.isDynamicDim(rhsIdx) &&
                  rhsType.getDimSize(rhsIdx) == 1;
  // Prefer the non-1 (the one actually providing the value).
  if (lhsHas && !lhsIsOne) {
    auto o = traceDimOrigin(lhs, lhsIdx);
    if (o)
      return o;
  }
  if (rhsHas && !rhsIsOne) {
    auto o = traceDimOrigin(rhs, rhsIdx);
    if (o)
      return o;
  }
  // Last resort: any operand that has the dim at all.
  if (lhsHas)
    return traceDimOrigin(lhs, lhsIdx);
  if (rhsHas)
    return traceDimOrigin(rhs, rhsIdx);
  return std::nullopt;
}

static std::optional<DimOrigin> traceDimOrigin(mlir::Value v, int64_t dim) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    return DimOrigin{static_cast<int64_t>(blockArg.getArgNumber()), dim};
  }
  mlir::Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  llvm::StringRef name = op->getName().getStringRef();

  // Unary same-shape ops.
  if (name == "onnx.Cast" || name == "onnx.Tanh" || name == "onnx.Softmax" ||
      name == "onnx.LayerNormalization" || name == "onnx.Sqrt" ||
      name == "onnx.Gelu" || name == "onnx.Sigmoid" || name == "onnx.Neg" ||
      name == "onnx.Erf") {
    if (op->getNumOperands() < 1)
      return std::nullopt;
    return traceDimOrigin(op->getOperand(0), dim);
  }

  if (name == "onnx.Transpose") {
    if (op->getNumOperands() != 1)
      return std::nullopt;
    auto inType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inType)
      return std::nullopt;
    int64_t rank = inType.getRank();
    llvm::SmallVector<int64_t> perm;
    if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
      for (mlir::Attribute a : permAttr) {
        auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
        if (!ia)
          return std::nullopt;
        perm.push_back(ia.getValue().getSExtValue());
      }
    } else {
      for (int64_t i = rank - 1; i >= 0; --i)
        perm.push_back(i);
    }
    if (dim < 0 || dim >= (int64_t)perm.size())
      return std::nullopt;
    return traceDimOrigin(op->getOperand(0), perm[dim]);
  }

  if (name == "onnx.Add" || name == "onnx.Sub" || name == "onnx.Mul" ||
      name == "onnx.Div" || name == "onnx.Pow") {
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!outType)
      return std::nullopt;
    return traceBroadcastDim(op->getOperand(0), op->getOperand(1),
                             outType.getRank(), dim);
  }

  if (name == "onnx.MatMul") {
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto lhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto rhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !outType)
      return std::nullopt;
    int64_t outRank = outType.getRank();
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    if (lhsRank < 2 || rhsRank < 2)
      return std::nullopt;
    if (dim == outRank - 2) // M from lhs[-2]
      return traceDimOrigin(op->getOperand(0), lhsRank - 2);
    if (dim == outRank - 1) // N from rhs[-1]
      return traceDimOrigin(op->getOperand(1), rhsRank - 1);
    // Outer-batch broadcast. Unlike generic binary broadcast (which
    // right-aligns the WHOLE operand to the output), matmul's outer
    // batch dims are the LEADING (lhsRank - 2) / (rhsRank - 2) dims of
    // each operand. Right-align WITHIN that outer slice only. For
    // dim `d` in [0, outOuter):
    //   lhsIdx = d - (outOuter - lhsOuter); rhsIdx = d - (outOuter - rhsOuter)
    int64_t outOuter = outRank - 2;
    int64_t lhsOuter = lhsRank - 2;
    int64_t rhsOuter = rhsRank - 2;
    int64_t lhsIdx = dim - (outOuter - lhsOuter);
    int64_t rhsIdx = dim - (outOuter - rhsOuter);
    bool lhsHas = lhsIdx >= 0 && lhsIdx < lhsOuter;
    bool rhsHas = rhsIdx >= 0 && rhsIdx < rhsOuter;
    bool lhsIsOne = lhsHas && !lhsType.isDynamicDim(lhsIdx) &&
                    lhsType.getDimSize(lhsIdx) == 1;
    bool rhsIsOne = rhsHas && !rhsType.isDynamicDim(rhsIdx) &&
                    rhsType.getDimSize(rhsIdx) == 1;
    if (lhsHas && !lhsIsOne) {
      auto o = traceDimOrigin(op->getOperand(0), lhsIdx);
      if (o)
        return o;
    }
    if (rhsHas && !rhsIsOne) {
      auto o = traceDimOrigin(op->getOperand(1), rhsIdx);
      if (o)
        return o;
    }
    if (lhsHas)
      return traceDimOrigin(op->getOperand(0), lhsIdx);
    if (rhsHas)
      return traceDimOrigin(op->getOperand(1), rhsIdx);
    return std::nullopt;
  }

  if (name == "onnx.Reshape") {
    // Trace dim 0 only — other output dims have been split/merged across
    // multiple input dims and can't be attributed to a single origin.
    //
    // ONNX Reshape preserves total element count, so for our dim-0 trace:
    //
    //   in_total  = in.dim[0]  * inOther
    //   out_total = out.dim[0] * outOther
    //   in_total == out_total
    //
    // therefore  out.dim[0] = in.dim[0] * inOther / outOther.
    //
    // Two cases we handle today on the OUTER product ratio:
    //   * inOther == outOther         → identity passthrough (mult *= 1.0)
    //     Example (Gemma-3 projector):
    //       in  <?x1152x16x16>  outer=1152*16*16=294912
    //       out <?x256x1152>    outer=256*1152=294912
    //
    //   * outOther > inOther          → divide (mult *= inOther/outOther)
    //     Example (Qwen3.5 vision patch merger):
    //       in  <num_patches, 1152>          outer=1152
    //       out <num_patches/4, 4608>        outer=4608
    //     out.dim[0] = in.dim[0] * 1152 / 4608 = in.dim[0] / 4
    //     Require outOther % inOther == 0 so the ratio is a clean integer.
    //
    // The third case `inOther > outOther` (Reshape splits dim 0 into more
    // elements per row, i.e. multiply-by-K) is representable by the same
    // `mult` field (`mult > 1.0`) but isn't enabled yet because no
    // shipping model needs it. Enable by dropping the
    // `outOther < inOther` refusal and adding the symmetric branch.
    //
    // outer == 0 OR any non-zero other-dim being dynamic means we cannot
    // compute the ratio at compile time; refuse to trace in that case too.
    if (dim != 0)
      return std::nullopt;
    auto inType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inType || !outType)
      return std::nullopt;
    if (inType.getRank() < 1 || outType.getRank() < 1)
      return std::nullopt;
    if (!inType.isDynamicDim(0) || !outType.isDynamicDim(0))
      return std::nullopt;

    int64_t inOther = 1, outOther = 1;
    for (int64_t i = 1; i < inType.getRank(); ++i) {
      if (inType.isDynamicDim(i))
        return std::nullopt;
      inOther *= inType.getDimSize(i);
    }
    for (int64_t i = 1; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i))
        return std::nullopt;
      outOther *= outType.getDimSize(i);
    }
    if (inOther <= 0 || outOther <= 0)
      return std::nullopt;
    if (outOther < inOther)
      return std::nullopt; // multiply-by-K: representable but not enabled
    if (outOther % inOther != 0)
      return std::nullopt; // non-integer ratio

    auto inner = traceDimOrigin(op->getOperand(0), 0);
    if (!inner)
      return std::nullopt;
    // 1.0 for identity, 1/K for divide-by-K. Power-of-2 K is exact in
    // IEEE 754; for integer K up to ~2^52 the rounded runtime result is
    // exact for shape values up to ~2^52.
    double newMult =
        static_cast<double>(inOther) / static_cast<double>(outOther);
    DimOrigin extended = *inner;
    extended.mult *= newMult;
    return extended;
  }

  if (name == "onnx.Conv" || name == "onnx.AveragePool" ||
      name == "onnx.MaxPool") {
    if (op->getNumOperands() < 1)
      return std::nullopt;
    if (dim == 0) // batch dim passes through these ops
      return traceDimOrigin(op->getOperand(0), 0);
    return std::nullopt;
  }

  // ReduceSum / ReduceMean / ReduceMax / ReduceProd: output dim `dim`
  // traces to a specific input dim iff that input dim survives the
  // reduction (i.e. is NOT in the axes list). With `keepdims=1` (default
  // in some opsets), all dims survive at the same position. With
  // `keepdims=0`, removed dims shift later dims up.
  //
  // For our purposes (batch dim 0 passing through projector ReduceSum):
  // if axes doesn't include 0 AND dim == 0, output dim 0 == input dim 0.
  // Handles both keepdims=0 and keepdims=1 (dim 0 stays at index 0 if
  // not reduced).
  if (name == "onnx.ReduceSum" || name == "onnx.ReduceMean" ||
      name == "onnx.ReduceMax" || name == "onnx.ReduceProd") {
    if (op->getNumOperands() < 1)
      return std::nullopt;
    auto inType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inType)
      return std::nullopt;
    int64_t inRank = inType.getRank();
    int64_t keepdims = 1;
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
      keepdims = a.getSInt();
    // Read axes: opset 18+ as operand 1; older as attr.
    llvm::SmallVector<int64_t> axes;
    if (op->getNumOperands() >= 2) {
      mlir::Operation *axesDef = op->getOperand(1).getDefiningOp();
      if (axesDef && axesDef->getName().getStringRef() == "onnx.Constant") {
        if (auto attr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
                axesDef->getAttr("value"))) {
          for (auto v : attr.getValues<llvm::APInt>())
            axes.push_back(v.getSExtValue());
        }
      }
    } else if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (mlir::Attribute a : axesAttr) {
        auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
        if (ia)
          axes.push_back(ia.getSInt());
      }
    }
    // Normalize negative axes; check membership for dim 0.
    bool zeroIsReduced = false;
    for (int64_t a : axes) {
      if (a < 0)
        a += inRank;
      if (a == 0) {
        zeroIsReduced = true;
        break;
      }
    }
    if (dim == 0 && !zeroIsReduced)
      return traceDimOrigin(op->getOperand(0), 0);
    return std::nullopt;
  }

  return std::nullopt;
}

/// Compute per-output, per-dim origins for the function. Outer vector
/// indexed by function result; inner vector by dim. For each dim, the
/// origin (if any) is a (graph_arg_index, dim_idx) pair into the
/// function arguments — the EP maps this directly to its
/// `dim_param_map[input_idx][dim_idx]` view.
static std::vector<std::vector<std::optional<DimOrigin>>>
traceOutputOrigins(mlir::func::FuncOp funcOp) {
  std::vector<std::vector<std::optional<DimOrigin>>> result;
  mlir::Operation *retOp = nullptr;
  funcOp.walk([&](mlir::Operation *op) {
    llvm::StringRef n = op->getName().getStringRef();
    if (n == "onnx.Return" || mlir::isa<mlir::func::ReturnOp>(op)) {
      retOp = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!retOp)
    return result;
  result.reserve(retOp->getNumOperands());
  for (mlir::Value retVal : retOp->getOperands()) {
    std::vector<std::optional<DimOrigin>> dims;
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(retVal.getType());
    if (!rt) {
      result.push_back({});
      continue;
    }
    dims.reserve(rt.getRank());
    for (int64_t d = 0; d < rt.getRank(); ++d) {
      if (rt.isDynamicDim(d))
        dims.push_back(traceDimOrigin(retVal, d));
      else
        dims.push_back(std::nullopt);
    }
    result.push_back(std::move(dims));
  }
  return result;
}

// Origin stash helper. Reads `traceOutputOrigins` (in this same
// anonymous namespace) and writes to the file-scope thread-local
// `g_last_output_origins` declared in `mlir::hip` (not in this anon
// ns) so the public accessor `getInferredOutputOrigins` below can
// read it.
// Snapshot the refined output shapes from the function signature. Must
// be called WHILE the function is still `func::FuncOp` — by the time
// CompilerDriver's `compileImpl` completes the LLVM-conversion phase,
// the function is `llvm.func` and can't be queried this way.
static void stashShapesFor(mlir::func::FuncOp funcOp) {
  g_last_refined_output_shapes.clear();
  auto funcType = funcOp.getFunctionType();
  g_last_refined_output_shapes.reserve(funcType.getNumResults());
  for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(funcType.getResult(i));
    if (!rt) {
      g_last_refined_output_shapes.push_back({});
      continue;
    }
    std::vector<int64_t> dims;
    dims.reserve(rt.getRank());
    for (int64_t d = 0; d < rt.getRank(); ++d)
      dims.push_back(rt.isDynamicDim(d) ? int64_t(-1) : rt.getDimSize(d));
    g_last_refined_output_shapes.push_back(std::move(dims));
  }
}

static void stashOriginsFor(mlir::func::FuncOp funcOp) {
  g_last_output_origins.clear();
  auto origins = traceOutputOrigins(funcOp);
  // The function may carry leading non-tensor args (notably `!hip.context`
  // added by AddHipContextArg). The EP-side metadata builder indexes
  // inputs by their position in the MORPHIZEN GRAPH, which excludes
  // those non-tensor args. Compute the offset so we can translate MLIR
  // arg positions to morphizen graph input positions before stashing.
  int64_t contextOffset = 0;
  auto funcType = funcOp.getFunctionType();
  for (mlir::Type t : funcType.getInputs()) {
    if (mlir::isa<mlir::hip::ContextType>(t))
      ++contextOffset;
    else
      break; // non-tensor args are always leading
  }
  g_last_output_origins.reserve(origins.size());
  for (const auto &dims : origins) {
    std::vector<DimOriginInfo> v;
    v.reserve(dims.size());
    for (const auto &o : dims) {
      if (o && o->arg_idx >= contextOffset)
        v.push_back({o->arg_idx - contextOffset, o->dim_idx, o->mult});
      else
        v.push_back({int64_t(-1), int64_t(-1), 1.0});
    }
    g_last_output_origins.push_back(std::move(v));
  }
  LLVM_DEBUG({
    for (size_t i = 0; i < g_last_output_origins.size(); ++i)
      for (size_t d = 0; d < g_last_output_origins[i].size(); ++d) {
        const auto &e = g_last_output_origins[i][d];
        llvm::dbgs() << "[" DEBUG_TYPE "] output[" << i << "].dim[" << d
                     << "] origin = (" << e.arg_idx << ", " << e.dim_idx
                     << ", *=" << e.mult << ")\n";
      }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

/// Public accessor for the thread-local origins stash. Read via the C
/// ABI export `hip_get_last_compile_output_dim_origins` in
/// `lib/CInterface/CompilerAPI.cpp`. Each entry is a triple
/// `(arg_idx, dim_idx, divisor)`.
const std::vector<std::vector<DimOriginInfo>> &getInferredOutputOrigins() {
  return g_last_output_origins;
}

/// Public accessor for the thread-local refined-shapes stash. Same
/// rationale as `getInferredOutputOrigins`. Read via the C ABI export
/// `hip_get_last_compile_output_shapes`.
const std::vector<std::vector<int64_t>> &getInferredOutputShapes() {
  return g_last_refined_output_shapes;
}

mlir::LogicalResult inferOnnxShapes(mlir::func::FuncOp funcOp, bool *changed) {
  // Walk is structural (post-order over regions, in IR order within a
  // region). For SSA dataflow that's def-before-use, which is what we want:
  // each op sees its operands' refined types when we visit it.
  bool anyChange = false;
  funcOp.walk([&](mlir::Operation *op) {
    if (!op->getName().getStringRef().starts_with("onnx."))
      return;
    if (op->getNumResults() != 1)
      return; // multi-result onnx ops are rare; skip rather than special-case
    if (mlir::Type refined = inferOpType(op)) {
      mlir::Type curType = op->getResult(0).getType();
      if (refined != curType) {
        op->getResult(0).setType(refined);
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << op->getName() << ": "
                                << curType << " -> " << refined << "\n");
        ++NumOpTypesRefined;
        anyChange = true;
      }
    }
  });
  if (refineFunctionSignature(funcOp))
    anyChange = true;
  if (changed && anyChange)
    *changed = true;
  // After refinement is done, capture per-output-dim refined shapes AND
  // SSA origins so the EP boundary can populate DimSource for dynamic
  // dims whose dim_param names don't match any input's dim_param (the
  // gemma3 vision case). Both stashes MUST happen here (while funcOp is
  // still `func::FuncOp`) — by the time `CompilerDriver::compileImpl`
  // returns from `runMLIRPasses`, the function has been converted to
  // `llvm.func` and the original signature is no longer queryable.
  stashShapesFor(funcOp);
  stashOriginsFor(funcOp);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Standalone pass wrapper
//===----------------------------------------------------------------------===//
//
// Production callers (ConvertOnnxToHipPass) drive `inferOnnxShapes` directly
// inside the per-function pre-lowering round loop. The standalone pass below
// is for LIT tests + pipeline composition: it runs ONE inference pass over
// the function and exits. Tests can chain it with hip-mlir-opt as
// `--hip-add-context-arg --infer-onnx-shapes` to verify per-op refinement
// rules without dragging in convert-onnx-to-hip's full lowering pipeline.

namespace {

struct InferOnnxShapesPass
    : public impl::InferOnnxShapesPassBase<InferOnnxShapesPass> {
  using InferOnnxShapesPassBase::InferOnnxShapesPassBase;

  void runOnOperation() override {
    if (mlir::failed(inferOnnxShapes(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

} // namespace hip
} // namespace mlir
