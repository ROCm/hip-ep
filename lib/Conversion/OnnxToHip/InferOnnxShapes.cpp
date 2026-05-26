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
//     output dim's runtime value. Results are attached to the enclosing
//     `mlir::ModuleOp` as `hip.refined_output_shapes` and
//     `hip.refined_output_dim_origins` ArrayAttrs. Those survive the
//     `func.func → llvm.func` conversion intact, so `CompilerDriver`
//     reads them after `runMLIRPasses` returns and packs the data into
//     the caller-supplied `CompilationOutputs` buffers on the C ABI.
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <cstring>
#include <functional>

#define DEBUG_TYPE "infer-onnx-shapes"

STATISTIC(NumOpTypesRefined,
          "Number of onnx.* op result types refined by InferOnnxShapes");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INFERONNXSHAPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

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
// Master dispatch — registry-based
//===----------------------------------------------------------------------===//
//
// `inferOpType` dispatches by op-name via an immortal StringMap keyed on
// the fully-qualified op string (e.g. "onnx.Reshape"). Adding a new op
// = one insert in `populateInferTypeRules`. Same pattern as the trace
// registry below; both registries share lifetime semantics (static
// local, initialised on first use, never torn down).

using InferTypeFn = mlir::Type (*)(mlir::Operation *);

static void populateInferTypeRules(llvm::StringMap<InferTypeFn> &r) {
  r["onnx.Reshape"] = inferReshape;
  r["onnx.Transpose"] = inferTranspose;
  r["onnx.MatMul"] = inferMatMul;
  r["onnx.Cast"] = inferCast;
  // Unary same-shape activations / norms.
  for (auto n :
       {"onnx.Tanh", "onnx.Softmax", "onnx.LayerNormalization", "onnx.Sqrt",
        "onnx.Gelu", "onnx.Sigmoid", "onnx.Neg", "onnx.Erf"})
    r[n] = inferUnarySameShape;
  // Binary broadcast elementwise.
  for (auto n : {"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Pow"})
    r[n] = inferBinaryBroadcast;
}

static const llvm::StringMap<InferTypeFn> &getInferTypeRegistry() {
  static const llvm::StringMap<InferTypeFn> kReg = [] {
    llvm::StringMap<InferTypeFn> m;
    populateInferTypeRules(m);
    return m;
  }();
  return kReg;
}

static mlir::Type inferOpType(mlir::Operation *op) {
  const auto &reg = getInferTypeRegistry();
  auto it = reg.find(op->getName().getStringRef());
  return it != reg.end() ? it->second(op) : nullptr;
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

/// Per-`traceOutputOrigins` shared state. Today holds the dedup set used
/// to suppress duplicate "no trace rule for op X" warnings (one per kind,
/// per function — without dedup, a deep model would emit thousands of
/// identical warnings for the same unknown producer reachable through
/// many forward paths). Threaded explicitly through `traceDimOrigin` /
/// `traceBroadcastDim` instead of a thread-local — keeps the helpers
/// reentrant if a future caller ever traces multiple function outputs
/// concurrently.
struct TraceContext {
  llvm::DenseSet<llvm::StringRef> warned;
};

static std::optional<DimOrigin> traceDimOrigin(mlir::Value v, int64_t dim,
                                               TraceContext &ctx);

/// Binary-broadcast helper: returns origin from the operand whose aligned
/// dim is non-1 (i.e. the one that *provides* the output dim). When both
/// dims are size-1 the result is also size-1 with no meaningful origin.
static std::optional<DimOrigin> traceBroadcastDim(mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  int64_t outRank, int64_t dim,
                                                  TraceContext &ctx) {
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
    auto o = traceDimOrigin(lhs, lhsIdx, ctx);
    if (o)
      return o;
  }
  if (rhsHas && !rhsIsOne) {
    auto o = traceDimOrigin(rhs, rhsIdx, ctx);
    if (o)
      return o;
  }
  // Last resort: any operand that has the dim at all.
  if (lhsHas)
    return traceDimOrigin(lhs, lhsIdx, ctx);
  if (rhsHas)
    return traceDimOrigin(rhs, rhsIdx, ctx);
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Per-op trace handlers
//===----------------------------------------------------------------------===//
//
// One handler per op kind (or per group of equivalent kinds, e.g. all
// unary same-shape ops share `traceUnary`). Registered via
// `populateTraceRules` into a file-static StringMap. The dispatch
// (`traceDimOrigin`) then becomes a single hash lookup plus a fall-
// through that warns once per function per unknown `onnx.*` producer.

using TraceFn = std::optional<DimOrigin> (*)(mlir::Operation *, int64_t,
                                             TraceContext &);

// Unary same-shape (Cast, Tanh, Softmax, LayerNorm, Sqrt, Gelu, Sigmoid,
// Neg, Erf): trace into operand 0 at the same dim.
static std::optional<DimOrigin> traceUnary(mlir::Operation *op, int64_t dim,
                                           TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  return traceDimOrigin(op->getOperand(0), dim, ctx);
}

// Transpose: trace into operand 0 at dim `perm[dim]`. Default perm is
// reverse-of-rank when the attribute is absent (ONNX spec).
static std::optional<DimOrigin> traceTranspose(mlir::Operation *op, int64_t dim,
                                               TraceContext &ctx) {
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
  return traceDimOrigin(op->getOperand(0), perm[dim], ctx);
}

// Binary broadcast (Add/Sub/Mul/Div/Pow): trace into the operand whose
// right-aligned dim is the non-`1` (non-broadcast) one. If both
// broadcast, give up (no unique origin).
static std::optional<DimOrigin>
traceBinaryBroadcast(mlir::Operation *op, int64_t dim, TraceContext &ctx) {
  if (op->getNumOperands() != 2)
    return std::nullopt;
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outType)
    return std::nullopt;
  return traceBroadcastDim(op->getOperand(0), op->getOperand(1),
                           outType.getRank(), dim, ctx);
}

// MatMul: M (output[-2]) from lhs[-2]; N (output[-1]) from rhs[-1];
// outer batch dims follow the binary-broadcast rule, RIGHT-ALIGNED WITHIN
// THE OUTER SLICE (NOT the full operand). The full-rank alignment trap
// silently mis-traces vision MatMul batch dims into K — see
// CLAUDE.md's "MatMul outer-batch dim alignment" gotcha.
static std::optional<DimOrigin> traceMatMul(mlir::Operation *op, int64_t dim,
                                            TraceContext &ctx) {
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
    return traceDimOrigin(op->getOperand(0), lhsRank - 2, ctx);
  if (dim == outRank - 1) // N from rhs[-1]
    return traceDimOrigin(op->getOperand(1), rhsRank - 1, ctx);
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
    auto o = traceDimOrigin(op->getOperand(0), lhsIdx, ctx);
    if (o)
      return o;
  }
  if (rhsHas && !rhsIsOne) {
    auto o = traceDimOrigin(op->getOperand(1), rhsIdx, ctx);
    if (o)
      return o;
  }
  if (lhsHas)
    return traceDimOrigin(op->getOperand(0), lhsIdx, ctx);
  if (rhsHas)
    return traceDimOrigin(op->getOperand(1), rhsIdx, ctx);
  return std::nullopt;
}

// Reshape: dim 0 keeps the original outer-product-ratio rule (passthrough
// when inOther == outOther; divide-by-K when outOther is an integer
// multiple of inOther). Dims > 0 are handled by a strict
// non-positional-passthrough check that requires the trailing products to
// balance EXACTLY.
//
// Mathematical contract — ONNX Reshape preserves total element count, so
// for any output dim d:
//   out.dim[d] = in.dim[d] * (prod(other in dims) / prod(other out dims))
// "passthrough" (mult=1) iff the OTHER-product ratio is 1. Both dim d
// values being dynamic is NOT sufficient — `<?x3> → <?x9>` matches at
// dim 0 (both dyn) but the actual runtime mapping is divide-by-3.
//
// What this trace handles today:
//   * dim 0, in.dim[0] dyn, out.dim[0] dyn, all other input AND output
//     dims static: passthrough when in_outer == out_outer (e.g.
//     <?x1152x16x16> → <?x256x1152>); divide-by-K when
//     out_outer == K * in_outer (e.g. <?x1152> → <?/4 x 4608>).
//   * dim d > 0, in.dim[d] dyn, out.dim[d] dyn, all other input AND
//     output dims static, product(other in) == product(other out):
//     passthrough only. The divide-by-K story for non-zero dims would
//     need to know which input dim is the dyn one (could be ≠ d), so
//     it stays explicitly unsupported until a model requires it.
//
// Multiply-by-K (in_outer > out_outer) is representable by the same
// `mult` field (> 1.0) but no shipping model currently needs it; enable
// by lifting the `outOther < inOther` refusal in the dim-0 branch.
static std::optional<DimOrigin> traceReshape(mlir::Operation *op, int64_t dim,
                                             TraceContext &ctx) {
  auto inType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inType || !outType)
    return std::nullopt;
  if (dim < 0 || dim >= outType.getRank())
    return std::nullopt;
  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();

  if (dim == 0) {
    if (inRank < 1 || !inType.isDynamicDim(0) || !outType.isDynamicDim(0))
      return std::nullopt;
    int64_t inOther = 1, outOther = 1;
    for (int64_t i : llvm::seq<int64_t>(1, inRank)) {
      if (inType.isDynamicDim(i))
        return std::nullopt;
      inOther *= inType.getDimSize(i);
    }
    for (int64_t i : llvm::seq<int64_t>(1, outRank)) {
      if (outType.isDynamicDim(i))
        return std::nullopt;
      outOther *= outType.getDimSize(i);
    }
    if (inOther <= 0 || outOther <= 0)
      return std::nullopt;
    if (outOther < inOther)
      return std::nullopt; // multiply-by-K: representable but not enabled
    if (outOther % inOther != 0)
      return std::nullopt;
    auto inner = traceDimOrigin(op->getOperand(0), 0, ctx);
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

  // dim > 0: strict passthrough requires both sides' dim d to be dyn AND
  // the products of EVERY other dim (input AND output) to be equal.
  if (dim >= inRank)
    return std::nullopt;
  if (!inType.isDynamicDim(dim) || !outType.isDynamicDim(dim))
    return std::nullopt;
  int64_t inOther = 1, outOther = 1;
  for (int64_t i : llvm::seq<int64_t>(0, inRank)) {
    if (i == dim)
      continue;
    if (inType.isDynamicDim(i))
      return std::nullopt;
    inOther *= inType.getDimSize(i);
  }
  for (int64_t i : llvm::seq<int64_t>(0, outRank)) {
    if (i == dim)
      continue;
    if (outType.isDynamicDim(i))
      return std::nullopt;
    outOther *= outType.getDimSize(i);
  }
  if (inOther != outOther)
    return std::nullopt;
  return traceDimOrigin(op->getOperand(0), dim, ctx);
}

// Conv / AveragePool / MaxPool: dim 0 (batch) passes through from input;
// spatial / channel dims have no SSA input origin.
static std::optional<DimOrigin>
traceConvLikeBatch(mlir::Operation *op, int64_t dim, TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  if (dim == 0)
    return traceDimOrigin(op->getOperand(0), 0, ctx);
  return std::nullopt;
}

// ReduceSum / ReduceMean / ReduceMax / ReduceProd: output dim 0 passes
// through from input dim 0 iff axes does not reduce dim 0 (handles both
// keepdims=0 and keepdims=1 — dim 0 stays at index 0 if not reduced).
// Other dims: conservative no-trace (would need keepdims-aware shift
// math for non-zero `dim`, not needed by today's models).
static std::optional<DimOrigin> traceReduce(mlir::Operation *op, int64_t dim,
                                            TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  auto inType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inType)
    return std::nullopt;
  int64_t inRank = inType.getRank();
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
    return traceDimOrigin(op->getOperand(0), 0, ctx);
  return std::nullopt;
}

// Helper: try to read a constant int operand into a SmallVector. Used by
// the Slice / Gather rules to peek at attribute-or-operand axes values.
// Returns false if the operand is not an onnx.Constant DenseElementsAttr.
static bool readConstIntOperand(mlir::Value v,
                                llvm::SmallVectorImpl<int64_t> &out) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return false;
  auto attr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  if (!attr)
    return false;
  for (llvm::APInt val : attr.getValues<llvm::APInt>())
    out.push_back(val.getSExtValue());
  return true;
}

// Squeeze: remove axes of size 1. Output rank = inRank - numSqueezedAxes.
// Output dim d corresponds to the d-th non-squeezed input axis. We
// require `axes` to be statically known (attribute in opset < 13 or a
// constant operand in opset >= 13); otherwise refuse (the axis-to-dim
// mapping is data-dependent).
static std::optional<DimOrigin> traceSqueeze(mlir::Operation *op, int64_t dim,
                                             TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  auto inType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inType)
    return std::nullopt;
  int64_t inRank = inType.getRank();
  llvm::SmallVector<int64_t> axes;
  if (op->getNumOperands() >= 2) {
    if (!readConstIntOperand(op->getOperand(1), axes))
      return std::nullopt;
  } else if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
    for (mlir::Attribute a : axesAttr) {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
      if (!ia)
        return std::nullopt;
      axes.push_back(ia.getSInt());
    }
  } else {
    // No axes specified: ONNX would squeeze all size-1 dims. Conservative.
    return std::nullopt;
  }
  llvm::DenseSet<int64_t> squeezed;
  for (int64_t a : axes)
    squeezed.insert(a < 0 ? a + inRank : a);
  // Find the dim-th non-squeezed input axis.
  int64_t outCursor = 0;
  for (int64_t i : llvm::seq<int64_t>(0, inRank)) {
    if (squeezed.contains(i))
      continue;
    if (outCursor == dim)
      return traceDimOrigin(op->getOperand(0), i, ctx);
    ++outCursor;
  }
  return std::nullopt;
}

// Unsqueeze: insert size-1 axes at the positions in `axes`. Output rank
// = inRank + numInsertedAxes. Output dim d:
//   * d ∈ axes  → inserted size-1, no SSA origin (refuse).
//   * d ∉ axes  → input dim d - (number of inserted axes < d).
// Same axes-must-be-static requirement as Squeeze.
static std::optional<DimOrigin> traceUnsqueeze(mlir::Operation *op, int64_t dim,
                                               TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outType)
    return std::nullopt;
  int64_t outRank = outType.getRank();
  llvm::SmallVector<int64_t> axes;
  if (op->getNumOperands() >= 2) {
    if (!readConstIntOperand(op->getOperand(1), axes))
      return std::nullopt;
  } else if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
    for (mlir::Attribute a : axesAttr) {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
      if (!ia)
        return std::nullopt;
      axes.push_back(ia.getSInt());
    }
  } else {
    return std::nullopt;
  }
  llvm::DenseSet<int64_t> inserted;
  for (int64_t a : axes)
    inserted.insert(a < 0 ? a + outRank : a);
  if (inserted.contains(dim))
    return std::nullopt;
  int64_t shift = 0;
  for (int64_t a : inserted)
    if (a < dim)
      ++shift;
  return traceDimOrigin(op->getOperand(0), dim - shift, ctx);
}

// Concat: output dim d:
//   * d == axis → sum across operands (no single origin) — refuse.
//   * d != axis → all operands' dim d must agree. Conservative:
//                 - if any pair has differing static value, refuse;
//                 - if mixed static/dynamic, refuse (can't prove
//                   they'd resolve to the same runtime value);
//                 - else trace operand 0's dim d.
static std::optional<DimOrigin> traceConcat(mlir::Operation *op, int64_t dim,
                                            TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto firstOpType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!outType || !firstOpType)
    return std::nullopt;
  int64_t outRank = outType.getRank();
  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  if (!axisAttr)
    return std::nullopt;
  int64_t axis = axisAttr.getSInt();
  if (axis < 0)
    axis += outRank;
  if (dim == axis)
    return std::nullopt;
  if (dim < 0 || dim >= firstOpType.getRank())
    return std::nullopt;
  // All operands must agree on dim `dim`.
  for (mlir::Value v : op->getOperands().drop_front()) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t || dim >= t.getRank())
      return std::nullopt;
    bool d0Dyn = firstOpType.isDynamicDim(dim);
    bool dnDyn = t.isDynamicDim(dim);
    if (d0Dyn != dnDyn)
      return std::nullopt; // mixed static/dynamic — refuse
    if (!d0Dyn && firstOpType.getDimSize(dim) != t.getDimSize(dim))
      return std::nullopt; // conflicting static values
  }
  return traceDimOrigin(op->getOperand(0), dim, ctx);
}

// Slice: opset 10+ uses operands (starts, ends, axes, steps). For our
// trace: if the requested output dim is NOT in the sliced axes set,
// it's an identity passthrough into the input at the same dim. If it
// IS in the sliced axes set, the runtime size depends on
// (end - start) / step — refuse unless we can prove identity (start=0,
// step=1, end large enough), which we currently don't, so refuse all
// sliced-axis cases.
//
// `axes` defaults to `[0, 1, ..., rank-1]` when the operand is absent —
// in that case EVERY axis is sliced and we can never passthrough; refuse.
static std::optional<DimOrigin> traceSlice(mlir::Operation *op, int64_t dim,
                                           TraceContext &ctx) {
  if (op->getNumOperands() < 1)
    return std::nullopt;
  // axes operand is index 3 in opset 10+; refuse if not present (default
  // = all axes sliced).
  if (op->getNumOperands() < 4)
    return std::nullopt;
  auto inType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inType)
    return std::nullopt;
  llvm::SmallVector<int64_t> axes;
  if (!readConstIntOperand(op->getOperand(3), axes))
    return std::nullopt;
  for (int64_t &a : axes)
    if (a < 0)
      a += inType.getRank();
  if (llvm::is_contained(axes, dim))
    return std::nullopt;
  return traceDimOrigin(op->getOperand(0), dim, ctx);
}

// Gather: data, indices, axis attr (default 0). Output rank =
// data_rank + indices_rank - 1.
//
// Only the SCALAR-indices case (indices rank 0) produces a clean dim
// mapping we can trace today:
//   output rank = data_rank - 1
//   output[d < axis] = data[d]
//   output[d >= axis] = data[d + 1]
// Other cases (vector / N-D indices) interleave indices dims with data
// dims and would require more elaborate rank arithmetic; conservatively
// refuse until a model needs it.
static std::optional<DimOrigin> traceGather(mlir::Operation *op, int64_t dim,
                                            TraceContext &ctx) {
  if (op->getNumOperands() != 2)
    return std::nullopt;
  auto dataType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto idxType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  if (!dataType || !idxType)
    return std::nullopt;
  if (idxType.getRank() != 0)
    return std::nullopt;
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = a.getSInt();
  if (axis < 0)
    axis += dataType.getRank();
  int64_t dataDim = dim < axis ? dim : dim + 1;
  if (dataDim < 0 || dataDim >= dataType.getRank())
    return std::nullopt;
  return traceDimOrigin(op->getOperand(0), dataDim, ctx);
}

//===----------------------------------------------------------------------===//
// Trace registry + dispatch
//===----------------------------------------------------------------------===//

static void populateTraceRules(llvm::StringMap<TraceFn> &r) {
  // Unary same-shape.
  for (auto n :
       {"onnx.Cast", "onnx.Tanh", "onnx.Softmax", "onnx.LayerNormalization",
        "onnx.Sqrt", "onnx.Gelu", "onnx.Sigmoid", "onnx.Neg", "onnx.Erf"})
    r[n] = traceUnary;
  r["onnx.Transpose"] = traceTranspose;
  // Binary broadcast.
  for (auto n : {"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Pow"})
    r[n] = traceBinaryBroadcast;
  r["onnx.MatMul"] = traceMatMul;
  r["onnx.Reshape"] = traceReshape;
  // Conv-like (batch dim 0 passthrough only).
  for (auto n : {"onnx.Conv", "onnx.AveragePool", "onnx.MaxPool"})
    r[n] = traceConvLikeBatch;
  for (auto n : {"onnx.ReduceSum", "onnx.ReduceMean", "onnx.ReduceMax",
                 "onnx.ReduceProd"})
    r[n] = traceReduce;
  // Shape-arithmetic ops surfaced by Qwen3.5 / Phi-4 vision exports.
  r["onnx.Squeeze"] = traceSqueeze;
  r["onnx.Unsqueeze"] = traceUnsqueeze;
  r["onnx.Concat"] = traceConcat;
  r["onnx.Slice"] = traceSlice;
  r["onnx.Gather"] = traceGather;
}

static const llvm::StringMap<TraceFn> &getTraceRegistry() {
  static const llvm::StringMap<TraceFn> kReg = [] {
    llvm::StringMap<TraceFn> m;
    populateTraceRules(m);
    return m;
  }();
  return kReg;
}

static std::optional<DimOrigin> traceDimOrigin(mlir::Value v, int64_t dim,
                                               TraceContext &ctx) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v))
    return DimOrigin{static_cast<int64_t>(blockArg.getArgNumber()), dim};
  mlir::Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  llvm::StringRef name = op->getName().getStringRef();
  const auto &reg = getTraceRegistry();
  auto it = reg.find(name);
  if (it != reg.end())
    return it->second(op, dim, ctx);
  // Unknown op kind. Warn once per function per kind so the next
  // model-bring-up sees exactly which trace rules to add. Limited to
  // `onnx.*` producers — unknown non-onnx ops (memref ops, hip dialect
  // ops, etc.) would be noise this trace was never expected to handle.
  if (name.starts_with("onnx.") && ctx.warned.insert(name).second) {
    op->emitWarning() << "InferOnnxShapes: no trace rule for op '" << name
                      << "' — output dims passing through this op may be "
                         "unresolvable at runtime";
  }
  return std::nullopt;
}

/// Compute per-output, per-dim origins for the function. Outer vector
/// indexed by function result; inner vector by dim. For each dim, the
/// origin (if any) is a (graph_arg_index, dim_idx) pair into the
/// function arguments — the EP maps this directly to its
/// `dim_param_map[input_idx][dim_idx]` view. A single shared TraceContext
/// is threaded through all per-dim traces so the "unknown op" warning
/// dedups across the whole function (not per-output-dim).
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
  TraceContext ctx;
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
        dims.push_back(traceDimOrigin(retVal, d, ctx));
      else
        dims.push_back(std::nullopt);
    }
    result.push_back(std::move(dims));
  }
  return result;
}

// Attach refined output shapes from the function signature as a
// module-level ArrayAttr. Layout:
//   ArrayAttr<ArrayAttr<IntegerAttr<i64>>>
//   outer = per function-result; inner = per dim (-1 = dynamic).
// Module attributes survive the `func.func → llvm.func` conversion that
// happens later in the pipeline; `CompilerDriver::readRefinedOutputsFromModule`
// reads them back into driver members after `runMLIRPasses` returns.
static void attachShapesAttr(mlir::func::FuncOp funcOp, mlir::ModuleOp module) {
  auto *ctx = funcOp.getContext();
  mlir::Builder b(ctx);
  auto funcType = funcOp.getFunctionType();
  llvm::SmallVector<mlir::Attribute> perOutputs;
  perOutputs.reserve(funcType.getNumResults());
  for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(funcType.getResult(i));
    llvm::SmallVector<mlir::Attribute> dims;
    if (rt) {
      dims.reserve(rt.getRank());
      for (int64_t d : llvm::seq<int64_t>(0, rt.getRank())) {
        int64_t v = rt.isDynamicDim(d) ? int64_t(-1) : rt.getDimSize(d);
        dims.push_back(b.getI64IntegerAttr(v));
      }
    }
    perOutputs.push_back(b.getArrayAttr(dims));
  }
  module->setAttr("hip.refined_output_shapes", b.getArrayAttr(perOutputs));
}

// Attach per-output, per-dim SSA origins as a module-level ArrayAttr.
// Layout:
//   ArrayAttr<ArrayAttr<IntegerAttr<i64>>>
//   outer = per function-result;
//   inner = flat (arg_idx, dim_idx, mult_bits) triples, 3 slots per dim.
// `mult_bits` is the IEEE 754 binary64 bit pattern of the `mult` field
// (single int64 stream, matches the encoding the EP-side metadata
// builder already expects). `(-1, -1, 0x3FF0000000000000 /* 1.0 */)` is
// the "no traceable origin" sentinel.
//
// The MLIR function may carry leading non-tensor args (notably
// `!hip.context` added by AddHipContextArg). The EP-side metadata
// builder indexes inputs by their position in the MORPHIZEN GRAPH,
// which excludes those non-tensor args. Subtract the context offset
// before writing so the EP can map arg_idx directly to input_idx.
static void attachOriginsAttr(mlir::func::FuncOp funcOp,
                              mlir::ModuleOp module) {
  auto *ctx = funcOp.getContext();
  mlir::Builder b(ctx);
  auto origins = traceOutputOrigins(funcOp);

  int64_t contextOffset = 0;
  auto funcType = funcOp.getFunctionType();
  for (mlir::Type t : funcType.getInputs()) {
    if (mlir::isa<mlir::hip::ContextType>(t))
      ++contextOffset;
    else
      break; // non-tensor args are always leading
  }

  const int64_t kOneBits =
      static_cast<int64_t>(0x3FF0000000000000ULL); // bit_cast<int64>(1.0)

  llvm::SmallVector<mlir::Attribute> perOutputs;
  perOutputs.reserve(origins.size());
  for (const auto &dims : origins) {
    llvm::SmallVector<mlir::Attribute> flat;
    flat.reserve(dims.size() * 3);
    for (const auto &o : dims) {
      int64_t arg = (o && o->arg_idx >= contextOffset)
                        ? (o->arg_idx - contextOffset)
                        : int64_t(-1);
      int64_t dim = (o && o->arg_idx >= contextOffset) ? o->dim_idx : -1;
      double mult = (o && o->arg_idx >= contextOffset) ? o->mult : 1.0;
      int64_t multBits;
      std::memcpy(&multBits, &mult, sizeof(double));
      if (arg < 0)
        multBits = kOneBits;
      flat.push_back(b.getI64IntegerAttr(arg));
      flat.push_back(b.getI64IntegerAttr(dim));
      flat.push_back(b.getI64IntegerAttr(multBits));
    }
    perOutputs.push_back(b.getArrayAttr(flat));
  }
  module->setAttr("hip.refined_output_dim_origins", b.getArrayAttr(perOutputs));

  LLVM_DEBUG({
    for (size_t i = 0; i < origins.size(); ++i)
      for (size_t d = 0; d < origins[i].size(); ++d) {
        const auto &o = origins[i][d];
        if (!o)
          continue;
        llvm::dbgs() << "[" DEBUG_TYPE "] output[" << i << "].dim[" << d
                     << "] origin = (" << o->arg_idx << ", " << o->dim_idx
                     << ", *=" << o->mult << ")\n";
      }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

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
  // dims whose dim_param names don't match any input's dim_param (typical
  // case: ViT-style exports whose output dim_params don't match the
  // input). Encoding goes on the enclosing module as ArrayAttrs — those
  // survive the `func.func → llvm.func` conversion that happens later in
  // the pipeline; reading them from the post-`runMLIRPasses` module is
  // safe.
  if (auto module = funcOp->getParentOfType<mlir::ModuleOp>()) {
    attachShapesAttr(funcOp, module);
    attachOriginsAttr(funcOp, module);
  }
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
