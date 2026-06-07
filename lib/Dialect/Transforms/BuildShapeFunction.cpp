/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- BuildShapeFunction.cpp - Emit @infer_shapes shape program ---------===//
//
// Emits a pure, data-independent `@infer_shapes` function that computes every
// `@main_graph` output dimension as index arithmetic over the input tensor
// dimensions. The MorphiZen EP calls the exported `inference_infer_shapes`
// wrapper (emitted later by GenerateInterface) before `ctx.GetOutput()` to
// size output buffers whose dims are not a verbatim copy of an input dim —
// flattens, reshapes, patch mergers — which `DimSource` cannot express.
// `DimSource` stays the O(1) fast path; this pass is best-effort and never
// hard-fails: any dim it cannot reduce to input arithmetic becomes the
// kDynamic sentinel and the EP falls back to `DimSource`.
//
// Algorithm
// ---------
// Every output dim sits at the top of a chain of shape-only ops (reshape,
// expand/collapse, dim arithmetic) layered over the heavy hip.* compute. We
// recover each one as a backward slice rooted at the function inputs:
//   1. Clone @main_graph into a throwaway scratch func (the real compute graph
//      is never mutated).
//   2. Replace the scratch terminator with `tensor.dim %ret, d` for every
//      output operand/dim, making those index values the scratch results.
//   3. Greedily apply the result-dim reification + reshape canonicalization
//      folds. Each `tensor.dim` collapses to arithmetic over `tensor.dim %arg,
//      c` leaves; the hip.* compute is not in that slice and folds away.
//   4. For each resolved dim, walk its operand DAG post-order. The dim is
//      RESOLVED iff every path bottoms out at a constant or a `tensor.dim
//      %inputArg, c` (an input shape dim), joined only by side-effect-free,
//      region-free arith/affine ops. Clone that slice into @infer_shapes,
//      remapping each shape-dim leaf to its dim arg. Any other leaf (a hip.*
//      result, a `tensor.dim` on a non-input value such as a hip.loop result,
//      a non-arith op such as math.ceil) makes the dim unresolvable → kDynamic.
//   5. Erase the scratch func.
//
// Signature contract (consumed by GenerateInterface):
//   (!hip.context, index×totalInputDims) -> (index×outputDims)
//   * arg 0 is an unused `!hip.context` (see below);
//   * one `index` arg per (input tensor, dim), row-major over all inputs;
//   * one `index` result per (output tensor, dim), row-major over all outputs.
//   Non-tensor inputs and the context contribute no dims.
//
// The `!hip.context` arg 0 exists only to uphold the module-wide invariant
// (established by `hip-add-context-arg`) that every `func.func` takes a context
// as arg 0; downstream passes assert it. `@infer_shapes` is synthesized after
// that pass, has no `hip.*` ops, never reads the context — it is lowered to
// `!llvm.ptr` and the GenerateInterface wrapper passes null.
//
// Before (@main_graph: [B,S,D] -> Reshape -> [B*S, D], D=4096):
//   func.func @main_graph(%ctx: !hip.context, %a: tensor<?x?x4096xf16>)
//       -> tensor<?x4096xf16> { ... return %r : tensor<?x4096xf16> }
// After (sibling added; @main_graph untouched):
//   func.func @infer_shapes(%ctx: !hip.context, %d0: index, %d1: index,
//                           %d2: index) -> (index, index) {
//     %bs = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%d0, %d1]
//     %c4096 = arith.constant 4096 : index
//     return %bs, %c4096 : index, index
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-build-shape-fn"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_BUILDSHAPEFUNCTIONPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Sentinel for a dim the shape fn cannot compute from input dims (the EP then
// falls back to DimSource). Matches ShapedType's dynamic marker (INT64_MIN).
static constexpr int64_t kUnknownDim = ShapedType::kDynamic;

/// The entry graph whose output shapes we model. Loop bodies
/// (`@main_graph_loop_body_n*`) are NOT the entry graph.
static func::FuncOp findMainGraph(ModuleOp module) {
  func::FuncOp result;
  module.walk([&](func::FuncOp f) -> WalkResult {
    auto nameAttr = f->getAttrOfType<StringAttr>("onnx.graph.name");
    if ((nameAttr && nameAttr.getValue() == "main_graph") ||
        f.getSymName() == "main_graph") {
      result = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Greedily fold the scratch func's `tensor.dim` queries into arithmetic over
/// `tensor.dim %arg, c` by running the result-dim reification patterns plus the
/// reshape (expand/collapse) canonicalizers. This is the same fold set as
/// --hip-resolve-tensor-dims, kept in lock-step so the shape program sees the
/// identical dim arithmetic the compute graph does. Best-effort: a fold that
/// does not converge just leaves some dims unresolved for the per-dim fallback,
/// so a non-clean run is benign rather than fatal.
static void runDimResolution(func::FuncOp f) {
  MLIRContext *ctx = f.getContext();
  RewritePatternSet patterns(ctx);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  tensor::DimOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(f, std::move(patterns));
}

/// Post-order walk of `v`'s operand DAG. Returns true iff `v` is a pure
/// function of constants and input-tensor dims. On success:
///   - `orderedOps` accumulates the pure arith/affine ops to clone, operands
///     before uses (topo order);
///   - `leaves` records each `tensor.dim %arg, %c` leaf (NOT cloned — remapped
///     to a scalar arg by the caller).
/// `memo` dedups shared subexpressions (SSA is acyclic, so no cycle guard).
static bool gatherSlice(Value v,
                        const DenseMap<BlockArgument, unsigned> &tensorArgPos,
                        llvm::SetVector<Operation *> &orderedOps,
                        SmallVectorImpl<tensor::DimOp> &leaves,
                        DenseMap<Operation *, bool> &memo) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false; // a block argument as a dim value is not shape-resolvable

  auto it = memo.find(def);
  if (it != memo.end())
    return it->second;

  // Leaf: `tensor.dim %inputArg, %constIdx`.
  if (auto dimOp = dyn_cast<tensor::DimOp>(def)) {
    auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
    std::optional<int64_t> idx = getConstantIntValue(dimOp.getIndex());
    bool ok = blockArg && tensorArgPos.contains(blockArg) && idx.has_value();
    memo[def] = ok;
    if (ok)
      leaves.push_back(dimOp);
    return ok;
  }

  // Interior: any side-effect-free, region-free arith/affine op (covers
  // arith.constant, muli/addi/divui/index_cast, affine.apply, …).
  StringRef ns = def->getDialect() ? def->getDialect()->getNamespace() : "";
  bool pureArithAffine = isMemoryEffectFree(def) && def->getNumRegions() == 0 &&
                         (ns == "arith" || ns == "affine");
  if (!pureArithAffine) {
    memo[def] = false;
    return false;
  }
  for (Value operand : def->getOperands()) {
    if (!gatherSlice(operand, tensorArgPos, orderedOps, leaves, memo)) {
      memo[def] = false;
      return false;
    }
  }
  orderedOps.insert(def); // post-order: operands already inserted
  memo[def] = true;
  return true;
}

struct BuildShapeFunctionPass
    : public impl::BuildShapeFunctionPassBase<BuildShapeFunctionPass> {
  void runOnOperation() override;
};

void BuildShapeFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp mainGraph = findMainGraph(module);
  if (!mainGraph || mainGraph.empty())
    return;
  // Idempotent: do not regenerate if already present.
  if (module.lookupSymbol("infer_shapes"))
    return;

  MLIRContext *ctx = module.getContext();
  Location loc = mainGraph.getLoc();
  auto idxTy = IndexType::get(ctx);

  // Every output must be a ranked tensor for the flattened-dims contract.
  auto mainRet =
      cast<func::ReturnOp>(mainGraph.getBody().front().getTerminator());
  for (Value out : mainRet.getOperands()) {
    if (!isa<RankedTensorType>(out.getType())) {
      LLVM_DEBUG(DBGS() << "skip: non-tensor @main_graph output\n");
      return;
    }
  }

  // 1. Clone @main_graph into a private scratch func (never mutate the real
  //    compute graph). Rename before insertion to avoid a duplicate symbol.
  func::FuncOp scratch = mainGraph.clone();
  scratch.setSymName("__hipdnn_shape_probe");
  scratch.setPrivate();
  module.push_back(scratch);

  Block &scratchBlock = scratch.getBody().front();
  auto scratchRet = cast<func::ReturnOp>(scratchBlock.getTerminator());

  // 2. Materialize `tensor.dim` on every scratch return operand/dim and make
  //    them the scratch func results (so the resolved values are readable off
  //    the persistent return op after the greedy fold).
  OpBuilder b(scratchRet);
  SmallVector<Value> dimQueries;
  for (Value out : scratchRet.getOperands()) {
    auto t = cast<RankedTensorType>(out.getType());
    for (int64_t d : llvm::seq<int64_t>(0, t.getRank()))
      dimQueries.push_back(tensor::DimOp::create(b, loc, out, d));
  }
  func::ReturnOp::create(b, loc, dimQueries);
  scratchRet.erase();
  SmallVector<Type> idxResults(dimQueries.size(), idxTy);
  scratch.setType(
      FunctionType::get(ctx, scratch.getArgumentTypes(), idxResults));

  // 3. Fold the dim queries to arithmetic over input-arg dims.
  runDimResolution(scratch);

  // Resolved per-output-dim values, flattened (read off the persistent return).
  auto resolvedRet =
      cast<func::ReturnOp>(scratch.getBody().front().getTerminator());
  SmallVector<Value> resolved(resolvedRet.getOperands().begin(),
                              resolvedRet.getOperands().end());

  // 4. Input-tensor arg layout on the scratch args (skip !hip.context and any
  //    non-tensor arg). param index of (input tensor t, dim d) = prefix[t] + d.
  DenseMap<BlockArgument, unsigned> tensorArgPos;
  SmallVector<unsigned> inRanks;
  for (BlockArgument a : scratch.getBody().front().getArguments()) {
    if (auto t = dyn_cast<RankedTensorType>(a.getType())) {
      tensorArgPos[a] = inRanks.size();
      inRanks.push_back(t.getRank());
    }
  }
  SmallVector<unsigned> inPrefix(inRanks.size() + 1, 0);
  for (size_t i : llvm::seq<size_t>(0, inRanks.size()))
    inPrefix[i + 1] = inPrefix[i] + inRanks[i];
  unsigned totalInDims = inPrefix.back();

  // 5. Create @infer_shapes(!hip.context, index x totalInDims)
  //                         -> (index x resolved.size()).
  //    arg 0 is the unused !hip.context (module-wide invariant; see header);
  //    the scalar dim args follow it, hence the `1 +` offset in paramFor.
  SmallVector<Type> argTypes;
  argTypes.push_back(ContextType::get(ctx));
  argTypes.append(totalInDims, idxTy);
  SmallVector<Type> resTypes(resolved.size(), idxTy);
  auto fn = func::FuncOp::create(loc, "infer_shapes",
                                 FunctionType::get(ctx, argTypes, resTypes));
  module.push_back(fn);
  Block *fnBlock = fn.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(fnBlock);

  auto paramFor = [&](unsigned tensorPos, int64_t dimIdx) -> Value {
    return fnBlock->getArgument(1 + inPrefix[tensorPos] + dimIdx);
  };

  // 6. For each resolved dim, clone its pure slice into @infer_shapes (or emit
  //    kDynamic on fallback).
  SmallVector<Value> results;
  results.reserve(resolved.size());
  for (Value rv : resolved) {
    llvm::SetVector<Operation *> ordered;
    SmallVector<tensor::DimOp> leaves;
    DenseMap<Operation *, bool> memo;
    if (!gatherSlice(rv, tensorArgPos, ordered, leaves, memo)) {
      results.push_back(arith::ConstantIndexOp::create(fb, loc, kUnknownDim));
      continue;
    }
    IRMapping map;
    for (tensor::DimOp dimOp : leaves) {
      auto blockArg = cast<BlockArgument>(dimOp.getSource());
      int64_t dimIdx = *getConstantIntValue(dimOp.getIndex());
      map.map(dimOp.getResult(), paramFor(tensorArgPos[blockArg], dimIdx));
    }
    for (Operation *op : ordered)
      fb.clone(*op, map);
    results.push_back(map.lookupOrDefault(rv));
  }
  func::ReturnOp::create(fb, loc, results);

  // 7. Drop the scratch func.
  scratch.erase();
}

} // namespace
} // namespace hip
} // namespace mlir
