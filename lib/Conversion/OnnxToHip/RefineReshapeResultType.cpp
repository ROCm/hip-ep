/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RefineReshapeResultType.cpp - Pin static onnx result dims ---------===//
//
// Pre-lowering pattern set: recover the static tensor dims that ONNX export
// dropped on a handful of shape-deterministic ops, so downstream pre-lowering
// rewrites that REQUIRE a static dim can fire. These patterns only ever make a
// result type MORE static (never change the value semantics), which is exactly
// what a full shape-inference pass would do — they exist because no ONNX-level
// shape-inference pass runs between the `convert-onnx-to-hip` pre-lowering
// rounds, so static info must be threaded forward by value-based rewrites.
//
// Relation to upstream shape inference
// ------------------------------------
// This is a deliberately narrow, value-based stand-in for a general ONNX
// shape-inference pass. The "result types only ever get MORE static, never
// change value semantics" contract is the same monotone-refinement invariant
// that MLIR's `InferTypeOpInterface`
// (mlir/include/mlir/Interfaces/InferTypeOpInterface.h) and StableHLO's
// `--stablehlo-refine-shapes` pass (`hlo::inferMostSpecificType` in
// StablehloRefineShapes.cpp) implement generically via op interfaces. We do
// NOT reuse those: the project matches ONNX ops by name through the generic
// `Operation` API and takes no onnx-mlir dependency, so onnx-mlir's
// `ShapeInferenceOpInterface` is unavailable, and the `onnx.*` ops here carry
// no `InferTypeOpInterface` model. Instead we re-derive the result extents for
// the handful of ops a norm/projector chain needs by reading inline constants
// and producer static dims directly — the targeted, dependency-free analogue
// of running shape inference over a `Reshape -> Transpose -> elementwise`
// region.
//
// Three patterns, all in `populateRefineReshapeResultTypePatterns`:
//
// 1. RefineReshapeResultFromShapeOperand — refine an `onnx.Reshape`'s RESULT
//    type from its (possibly partially-dynamic) shape operand.
// 2. RefineTransposeResultType — refine an `onnx.Transpose`'s result from its
//    (now-more-static) input + perm.
// 3. RefineElementwiseResultType — refine an elementwise op's result from the
//    numpy-broadcast of its (now-more-static) operands.
//
// Why this matters
// ----------------
// Some exporters emit a Reshape whose target shape is assembled at "runtime"
// from a `Shape -> Gather/Slice -> Concat` chain mixing dynamic dims (e.g.
// batch) with literal constants and a single inferred `-1`. ONNX shape
// inference at export leaves the Reshape result fully dynamic (`?x?x?`) because
// the Concat operand is not a single constant. Downstream rewrites then bail:
//   - `AveragePoolToReshapeMean` requires static channel + spatial dims;
//   - `ReduceMeanToReduceSumDiv` requires a static reduce axis.
// With the chain left dynamic those ops survive to one-shot-bufferize, which
// aborts with "op was not bufferized" (the EP then silently declines the
// graph). Yet the missing dims ARE statically provable: a Concat element that
// is a literal constant pins that output dim; a `Gather/Slice(Shape(src))`
// element pins it to `src`'s static dim(s); and a single `-1` is recoverable
// by element-count arithmetic (with batch-dim cancellation when the kept
// prefix dims come from the reshape's own input — the classic
// `Reshape(x, Concat(Slice(Shape(x), [0:k]), -1))` norm/projector idiom).
//
// A norm chain is `Reshape -> Transpose -> Mul(square) -> ReduceMean`; refining
// only the Reshape is not enough because `ReduceMean`'s matcher reads ITS input
// type, and Transpose/Mul drop the static dims again unless refined. Patterns 2
// and 3 thread the statics forward so ReduceMean's reduce dim becomes static in
// a later round and the decomposition fires.
//
// Must run BEFORE `lowerOnnxConstants` so the literal `onnx.Constant` values in
// the Concat (and the Gather/Slice index / Shape source) are still inline.
//
// Before:
//   %ap   = onnx.AveragePool(...)  : tensor<?x1152x16x16xf16>
//   %sh   = onnx.Shape(%ap)        : tensor<4xi64>
//   %sl   = onnx.Slice(%sh,0,2,0)  : tensor<2xi64>          // = [B, 1152]
//   %m1   = onnx.Constant -1       : tensor<1xi64>
//   %shp  = onnx.Concat(%sl, %m1)  : tensor<3xi64>          // = [B, 1152, -1]
//   %r    = onnx.Reshape(%ap, %shp) : (...) -> tensor<?x?x?xf16>
//   %t    = onnx.Transpose(%r) {perm=[0,2,1]} : tensor<?x?x?xf16>
//   %sq   = onnx.Mul(%t, %t)       : tensor<?x?x?xf16>
//   %rm   = onnx.ReduceMean(%sq, -1) {keepdims} : tensor<?x?x1xf16>
//
// After (only result types change; operands untouched):
//   %r    = onnx.Reshape(%ap, %shp) : (...) -> tensor<?x1152x256xf16>
//   %t    = onnx.Transpose(%r) {perm=[0,2,1]} : tensor<?x256x1152xf16>
//   %sq   = onnx.Mul(%t, %t)       : tensor<?x256x1152xf16>
//   %rm   = onnx.ReduceMean(%sq, -1) {keepdims} : tensor<?x256x1xf16>
//   (ReduceMeanToReduceSumDiv now fires: reduce dim 1152 is static.)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "refine-reshape-result-type"

STATISTIC(NumReshapeResultRefinements,
          "Number of onnx.Reshape result types refined with statically "
          "resolvable shape-operand dims");
STATISTIC(NumTransposeResultRefinements,
          "Number of onnx.Transpose result types refined from a more-static "
          "input");
STATISTIC(NumElementwiseResultRefinements,
          "Number of elementwise op result types refined from more-static "
          "operands");

namespace mlir {
namespace hip {

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Read a single integer from a 1-element (rank-0 or rank-1) onnx.Constant.
static std::optional<int64_t> getConstScalarInt(mlir::Operation *op) {
  if (!op || op->getName().getStringRef() != "onnx.Constant")
    return std::nullopt;
  auto attr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op->getAttr("value"));
  if (!attr || attr.getNumElements() != 1 ||
      !attr.getElementType().isIntOrIndex())
    return std::nullopt;
  return (*attr.getValues<llvm::APInt>().begin()).getSExtValue();
}

/// True iff `t` is a ranked tensor with at least one dynamic dim.
static bool isPartlyDynamic(mlir::Type t) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
  return rt && !rt.hasStaticShape();
}

//===----------------------------------------------------------------------===//
// onnx.Reshape result-type refinement
//===----------------------------------------------------------------------===//

/// Per-output-position description of a Reshape shape-operand element.
struct DimDesc {
  // Resolution kinds (mutually exclusive):
  bool isStatic = false; ///< pinned literal >= 0 (staticVal valid; 0 = copy)
  int64_t staticVal = 0;
  bool isNegOne = false; ///< the single inferred (-1) position
  // Dynamic position equal to the reshape DATA tensor's dim[dataAxis]
  // (provable only when the Shape source is SSA-identical to the data).
  // Used to cancel that data dim out of the -1 element-count arithmetic.
  bool cancellable = false;
  int64_t dataAxis = -1;
};

/// Parse `op` as `onnx.Shape(src)` (optionally with a `start`/`end` sub-range).
/// On success fills `srcType` / `srcVal` and the `start` offset (shape-vector
/// position i maps to src dim `start + i`). Only the common case where the
/// upper bound spans to the end is supported; anything else bails.
static bool parseShapeOp(mlir::Operation *op, mlir::RankedTensorType &srcType,
                         mlir::Value &srcVal, int64_t &start) {
  if (!op || op->getName().getStringRef() != "onnx.Shape" ||
      op->getNumOperands() != 1)
    return false;
  srcVal = op->getOperand(0);
  srcType = mlir::dyn_cast<mlir::RankedTensorType>(srcVal.getType());
  if (!srcType)
    return false;
  int64_t rank = srcType.getRank();
  start = 0;
  if (auto s = op->getAttrOfType<mlir::IntegerAttr>("start"))
    start = s.getSInt();
  if (start < 0)
    start += rank;
  if (auto e = op->getAttrOfType<mlir::IntegerAttr>("end")) {
    int64_t end = e.getSInt();
    if (end < 0)
      end += rank;
    if (end != rank)
      return false;
  }
  return start >= 0 && start <= rank;
}

/// Build the descriptor for the value of `src`'s dim at `srcAxis`, knowing
/// whether `src` is the same SSA value as the reshape's data operand.
static DimDesc dimDescFromShapeAxis(mlir::RankedTensorType srcType,
                                    mlir::Value srcVal,
                                    mlir::Value reshapeData, int64_t srcAxis) {
  DimDesc d;
  if (srcAxis < 0 || srcAxis >= srcType.getRank())
    return d; // unknown
  if (!srcType.isDynamicDim(srcAxis)) {
    d.isStatic = true;
    d.staticVal = srcType.getDimSize(srcAxis);
    return d;
  }
  // Dynamic src dim: only cancellable when the shape source IS the reshape's
  // own data tensor (then output dim == data dim[srcAxis] exactly).
  if (srcVal == reshapeData) {
    d.cancellable = true;
    d.dataAxis = srcAxis;
  }
  return d; // else unknown dynamic
}

/// Resolve a single 1-element shape-vector value to a DimDesc.
static DimDesc resolveOneElem(mlir::Value v, mlir::Value reshapeData) {
  DimDesc d;
  mlir::Operation *op = v.getDefiningOp();
  if (!op)
    return d;
  llvm::StringRef name = op->getName().getStringRef();

  if (name == "onnx.Constant") {
    if (auto c = getConstScalarInt(op)) {
      if (*c == -1)
        d.isNegOne = true;
      else {
        d.isStatic = true;
        d.staticVal = *c;
      }
    }
    return d;
  }
  if ((name == "onnx.Unsqueeze" || name == "onnx.Squeeze") &&
      op->getNumOperands() >= 1)
    return resolveOneElem(op->getOperand(0), reshapeData);

  if (name == "onnx.Gather" && op->getNumOperands() == 2) {
    mlir::RankedTensorType srcType;
    mlir::Value srcVal;
    int64_t start;
    if (!parseShapeOp(op->getOperand(0).getDefiningOp(), srcType, srcVal,
                      start))
      return d;
    std::optional<int64_t> idx =
        getConstScalarInt(op->getOperand(1).getDefiningOp());
    if (!idx)
      return d;
    int64_t shapeLen = srcType.getRank() - start;
    int64_t i = *idx;
    if (i < 0)
      i += shapeLen;
    return dimDescFromShapeAxis(srcType, srcVal, reshapeData, start + i);
  }
  return d; // unknown
}

/// Expand an `onnx.Slice(onnx.Shape(src), starts, ends [, axes [, steps]])`
/// operand of a Concat into one DimDesc per sliced shape position. Returns
/// false when the slice is not a simple contiguous (step 1, axis 0) slice of a
/// Shape op.
static bool expandSliceOfShape(mlir::Operation *sliceOp,
                               mlir::Value reshapeData,
                               llvm::SmallVectorImpl<DimDesc> &out) {
  if (!sliceOp || sliceOp->getName().getStringRef() != "onnx.Slice" ||
      sliceOp->getNumOperands() < 3)
    return false;
  mlir::RankedTensorType srcType;
  mlir::Value srcVal;
  int64_t start;
  if (!parseShapeOp(sliceOp->getOperand(0).getDefiningOp(), srcType, srcVal,
                    start))
    return false;
  int64_t shapeLen = srcType.getRank() - start;

  std::optional<int64_t> s =
      getConstScalarInt(sliceOp->getOperand(1).getDefiningOp());
  std::optional<int64_t> e =
      getConstScalarInt(sliceOp->getOperand(2).getDefiningOp());
  if (!s || !e)
    return false;
  // axes operand (optional) must be [0] (slicing the shape vector itself).
  if (sliceOp->getNumOperands() >= 4) {
    std::optional<int64_t> ax =
        getConstScalarInt(sliceOp->getOperand(3).getDefiningOp());
    if (!ax || *ax != 0)
      return false;
  }
  // steps operand (optional) must be [1].
  if (sliceOp->getNumOperands() >= 5) {
    std::optional<int64_t> st =
        getConstScalarInt(sliceOp->getOperand(4).getDefiningOp());
    if (!st || *st != 1)
      return false;
  }

  auto clamp = [&](int64_t x) -> int64_t {
    if (x < 0)
      x += shapeLen;
    if (x < 0)
      x = 0;
    if (x > shapeLen)
      x = shapeLen;
    return x;
  };
  int64_t lo = clamp(*s), hi = clamp(*e);
  if (hi < lo)
    return false;
  for (int64_t p = lo; p < hi; ++p)
    out.push_back(dimDescFromShapeAxis(srcType, srcVal, reshapeData, start + p));
  return true;
}

/// Collect a DimDesc per output position from the Reshape shape operand, which
/// is either an `onnx.Concat(axis=0)` of {1-element pieces | Slice(Shape) |
/// multi-element Constant} or a rank-1 `onnx.Constant`. Returns false when the
/// operand is neither (or any sub-operand is unrecognized).
static bool collectDims(mlir::Value shapeOperand, mlir::Value reshapeData,
                        llvm::SmallVectorImpl<DimDesc> &dims) {
  mlir::Operation *def = shapeOperand.getDefiningOp();
  if (!def)
    return false;
  llvm::StringRef name = def->getName().getStringRef();

  auto pushConstVector = [&](mlir::Operation *cst) -> bool {
    auto attr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(cst->getAttr("value"));
    if (!attr || attr.getType().getRank() != 1 ||
        !attr.getElementType().isIntOrIndex())
      return false;
    for (const llvm::APInt &v : attr.getValues<llvm::APInt>()) {
      DimDesc d;
      int64_t cv = v.getSExtValue();
      if (cv == -1)
        d.isNegOne = true;
      else {
        d.isStatic = true;
        d.staticVal = cv;
      }
      dims.push_back(d);
    }
    return true;
  };

  if (name == "onnx.Constant")
    return pushConstVector(def) && !dims.empty();

  if (name == "onnx.Concat") {
    if (auto axis = def->getAttrOfType<mlir::IntegerAttr>("axis"))
      if (axis.getSInt() != 0)
        return false;
    for (mlir::Value operand : def->getOperands()) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (!t || t.getRank() != 1 || t.isDynamicDim(0))
        return false;
      int64_t n = t.getDimSize(0);
      if (n == 1) {
        dims.push_back(resolveOneElem(operand, reshapeData));
        continue;
      }
      // Multi-element operand: a contiguous Slice(Shape(src)) or a literal
      // constant vector. Both expand to `n` positions.
      mlir::Operation *od = operand.getDefiningOp();
      if (od && od->getName().getStringRef() == "onnx.Slice") {
        llvm::SmallVector<DimDesc> sliced;
        if (!expandSliceOfShape(od, reshapeData, sliced) ||
            static_cast<int64_t>(sliced.size()) != n)
          return false;
        dims.append(sliced.begin(), sliced.end());
        continue;
      }
      if (od && od->getName().getStringRef() == "onnx.Constant") {
        size_t before = dims.size();
        if (!pushConstVector(od) ||
            static_cast<int64_t>(dims.size() - before) != n)
          return false;
        continue;
      }
      return false; // unrecognized multi-element operand
    }
    return !dims.empty();
  }

  return false;
}

struct RefineReshapeResultFromShapeOperand : public mlir::RewritePattern {
  RefineReshapeResultFromShapeOperand(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp->getNumOperands() < 2 || reshapeOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.arity");

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        reshapeOp->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.result_unranked");
    if (resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.already_static");

    mlir::Value data = reshapeOp->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());

    // allowzero == 0 (default): a 0 in the shape vector copies the input dim
    // at the same index. allowzero != 0: a 0 is a literal dim (left dynamic).
    bool allowZero = false;
    if (auto a = reshapeOp->getAttrOfType<mlir::IntegerAttr>("allowzero"))
      allowZero = a.getSInt() != 0;

    llvm::SmallVector<DimDesc> dims;
    if (!collectDims(reshapeOp->getOperand(1), data, dims))
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.shape_operand");
    if (static_cast<int64_t>(dims.size()) != resultType.getRank())
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.rank_mismatch");

    llvm::SmallVector<int64_t> newShape(resultType.getShape().begin(),
                                        resultType.getShape().end());

    // Pin every directly-resolvable static position first.
    int64_t negOnePos = -1;
    int negOneCount = 0;
    for (auto idx : llvm::seq<int64_t>(0, resultType.getRank())) {
      const DimDesc &d = dims[idx];
      if (d.isNegOne) {
        ++negOneCount;
        negOnePos = idx;
        continue;
      }
      if (!resultType.isDynamicDim(idx))
        continue; // result already pins it
      if (!d.isStatic)
        continue; // dynamic / cancellable / unknown -> leave dynamic
      int64_t v = d.staticVal;
      if (v == 0 && !allowZero) {
        if (inputType && idx < inputType.getRank() &&
            !inputType.isDynamicDim(idx))
          v = inputType.getDimSize(idx);
        else
          continue;
      }
      if (v <= 0)
        continue;
      newShape[idx] = v;
    }

    // Resolve a single -1 by element-count arithmetic, cancelling any dynamic
    // data dims that flow bijectively into output positions (so the batch dim
    // drops out of both the numerator and the kept-dim product).
    if (negOneCount == 1 && inputType) {
      llvm::SmallSet<int64_t, 4> cancelled;
      bool bijective = true;
      for (const DimDesc &d : dims) {
        if (!d.cancellable)
          continue;
        if (!cancelled.insert(d.dataAxis).second) {
          bijective = false; // same data dim mapped twice -> not bijective
          break;
        }
      }
      if (bijective) {
        int64_t numer = 1;
        bool numerOk = true;
        for (auto a : llvm::seq<int64_t>(0, inputType.getRank())) {
          if (cancelled.count(a))
            continue;
          if (inputType.isDynamicDim(a)) {
            numerOk = false;
            break;
          }
          numer *= inputType.getDimSize(a);
        }
        int64_t denom = 1;
        for (const DimDesc &d : dims)
          if (d.isStatic && d.staticVal > 0)
            denom *= d.staticVal;
        if (numerOk && denom > 0 && numer % denom == 0) {
          int64_t inferred = numer / denom;
          if (inferred > 0)
            newShape[negOnePos] = inferred;
        }
      }
    }

    if (llvm::ArrayRef<int64_t>(newShape) == resultType.getShape())
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.no_new_static");

    auto newType =
        mlir::RankedTensorType::get(newShape, resultType.getElementType());
    rewriter.modifyOpInPlace(
        reshapeOp, [&] { reshapeOp->getResult(0).setType(newType); });
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] refined onnx.Reshape result "
                            << resultType << " -> " << newType << "\n");
    ++NumReshapeResultRefinements;
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Transpose result-type refinement
//===----------------------------------------------------------------------===//

// Pull each result dim from the permuted input dim when the input is more
// static than the (export-dropped) result. Only the result type changes.
//
// Before (%r already refined to tensor<?x1152x256xf16> by pattern 1):
//   %t = onnx.Transpose(%r) {perm=[0,2,1]} : tensor<?x?x?xf16>
// After:
//   %t = onnx.Transpose(%r) {perm=[0,2,1]} : tensor<?x256x1152xf16>
struct RefineTransposeResultType : public mlir::RewritePattern {
  RefineTransposeResultType(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "transpose.arity");
    auto inT =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outT =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inT || !outT || inT.getRank() != outT.getRank())
      return rewriter.notifyMatchFailure(op, "transpose.types");
    if (outT.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "transpose.already_static");

    int64_t rank = inT.getRank();
    llvm::SmallVector<int64_t> perm;
    if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
      for (mlir::Attribute a : permAttr) {
        auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
        if (!ia)
          return rewriter.notifyMatchFailure(op, "transpose.perm_kind");
        perm.push_back(ia.getSInt());
      }
    } else {
      for (auto i : llvm::seq<int64_t>(0, rank))
        perm.push_back(rank - 1 - i); // default: reverse
    }
    if (static_cast<int64_t>(perm.size()) != rank)
      return rewriter.notifyMatchFailure(op, "transpose.perm_rank");

    llvm::SmallVector<int64_t> newShape(outT.getShape().begin(),
                                        outT.getShape().end());
    for (auto i : llvm::seq<int64_t>(0, rank)) {
      int64_t p = perm[i];
      if (p < 0 || p >= rank)
        return rewriter.notifyMatchFailure(op, "transpose.perm_oob");
      if (mlir::ShapedType::isDynamic(newShape[i]) && !inT.isDynamicDim(p))
        newShape[i] = inT.getDimSize(p);
    }
    if (llvm::ArrayRef<int64_t>(newShape) == outT.getShape())
      return rewriter.notifyMatchFailure(op, "transpose.no_new_static");

    auto newType =
        mlir::RankedTensorType::get(newShape, outT.getElementType());
    rewriter.modifyOpInPlace(op, [&] { op->getResult(0).setType(newType); });
    ++NumTransposeResultRefinements;
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise op result-type refinement (numpy broadcast)
//===----------------------------------------------------------------------===//

/// ONNX ops that are pure elementwise with numpy-broadcast semantics. The
/// result shape is fully determined by the operand shapes; the result dtype is
/// preserved as-is (covers value-preserving Cast etc.). Restricted to the ops a
/// norm / projector chain needs, plus common siblings — adding an op here only
/// recovers static dims it would otherwise lose, never changes semantics.
static bool isBroadcastElementwise(llvm::StringRef name) {
  static constexpr llvm::StringLiteral kOps[] = {
      "onnx.Add",        "onnx.Sub",  "onnx.Mul",  "onnx.Div",
      "onnx.Pow",        "onnx.Sqrt", "onnx.Cast", "onnx.Relu",
      "onnx.Sigmoid",    "onnx.Tanh", "onnx.Exp",  "onnx.Log",
      "onnx.Reciprocal", "onnx.Abs",  "onnx.Neg",  "onnx.Erf"};
  return llvm::is_contained(kOps, name);
}

// Resolve each dynamic result dim from the numpy-broadcast of the operands'
// (now-more-static) dims: a single distinct static value > 1 wins; all-unit
// contributors give 1; a static conflict or a dynamic non-unit leaves it
// dynamic. Only the result type changes.
//
// Before (%a, %b refined upstream; result still export-dropped):
//   %a = ... : tensor<?x256x1152xf16>
//   %b = ... : tensor<?x256x1xf16>     // broadcasts over the last dim
//   %m = onnx.Mul(%a, %b) : tensor<?x?x?xf16>
// After:
//   %m = onnx.Mul(%a, %b) : tensor<?x256x1152xf16>
struct RefineElementwiseResultType : public mlir::RewritePattern {
  RefineElementwiseResultType(mlir::MLIRContext *ctx)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isBroadcastElementwise(op->getName().getStringRef()))
      return rewriter.notifyMatchFailure(op, "ew.not_elementwise");
    if (op->getNumResults() != 1 || op->getNumOperands() < 1)
      return rewriter.notifyMatchFailure(op, "ew.arity");
    auto outT =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!outT || outT.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "ew.out");
    int64_t rank = outT.getRank();

    // Gather ranked operands aligned to the result rank from the trailing end
    // (numpy broadcast). An operand of lower rank contributes to the trailing
    // positions only; missing leading positions are implicitly 1.
    llvm::SmallVector<mlir::RankedTensorType> operands;
    for (mlir::Value v : op->getOperands()) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
      if (!t || t.getRank() > rank)
        return rewriter.notifyMatchFailure(op, "ew.operand_rank");
      operands.push_back(t);
    }
    if (operands.empty())
      return rewriter.notifyMatchFailure(op, "ew.no_operands");

    llvm::SmallVector<int64_t> newShape(outT.getShape().begin(),
                                        outT.getShape().end());
    for (auto i : llvm::seq<int64_t>(0, rank)) {
      if (!mlir::ShapedType::isDynamic(newShape[i]))
        continue;
      // Resolve broadcast dim i: a single distinct static value > 1 wins; if
      // every contributing operand is static (all 1) the result is 1; a static
      // conflict or any dynamic non-1 contributor leaves it dynamic.
      int64_t resolved = mlir::ShapedType::kDynamic;
      bool conflict = false;
      bool allStatic = true;
      for (mlir::RankedTensorType t : operands) {
        int64_t off = rank - t.getRank();
        int64_t ax = i - off;
        if (ax < 0)
          continue; // implicit leading 1
        if (t.isDynamicDim(ax)) {
          allStatic = false;
          continue;
        }
        int64_t dv = t.getDimSize(ax);
        if (dv == 1)
          continue; // unit broadcasts; does not pin
        if (mlir::ShapedType::isDynamic(resolved))
          resolved = dv;
        else if (resolved != dv)
          conflict = true;
      }
      if (conflict)
        continue;
      if (!mlir::ShapedType::isDynamic(resolved))
        newShape[i] = resolved;
      else if (allStatic)
        newShape[i] = 1; // every contributor static and == 1
    }

    if (llvm::ArrayRef<int64_t>(newShape) == outT.getShape())
      return rewriter.notifyMatchFailure(op, "ew.no_new_static");
    auto newType =
        mlir::RankedTensorType::get(newShape, outT.getElementType());
    rewriter.modifyOpInPlace(op, [&] { op->getResult(0).setType(newType); });
    ++NumElementwiseResultRefinements;
    return mlir::success();
  }
};

} // namespace

void populateRefineReshapeResultTypePatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *ctx) {
  patterns.add<RefineReshapeResultFromShapeOperand, RefineTransposeResultType,
               RefineElementwiseResultType>(ctx);
}

} // namespace hip
} // namespace mlir
