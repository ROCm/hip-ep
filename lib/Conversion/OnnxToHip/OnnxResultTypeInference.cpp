/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInference.cpp - Pure result-type rules for onnx.* ----===//
//
// Implementation of the helpers declared in OnnxResultTypeInference.h.
// See that header for the API contract; this TU is internal to the
// OnnxToHip library.
//
// Two callers share these rules: the `InferOnnxShapes` pass uses them to
// refine existing onnx-op result types in-place, and the pre-lowering
// rewriters (ProjectorOpsRewrites, FastGeluFusion) use them to type
// newly-emitted ops without repeating the derivation logic.
//
//===----------------------------------------------------------------------===//

#include "OnnxResultTypeInference.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

namespace {

//===----------------------------------------------------------------------===//
// Shape-operand resolver (used by inferReshapeResultType)
//===----------------------------------------------------------------------===//

/// One resolved element of a 1-D shape vector. Either a compile-time
/// constant value (`staticValue`) is set, or the entry refers to dim
/// `dimIndex` of the SSA value `dimSource`. An entry with both fields
/// empty is unresolvable — caller must abort.
struct ResolvedDim {
  std::optional<int64_t> staticValue;
  Value dimSource;
  int64_t dimIndex = -1;
};

static bool readConstantIntValues(Operation *def,
                                  llvm::SmallVectorImpl<int64_t> &out) {
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return false;
  auto valueAttr = dyn_cast_or_null<DenseElementsAttr>(def->getAttr("value"));
  if (!valueAttr)
    return false;
  if (!valueAttr.getElementType().isIntOrIndex())
    return false;
  out.clear();
  out.reserve(valueAttr.getNumElements());
  for (auto v : valueAttr.getValues<APInt>())
    out.push_back(v.getSExtValue());
  return true;
}

/// Recursively resolve a 1-D shape vector value into a list of per-element
/// `ResolvedDim`s. Handles the shape-arithmetic chain that vision and LLM
/// exporters emit: `onnx.Concat` of `onnx.Slice(onnx.Shape(x), …)` legs
/// and `onnx.Constant` legs.
static bool resolveShapeValue(Value v,
                              llvm::SmallVectorImpl<ResolvedDim> &out) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  llvm::StringRef opName = def->getName().getStringRef();

  if (opName == "onnx.Constant") {
    llvm::SmallVector<int64_t> values;
    if (!readConstantIntValues(def, values))
      return false;
    for (int64_t val : values) {
      ResolvedDim rd;
      rd.staticValue = val;
      out.push_back(rd);
    }
    return true;
  }

  if (opName == "onnx.Shape") {
    if (def->getNumOperands() < 1)
      return false;
    Value input = def->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return false;
    int64_t rank = inputType.getRank();
    int64_t start = 0, end = rank;
    if (auto a = def->getAttrOfType<IntegerAttr>("start"))
      start = a.getSInt();
    if (auto a = def->getAttrOfType<IntegerAttr>("end"))
      end = a.getSInt();
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::max(start, int64_t(0));
    end = std::min(end, rank);
    if (start > end)
      return false;
    for (int64_t i : llvm::seq<int64_t>(start, end)) {
      ResolvedDim rd;
      if (!inputType.isDynamicDim(i))
        rd.staticValue = inputType.getDimSize(i);
      else {
        rd.dimSource = input;
        rd.dimIndex = i;
      }
      out.push_back(rd);
    }
    return true;
  }

  if (opName == "onnx.Concat") {
    int64_t axis = 0;
    if (auto a = def->getAttrOfType<IntegerAttr>("axis"))
      axis = a.getSInt();
    if (axis != 0)
      return false;
    for (Value operand : def->getOperands())
      if (!resolveShapeValue(operand, out))
        return false;
    return true;
  }

  if (opName == "onnx.Unsqueeze" || opName == "onnx.Cast") {
    // Both are identity on the SCALAR sequence of values produced by a
    // shape-arithmetic chain. (Unsqueeze wraps; Cast preserves integer
    // values for int->int.)
    if (def->getNumOperands() < 1)
      return false;
    return resolveShapeValue(def->getOperand(0), out);
  }

  if (opName == "onnx.Gather") {
    if (def->getNumOperands() != 2)
      return false;
    int64_t axis = 0;
    if (auto a = def->getAttrOfType<IntegerAttr>("axis"))
      axis = a.getSInt();
    if (axis != 0)
      return false;
    llvm::SmallVector<ResolvedDim> src;
    if (!resolveShapeValue(def->getOperand(0), src))
      return false;
    llvm::SmallVector<int64_t> indices;
    if (!readConstantIntValues(def->getOperand(1).getDefiningOp(), indices))
      return false;
    int64_t srcLen = static_cast<int64_t>(src.size());
    for (int64_t idx : indices) {
      if (idx < 0)
        idx += srcLen;
      if (idx < 0 || idx >= srcLen)
        return false;
      out.push_back(src[idx]);
    }
    return true;
  }

  if (opName == "onnx.Slice") {
    if (def->getNumOperands() < 3)
      return false;
    llvm::SmallVector<ResolvedDim> src;
    if (!resolveShapeValue(def->getOperand(0), src))
      return false;
    llvm::SmallVector<int64_t> startsArr, endsArr;
    if (!readConstantIntValues(def->getOperand(1).getDefiningOp(), startsArr))
      return false;
    if (!readConstantIntValues(def->getOperand(2).getDefiningOp(), endsArr))
      return false;
    if (startsArr.size() != 1 || endsArr.size() != 1)
      return false;
    if (def->getNumOperands() >= 4) {
      llvm::SmallVector<int64_t> axesArr;
      if (!readConstantIntValues(def->getOperand(3).getDefiningOp(), axesArr))
        return false;
      if (axesArr.size() != 1 || axesArr[0] != 0)
        return false;
    }
    if (def->getNumOperands() >= 5) {
      llvm::SmallVector<int64_t> stepsArr;
      if (!readConstantIntValues(def->getOperand(4).getDefiningOp(), stepsArr))
        return false;
      if (stepsArr.size() != 1 || stepsArr[0] != 1)
        return false;
    }
    int64_t srcLen = static_cast<int64_t>(src.size());
    int64_t s = startsArr[0], e = endsArr[0];
    if (s < 0)
      s += srcLen;
    if (e < 0)
      e += srcLen;
    s = std::max(s, int64_t(0));
    e = std::min(e, srcLen);
    if (s > e)
      return false;
    for (int64_t i : llvm::seq<int64_t>(s, e))
      out.push_back(src[i]);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Broadcast helpers
//===----------------------------------------------------------------------===//

static int64_t alignedDim(RankedTensorType t, int64_t outRank, int64_t i) {
  int64_t shift = outRank - t.getRank();
  int64_t srcIdx = i - shift;
  if (srcIdx < 0)
    return 1; // implicit broadcast dim of size 1
  return t.isDynamicDim(srcIdx) ? ShapedType::kDynamic : t.getDimSize(srcIdx);
}

static int64_t broadcastDim(int64_t l, int64_t r) {
  if (l == ShapedType::kDynamic && r == ShapedType::kDynamic)
    return ShapedType::kDynamic;
  if (l == ShapedType::kDynamic)
    return r > 1 ? r : ShapedType::kDynamic;
  if (r == ShapedType::kDynamic)
    return l > 1 ? l : ShapedType::kDynamic;
  return std::max(l, r);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public helpers
//===----------------------------------------------------------------------===//

RankedTensorType inferReshapeResultType(RankedTensorType inputType,
                                        Value shapeOperand, int64_t outputRank,
                                        int64_t allowzero) {
  if (!inputType || outputRank < 0)
    return {};

  // Worst-case fallback shape: all-dynamic of the requested rank. The
  // helper falls back to this whenever the resolver can't produce a
  // shape vector of length `outputRank` — useful for the rewriter
  // emission case (the next round of InferOnnxShapes can refine the
  // result once the SSA dataflow stabilises).
  auto fallback = [&]() {
    llvm::SmallVector<int64_t> dyn(outputRank, ShapedType::kDynamic);
    return RankedTensorType::get(dyn, inputType.getElementType());
  };

  llvm::SmallVector<ResolvedDim> resolved;
  if (!shapeOperand || !resolveShapeValue(shapeOperand, resolved))
    return fallback();
  if (static_cast<int64_t>(resolved.size()) != outputRank)
    return fallback();

  // Pre-compute input static element-count and dyn-dim count for `-1`
  // inference (when shape contains exactly one -1 and the dyn-dim counts
  // match on both sides, the unknown dim is computable from
  // prod(input_static) / prod(output_resolved_static).)
  int64_t inputStaticProduct = 1, inputDynCount = 0;
  for (int64_t i : llvm::seq<int64_t>(inputType.getRank())) {
    if (inputType.isDynamicDim(i))
      ++inputDynCount;
    else
      inputStaticProduct *= inputType.getDimSize(i);
  }

  llvm::SmallVector<int64_t> refinedShape;
  refinedShape.reserve(outputRank);
  int64_t minusOneIdx = -1;
  int64_t resolvedStaticProduct = 1, resolvedDynCount = 0;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    const ResolvedDim &rd = resolved[i];
    int64_t refined = ShapedType::kDynamic;
    if (rd.staticValue.has_value()) {
      int64_t s = *rd.staticValue;
      if (s > 0) {
        refined = s;
      } else if (s == 0 && allowzero == 0 && i < inputType.getRank() &&
                 !inputType.isDynamicDim(i)) {
        refined = inputType.getDimSize(i);
      } else if (s == -1) {
        minusOneIdx = i;
      }
    } else if (rd.dimSource) {
      auto srcType = dyn_cast<RankedTensorType>(rd.dimSource.getType());
      if (srcType && rd.dimIndex < srcType.getRank() &&
          !srcType.isDynamicDim(rd.dimIndex))
        refined = srcType.getDimSize(rd.dimIndex);
    }
    refinedShape.push_back(refined);
    if (i != minusOneIdx) {
      if (refined != ShapedType::kDynamic && refined > 0)
        resolvedStaticProduct *= refined;
      else
        ++resolvedDynCount;
    }
  }
  // -1 inference: requires (1) exactly one -1 slot, (2) same dyn count on
  // both sides so symbolic dims cancel, (3) input static product divisible
  // by other-output static product.
  if (minusOneIdx >= 0 && resolvedDynCount == inputDynCount &&
      inputStaticProduct > 0 && resolvedStaticProduct > 0 &&
      inputStaticProduct % resolvedStaticProduct == 0) {
    int64_t inferred = inputStaticProduct / resolvedStaticProduct;
    if (inferred > 0)
      refinedShape[minusOneIdx] = inferred;
  }
  return RankedTensorType::get(refinedShape, inputType.getElementType());
}

RankedTensorType inferTransposeResultType(RankedTensorType inputType,
                                          ArrayRef<int64_t> perm) {
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return {};
  llvm::SmallVector<int64_t> dims;
  dims.reserve(rank);
  for (int64_t src : perm) {
    if (src < 0 || src >= rank)
      return {};
    dims.push_back(inputType.isDynamicDim(src) ? ShapedType::kDynamic
                                               : inputType.getDimSize(src));
  }
  return RankedTensorType::get(dims, inputType.getElementType());
}

RankedTensorType inferMatMulResultType(RankedTensorType lhsType,
                                       RankedTensorType rhsType) {
  if (!lhsType || !rhsType)
    return {};
  int64_t lhsRank = lhsType.getRank(), rhsRank = rhsType.getRank();
  if (lhsRank < 2 || rhsRank < 2)
    return {}; // rank-1 matmul handling is unusual; bail conservatively
  int64_t outRank = std::max(lhsRank, rhsRank);
  int64_t lhsBatch = lhsRank - 2, rhsBatch = rhsRank - 2;
  int64_t outBatch = outRank - 2;
  llvm::SmallVector<int64_t> dims;
  dims.reserve(outRank);
  // Outer broadcast — right-align over the OUTER slice only. (See header
  // for the canonical pitfall on full-rank alignment.)
  for (int64_t i : llvm::seq<int64_t>(outBatch)) {
    int64_t lShift = outBatch - lhsBatch, rShift = outBatch - rhsBatch;
    int64_t lIdx = i - lShift, rIdx = i - rShift;
    int64_t l = (lIdx < 0)
                    ? int64_t(1)
                    : (lhsType.isDynamicDim(lIdx) ? ShapedType::kDynamic
                                                  : lhsType.getDimSize(lIdx));
    int64_t r = (rIdx < 0)
                    ? int64_t(1)
                    : (rhsType.isDynamicDim(rIdx) ? ShapedType::kDynamic
                                                  : rhsType.getDimSize(rIdx));
    dims.push_back(broadcastDim(l, r));
  }
  // M from lhs[-2], N from rhs[-1].
  dims.push_back(lhsType.isDynamicDim(lhsRank - 2)
                     ? ShapedType::kDynamic
                     : lhsType.getDimSize(lhsRank - 2));
  dims.push_back(rhsType.isDynamicDim(rhsRank - 1)
                     ? ShapedType::kDynamic
                     : rhsType.getDimSize(rhsRank - 1));
  return RankedTensorType::get(dims, lhsType.getElementType());
}

RankedTensorType inferCastResultType(RankedTensorType inputType,
                                     Type targetElemType) {
  if (!inputType || !targetElemType)
    return {};
  llvm::SmallVector<int64_t> dims;
  dims.reserve(inputType.getRank());
  for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
    dims.push_back(inputType.isDynamicDim(i) ? ShapedType::kDynamic
                                             : inputType.getDimSize(i));
  return RankedTensorType::get(dims, targetElemType);
}

RankedTensorType inferUnarySameShapeResultType(RankedTensorType inputType) {
  if (!inputType)
    return {};
  return inputType;
}

RankedTensorType inferBinaryBroadcastResultType(RankedTensorType lhsType,
                                                RankedTensorType rhsType) {
  if (!lhsType || !rhsType)
    return {};
  int64_t outRank = std::max(lhsType.getRank(), rhsType.getRank());
  llvm::SmallVector<int64_t> dims;
  dims.reserve(outRank);
  for (int64_t i : llvm::seq<int64_t>(outRank))
    dims.push_back(broadcastDim(alignedDim(lhsType, outRank, i),
                                alignedDim(rhsType, outRank, i)));
  return RankedTensorType::get(dims, lhsType.getElementType());
}

RankedTensorType inferConcatResultType(ValueRange operands, int64_t axis) {
  if (operands.empty())
    return {};
  // First ranked operand seeds rank + element type.
  RankedTensorType refType;
  for (Value v : operands) {
    if (auto rt = dyn_cast<RankedTensorType>(v.getType())) {
      refType = rt;
      break;
    }
  }
  if (!refType)
    return {};
  int64_t rank = refType.getRank();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  // Sum sizes along `axis`; tighten other axes from the first operand that
  // is static at that position. An operand with mismatched rank or
  // dynamic-along-axis collapses the axis result to dynamic but does not
  // poison the non-axis tightening (other ranked operands may still
  // supply static dims).
  llvm::SmallVector<int64_t> outShape(rank, ShapedType::kDynamic);
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    if (i == axis)
      continue;
    for (Value v : operands) {
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt || rt.getRank() != rank || rt.isDynamicDim(i))
        continue;
      outShape[i] = rt.getDimSize(i);
      break;
    }
  }
  bool axisDyn = false;
  int64_t axisSum = 0;
  for (Value v : operands) {
    auto rt = dyn_cast<RankedTensorType>(v.getType());
    if (!rt || rt.getRank() != rank || rt.isDynamicDim(axis)) {
      axisDyn = true;
      continue;
    }
    axisSum += rt.getDimSize(axis);
  }
  outShape[axis] = axisDyn ? ShapedType::kDynamic : axisSum;

  return RankedTensorType::get(outShape, refType.getElementType());
}

} // namespace hip
} // namespace mlir
