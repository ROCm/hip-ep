/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReduceConversionUtils.h - Shared onnx.Reduce* -> hip.reduce_* -----===//
//
// onnx.ReduceMean / ReduceSum / ReduceMax / ReduceMin all lower to a
// hip.reduce_* DPS op with the same operands and attributes. With constant
// axes the destination takes its extents from reifyReductionWithKeepdims.
//
// The runtime reduces one contiguous run of axes. Mean, sum, max and min are
// separable, so constant axes made of several runs become a chain, one
// hip.reduce_* per run, from the last run to the first. Going right to left
// keeps the earlier runs' indices valid when keepdims=0 drops axes:
//
//   %y = onnx.ReduceMean(%x, [2, 4]) {keepdims = 0}
//          : (tensor<?x?x2x?x2xCxf16>) -> tensor<?x?x?xCxf16>
// becomes
//   %t = hip.reduce_mean ins(%x, [4]) {reduced_axes = [4]} : ?x?x2x?xC
//   %y = hip.reduce_mean ins(%t, [2]) {reduced_axes = [2]} : ?x?x?xC
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_REDUCECONVERSIONUTILS_H
#define HIP_CONVERSION_ONNXTOHIP_REDUCECONVERSIONUTILS_H

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {

/// Split sorted, unique axes into maximal runs of consecutive axes.
inline llvm::SmallVector<llvm::SmallVector<int64_t>>
splitContiguousAxes(llvm::ArrayRef<int64_t> axes) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> runs;
  for (int64_t axis : axes) {
    if (runs.empty() || runs.back().back() + 1 != axis)
      runs.emplace_back();
    runs.back().push_back(axis);
  }
  return runs;
}

/// The type of reducing `type` over sorted, unique, non-negative `axes`.
inline mlir::RankedTensorType reducedType(mlir::RankedTensorType type,
                                          llvm::ArrayRef<int64_t> axes,
                                          int64_t keepdims) {
  llvm::SmallVector<int64_t> shape;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (!llvm::is_contained(axes, i))
      shape.push_back(type.getDimSize(i));
    else if (keepdims)
      shape.push_back(1);
  }
  return mlir::RankedTensorType::get(shape, type.getElementType());
}

/// onnx.Reduce{Mean,Sum,Max,Min} -> hip.reduce_*. `separable` enables the
/// per-run chain for constant axes that are not one contiguous run.
template <typename HipOp>
mlir::LogicalResult convertOnnxReduction(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter,
                                         llvm::StringRef name, bool separable) {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  int64_t noopWithEmptyAxes = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes"))
    noopWithEmptyAxes = attr.getSInt();
  int64_t keepdims = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = attr.getSInt();

  // Axes from the attribute (opset < 18) or the all-axes default; an operand
  // may still be a constant, which getConstantReducedAxesAttr resolves.
  bool axesStaticallyKnown = op->getNumOperands() <= 1;
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  llvm::SmallVector<int64_t> axesVec;
  if (axesStaticallyKnown) {
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(
            mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
    } else if (noopWithEmptyAxes == 0 && inputType) {
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
    }
  }

  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, name + ": cannot infer unranked result (need ranked input and "
                   "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  // Normalized constant axes, or null for runtime axes. Resolved before any
  // op is created so that a rejection leaves the IR untouched.
  mlir::DenseI64ArrayAttr reduced;
  if (!axesStaticallyKnown) {
    reduced = getConstantReducedAxesAttr(rewriter, data, op->getOperand(1),
                                         noopWithEmptyAxes);
  } else if (inputType) {
    llvm::SmallVector<int64_t> list;
    for (int64_t axis : axesVec) {
      if (axis < 0)
        axis += inputType.getRank();
      if (axis < 0 || axis >= inputType.getRank())
        return rewriter.notifyMatchFailure(op, name + ": axis out of range");
      list.push_back(axis);
    }
    if (list.empty() && noopWithEmptyAxes == 0)
      llvm::append_range(list, llvm::seq<int64_t>(0, inputType.getRank()));
    llvm::sort(list);
    list.erase(std::unique(list.begin(), list.end()), list.end());
    reduced = rewriter.getDenseI64ArrayAttr(list);
  }
  if (reduced && inputType) {
    mlir::RankedTensorType expected =
        reducedType(inputType, reduced.asArrayRef(), keepdims);
    if (!isResultTypeCompatibleWithInferredShape(resultType,
                                                 expected.getShape()))
      return rewriter.notifyMatchFailure(
          op, name + ": result type conflicts with the reduced axes");
  }

  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    axesOperand = op->getOperand(1);
  } else {
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    axesOperand = mlir::arith::ConstantOp::create(
        rewriter, loc, axesType,
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec)));
  }

  auto createOp = [&](mlir::Value input, mlir::Value axes, mlir::Value init,
                      int64_t noop, mlir::DenseI64ArrayAttr reducedAxes) {
    return HipOp::create(rewriter, loc, context, input, axes, init,
                         rewriter.getI64IntegerAttr(keepdims),
                         rewriter.getI64IntegerAttr(noop), reducedAxes)
        ->getResult(0);
  };

  // Runtime axes: the destination can only follow the data positionally.
  if (!reduced) {
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);
    rewriter.replaceOp(
        op, createOp(data, axesOperand, init, noopWithEmptyAxes, nullptr));
    return mlir::success();
  }

  auto runs = splitContiguousAxes(reduced.asArrayRef());
  if (runs.size() <= 1 || !separable) {
    mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
        rewriter, loc, resultType, data, reduced.asArrayRef(), keepdims,
        noopWithEmptyAxes);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, name + ": destination shape conflicts with the result type");
    rewriter.replaceOp(
        op, createOp(data, axesOperand, *init, noopWithEmptyAxes, reduced));
    return mlir::success();
  }

  if (!inputType)
    return rewriter.notifyMatchFailure(op, name + ": unranked input");
  mlir::Value current = data;
  for (size_t i = runs.size(); i-- > 0;) {
    llvm::ArrayRef<int64_t> run = runs[i];
    auto currentType = mlir::cast<mlir::RankedTensorType>(current.getType());
    mlir::RankedTensorType stepType =
        i == 0 ? resultType : reducedType(currentType, run, keepdims);
    mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
        rewriter, loc, stepType, current, run, keepdims, /*noop=*/0);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, name + ": destination shape conflicts with the result type");
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(run.size())}, rewriter.getI64Type());
    mlir::Value runAxes = mlir::arith::ConstantOp::create(
        rewriter, loc, axesType,
        mlir::DenseIntElementsAttr::get(axesType, run));
    current = createOp(current, runAxes, *init, /*noop=*/0,
                       rewriter.getDenseI64ArrayAttr(run));
  }
  rewriter.replaceOp(op, current);
  return mlir::success();
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_REDUCECONVERSIONUTILS_H
