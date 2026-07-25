/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace hip {
namespace {

/// Try to recognise \p v as a compile-time 1-D integer constant tensor.
/// Mirrors the helper in SliceConversion.cpp / PadConversion.cpp.
static mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defOp)) {
    auto bufDef =
        toTensor.getBuffer().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!bufDef)
      return nullptr;
    auto module = bufDef->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return nullptr;
    auto global =
        module.lookupSymbol<mlir::memref::GlobalOp>(bufDef.getNameAttr());
    if (!global)
      return nullptr;
    return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        global.getInitialValueAttr());
  }
  return nullptr;
}

static mlir::LogicalResult
extractIntVector(mlir::Value v, llvm::SmallVectorImpl<int64_t> &out) {
  if (!v)
    return mlir::failure();
  auto dense = getCompileTimeConstantTensor(v);
  if (!dense)
    return mlir::failure();
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() != 1)
    return mlir::failure();
  auto elemTy = tensorType.getElementType();
  if (!elemTy.isInteger(64) && !elemTy.isInteger(32))
    return mlir::failure();
  for (mlir::APInt entry : dense.getValues<mlir::APInt>())
    out.push_back(entry.getSExtValue());
  return mlir::success();
}

/// Build the destination `tensor.empty` for ReduceProd.
///
/// Output shape semantics:
///   * keepdims=1: out_rank == in_rank; out[i] = in[i] when i is NOT
///     reduced, otherwise 1.
///   * keepdims=0: out_rank == in_rank - #axes; out skips the reduced
///     axes; out[j] = in[non_reduced_axis_at_position_j].
///
/// For dynamic output dims we map back to the source `data` dim using
/// the known axes set (which must be a compile-time constant -- a dynamic
/// `axes` would make the mapping data-dependent). When `axes` is not
/// known, we fall back to positional alignment (correct only for
/// keepdims=1 and the all-reduce case).
static mlir::Value buildReduceProdInit(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc,
                                       mlir::RankedTensorType resultType,
                                       mlir::Value data,
                                       llvm::ArrayRef<int64_t> axesVec,
                                       bool axesKnown, int64_t keepdims) {
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  int64_t inRank = dataType.getRank();

  // Build the lookup: for each output dim, which input dim is its source?
  // Static -> reduced axis (resulting size 1) gets no entry. Identity-
  // mapped output dim points back to the corresponding input dim.
  llvm::SmallSet<int64_t, 8> reducedAxes;
  for (int64_t a : axesVec) {
    if (a < 0)
      a += inRank;
    reducedAxes.insert(a);
  }

  // Map outIdx -> inIdx (or -1 for "reduced -> size 1").
  llvm::SmallVector<int64_t> outToIn(resultType.getRank(), -1);
  if (axesKnown) {
    if (keepdims) {
      for (int64_t i = 0; i < resultType.getRank(); ++i)
        outToIn[i] = reducedAxes.contains(i) ? -1 : i;
    } else {
      int64_t outIdx = 0;
      for (int64_t i = 0; i < inRank; ++i) {
        if (reducedAxes.contains(i))
          continue;
        if (outIdx < resultType.getRank())
          outToIn[outIdx] = i;
        ++outIdx;
      }
    }
  } else {
    // Fallback: positional alignment.
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      outToIn[i] = i < inRank ? i : -1;
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (!resultType.isDynamicDim(i))
      continue;
    int64_t inIdx = outToIn[i];
    if (inIdx < 0) {
      // Reduced axis -- size 1.
      dynSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
    } else if (dataType.isDynamicDim(inIdx)) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, data, inIdx));
    } else {
      dynSizes.push_back(mlir::arith::ConstantIndexOp::create(
          rewriter, loc, dataType.getDimSize(inIdx)));
    }
  }
  return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

/// onnx.ReduceProd -> hip.reduce_prod
///
/// Mirrors the existing ReduceSum/ReduceMax conversion: lifts an optional
/// `axes` operand or attribute into a required tensor operand, threads
/// `keepdims` and `noop_with_empty_axes` through, and shares lowering with
/// other reduction ops via the unified ReduceLowering template.
struct ReduceProdToHip : public mlir::RewritePattern {
  ReduceProdToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceProd", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceProdToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr =
          op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes")) {
    noopWithEmptyAxes = noopAttr.getSInt();
  }

  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getSInt();

  // Materialise the axes vector (and remember whether it is known at
  // compile time) so we can build the destination tensor with the right
  // dynamic dim sources even when the result is partially dynamic.
  llvm::SmallVector<int64_t> axesVec;
  bool axesKnown = false;
  mlir::Value axesOperand;
  if (op->getNumOperands() > 1 &&
      !mlir::isa<mlir::NoneType>(op->getOperand(1).getType())) {
    axesOperand = op->getOperand(1);
    if (mlir::succeeded(extractIntVector(axesOperand, axesVec)))
      axesKnown = true;
  } else {
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(
            mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
      axesKnown = true;
    } else if (noopWithEmptyAxes == 0) {
      auto inputType = mlir::cast<mlir::RankedTensorType>(data.getType());
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
      axesKnown = true;
    } else {
      axesKnown = true; // empty axes, noop
    }
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  // Resolve the result type (infer if the importer left it unranked). axesKnown
  // also covers the compile-time-constant axes-operand case extracted above.
  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceProd: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::Value init = buildReduceProdInit(rewriter, loc, resultType, data,
                                         axesVec, axesKnown, keepdims);

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceProdOp::create(
      rewriter, loc, context, data, axesOperand, init, keepdimsAttr, noopAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceProdConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ReduceProdToHip>(ctx);
}

} // namespace hip
} // namespace mlir
