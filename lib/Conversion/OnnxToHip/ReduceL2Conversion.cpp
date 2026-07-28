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
/// Mirrors the helper in SliceConversion.cpp / ReduceProdConversion.cpp.
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

/// Build the destination `tensor.empty` for ReduceL2.
///
/// Output shape semantics match other ONNX reduce ops:
///   * keepdims=1: out_rank == in_rank; reduced axes become size 1.
///   * keepdims=0: out_rank == in_rank - #axes; reduced axes are dropped.
///
/// For dynamic output dims we map back to the source `data` dim using the
/// known axes set (compile-time constant). When axes are not known, fall
/// back to positional alignment (correct for keepdims=1 and all-reduce).
static mlir::Value buildReduceL2Init(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value data,
                                     llvm::ArrayRef<int64_t> axesVec,
                                     bool axesKnown, int64_t keepdims) {
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  int64_t inRank = dataType.getRank();

  llvm::SmallSet<int64_t, 8> reducedAxes;
  for (int64_t a : axesVec) {
    if (a < 0)
      a += inRank;
    reducedAxes.insert(a);
  }

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
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      outToIn[i] = i < inRank ? i : -1;
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (!resultType.isDynamicDim(i))
      continue;
    int64_t inIdx = outToIn[i];
    if (inIdx < 0) {
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

/// onnx.ReduceL2 -> hip.reduce_l2
///
/// Direct, dim-tolerant conversion: sqrt(sum(x^2)) along the reduced axes
/// happens inside the runtime kernel, so no static reduce axis is required
/// at runtime. Compile-time shape inference still needs ranked input/output
/// or statically-known axes to build the DPS init tensor.
struct ReduceL2ToHip : public mlir::RewritePattern {
  ReduceL2ToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceL2", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceL2ToHip::matchAndRewrite(mlir::Operation *op,
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
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims")) {
    keepdims = keepdimsAttr.getSInt();
  }

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

  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceL2: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::Value init = buildReduceL2Init(rewriter, loc, resultType, data, axesVec,
                                       axesKnown, keepdims);

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp =
      mlir::hip::ReduceL2Op::create(rewriter, loc, context, data, axesOperand,
                                    init, keepdimsAttr, noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceL2ConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<ReduceL2ToHip>(ctx);
}

} // namespace hip
} // namespace mlir
