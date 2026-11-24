/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "ReduceConversionUtils.h"

#include "llvm/ADT/APInt.h"

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
  if (mlir::failed(checkReductionAttributes(op, rewriter, "ReduceL2")))
    return mlir::failure();
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
  }

  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceL2: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::Value init;
  if (axesKnown) {
    mlir::FailureOr<mlir::Value> empty = createReductionEmptyTensor(
        rewriter, loc, resultType, data, axesVec, keepdims, noopWithEmptyAxes);
    if (mlir::failed(empty))
      return rewriter.notifyMatchFailure(
          op, "ReduceL2: destination shape conflicts with the result type");
    init = *empty;
  } else {
    // Runtime axes: the destination can only follow the data positionally.
    init = createEmptyTensor(rewriter, loc, resultType, data);
  }

  if (!axesOperand) {
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceL2Op::create(
      rewriter, loc, context, data, axesOperand, init, keepdimsAttr,
      noopWithEmptyAxesAttr,
      getConstantReducedAxesAttr(rewriter, data, axesOperand,
                                 noopWithEmptyAxes));

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
