/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ConstantOfShape -> arith.constant (compile-time fold)
//===----------------------------------------------------------------------===//
//
// `onnx.ConstantOfShape` takes a rank-1 int64 tensor describing the output
// shape and produces a tensor of that shape filled with the optional `value`
// attribute (defaulting to a single fp32 zero).
//
// Static-shape models always feed a compile-time constant into the shape
// input (most commonly an `onnx.Constant` that has already been lowered to
// `arith.constant` by the time this pattern runs), so the whole op collapses
// to a single splat `arith.constant`.  No runtime support is needed.
//
// If the shape input is not a recognised compile-time constant we bail out
// via `notifyMatchFailure`; in production this should never happen for
// static-shape models, but failing safely lets unit tests detect regressions.

/// Return the dense-elements attribute backing \p value if it can be
/// determined at compile time (the same forms recognised by ReshapeConversion
/// for "axes" inputs).  Returns null otherwise.
static mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());

  // onnx.Constant carrying a dense `value` attribute (pre-lowerOnnxConstants).
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;

  // Externalised path: bufferization.to_tensor of a memref.get_global whose
  // global has a dense initial_value.  Used when the shape input came in
  // through the constant externalisation route.
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

struct ConstantOfShapeFold : public mlir::RewritePattern {
  ConstantOfShapeFold(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    mlir::Value shapeInput = op->getOperand(0);
    mlir::DenseElementsAttr shapeAttr =
        getCompileTimeConstantTensor(shapeInput);
    if (!shapeAttr)
      return rewriter.notifyMatchFailure(
          op, "shape input must be a compile-time constant tensor");

    auto shapeTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(shapeAttr.getType());
    if (!shapeTensorType || shapeTensorType.getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "shape input must be a rank-1 tensor");
    if (!shapeTensorType.getElementType().isInteger(64) &&
        !shapeTensorType.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "shape input must have int32 or int64 element type");

    llvm::SmallVector<int64_t> outShape;
    outShape.reserve(shapeTensorType.getDimSize(0));
    for (mlir::APInt v : shapeAttr.getValues<mlir::APInt>()) {
      int64_t d = v.getSExtValue();
      if (d < 0)
        return rewriter.notifyMatchFailure(
            op, "negative dimension in ConstantOfShape input");
      outShape.push_back(d);
    }

    // Resolve the fill value.  The ONNX `value` attribute is a single-element
    // tensor; default is fp32 zero when absent.
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    mlir::Type elemType = resultType.getElementType();

    // Synthesize the splat attribute matching the result element type.  The
    // ONNX `value` attribute, when present, is itself a DenseElementsAttr
    // holding the single fill value; we copy its scalar splat into the new
    // shape.  When absent the default per the spec is +0.0 fp32.
    auto outTensorType = mlir::RankedTensorType::get(outShape, elemType);
    mlir::DenseElementsAttr splatAttr;

    if (auto valueAttr = op->getAttrOfType<mlir::ElementsAttr>("value")) {
      auto valueDense = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
      if (!valueDense)
        return rewriter.notifyMatchFailure(
            op, "non-dense value attribute is not supported");
      auto valueTensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(valueDense.getType());
      if (!valueTensorType || valueTensorType.getElementType() != elemType)
        return rewriter.notifyMatchFailure(
            op, "value attribute element type does not match result");

      // Splat-construct an attribute of the output shape from the scalar in
      // `value`.  Element-wise reading covers both splat and non-splat 1x
      // tensors uniformly.
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
        mlir::APFloat scalar = *valueDense.getValues<mlir::APFloat>().begin();
        splatAttr = mlir::DenseElementsAttr::get(outTensorType, scalar);
      } else if (mlir::isa<mlir::IntegerType>(elemType)) {
        mlir::APInt scalar = *valueDense.getValues<mlir::APInt>().begin();
        splatAttr = mlir::DenseElementsAttr::get(outTensorType, scalar);
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported result element type");
      }
    } else {
      // Spec default: 0.0 fp32.  We honour the result tensor's actual element
      // type to keep the IR well-typed even if upstream shape inference picked
      // a different default.
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
        mlir::APFloat zero(floatTy.getFloatSemantics(), 0);
        splatAttr = mlir::DenseElementsAttr::get(outTensorType, zero);
      } else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        mlir::APInt zero(intTy.getWidth(), 0);
        splatAttr = mlir::DenseElementsAttr::get(outTensorType, zero);
      } else {
        return rewriter.notifyMatchFailure(
            op, "unsupported default result element type");
      }
    }

    mlir::Value cst =
        mlir::arith::ConstantOp::create(rewriter, op->getLoc(), splatAttr);
    rewriter.replaceOp(op, cst);
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateConstantOfShapeConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ConstantOfShapeFold>(ctx);
}

} // namespace hip
} // namespace mlir
