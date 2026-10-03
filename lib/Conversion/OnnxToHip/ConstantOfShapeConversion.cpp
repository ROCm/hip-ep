/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ConstantOfShape lowering
//===----------------------------------------------------------------------===//
//
// `onnx.ConstantOfShape` takes a rank-1 int tensor describing the output
// shape and produces a tensor of that shape filled with the optional `value`
// attribute (defaulting to a single fp32 zero).
//
// Three paths are provided, in benefit order:
//
//   1. ConstantOfShapeAsScalar (benefit=3): when the result is only consumed
//      by ops that already broadcast a scalar input across the result shape
//      (today: `onnx.Where`), replace the op with a rank-0 fill-value buffer.
//      The consumer broadcasts the scalar at no extra memory cost (stride==0
//      along every output axis in the runtime kernel), avoiding the
//      `tensor.splat` -> `linalg.map` -> `scf.for` bufferization chain that
//      the dynamic path would otherwise produce for a full-size buffer.
//
//   2. ConstantOfShapeFold (benefit=2): when the shape input is a recognised
//      compile-time constant AND the result type is fully static, the op
//      collapses to a single splat `arith.constant`. No runtime work; the
//      whole computation is materialised at compile time. This is the
//      common case for transformer-style "allocate a zero KV/mask tensor"
//      patterns once shape inference + constant folding have run.
//
//   3. ConstantOfShapeDynamic (benefit=1): fallback for non-constant shape
//      input OR a result type with at least one dynamic dim. Each dynamic
//      dim is materialised from the shape input via `tensor.extract` +
//      `arith.index_cast`, the fill value is built as a scalar
//      `arith.constant`, and the output tensor is produced via
//      `tensor.splat` (which broadcasts a scalar to a ranked tensor with
//      arbitrary static + dynamic dims). No HIP dialect op or runtime
//      function is needed -- `tensor.splat` bufferizes naturally and lowers
//      through the standard MLIR pipeline.

/// Extract ConstantOfShape's rank-1 integer shape. Ordinary imported constants
/// arrive here as dense `hip.constant` carriers and use the shared extractor.
/// `onnx.Shape` is the one dedicated extension because its payload is implicit
/// in its source type.
static bool extractConstantOfShapeVector(mlir::Value value,
                                         llvm::SmallVectorImpl<int64_t> &out) {
  if (extractConstantIntVector(value, out))
    return true;
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  // `onnx.Shape(static-tensor)` is itself a compile-time constant -- the
  // transformer-emitted `Shape -> ConstantOfShape` pattern (zero-initialised
  // KV / mask buffers) relies on this fold to collapse to a single splat
  // constant when the source tensor has fully static shape.
  if (defOp->getName().getStringRef() == "onnx.Shape") {
    if (defOp->getNumOperands() != 1)
      return false;
    auto srcType =
        mlir::dyn_cast<mlir::RankedTensorType>(defOp->getOperand(0).getType());
    if (!srcType || !srcType.hasStaticShape())
      return false;
    // ONNX Shape supports start/end attributes for slicing the shape vector.
    int64_t rank = srcType.getRank();
    int64_t start = 0;
    int64_t end = rank;
    if (auto a = defOp->getAttrOfType<mlir::IntegerAttr>("start"))
      start = a.getInt();
    if (auto a = defOp->getAttrOfType<mlir::IntegerAttr>("end"))
      end = a.getInt();
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::max<int64_t>(0, std::min<int64_t>(rank, start));
    end = std::max<int64_t>(0, std::min<int64_t>(rank, end));
    if (end < start)
      end = start;
    out.clear();
    out.reserve(end - start);
    for (int64_t i = start; i < end; ++i)
      out.push_back(srcType.getDimSize(i));
    return true;
  }
  return false;
}

/// Build the scalar fill value for ConstantOfShape, respecting the optional
/// `value` ONNX attribute and the spec default (fp32 zero, retyped to the
/// result element type when they differ).
static mlir::LogicalResult buildScalarFillValue(mlir::Operation *op,
                                                mlir::PatternRewriter &rewriter,
                                                mlir::Location loc,
                                                mlir::Type elemType,
                                                mlir::Value &scalarOut) {
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

    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
      mlir::APFloat scalar = *valueDense.getValues<mlir::APFloat>().begin();
      scalarOut =
          mlir::arith::ConstantFloatOp::create(rewriter, loc, floatTy, scalar);
      return mlir::success();
    }
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
      mlir::APInt scalar = *valueDense.getValues<mlir::APInt>().begin();
      scalarOut =
          mlir::arith::ConstantIntOp::create(rewriter, loc, intTy, scalar);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported result element type");
  }

  // Spec default: 0.0 fp32, retyped to the result element type to keep the
  // IR well-typed even when shape inference picked a different default.
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
    mlir::APFloat zero(floatTy.getFloatSemantics(), 0);
    scalarOut =
        mlir::arith::ConstantFloatOp::create(rewriter, loc, floatTy, zero);
    return mlir::success();
  }
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
    mlir::APInt zero(intTy.getWidth(), 0);
    scalarOut = mlir::arith::ConstantIntOp::create(rewriter, loc, intTy, zero);
    return mlir::success();
  }
  return rewriter.notifyMatchFailure(op,
                                     "unsupported default result element type");
}

/// `onnx.Where(mask, _, _)` already broadcasts every input across the output
/// shape (stride==0 along axes where the operand has dim 1 or smaller rank)
/// in the runtime where-kernel.  So when the ConstantOfShape's only purpose is
/// to materialise a fill-value buffer for such an op, the full-size buffer can
/// be elided and the consumer can read from a rank-0 scalar instead.  This
/// saves the tensor.splat -> linalg.map -> scf.for -> cf chain (and the
/// full-size GPU allocation) on every inference.
///
/// Before:
///   %s = onnx.Shape %ids                       : tensor<2xi64>
///   %c = onnx.ConstantOfShape %s {value = -100} : tensor<?x?xi64>
///   %o = onnx.Where %mask, %c, %ids            : tensor<?x?xi64>
/// After:
///   %e = tensor.empty()                  : tensor<i64>
///   %c = linalg.fill ins(-100) outs(%e)  : tensor<i64>   // rank-0
///   %o = onnx.Where %mask, %c, %ids      : tensor<?x?xi64>  // broadcasts
///
/// IMPORTANT: we do NOT emit `arith.constant dense<v> : tensor<>` here.
/// After bufferization that lowers to a `memref.global "private" constant`
/// that lives in the compiled DLL's `.data` section -- a *host* pointer the
/// GPU where-kernel cannot dereference even on UMA targets.  A text-only path
/// masks the bug because the mask is all-false and the kernel never reads
/// `x[0]`; the moment any mask bit is true, `x[0]` is read on-device and the
/// kernel faults.  `tensor.empty + linalg.fill` survives canonicalization and
/// bufferizes to `memref.alloc + linalg.fill`.  For the rank-0 case the fill
/// is a single store (no loops), and `hip-materialize-host-scalars` then
/// redirects the small static-shape alloc to host-mapped scratch
/// (host-writable AND GPU-readable at the same VA on UMA), so the where kernel
/// reads a valid device-side address.
struct ConstantOfShapeAsScalar : public mlir::RewritePattern {
  ConstantOfShapeAsScalar(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/3, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 result");
    mlir::Value result = op->getResult(0);

    // Only fold when every consumer is `onnx.Where`.  Future broadcasts
    // (Mul / Add / ...) can be added incrementally as their lowerings are
    // verified to handle a rank-0 operand.
    if (result.use_empty())
      return rewriter.notifyMatchFailure(op, "dead op, leave to DCE");
    for (auto &use : result.getUses()) {
      if (use.getOwner()->getName().getStringRef() != "onnx.Where")
        return rewriter.notifyMatchFailure(
            op, "consumer is not onnx.Where; scalar broadcast not yet "
                "supported for this consumer");
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");

    mlir::Type elemType = resultType.getElementType();
    mlir::Location loc = op->getLoc();
    mlir::Value scalar;
    if (mlir::failed(buildScalarFillValue(op, rewriter, loc, elemType, scalar)))
      return mlir::failure();

    // rank-0 buffer via `tensor.empty + linalg.fill` -- deliberately NOT
    // `tensor.splat(scalar_const)`, which the canonicalizer folds to
    // `arith.constant dense<v> : tensor<>` (a host `memref.global`; see the
    // pattern doc above for the device-fault rationale).
    auto scalarTensorTy = mlir::RankedTensorType::get({}, elemType);
    mlir::Value empty =
        mlir::tensor::EmptyOp::create(rewriter, loc, scalarTensorTy,
                                      /*dynamicSizes=*/mlir::ValueRange{});
    auto fill = mlir::linalg::FillOp::create(
        rewriter, loc, mlir::ValueRange{scalar}, mlir::ValueRange{empty});
    rewriter.replaceOp(op, fill.getResult(0));
    return mlir::success();
  }
};

struct ConstantOfShapeFold : public mlir::RewritePattern {
  ConstantOfShapeFold(mlir::MLIRContext *ctx,
                      bool staticShapeSourceOnly = false)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/2, ctx),
        staticShapeSourceOnly(staticShapeSourceOnly) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");
    if (staticShapeSourceOnly) {
      mlir::Operation *source = op->getOperand(0).getDefiningOp();
      if (!source || source->getName().getStringRef() != "onnx.Shape")
        return rewriter.notifyMatchFailure(
            op, "pre-carrier fold is reserved for onnx.Shape");
    }

    llvm::SmallVector<int64_t> outShape;
    if (!extractConstantOfShapeVector(op->getOperand(0), outShape))
      return rewriter.notifyMatchFailure(
          op, "shape input is not a compile-time constant; "
              "falling back to ConstantOfShapeDynamic");
    for (int64_t d : outShape)
      if (d < 0)
        return rewriter.notifyMatchFailure(
            op, "negative dimension in ConstantOfShape input");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    mlir::Type elemType = resultType.getElementType();

    // The fold path replaces the op with a `arith.constant` whose type
    // is `tensor<outShape x elemType>`. If the original result type has
    // dynamic dims, the SSA value type would mismatch its uses and IR
    // verification would fail. Defer to the dynamic-shape pattern in
    // that case.
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "result type has dynamic dims; "
              "falling back to ConstantOfShapeDynamic");

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

private:
  bool staticShapeSourceOnly;
};

/// Dynamic-shape fallback: produce a `tensor.splat` whose dynamic dim sizes
/// are read out of the shape input at runtime via `tensor.extract`.
/// Handles either (a) a non-constant shape input or (b) a result type with
/// at least one dynamic dim.
struct ConstantOfShapeDynamic : public mlir::RewritePattern {
  ConstantOfShapeDynamic(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    mlir::Value shapeInput = op->getOperand(0);
    auto shapeTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(shapeInput.getType());
    if (!shapeTensorType || shapeTensorType.getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "shape input must be a rank-1 tensor");
    if (!shapeTensorType.getElementType().isInteger(64) &&
        !shapeTensorType.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "shape input must have int32 or int64 element type");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    mlir::Type elemType = resultType.getElementType();

    mlir::Location loc = op->getLoc();

    // For each dynamic result dim, read the corresponding entry from the
    // shape tensor and cast to `index`. The shape tensor's length must
    // equal the result rank; we trust shape inference and only emit an
    // extract per dynamic dim (static dims are encoded directly in the
    // result type).
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      mlir::Value idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      mlir::Value extracted = mlir::tensor::ExtractOp::create(
          rewriter, loc, shapeInput, mlir::ValueRange{idx});
      mlir::Value asIndex = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), extracted);
      dynSizes.push_back(asIndex);
    }

    mlir::Value scalar;
    if (mlir::failed(buildScalarFillValue(op, rewriter, loc, elemType, scalar)))
      return mlir::failure();

    // tensor.splat broadcasts a scalar to a ranked (possibly dynamic) shape;
    // dynamic dims must be supplied positionally in the same order they
    // appear in the result type.
    mlir::Value splat = mlir::tensor::SplatOp::create(rewriter, loc, scalar,
                                                      resultType, dynSizes);
    rewriter.replaceOp(op, splat);
    return mlir::success();
  }
};

} // namespace

void populateConstantOfShapeConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ConstantOfShapeAsScalar, ConstantOfShapeFold,
               ConstantOfShapeDynamic>(ctx);
}

void populateConstantOfShapePreLoweringPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<ConstantOfShapeAsScalar>(ctx);
  patterns.add<ConstantOfShapeFold>(ctx, /*staticShapeSourceOnly=*/true);
}

} // namespace hip
} // namespace mlir
