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
//      (today: `onnx.Where`), replace the op with a rank-0 `arith.constant`
//      of the fill value.  The consumer broadcasts the scalar at no extra
//      memory cost (stride==0 along every output axis in the runtime
//      kernel), avoiding the `tensor.splat` -> `linalg.map` -> `scf.for`
//      bufferization chain that the dynamic path would otherwise produce.
//      This is the Qwen3.5-35B `embedding.onnx` mask-fill pattern
//      (`Shape -> ConstantOfShape(value=V) -> Where(mask, _, input_ids)`).
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

/// Return the dense-elements attribute backing \p value if it can be
/// determined at compile time (the same forms recognised by ReshapeConversion
/// for "axes" inputs).  Returns null otherwise.
static mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());

  // `onnx.Shape(static-tensor)` is itself a compile-time constant -- the
  // transformer-emitted `Shape -> ConstantOfShape` pattern (zero-initialised
  // KV / mask buffers) relies on this fold to collapse to a single splat
  // constant when the source tensor has fully static shape.
  if (defOp->getName().getStringRef() == "onnx.Shape") {
    if (defOp->getNumOperands() != 1)
      return nullptr;
    auto srcType =
        mlir::dyn_cast<mlir::RankedTensorType>(defOp->getOperand(0).getType());
    if (!srcType || !srcType.hasStaticShape())
      return nullptr;
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
    llvm::SmallVector<int64_t> dims;
    dims.reserve(end - start);
    for (int64_t i = start; i < end; ++i)
      dims.push_back(srcType.getDimSize(i));
    auto i64 = mlir::IntegerType::get(value.getContext(), 64);
    auto outTy = mlir::RankedTensorType::get({end - start}, i64);
    llvm::SmallVector<mlir::APInt> apDims;
    apDims.reserve(dims.size());
    for (int64_t d : dims)
      apDims.emplace_back(64, d);
    return mlir::DenseElementsAttr::get(outTy, apDims);
  }

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

/// Build a rank-0 DenseElementsAttr from a `value` attribute that may have
/// been emitted by different upstream paths:
///   * `DenseElementsAttr` -- the canonical ONNX form (TensorProto value
///     unpacked by the importer).
///   * `IntegerAttr` / `FloatAttr` -- the form the morphizen ir-converter
///     occasionally emits for single-element TensorProtos
///     (e.g. Qwen3.5-35B `embedding.onnx` ConstantOfShape value).
///   * `nullptr` (attribute missing) -- ONNX spec default of zero, retyped
///     to the result element type.
///
/// Returns std::nullopt if no scalar form can be derived.
static std::optional<mlir::DenseElementsAttr>
buildRank0ScalarAttr(mlir::Attribute valueAttr, mlir::Type elemType) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);

  if (auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(valueAttr)) {
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
      mlir::APFloat scalar = *dense.getValues<mlir::APFloat>().begin();
      return mlir::DenseElementsAttr::get(scalarType, scalar);
    }
    if (mlir::isa<mlir::IntegerType>(elemType)) {
      mlir::APInt scalar = *dense.getValues<mlir::APInt>().begin();
      return mlir::DenseElementsAttr::get(scalarType, scalar);
    }
    return std::nullopt;
  }

  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(valueAttr)) {
    auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType);
    if (!intTy)
      return std::nullopt;
    mlir::APInt scalar = intAttr.getValue();
    // Re-fit to result element bit-width; the attribute is most commonly
    // i64 because that's what ORT/morphizen unpacks for single-element
    // TensorProtos, but the result element type can be any integer width.
    scalar = scalar.sextOrTrunc(intTy.getWidth());
    return mlir::DenseElementsAttr::get(scalarType, scalar);
  }

  if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(valueAttr)) {
    auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType);
    if (!floatTy)
      return std::nullopt;
    mlir::APFloat scalar = floatAttr.getValue();
    bool losesInfo = false;
    scalar.convert(floatTy.getFloatSemantics(),
                   mlir::APFloat::rmNearestTiesToEven, &losesInfo);
    return mlir::DenseElementsAttr::get(scalarType, scalar);
  }

  if (!valueAttr) {
    // ONNX spec default: f32 0, retyped to the result element type.
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
      mlir::APFloat zero(floatTy.getFloatSemantics(), 0);
      return mlir::DenseElementsAttr::get(scalarType, zero);
    }
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
      return mlir::DenseElementsAttr::get(
          scalarType, mlir::APInt(intTy.getWidth(), 0));
    }
  }

  return std::nullopt;
}

/// `onnx.Where(mask, _, _)` already broadcasts every input across the output
/// shape (stride==0 along axes where the operand has dim 1 or smaller rank)
/// via `elementwise_where_kernel.hip::make_broadcast_layout`. So when the
/// ConstantOfShape's only purpose is to materialise a fill-value buffer for
/// such an op, the buffer can be elided entirely and the consumer can read
/// from a rank-0 scalar instead.  This saves the tensor.splat -> linalg.map
/// -> scf.for -> cf chain on every inference.
///
/// IMPORTANT: we do NOT emit `arith.constant dense<v> : tensor<>` here.
/// After bufferization that lowers to a `memref.global "private" constant`
/// that lives in the DLL's `.data` section -- a *host* pointer that the GPU
/// where-kernel cannot dereference.  The text-only path masked the bug
/// because the mask was all-false, so the kernel never actually read
/// `x[0]`; the moment a multimodal prompt sets even one mask bit, `x[0]`
/// is read on-device and the kernel faults with `unspecified launch
/// failure (719)`.
///
/// Instead, build the rank-0 constant via `tensor.splat`, which bufferizes
/// to a fresh `memref.alloc()` + `linalg.fill` placed in the GPU pool by
/// the standard pipeline.  The where kernel then reads the scalar at a
/// device-side virtual address, which is correct.  For rank-0 the
/// downstream chain stays trivial: the fill is a single store, no nested
/// loops, no scf-for / linalg-map blow-up.
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

    // rank-0 buffer materialised via `tensor.empty + linalg.fill`. This
    // intentionally avoids `tensor.splat(scalar_const)` because the MLIR
    // canonicalizer would fold that to `arith.constant dense<v> : tensor<>`,
    // which bufferizes to a *host* `memref.global "private" constant`. That
    // global lives in the compiled DLL's `.data` section -- a virtual
    // address the GPU cannot dereference even on UMA targets (gfx1151), so
    // any consumer kernel that actually reads `x[0]` (e.g. `hip.where` on
    // a multimodal prompt where the mask has a True bit) faults with HIP
    // 719 ("unspecified launch failure"). The text-only path masked the
    // bug because the mask is all-False there and the kernel never reads
    // the operand.
    //
    // `tensor.empty + linalg.fill` survives canonicalization and bufferizes
    // to `memref.alloc + linalg.fill`. `hip-materialize-host-scalars` then
    // catches the small static-shape memref.alloc with a host store on it
    // and redirects it to host-mapped scratch (`hipHostMalloc(Mapped)`),
    // giving us a buffer that is *both* host-writable AND GPU-readable at
    // the same VA on UMA targets. The end result is identical perf-wise
    // (one i64 store at inference start, no per-step overhead) while
    // remaining correct under the where kernel's `x[0]` read.
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
  ConstantOfShapeFold(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/2, ctx) {}

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
          op, "shape input is not a compile-time constant; "
              "falling back to ConstantOfShapeDynamic");

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

} // namespace hip
} // namespace mlir
