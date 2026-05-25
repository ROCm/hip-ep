/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "hip/Dialect/IR/HipShapeInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include <cstring>

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
// Two paths are provided, in benefit order:
//
//   1. ConstantOfShapeFold (benefit=2): when the shape input is a recognised
//      compile-time constant AND the result type is fully static, the op
//      collapses to a single splat `arith.constant`. No runtime work; the
//      whole computation is materialised at compile time. This is the
//      common case for transformer-style "allocate a zero KV/mask tensor"
//      patterns once shape inference + constant folding have run.
//
//   2. ConstantOfShapeDynamic (benefit=1): fallback for non-constant shape
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
    // ONNX importer uses signed (si64) attributes; `IntegerAttr::getInt()`
    // asserts signless, so go through getValue().getSExtValue() which works
    // for both signed and signless.
    int64_t rank = srcType.getRank();
    int64_t start = 0;
    int64_t end = rank;
    if (auto a = defOp->getAttrOfType<mlir::IntegerAttr>("start"))
      start = a.getValue().getSExtValue();
    if (auto a = defOp->getAttrOfType<mlir::IntegerAttr>("end"))
      end = a.getValue().getSExtValue();
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

/// Encode the ONNX `value` attribute into the i64 bit-pattern accepted by
/// `hip.constant_of_shape`'s `fill_value`. The runtime kernel re-interprets
/// the low bits per `output_data_type`.  Returns failure if the value
/// attribute is non-dense / unsupported.
static mlir::LogicalResult
encodeFillValueBits(mlir::Operation *op, mlir::Type elemType,
                    int64_t &bits_out) {
  bits_out = 0;
  auto valueAttr = op->getAttrOfType<mlir::ElementsAttr>("value");
  if (!valueAttr) {
    // Spec default: 0 (fp32). Bit-pattern of +0.0 in any IEEE/integer type
    // is all-zero, so 0 works for every supported elemType.
    return mlir::success();
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!dense)
    return mlir::failure();

  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
    mlir::APFloat v = *dense.getValues<mlir::APFloat>().begin();
    if (floatTy.isF32()) {
      float f = v.convertToFloat();
      uint32_t u = 0;
      std::memcpy(&u, &f, sizeof(u));
      bits_out = static_cast<int64_t>(u);
    } else if (floatTy.isF64()) {
      double d = v.convertToDouble();
      uint64_t u = 0;
      std::memcpy(&u, &d, sizeof(u));
      bits_out = static_cast<int64_t>(u);
    } else {
      // fp16 / bf16: APFloat already holds the 16-bit pattern; bitcast via
      // bitcastToAPInt to recover the raw bits unambiguously.
      bits_out = static_cast<int64_t>(v.bitcastToAPInt().getZExtValue());
    }
    return mlir::success();
  }
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
    mlir::APInt v = *dense.getValues<mlir::APInt>().begin();
    bits_out = static_cast<int64_t>(v.getSExtValue());
    return mlir::success();
  }
  return mlir::failure();
}

/// Map an MLIR element type to the runtime's HIPDNN_EP_DATATYPE_* enum that
/// `hip.constant_of_shape::output_data_type` carries. Returns -1 for
/// unsupported types. The set matches the runtime kernel's switch and the
/// table in `getHipdnnDataType` (HipToLLVMUtils.h) -- duplicated here so
/// the conversion has no dep on HipToLLVMUtils.
static int64_t mapElemTypeToHipdnnDtype(mlir::Type elemType) {
  if (elemType.isF32())
    return 0; // HIPDNN_EP_DATATYPE_FLOAT
  if (elemType.isF16())
    return 1; // HIPDNN_EP_DATATYPE_HALF
  if (elemType.isBF16())
    return 2; // HIPDNN_EP_DATATYPE_BFLOAT16
  if (elemType.isInteger(32))
    return 3; // HIPDNN_EP_DATATYPE_INT32
  if (elemType.isInteger(64))
    return 4; // HIPDNN_EP_DATATYPE_INT64
  if (elemType.isSignlessInteger(8) || elemType.isSignedInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8 (also covers ONNX bool)
  if (elemType.isF64())
    return 6; // HIPDNN_EP_DATATYPE_DOUBLE
  if (elemType.isUnsignedInteger(8))
    return 7; // HIPDNN_EP_DATATYPE_UINT8
  return -1;
}

/// Allocate a contiguous block of slot ids on the parent module. Mirrors
/// the bookkeeping pattern used by NonZeroConversion / RangeConversion --
/// we increment a per-module counter so each Category-C site gets unique
/// slot ids.
static llvm::SmallVector<int32_t, 4>
reserveSlotIds(mlir::Operation *op, mlir::PatternRewriter &rewriter,
               int64_t count) {
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  int32_t base = 0;
  if (auto a = moduleOp->getAttrOfType<mlir::IntegerAttr>(
          "hipdnn.next_dyn_slot_id"))
    base = static_cast<int32_t>(a.getInt());
  moduleOp->setAttr(
      "hipdnn.next_dyn_slot_id",
      rewriter.getI32IntegerAttr(base + static_cast<int32_t>(count)));
  llvm::SmallVector<int32_t, 4> ids;
  ids.reserve(count);
  for (int64_t i = 0; i < count; ++i)
    ids.push_back(base + static_cast<int32_t>(i));
  return ids;
}

/// Dynamic-shape fallback: emit `hip.constant_of_shape` and attach the
/// appropriate `output_dim_specs` (Category B) or `slot_ids` +
/// RuntimeSlot specs (Category C). The op replaces both the legacy
/// `tensor.splat` lowering path and the explicit `tensor.extract` chain
/// -- ConstantOfShape always goes through the runtime now (one launch),
/// keeping the dyn-shape tracking on a single op attribute.
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

    int64_t fillBits = 0;
    if (mlir::failed(encodeFillValueBits(op, elemType, fillBits)))
      return rewriter.notifyMatchFailure(op,
                                         "could not encode fill_value bits");
    int64_t hipdnnDtype = mapElemTypeToHipdnnDtype(elemType);
    if (hipdnnDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported result element type");

    mlir::Location loc = op->getLoc();
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value ctx = *ctxOrFailure;

    int64_t outRank = resultType.getRank();

    // ------------------------------------------------------------------
    // Operand-provenance dispatch.
    // ------------------------------------------------------------------
    // The shape tensor traces to a func-arg => Category B: every output
    // dim is `InputValueI64(arg_idx, i)`. We can size the destination at
    // compile time via tensor.empty's dynamic-dim values read out of the
    // shape tensor (which the EP marshals into host-readable memory), and
    // the EP resolves the output OrtValue shape from the DimSpec tree
    // pre-compute. No slot publish needed.
    //
    // The shape tensor is intermediate => Category C: the EP cannot read
    // it pre-compute. Allocate `outRank` slots (one per dim), attach a
    // RuntimeSlot DimSpec per dim, and have wrap_constant_of_shape_dyn
    // publish them at runtime.
    const bool shapeIsFuncArg = operandIsFuncEntryBlockArg(shapeInput);

    // Build the destination via tensor.empty.  For Category B we use
    // tensor.extract over the host-resident shape input to size dynamic
    // dims; for Category C we emit a sentinel-sized tensor.empty (any
    // dyn dim becomes a small placeholder constant: the actual buffer is
    // allocated by the wrapper out of the dyn pool, so the destination
    // SSA value only needs a valid rank/shape so memref bufferization
    // doesn't trip up).
    llvm::SmallVector<mlir::Value> dynSizes;
    if (shapeIsFuncArg) {
      for (int64_t i = 0; i < outRank; ++i) {
        if (!resultType.isDynamicDim(i))
          continue;
        mlir::Value idx =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        mlir::Value extracted = mlir::tensor::ExtractOp::create(
            rewriter, loc, shapeInput, mlir::ValueRange{idx});
        mlir::Value asIndex = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), extracted);
        dynSizes.push_back(asIndex);
      }
    } else {
      // Category C: substitute size-1 placeholders for every dynamic
      // dim. The wrapper publishes the actual buffer via slot
      // mechanism; the destination ptr is dropped at runtime in favour
      // of the slot buffer. tensor.empty needs a valid index for every
      // ? dim, so we feed it a constant 1 -- it produces a tiny placeholder
      // memref that bufferization can allocate and immediately leak; the
      // EP's tensor_t marshalling for Category-C outputs already skips
      // memcpy via the sentinel data=null path.
      mlir::Value one =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (int64_t i = 0; i < outRank; ++i) {
        if (resultType.isDynamicDim(i))
          dynSizes.push_back(one);
      }
    }
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), elemType, dynSizes);

    // ------------------------------------------------------------------
    // Build the DimSpec attribute tree.
    // ------------------------------------------------------------------
    // We always emit one inner DimSpec list per output dim.  Static dims
    // get a Static leaf; dynamic dims get either InputValueI64 (Category
    // B) or RuntimeSlot (Category C).
    auto *ctxRaw = rewriter.getContext();
    llvm::SmallVector<int32_t, 4> slot_ids;
    llvm::SmallVector<mlir::Attribute, 4> perDimSpecs;
    perDimSpecs.reserve(outRank);

    if (!shapeIsFuncArg) {
      slot_ids = reserveSlotIds(op, rewriter, outRank);
    }
    int32_t slot_cursor = 0;

    for (int64_t d = 0; d < outRank; ++d) {
      DimSpec spec;
      if (!resultType.isDynamicDim(d)) {
        spec = DimSpec::makeStatic(resultType.getDimSize(d));
      } else if (shapeIsFuncArg) {
        // Category B: the shape tensor is operand[0] of this op
        // (`shape`); after rewriting, ComposeDimSpecs walks
        // `op->getOperand(idx)` to substitute operand-relative leaves
        // into func-arg-relative leaves. `Hip_ConstantOfShapeOp` has
        // ctx@0, shape@1, output@2 -- so input_index=1 in the local
        // operand frame.
        spec = DimSpec::makeInputValueI64(/*input_index=*/1,
                                          /*flat_offset=*/d);
      } else {
        spec = DimSpec::makeRuntimeSlot(slot_ids[slot_cursor++]);
      }
      perDimSpecs.push_back(spec.serializeAsArrayAttr(ctxRaw));
    }
    auto outputDimSpecsAttr =
        rewriter.getArrayAttr({rewriter.getArrayAttr(perDimSpecs)});

    mlir::DenseI32ArrayAttr slotIdsAttr;
    if (!shapeIsFuncArg)
      slotIdsAttr = rewriter.getDenseI32ArrayAttr(slot_ids);

    auto cofOp = mlir::hip::ConstantOfShapeOp::create(
        rewriter, loc, resultType, ctx, shapeInput, init,
        rewriter.getI64IntegerAttr(fillBits),
        rewriter.getI64IntegerAttr(hipdnnDtype),
        /*output_dim_specs=*/outputDimSpecsAttr,
        /*slot_ids=*/slotIdsAttr);
    rewriter.replaceOp(op, cofOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateConstantOfShapeConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ConstantOfShapeFold, ConstantOfShapeDynamic>(ctx);
}

} // namespace hip
} // namespace mlir
