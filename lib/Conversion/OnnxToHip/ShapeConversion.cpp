/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "hip/Dialect/IR/HipShapeInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX `Shape` lowering. Two patterns share the slot:
//
//   1. ShapeToConstant — input is fully static. The result is a literal
//      `arith.constant` of i64s. No HIP op, no runtime function, no
//      per-inference work. This is the path every static-shape test in
//      tree (and every model produced via `fix_shapes()`) hits.
//
//   2. ShapeToHip — at least one input dim is dynamic, typically because
//      the input was produced by a Category-C op (e.g. `NonZero`). We emit
//      a `hip.shape` op that carries one DimSpec per output element. The
//      lowering pass evaluates each DimSpec inline (`arith.constant` for
//      Static leaves, `hipdnn_ep_state_read_dim` for RuntimeSlot leaves,
//      `arith.{add,mul,...}` for binary nodes), packs the resulting i64s
//      into a stack buffer and copies them H2D into the destination GPU
//      memref. Shape is a *slot reader*, not a slot publisher — its own
//      result length is statically known (`end - start`).
//
// Pattern benefits: ShapeToConstant runs at benefit=2 so it preempts the
// HIP-emitting path whenever the static fold applies; ShapeToHip is the
// fallback at benefit=1.
//===----------------------------------------------------------------------===//

// Compute the clamped [start, end) slice on the input shape vector, in
// the same way ONNX opset 15+ defines for the `Shape` attributes. Shared
// between the two patterns to keep the math identical.
static void computeShapeSlice(mlir::Operation *op, int64_t rank, int64_t &start,
                              int64_t &end) {
  start = 0;
  end = rank;
  // ONNX importer emits si64 (signed) attributes; IntegerAttr::getInt()
  // asserts signless, so reach in via the APInt and use signed extension.
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("start"))
    start = a.getValue().getSExtValue();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("end"))
    end = a.getValue().getSExtValue();
  if (start < 0)
    start += rank;
  if (end < 0)
    end += rank;
  start = std::max<int64_t>(0, std::min<int64_t>(rank, start));
  end = std::max<int64_t>(0, std::min<int64_t>(rank, end));
  if (end < start)
    end = start;
}

struct ShapeToConstant : public mlir::RewritePattern {
  // benefit=2: prefer the static fold whenever applicable.
  ShapeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "ShapeToConstant only handles statically-shaped inputs; the "
              "dynamic path is handled by ShapeToHip");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must have i64 elements");
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "result must be rank-1");

    int64_t rank = inputType.getRank();
    int64_t start, end;
    computeShapeSlice(op, rank, start, end);

    int64_t expectedLen = end - start;
    if (resultType.getDimSize(0) != mlir::ShapedType::kDynamic &&
        resultType.getDimSize(0) != expectedLen)
      return rewriter.notifyMatchFailure(
          op, "result length does not match start/end attributes");

    llvm::SmallVector<mlir::APInt> dims;
    dims.reserve(expectedLen);
    for (int64_t i = start; i < end; ++i)
      dims.emplace_back(64, inputType.getDimSize(i), /*isSigned=*/true);

    auto i64Ty = rewriter.getIntegerType(64);
    auto outTy = mlir::RankedTensorType::get({expectedLen}, i64Ty);
    auto attr = mlir::DenseElementsAttr::get(outTy, dims);

    mlir::Location loc = op->getLoc();
    mlir::Value cst = mlir::arith::ConstantOp::create(rewriter, loc, attr);
    rewriter.replaceOp(op, cst);
    return mlir::success();
  }
};

// ONNX Shape -> hip.shape (dynamic-input path).
//
// Each output element corresponds to one input dim within `[start, end)`.
// We walk the input via `shape_interface::resolveDimFromValue` to extract
// the per-dim DimSpec (Static / InputDim / RuntimeSlot / Binary) and pack
// the result into a per-element `output_dim_specs`-shaped ArrayAttr that
// the HipToLLVM pass evaluates.
struct ShapeToHip : public mlir::RewritePattern {
  ShapeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must have i64 elements");
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "result must be rank-1");

    // For each output element we need a non-empty DimSpec. Walk back
    // through the producer chain (NonZero etc.) to resolve dynamic dims;
    // bail out if any dim can't be expressed (the conversion will be
    // retried after more passes run).
    int64_t rank = inputType.getRank();
    int64_t start, end;
    computeShapeSlice(op, rank, start, end);
    int64_t length = end - start;

    if (resultType.getDimSize(0) != mlir::ShapedType::kDynamic &&
        resultType.getDimSize(0) != length)
      return rewriter.notifyMatchFailure(
          op, "result length does not match start/end attributes");

    auto *ctx = rewriter.getContext();
    llvm::SmallVector<mlir::Attribute, 8> elemSpecAttrs;
    elemSpecAttrs.reserve(length);
    for (int64_t i = start; i < end; ++i) {
      // Static dim: short-circuit, no need to walk the producer.
      DimSpec spec;
      if (!inputType.isDynamicDim(i)) {
        spec = DimSpec::makeStatic(inputType.getDimSize(i));
      } else {
        spec = shape_interface::resolveDimFromValue(input, (unsigned)i);
        if (spec.nodes().empty())
          return rewriter.notifyMatchFailure(
              op, "could not resolve DimSpec for a dynamic input dim — the "
                  "producer chain does not implement HipShapeOpInterface or "
                  "is not yet lowered");
      }
      elemSpecAttrs.push_back(spec.serializeAsArrayAttr(ctx));
    }
    mlir::ArrayAttr elementDimSpecs = rewriter.getArrayAttr(elemSpecAttrs);

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Output type is fully static — length is a compile-time constant.
    auto outTy =
        mlir::RankedTensorType::get({length}, resultType.getElementType());
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, outTy.getShape(), outTy.getElementType());

    auto hipOp = mlir::hip::ShapeOp::create(rewriter, loc, outTy, context,
                                            input, init, elementDimSpecs);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateShapeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<ShapeToConstant>(ctx);
  patterns.add<ShapeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
