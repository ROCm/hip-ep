/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ExpandConversion.cpp - lower onnx.Expand to hip.expand
//--------------===//
//
// Lowers `onnx.Expand` to `tensor.empty` + `hip.expand`.  For dynamic result
// dims that come from the 1-D `shape` operand, we avoid `tensor.extract` on a
// `tensor.from_elements` shape vector when each element traces to
// `tensor.dim(source, k)` or a compile-time constant.  That keeps the
// bufferized size chain in the `memref.dim` / `arith.*` form
// `hip-pool-allocs` can hoist, instead of `memref.load` from a shared shape
// scratch buffer (dominance failure when two Expands reuse one Shape result).
//
// Symmetric to `GatherShapeFold.cpp`, which collapses Gather(Shape, const) ->
// tensor.dim.  Here the idiom is Shape(x) -> from_elements -> Expand, and we
// read dims directly from x for `tensor.empty` sizing.
//
// When `shape` is `tensor.from_elements` but a needed entry is not traceable,
// we fail the match so the pipeline surfaces a clear error rather than
// emitting extract/load IR that breaks pool-allocs.
//
// ONNX Expand broadcast semantics (v8/v13 spec):
//   output[k] = numpy.array(input) * numpy.ones(shape) -- right-aligned,
//   per-axis output_dim[k] = max(input_dim[k'], shape_value[s']).
// Critically: ONNX Expand is NOT numpy.broadcast_to.  When shape[s] == 1 but
// input has input_dim > 1 at the corresponding axis, the spec says the output
// dim is input_dim (identity broadcast), NOT 1.  This is rare in LLM-style
// exports (which build shape from `Shape(x)` so SSA-traced values already
// equal input_dim), but HF vision encoders (Qwen3.5-VL, ViT patch flows)
// emit literal `Expand(x, [1, 1, ...])` with x rank > 0 -- a pure no-op at
// runtime, but only if we honour max(input_dim, shape_value).  When the shape
// operand is an opaque constant we cannot peek into (case (C) below), we now
// emit `arith.maxsi(tensor.dim(input, k), index_cast(extract(shape, k)))`
// instead of just `index_cast(extract(shape, k))`.
//
// Before (broken on `Expand([3136,1152]xf16, dense<[1,1]>) -> [?, 1152]`):
//   %v = tensor.extract %shape[%c0] : tensor<2xi64>     // = 1
//   %d = arith.index_cast %v : i64 to index             // = 1
//   %t = tensor.empty(%d) : tensor<?x1152xf16>          // = tensor<1x1152>
//
// After:
//   %sv = tensor.extract %shape[%c0] : tensor<2xi64>    // = 1
//   %sd = arith.index_cast %sv : i64 to index           // = 1
//   %id = tensor.dim %input, %c0 : tensor<3136x1152xf16> // = 3136
//   %d  = arith.maxsi %id, %sd : index                  // = 3136
//   %t  = tensor.empty(%d) : tensor<?x1152xf16>         // = tensor<3136x1152>
//
// Compile-time fast paths (skip the `tensor.dim` + `maxsi` op pair):
//   - input_idx < 0  (shape rank > input rank, leading output axis with no
//                     input counterpart): output_dim = shape_value.
//   - input_dim is statically 1: max(1, shape) == shape, so output_dim =
//                                shape_value.
// Both compile-time-static-equivalent to the pre-fix path, so they preserve
// IR shape for the canonical attention-mask Expand idiom (Llama-style:
// input `[1, 1, S]`, shape via `from_elements(Shape(x))`).
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"

STATISTIC(NumExpandShapeTraces,
          "Number of Expand dynamic dims resolved via tensor.dim traceback "
          "instead of tensor.extract on a shape vector");

namespace mlir {
namespace hip {
namespace {

/// If \p value is `arith.index_cast` of an index-typed SSA value, return the
/// operand; otherwise return \p value unchanged.
static mlir::Value skipIndexCastToIndex(mlir::Value value) {
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
    if (castOp.getIn().getType().isIndex())
      return castOp.getIn();
  }
  return value;
}

/// Try to resolve `shape[elemIdx]` to an index SSA value without
/// `tensor.extract`, when `shape` is `tensor.from_elements(...)`.
///
/// Supported element producers (after optional `arith.index_cast` i64 <-
/// index):
///   - `tensor.dim %tensor, %k`
///   - `arith.constant` i64 (static dim baked into Shape lowering)
static std::optional<mlir::Value>
tryResolveIndexFromShapeVector(mlir::Value shape, int64_t elemIdx,
                               mlir::Location loc,
                               mlir::PatternRewriter &rewriter) {
  auto fromElts = shape.getDefiningOp<tensor::FromElementsOp>();
  if (!fromElts)
    return std::nullopt;
  if (elemIdx < 0 || elemIdx >= static_cast<int64_t>(fromElts.getNumOperands()))
    return std::nullopt;

  mlir::Value elem = skipIndexCastToIndex(fromElts.getOperand(elemIdx));

  if (auto dimOp = elem.getDefiningOp<tensor::DimOp>())
    return dimOp.getResult();

  if (auto cst = elem.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return arith::ConstantIndexOp::create(rewriter, loc,
                                            intAttr.getValue().getSExtValue());
  }
  return std::nullopt;
}

/// When `shape` is still `onnx.Shape(%tensor)` (Expand lowered before
/// ShapeConversion in the same greedy pass), resolve entry \p elemIdx in the
/// shape vector directly from \p tensor's dims (same start/end normalization
/// as ShapeConversion / GatherShapeFold).
static std::optional<mlir::Value>
tryResolveIndexFromOnnxShape(mlir::Value shape, int64_t elemIdx,
                             mlir::Location loc,
                             mlir::PatternRewriter &rewriter) {
  auto *shapeOp = shape.getDefiningOp();
  if (!shapeOp || shapeOp->getName().getStringRef() != "onnx.Shape")
    return std::nullopt;

  mlir::Value shapeInput = shapeOp->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(shapeInput.getType());
  if (!inputType)
    return std::nullopt;

  int64_t rank = inputType.getRank();
  int64_t start = 0;
  int64_t end = rank;
  if (auto startAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("start"))
    start = startAttr.getSInt();
  if (auto endAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("end"))
    end = endAttr.getSInt();
  if (start < 0)
    start += rank;
  if (end < 0)
    end += rank;
  start = std::max(start, int64_t(0));
  end = std::min(end, rank);
  int64_t rangeLen = std::max(end - start, int64_t(0));
  if (elemIdx < 0 || elemIdx >= rangeLen)
    return std::nullopt;

  int64_t absDim = start + elemIdx;
  if (inputType.isDynamicDim(absDim))
    return mlir::tensor::DimOp::create(rewriter, loc, shapeInput, absDim);
  return arith::ConstantIndexOp::create(rewriter, loc,
                                        inputType.getDimSize(absDim));
}

/// onnx.Expand -> hip.expand
struct ExpandToHip : public mlir::RewritePattern {
  ExpandToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value shape = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    int64_t resultRank = resultType.getRank();
    int64_t inputRank = inputType.getRank();

    auto shapeType = mlir::cast<mlir::RankedTensorType>(shape.getType());
    if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op, "expand shape input must have static rank-1 type");
    int64_t shapeLen = shapeType.getDimSize(0);

    mlir::Operation *shapeDef = shape.getDefiningOp();
    const bool shapeIsFromElements =
        shapeDef && isa<tensor::FromElementsOp>(shapeDef);
    const bool shapeIsOnnxShape =
        shapeDef && shapeDef->getName().getStringRef() == "onnx.Shape";

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      int64_t shapeIdx = i - (resultRank - shapeLen);
      // Right-aligned input axis matching this output axis (-1 if the output
      // axis has no input counterpart -- leading dims when shape rank is
      // larger than input rank).
      int64_t inputIdx = i - (resultRank - inputRank);
      mlir::Value dim;
      if (shapeIdx >= 0) {
        std::optional<mlir::Value> traced =
            tryResolveIndexFromShapeVector(shape, shapeIdx, loc, rewriter);
        if (!traced)
          traced = tryResolveIndexFromOnnxShape(shape, shapeIdx, loc, rewriter);
        if (traced) {
          dim = *traced;
          ++NumExpandShapeTraces;
        } else if (shapeIsFromElements || shapeIsOnnxShape) {
          return rewriter.notifyMatchFailure(
              op, "expand shape is Shape/from_elements but entry is not "
                  "tensor.dim or constant; cannot size empty() without "
                  "tensor.extract (breaks hip-pool-allocs dominance)");
        } else {
          // Opaque shape operand (initializer constant tensor or function
          // argument). Honour the ONNX Expand spec rule that
          //   output_dim[i] = max(input_dim[input_idx], shape_value[shape_idx])
          // not just `shape_value`. The pre-fix code used shape_value
          // directly, which silently shrank inputs when shape carries 1 at an
          // axis where input has > 1 (HF Qwen3.5-VL `Expand(x, [1,1])`
          // identity-broadcast pattern). See file header for the IR diff.
          mlir::Value idx =
              arith::ConstantIndexOp::create(rewriter, loc, shapeIdx);
          mlir::Value extracted = tensor::ExtractOp::create(
              rewriter, loc, shape, mlir::ValueRange{idx});
          mlir::Value shapeDim = arith::IndexCastOp::create(
              rewriter, loc, rewriter.getIndexType(), extracted);
          if (inputIdx < 0) {
            // Leading output axis with no input dim: output = shape_value.
            dim = shapeDim;
          } else if (!inputType.isDynamicDim(inputIdx) &&
                     inputType.getDimSize(inputIdx) == 1) {
            // Static input dim is 1: max(1, x) == x, so output = shape_value.
            // Preserves IR shape for the canonical attention-mask idiom
            // (input `[1,1,S]` Expand to `[B,H,S]`).
            dim = shapeDim;
          } else {
            // input_dim could be > 1 -- emit the runtime max so an
            // identity-broadcast shape operand does not collapse the output.
            mlir::Value inputDim =
                tensor::DimOp::create(rewriter, loc, input, inputIdx);
            dim = arith::MaxSIOp::create(rewriter, loc, inputDim, shapeDim);
          }
        }
      } else {
        // Output axis with no shape entry (shape rank < output rank): the
        // missing leading dims come straight from the input.
        if (inputIdx < 0)
          return rewriter.notifyMatchFailure(
              op, "cannot resolve dynamic dim from input or shape");
        dim = tensor::DimOp::create(rewriter, loc, input, inputIdx);
      }
      dynSizes.push_back(dim);
    }

    mlir::Value init =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                resultType.getElementType(), dynSizes);

    auto hipOp = hip::ExpandOp::create(rewriter, loc, resultType, context,
                                       input, shape, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::MLIRContext *ctx) {
  patterns.add<ExpandToHip>(ctx);
}

} // namespace hip
} // namespace mlir
