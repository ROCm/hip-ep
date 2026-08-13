/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shape Operations Helpers (Reshape, Unsqueeze, Squeeze)
//===----------------------------------------------------------------------===//

/// Validate common requirements for Unsqueeze/Squeeze operations.
///
/// Requires only: ranked data + result tensors with matching element type. The
/// `axes` operand's VALUE is intentionally NOT required to be present or
/// constant: both lowerings (Squeeze -> collapse_shape / rank-0 slice,
/// Unsqueeze -> expand_shape) derive the reassociation purely from the
/// input/output SHAPES (getReassociationIndicesForReshape), never from the axes
/// values. Demanding a compile-time-constant axes operand only gate-kept valid
/// zero-cost reshapes: a graph slice can route a folded `axes` constant in as a
/// block-argument graph input (a mid-model part fed a shared `const_1d_*` axes
/// tensor), and opset<13 carries axes as an attribute (1 operand) rather than
/// an operand -- both were wrongly rejected, leaving the op unconverted ->
/// `error: op was not bufferized`. `axes` is still returned (for callers that
/// want it) when present.
mlir::LogicalResult
validateSqueezeUnsqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const char *tensorOpName, mlir::Value &data,
                           mlir::Value &axes, mlir::RankedTensorType &inputType,
                           mlir::RankedTensorType &outputType) {
  (void)tensorOpName;
  if (op->getNumOperands() < 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected at least 1 operand (data)");

  data = op->getOperand(0);
  axes = op->getNumOperands() >= 2 ? op->getOperand(1) : mlir::Value();

  inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  if (inputType.getElementType() != outputType.getElementType())
    return rewriter.notifyMatchFailure(op, "element type mismatch");

  return mlir::success();
}

/// Count the dynamic output dims a reassociation group covers.
static int64_t countDynOutDims(mlir::RankedTensorType outputType,
                               const mlir::ReassociationIndices &group) {
  return llvm::count_if(
      group, [&](int64_t idx) { return outputType.isDynamicDim(idx); });
}

/// Resolve a Reshape's `shape` operand into one extent per output dim.
///
/// Only a host-visible shape vector is honoured, i.e. the
/// `tensor.from_elements` that ReshapeShapeFold leaves behind for the
/// `Reshape(_, Shape(x))` idiom. A shape tensor that is still device-resident
/// yields nullopt instead of paying for a readback; the caller then falls back
/// to `tensor.reshape`, which consumes the runtime shape vector directly.
static std::optional<llvm::SmallVector<mlir::OpFoldResult>>
resolveShapeOperandExtents(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value shapeOperand, int64_t outputRank) {
  if (!shapeOperand)
    return std::nullopt;
  auto fromElements =
      shapeOperand.getDefiningOp<mlir::tensor::FromElementsOp>();
  if (!fromElements ||
      static_cast<int64_t>(fromElements.getElements().size()) != outputRank)
    return std::nullopt;

  // Resolve every element before building anything, so a late unresolvable
  // entry cannot leave a half-materialized cast behind in the IR.
  llvm::SmallVector<mlir::OpFoldResult> extents;
  llvm::SmallVector<mlir::Value> needsCast(outputRank);
  extents.reserve(outputRank);
  for (auto [i, element] : llvm::enumerate(fromElements.getElements())) {
    // ReshapeShapeFold emits `arith.index_cast %tensor.dim`. Reusing the
    // pre-cast index keeps the extent on the same SSA value the rest of the
    // conversion derives dims from, instead of a round trip through i64.
    if (auto cast = element.getDefiningOp<mlir::arith::IndexCastOp>();
        cast && mlir::isa<mlir::IndexType>(cast.getIn().getType())) {
      extents.push_back(cast.getIn());
      continue;
    }
    if (auto constant = element.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
      // ONNX gives 0 ("keep the input dim") and -1 ("infer") meanings that
      // this helper does not resolve; neither is usable as an extent.
      if (!intAttr || intAttr.getInt() <= 0)
        return std::nullopt;
      extents.push_back(rewriter.getIndexAttr(intAttr.getInt()));
      continue;
    }
    needsCast[i] = element;
    extents.push_back(mlir::OpFoldResult{});
  }

  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!needsCast[i])
      continue;
    mlir::Value asIndex = mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), needsCast[i]);
    extents[i] = asIndex;
  }
  return extents;
}

/// Build output shape for expand_shape operations.
/// Used by both Reshape and Unsqueeze when expanding dimensions.
///
/// For static dimensions: use compile-time size from outputType.
/// For dynamic dimensions: extract from input via DimOp, dividing out any
/// static dimensions in the same reassociation group.
///
/// That derivation only holds while a group covers at most ONE dynamic output
/// dim. When a group splits one dynamic source dim into several dynamic output
/// dims, the source extent is their PRODUCT and says nothing about the split,
/// so those extents have to come from \p shapeOperand (the Reshape's `shape`
/// input). Returns nullopt when that is needed but unreadable, letting the
/// caller fall back to `tensor.reshape` instead of emitting an expand_shape
/// that claims each dynamic output dim is the whole product -- which is how
/// `[bs*ss, 2816] -> [bs, ss, 2816]` became `[ss, ss, 2816]` and dispatched
/// every Gemma-4 layer's input norm over ss^2 rows.
std::optional<llvm::SmallVector<mlir::OpFoldResult>>
buildExpandShapeOutputShape(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value data, mlir::RankedTensorType outputType,
                            llvm::ArrayRef<mlir::ReassociationIndices> reassoc,
                            mlir::Value shapeOperand = {}) {
  int64_t outputRank = outputType.getRank();

  llvm::SmallVector<int64_t> outDimToInDim(outputRank, -1);
  for (auto [g, group] : llvm::enumerate(reassoc))
    for (int64_t idx : group)
      outDimToInDim[idx] = g;

  std::optional<llvm::SmallVector<mlir::OpFoldResult>> shapeExtents;
  if (llvm::any_of(reassoc, [&](const mlir::ReassociationIndices &group) {
        return countDynOutDims(outputType, group) > 1;
      })) {
    shapeExtents =
        resolveShapeOperandExtents(rewriter, loc, shapeOperand, outputRank);
    if (!shapeExtents)
      return std::nullopt;
  }

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    int64_t srcDim = outDimToInDim[i];
    const auto &group = reassoc[srcDim];

    if (countDynOutDims(outputType, group) > 1) {
      outputShape.push_back((*shapeExtents)[i]);
      continue;
    }

    int64_t staticProduct = 1;
    for (int64_t idx : group)
      if (!outputType.isDynamicDim(idx))
        staticProduct *= outputType.getDimSize(idx);

    mlir::Value inputSize =
        mlir::tensor::DimOp::create(rewriter, loc, data, srcDim);
    if (staticProduct == 1) {
      outputShape.push_back(inputSize);
    } else {
      mlir::Value divisor =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, staticProduct);
      mlir::Value dynSize =
          mlir::arith::DivUIOp::create(rewriter, loc, inputSize, divisor);
      outputShape.push_back(dynSize);
    }
  }

  return outputShape;
}

//===----------------------------------------------------------------------===//
// Reshape -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Reshape -> tensor.expand_shape / tensor.collapse_shape.
///
/// Reshape is a zero-cost metadata operation: it reinterprets shape/strides
/// without moving data.  We lower to standard MLIR tensor ops which
/// bufferize to memref.expand_shape / memref.collapse_shape (zero-copy
/// alias) and then to LLVM struct manipulation (same data pointer, new
/// sizes/strides).  No HIP kernel is needed.
struct ReshapeToStdTensor : public mlir::RewritePattern {
  ReshapeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "element type mismatch");

    mlir::Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = outputType.getRank();

    // No-op: same type
    if (inputType == outputType) {
      rewriter.replaceOp(op, data);
      return mlir::success();
    }

    // Rank-0 → rank-N where every output dim is static and the total element
    // count equals 1 (a scalar wrapped into a degenerate-shape tensor). The
    // `tensor.expand_shape` op cannot express a 0→N source-to-result rank
    // change because reassociation groups need at least one source dim per
    // group. `tensor.from_elements` is the canonical lowering: extract the
    // scalar value, then re-emit it as the single element of an N-D tensor.
    //
    // Canonical site: integer shape-arithmetic chains in dynamic-shape vision
    // encoders (Qwen-style image grid arithmetic). A `ReduceMax/Gather/Squeeze`
    // produces a `tensor<i64>` scalar that the model then `Reshape`s to
    // `tensor<1xi64>` so it can be `Concat`-ed into a larger shape vector.
    //
    // The scalar must reach the host to re-emit via tensor.from_elements. For a
    // runtime, GPU-produced scalar a plain tensor.extract races the kernel (see
    // ReadbackScalar.h); we read it back through hip.readback_scalar
    // (D2H + stream sync) instead. Compile-time constants fold.
    //
    // Before:
    //   %r = onnx.Reshape %s : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    // After (runtime scalar):
    //   %v = hip.readback_scalar(%ctx, %s : tensor<i64>) -> i64
    //   %r = tensor.from_elements %v : tensor<1xi64>
    if (inputRank == 0 && outputRank > 0 && outputType.hasStaticShape() &&
        outputType.getNumElements() == 1) {
      mlir::Value scalar =
          readbackScalarToHostOrExtract(rewriter, loc, op, data);
      mlir::Value flat = mlir::tensor::FromElementsOp::create(
          rewriter, loc,
          mlir::RankedTensorType::get({1}, inputType.getElementType()),
          mlir::ValueRange{scalar});
      if (outputRank == 1) {
        rewriter.replaceOp(op, flat);
      } else {
        // Promote 1-element rank-1 to rank-N (all unit dims) via a single
        // expand_shape group covering every output dim.
        mlir::ReassociationIndices all;
        for (int64_t i : llvm::seq<int64_t>(outputRank))
          all.push_back(i);
        llvm::SmallVector<mlir::ReassociationIndices> reassoc = {all};
        llvm::SmallVector<mlir::OpFoldResult> shape;
        for (int64_t i : llvm::seq<int64_t>(outputRank))
          shape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
        auto expanded = mlir::tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, flat, reassoc, shape);
        rewriter.replaceOp(op, expanded.getResult());
      }
      return mlir::success();
    }

    // Different rank: expand or collapse via reassociation when MLIR's helper
    // can prove a clean mapping. Otherwise fall through to the
    // tensor.reshape fallback below — common for fully-dynamic inputs from
    // loop-body block arguments where the helper has no static dim to anchor
    // groups against.
    if (outputRank != inputRank) {
      if (auto reassocOpt =
              mlir::getReassociationIndicesForReshape(inputType, outputType)) {
        if (outputRank > inputRank) {
          // The `shape` operand is the only place a multi-dynamic split's
          // per-dim extents exist; ReshapeShapeFold has already folded
          // `Shape(x)` into a host-visible tensor.from_elements for it.
          mlir::Value shapeOperand =
              op->getNumOperands() >= 2 ? op->getOperand(1) : mlir::Value();
          if (auto outputShape = buildExpandShapeOutputShape(
                  rewriter, loc, data, outputType, *reassocOpt, shapeOperand)) {
            auto expandOp = mlir::tensor::ExpandShapeOp::create(
                rewriter, loc, outputType, data, *reassocOpt, *outputShape);
            rewriter.replaceOp(op, expandOp.getResult());
            return mlir::success();
          }
          // Multi-dynamic split with an unreadable shape operand: fall through
          // to the tensor.reshape fallback, which takes the runtime shape
          // vector verbatim.
        } else {
          auto collapseOp = mlir::tensor::CollapseShapeOp::create(
              rewriter, loc, outputType, data, *reassocOpt);
          rewriter.replaceOp(op, collapseOp.getResult());
          return mlir::success();
        }
      }
      // Fall through to tensor.reshape fallback (no structured reassoc).
    }

    // Same rank, dynamic — decompose to expand_shape + collapse_shape when
    // the difference can be explained by ONE static dim splitting a factor K
    // and an adjacent dynamic dim absorbing it. Canonical case (a same-rank
    // dynamic Reshape pair around a per-head norm op):
    //   Reshape_1 (split):   <?x?xH*D> -> <?x?xD>      (last dim shrinks K=H)
    //   Reshape_2 (combine): <?x?xD>   -> <?x?xH*D>    (last dim grows K=H)
    // Both decompose without a kernel; the bufferized expand/collapse are
    // pure descriptor (offset/stride) edits.
    //
    // IR example (split direction, H = 8 absorbed by the leading dyn dim):
    //
    //   Before:
    //     %r = onnx.Reshape %x : tensor<?x?x40xf16> to tensor<?x?x5xf16>
    //
    //   After:
    //     %e = tensor.expand_shape %x [[0], [1], [2, 3]]
    //              : tensor<?x?x40xf16> into tensor<?x?x8x5xf16>
    //     %r = tensor.collapse_shape %e [[0], [1, 2], [3]]
    //              : tensor<?x?x8x5xf16> into tensor<?x?x5xf16>
    //
    // IR example (combine direction, K = 8 reabsorbed back into the static
    // dim — requires an arith.divui in the expand `output_shape`):
    //
    //   Before:
    //     %r = onnx.Reshape %x : tensor<?x?x5xf16> to tensor<?x?x40xf16>
    //
    //   After:
    //     %d   = tensor.dim %x, %c1 : tensor<?x?x5xf16>
    //     %k   = arith.constant 8 : index
    //     %d8  = arith.divui %d, %k : index
    //     %d0  = tensor.dim %x, %c0 : tensor<?x?x5xf16>
    //     %e   = tensor.expand_shape %x [[0], [1, 2], [3]]
    //              output_shape [%d0, %d8, 8, 5]
    //              : tensor<?x?x5xf16> into tensor<?x?x8x5xf16>
    //     %r   = tensor.collapse_shape %e [[0], [1], [2, 3]]
    //              : tensor<?x?x8x5xf16> into tensor<?x?x40xf16>
    //
    // Pattern requirements (otherwise fall through to the 1-D-flatten path
    // which only works for fully-static shapes):
    //   1. Exactly ONE static dim differs between input and output, with
    //      sizes related by an integer factor K (= max/min, exact divide).
    //   2. The factor-bearing static position is adjacent to a position
    //      that is dynamic in BOTH input and output (the "absorber").
    //   3. All other positions match exactly (static==static same size,
    //      dyn==dyn).
    // Same-rank dynamic structured path: scoped in a do-while(0) so that
    // pattern-not-recognised cases `break` out and fall through to the
    // tensor.reshape fallback below, instead of returning a hard failure.
    if (inputRank == outputRank &&
        (!inputType.hasStaticShape() || !outputType.hasStaticShape()))
      do {
        // (a) Locate the unique static-different position and validate the
        // rest.
        int64_t staticIdx = -1;
        bool valid = true;
        for (int64_t i = 0; i < inputRank && valid; ++i) {
          bool inDyn = inputType.isDynamicDim(i);
          bool outDyn = outputType.isDynamicDim(i);
          if (inDyn != outDyn) {
            valid = false; // dyn↔static at the same position — unsupported
            break;
          }
          if (inDyn)
            continue;
          if (inputType.getDimSize(i) == outputType.getDimSize(i))
            continue;
          if (staticIdx >= 0) {
            valid = false; // more than one static-different position
            break;
          }
          staticIdx = i;
        }
        if (!valid || staticIdx < 0)
          break; // fall through to tensor.reshape fallback

        int64_t inStatic = inputType.getDimSize(staticIdx);
        int64_t outStatic = outputType.getDimSize(staticIdx);
        int64_t k = 0;
        bool splitDir = false;
        if (inStatic > outStatic && inStatic % outStatic == 0) {
          k = inStatic / outStatic;
          splitDir = true;
        } else if (outStatic > inStatic && outStatic % inStatic == 0) {
          k = outStatic / inStatic;
          splitDir = false;
        }
        if (k <= 1)
          break; // fall through to tensor.reshape fallback

        // (b) Find the absorber: an adjacent position that is dynamic in both.
        auto isDynBoth = [&](int64_t i) {
          return i >= 0 && i < inputRank && inputType.isDynamicDim(i) &&
                 outputType.isDynamicDim(i);
        };
        int64_t dynIdx = -1;
        if (isDynBoth(staticIdx - 1))
          dynIdx = staticIdx - 1;
        else if (isDynBoth(staticIdx + 1))
          dynIdx = staticIdx + 1;
        if (dynIdx < 0)
          break; // fall through to tensor.reshape fallback

        // (c) Build the rank-(N+1) intermediate shape and the expand
        // reassociation. The "split" position is the input dim that grows from
        // 1 sub-dim to 2:
        //   * split direction:  splitInputDim = staticIdx (K*small -> K, small)
        //   * combine direction: splitInputDim = dynIdx   (dyn   -> dyn/K, K)
        int64_t splitInputDim = splitDir ? staticIdx : dynIdx;
        llvm::SmallVector<int64_t> intShape;
        intShape.reserve(inputRank + 1);
        llvm::SmallVector<mlir::ReassociationIndices> expandReassoc;
        int64_t cursor = 0;
        for (int64_t i = 0; i < inputRank; ++i) {
          if (i == splitInputDim) {
            if (splitDir) {
              intShape.push_back(k);         // outer (K)
              intShape.push_back(outStatic); // inner (smaller static)
            } else {
              intShape.push_back(mlir::ShapedType::kDynamic); // outer (dyn/K)
              intShape.push_back(k);                          // inner (K)
            }
            expandReassoc.push_back({cursor, cursor + 1});
            cursor += 2;
          } else {
            intShape.push_back(inputType.isDynamicDim(i)
                                   ? mlir::ShapedType::kDynamic
                                   : inputType.getDimSize(i));
            expandReassoc.push_back({cursor});
            cursor += 1;
          }
        }
        auto intType =
            mlir::RankedTensorType::get(intShape, inputType.getElementType());

        // (d) Reuse the existing helper to compute output_shape values for the
        // expand. It emits tensor.dim for dynamic dims and arith.divui when a
        // dynamic input dim is split into (dyn, static_factor) — exactly the
        // combine direction here. (PoolAllocs's hoistable whitelist must
        // include arith.divui for the resulting dim arithmetic to survive
        // pool-base hoisting.)
        // Each group here pairs one dynamic extent with the static factor K,
        // so no group is multi-dynamic and the shape operand is never needed.
        auto intOutShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                       intType, expandReassoc);
        if (!intOutShape)
          break; // fall through to tensor.reshape fallback

        auto expanded = mlir::tensor::ExpandShapeOp::create(
            rewriter, loc, intType, data, expandReassoc, *intOutShape);

        // (e) Collapse: the OUTPUT dim that absorbs the factor pair maps to the
        // two intermediate dims; everything else is identity.
        //   * split direction:  collapse target = dynIdx (absorbs K into dyn)
        //   * combine direction: collapse target = staticIdx (forms K*small)
        int64_t collapseTarget = splitDir ? dynIdx : staticIdx;
        llvm::SmallVector<mlir::ReassociationIndices> collapseReassoc;
        cursor = 0;
        for (int64_t i = 0; i < outputRank; ++i) {
          if (i == collapseTarget) {
            collapseReassoc.push_back({cursor, cursor + 1});
            cursor += 2;
          } else {
            collapseReassoc.push_back({cursor});
            cursor += 1;
          }
        }

        auto collapsed = mlir::tensor::CollapseShapeOp::create(
            rewriter, loc, outputType, expanded.getResult(), collapseReassoc);
        rewriter.replaceOp(op, collapsed.getResult());
        return mlir::success();
      } while (false);

    // Dynamic-shape fallback: when the structured expand/collapse paths above
    // don't recognise the pattern (e.g. fully-dynamic input from a loop-body
    // block argument, or rank-changing with a leading static 1 and no static
    // dim to anchor reassoc groups against), lower to tensor.reshape which
    // accepts the runtime shape operand directly. tensor.reshape bufferizes
    // to memref.reshape (zero-copy when source is contiguous) and consumes
    // the existing tensor<Nxi64> shape operand without conversion.
    //
    // Canonical site: dynamic-shape vision encoders inside hip.loop bodies
    // where loop block arguments arrive with under-refined types
    // (tensor<?x?x?xf16>) and the body's Reshape lifts them back to
    // higher-rank shapes (tensor<1x?x16x72xf16>) using a runtime-built shape
    // operand. The static-shape reassoc helper can't prove dim alignment when
    // the source is fully dynamic, so we hand the runtime shape verbatim to
    // tensor.reshape.
    //
    // Before:
    //   %r = onnx.Reshape %x, %shape :
    //          (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<1x?x16x72xf16>
    // After:
    //   %r = tensor.reshape %x(%shape) :
    //          (tensor<?x?x?xf16>, tensor<4xi64>) -> tensor<1x?x16x72xf16>
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      mlir::Value shapeOperand = op->getOperand(1);
      auto shapeTy =
          mlir::dyn_cast<mlir::RankedTensorType>(shapeOperand.getType());
      if (shapeTy && shapeTy.getRank() == 1 &&
          shapeTy.getElementType().isInteger() &&
          shapeTy.getDimSize(0) == outputRank) {
        // ONNX `Reshape` permits one shape entry to be `-1`, meaning "infer
        // this dim so the total element count is preserved". `memref.reshape`
        // (the bufferization target of `tensor.reshape`) does NOT interpret
        // `-1` -- it uses the shape value verbatim as the dim. If we pass
        // `-1` through, the alloc downstream sees a negative size, casts to
        // size_t, hipMalloc returns NULL, and the model SEGVs.
        //
        // Resolve `-1` here, before tensor.reshape: build a new shape tensor
        // where each `-1` entry is replaced by
        //   total_input_elements / product_of_other_dims_treating_-1_as_1.
        // Bufferization of tensor.from_elements + tensor.reshape stays
        // zero-copy for the data path; only the shape side gets the extra
        // arith chain.
        //
        // Before:
        //   %r = onnx.Reshape %x, %shape :
        //          (tensor<?x1152xf16>, tensor<6xi64>) ->
        //          tensor<?x?x2x?x2x?xf16>
        //   // %shape may contain a literal -1 at some index
        // After:
        //   %total  = product of tensor.dim(%x, i) for i in 0..inputRank
        //   %d[0..outRank-1] = tensor.extract %shape[i]
        //   %pp     = product of max(%d[i], 1) for i in 0..outRank
        //   %inf    = %total / %pp
        //   %d'[i]  = select(%d[i] == -1, %inf, %d[i])
        //   %newsh  = tensor.from_elements %d'[0..outRank-1] : tensor<NxI64>
        //   %r      = tensor.reshape %x(%newsh)
        mlir::Type elemTy = shapeTy.getElementType();
        unsigned bits = elemTy.getIntOrFloatBitWidth();
        mlir::Value cMinusOne = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(elemTy, mlir::APInt(bits, /*val=*/-1,
                                                        /*isSigned=*/true)));
        mlir::Value cOne = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(elemTy, 1));

        // total = product of input dim sizes (as elemTy).
        mlir::Value total = cOne;
        for (int64_t i : llvm::seq<int64_t>(inputRank)) {
          mlir::Value dimIdx =
              mlir::tensor::DimOp::create(rewriter, loc, data, i);
          mlir::Value dimI =
              mlir::arith::IndexCastOp::create(rewriter, loc, elemTy, dimIdx);
          total = mlir::arith::MulIOp::create(rewriter, loc, total, dimI);
        }

        // Extract each shape entry; compute product of positives (treat
        // negative / zero as 1 for the divisor) to find the inferred dim.
        llvm::SmallVector<mlir::Value> dims;
        dims.reserve(outputRank);
        mlir::Value posProduct = cOne;
        for (int64_t i : llvm::seq<int64_t>(outputRank)) {
          // Read each shape entry to the host with a stream sync (constants
          // fold) instead of a bare host load of device memory. A bare
          // tensor.extract here races a GPU-computed shape tensor and yields a
          // garbage dim that collapses the reshape. See ReadbackScalar.h.
          mlir::Value v = readbackShapeEntryToHostOrExtract(rewriter, loc, op,
                                                            shapeOperand, i);
          dims.push_back(v);
          mlir::Value isPositive = mlir::arith::CmpIOp::create(
              rewriter, loc, mlir::arith::CmpIPredicate::sgt, v, cOne);
          // sgt 1 catches >=2; combine with == 1 to keep 1 too.
          mlir::Value isOne = mlir::arith::CmpIOp::create(
              rewriter, loc, mlir::arith::CmpIPredicate::eq, v, cOne);
          mlir::Value keep =
              mlir::arith::OrIOp::create(rewriter, loc, isPositive, isOne);
          mlir::Value vForProd =
              mlir::arith::SelectOp::create(rewriter, loc, keep, v, cOne);
          posProduct =
              mlir::arith::MulIOp::create(rewriter, loc, posProduct, vForProd);
        }
        mlir::Value inferred =
            mlir::arith::DivSIOp::create(rewriter, loc, total, posProduct);

        // Replace -1 entries with the inferred dim.
        for (int64_t i : llvm::seq<int64_t>(outputRank)) {
          mlir::Value isMinusOne = mlir::arith::CmpIOp::create(
              rewriter, loc, mlir::arith::CmpIPredicate::eq, dims[i],
              cMinusOne);
          dims[i] = mlir::arith::SelectOp::create(rewriter, loc, isMinusOne,
                                                  inferred, dims[i]);
        }

        // Build the resolved shape tensor and pass to tensor.reshape.
        auto newShapeTy =
            mlir::RankedTensorType::get({(int64_t)outputRank}, elemTy);
        mlir::Value newShape = mlir::tensor::FromElementsOp::create(
            rewriter, loc, newShapeTy, dims);

        auto reshapeOp = mlir::tensor::ReshapeOp::create(
            rewriter, loc, outputType, data, newShape);
        rewriter.replaceOp(op, reshapeOp.getResult());
        return mlir::success();
      }
      return rewriter.notifyMatchFailure(
          op, "dynamic reshape: shape operand not a rank-1 integer tensor "
              "matching the output rank");
    }

    int64_t numElems = inputType.getNumElements();
    if (numElems != outputType.getNumElements())
      return rewriter.notifyMatchFailure(op, "element count mismatch");

    auto flatType =
        mlir::RankedTensorType::get({numElems}, inputType.getElementType());

    mlir::ReassociationIndices allInputDims;
    for (int64_t i : llvm::seq<int64_t>(inputRank))
      allInputDims.push_back(i);
    llvm::SmallVector<mlir::ReassociationIndices> collapseReassoc = {
        allInputDims};
    auto collapsed = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, flatType, data, collapseReassoc);

    mlir::ReassociationIndices allOutputDims;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      allOutputDims.push_back(i);
    llvm::SmallVector<mlir::ReassociationIndices> expandReassoc = {
        allOutputDims};
    llvm::SmallVector<mlir::OpFoldResult> flatOutputShape;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      flatOutputShape.push_back(
          rewriter.getIndexAttr(outputType.getDimSize(i)));
    auto expanded = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, collapsed.getResult(), expandReassoc,
        flatOutputShape);

    rewriter.replaceOp(op, expanded.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Unsqueeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Unsqueeze -> tensor.expand_shape (zero-cost metadata operation).
///
/// The reassociation is computed from the input/output SHAPES, so the `axes`
/// operand is not read (it may even be a block-argument graph input after a
/// graph slice routes a folded `const_1d_*` axes tensor across a part
/// boundary).
///
/// Before:
///   %r = onnx.Unsqueeze %x, %axes
///          : (tensor<?x?x128xf16>, tensor<1xi64>) -> tensor<?x?x128x1xf16>
/// After:
///   %r = tensor.expand_shape %x [[0], [1], [2, 3]]
///          output_shape [%b, %s, 128, 1]
///          : tensor<?x?x128xf16> into tensor<?x?x128x1xf16>
struct UnsqueezeToStdTensor : public mlir::RewritePattern {
  UnsqueezeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Unsqueeze", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data, axes;
    mlir::RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "expand_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    auto reassocOpt =
        mlir::getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute unsqueeze reassociation");

    mlir::Location loc = op->getLoc();
    // Unsqueeze only inserts unit dims, so every group keeps at most one
    // dynamic dim and no shape operand is needed.
    auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                   outputType, *reassocOpt);
    if (!outputShape)
      return rewriter.notifyMatchFailure(
          op, "unsqueeze: multi-dynamic reassociation group");
    auto expandOp = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt, *outputShape);
    rewriter.replaceOp(op, expandOp.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Squeeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Squeeze -> tensor.collapse_shape (zero-cost metadata operation).
struct SqueezeToStdTensor : public mlir::RewritePattern {
  SqueezeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Squeeze", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data, axes;
    mlir::RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "collapse_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    mlir::Location loc = op->getLoc();

    // Squeeze to a rank-0 scalar (typical case: a pooler reduces a [B,1]
    // statistic to a single value that is then consumed as a scalar
    // depth/index). A `tensor.collapse_shape` into rank-0 is only legal when
    // every collapsed input dim is a STATIC 1; an input with a dynamic or
    // non-unit dim (e.g. tensor<?x1xf16>) makes the verifier reject with "rank
    // 0 memrefs can only be extended/collapsed with/from ones". Squeeze
    // semantics guarantee every squeezed axis is size 1 at runtime, so slice
    // out the leading unit corner to a fully-static all-ones shape first
    // (zero-copy tensor.extract_slice), then collapse that to rank-0. The
    // full-extent slice folds away for an already-static all-ones input.
    //
    // Before:
    //   %r = onnx.Squeeze %x, %axes : (tensor<?x1xf16>, tensor<2xi64>)
    //          -> tensor<f16>
    // After:
    //   %s = tensor.extract_slice %x[0, 0] [1, 1] [1, 1]
    //          : tensor<?x1xf16> to tensor<1x1xf16>
    //   %r = tensor.collapse_shape %s [] : tensor<1x1xf16> into tensor<f16>
    if (outputType.getRank() == 0) {
      int64_t inputRank = inputType.getRank();
      llvm::SmallVector<mlir::OpFoldResult> offsets(inputRank,
                                                    rewriter.getIndexAttr(0));
      llvm::SmallVector<mlir::OpFoldResult> sizes(inputRank,
                                                  rewriter.getIndexAttr(1));
      llvm::SmallVector<mlir::OpFoldResult> strides(inputRank,
                                                    rewriter.getIndexAttr(1));
      llvm::SmallVector<int64_t> onesShape(inputRank, 1);
      auto onesType =
          mlir::RankedTensorType::get(onesShape, inputType.getElementType());
      mlir::Value unitCorner = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc, onesType, data, offsets, sizes, strides);
      auto collapseOp = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, outputType, unitCorner,
          llvm::ArrayRef<mlir::ReassociationIndices>{});
      rewriter.replaceOp(op, collapseOp.getResult());
      return mlir::success();
    }

    auto reassocOpt =
        mlir::getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute squeeze reassociation");

    auto collapseOp = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt);
    rewriter.replaceOp(op, collapseOp.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Split -> standard tensor ops (zero-cost slice views)
//===----------------------------------------------------------------------===//

/// onnx.Split -> tensor.extract_slice operations (zero-cost metadata
/// operation).
///
/// Split divides a tensor into multiple chunks along a specified axis.
/// This is a zero-cost operation: it creates views into the input tensor
/// without copying data. We lower to standard MLIR tensor.extract_slice ops
/// which bufferize to memref.subview (zero-copy alias) and then to LLVM
/// pointer arithmetic. No HIP kernel is needed.
struct SplitToStdTensor : public mlir::RewritePattern {
  SplitToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Split", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    mlir::Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    unsigned numOutputs = op->getNumResults();

    // Edge case: single output is identity operation
    if (numOutputs == 1) {
      rewriter.replaceOp(op, input);
      return mlir::success();
    }

    // Extract axis attribute (default 0)
    int64_t axis = 0;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();

    // Normalize negative axis
    if (axis < 0)
      axis += inputRank;
    if (axis < 0 || axis >= inputRank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Determine split mode: equal splits or custom splits
    llvm::SmallVector<mlir::OpFoldResult> splitLengths;
    bool isEqualSplit = true;

    if (op->getNumOperands() == 2) {
      // Check if second operand is onnx.NoValue (representing none/optional)
      mlir::Value splitInput = op->getOperand(1);
      auto splitDefOp = splitInput.getDefiningOp();

      // Handle onnx.NoValue: treat as equal split
      if (splitDefOp &&
          splitDefOp->getName().getStringRef() == "onnx.NoValue") {
        isEqualSplit = true;
      } else {
        // Custom splits: prefer inspectable compile-time constants/carriers
        // when available, but also support runtime split-length tensors.
        mlir::DenseElementsAttr splitAttr;
        if (splitDefOp && mlir::isa<mlir::arith::ConstantOp>(splitDefOp))
          if (auto constOp =
                  mlir::dyn_cast<mlir::arith::ConstantOp>(splitDefOp))
            splitAttr =
                mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
        if (!splitAttr && splitDefOp && splitDefOp->hasAttr("value"))
          splitAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(
              splitDefOp->getAttr("value"));

        if (splitAttr) {
          // Extract split lengths as index attributes
          for (auto val : splitAttr.getValues<mlir::APInt>())
            splitLengths.push_back(rewriter.getIndexAttr(val.getSExtValue()));

          if (splitLengths.size() != numOutputs)
            return rewriter.notifyMatchFailure(
                op, "split lengths count must match number of outputs");

          // Validate sum of splits equals axis dimension (if axis is static)
          if (!inputType.isDynamicDim(axis)) {
            int64_t axisDimSize = inputType.getDimSize(axis);
            int64_t splitSum = 0;
            for (const auto &length : splitLengths) {
              if (auto attr =
                      llvm::dyn_cast_if_present<mlir::Attribute>(length)) {
                auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
                if (intAttr) {
                  splitSum += intAttr.getInt();
                } else {
                  splitSum = -1;
                  break;
                }
              } else {
                splitSum = -1;
                break;
              }
            }
            if (splitSum >= 0 && splitSum != axisDimSize) {
              return rewriter.notifyMatchFailure(
                  op, "sum of split lengths must equal axis dimension size");
            }
          }
        } else {
          // Runtime path: read split lengths element-by-element.
          auto splitType =
              mlir::dyn_cast<mlir::RankedTensorType>(splitInput.getType());
          if (!splitType || splitType.getRank() != 1)
            return rewriter.notifyMatchFailure(
                op, "split input must be a rank-1 tensor");
          if (!splitType.getElementType().isIntOrIndex())
            return rewriter.notifyMatchFailure(
                op, "split input element type must be integer or index");
          if (!splitType.isDynamicDim(0) &&
              splitType.getDimSize(0) != static_cast<int64_t>(numOutputs))
            return rewriter.notifyMatchFailure(
                op, "split lengths count must match number of outputs");

          auto indexType = rewriter.getIndexType();
          for (unsigned i = 0; i < numOutputs; ++i) {
            mlir::Value idx =
                rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
            mlir::Value len = rewriter.create<mlir::tensor::ExtractOp>(
                loc, splitInput, mlir::ValueRange{idx});
            if (len.getType() != indexType)
              len = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType,
                                                              len);
            splitLengths.push_back(len);
          }
        }

        isEqualSplit = false;
      }
    }

    // Handle equal splits
    if (isEqualSplit) {
      mlir::Value axisDim =
          rewriter.create<mlir::tensor::DimOp>(loc, input, axis);
      mlir::Value numOutputsVal =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, numOutputs);
      mlir::Value chunkSize =
          rewriter.create<mlir::arith::DivUIOp>(loc, axisDim, numOutputsVal);

      // For equal splits: all outputs except possibly the last have size
      // chunkSize The last output gets the remainder: axis_size -
      // (num_outputs-1) * chunkSize
      for (unsigned i = 0; i < numOutputs - 1; ++i)
        splitLengths.push_back(chunkSize);

      // Last chunk size = axis_size - sum(previous chunks)
      // = axis_size - (num_outputs - 1) * chunkSize
      // Note: numOutputs is always >= 2 here (single output case returns early)
      mlir::Value numPrevChunks =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, numOutputs - 1);
      mlir::Value prevTotal =
          rewriter.create<mlir::arith::MulIOp>(loc, chunkSize, numPrevChunks);
      mlir::Value lastChunkSize =
          rewriter.create<mlir::arith::SubIOp>(loc, axisDim, prevTotal);
      splitLengths.push_back(lastChunkSize);
    }

    // Generate extract_slice for each output
    llvm::SmallVector<mlir::Value> replacements;
    mlir::OpFoldResult currentOffset = rewriter.getIndexAttr(0);

    for (unsigned i = 0; i < numOutputs; ++i) {
      auto outputType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(i).getType());
      if (!outputType)
        return rewriter.notifyMatchFailure(op, "expected ranked output type");

      // Track the actual slice size used for this output (for offset
      // calculation)
      mlir::OpFoldResult actualSliceSize;

      // Build offsets, sizes, strides arrays
      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      for (int64_t dim = 0; dim < inputRank; ++dim) {
        if (dim == axis) {
          // Axis dimension: use computed offset and split length
          offsets.push_back(currentOffset);

          // For the size: if the output dimension is static, use static attr;
          // otherwise use the dynamic value
          if (!outputType.isDynamicDim(dim)) {
            int64_t staticSize = outputType.getDimSize(dim);
            sizes.push_back(rewriter.getIndexAttr(staticSize));
            actualSliceSize = rewriter.getIndexAttr(staticSize);

            // Validate consistency: static output size must match split length
            if (!isEqualSplit) {
              // For custom splits, verify the split length matches
              if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(
                      splitLengths[i])) {
                if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                  if (intAttr.getInt() != staticSize) {
                    return rewriter.notifyMatchFailure(
                        op, "custom split length does not match static output "
                            "dimension");
                  }
                }
              }
            }
          } else {
            sizes.push_back(splitLengths[i]);
            actualSliceSize = splitLengths[i];
          }
        } else {
          // Other dimensions: identity (offset=0, size=dim_size, stride=1)
          offsets.push_back(rewriter.getIndexAttr(0));
          if (outputType.isDynamicDim(dim)) {
            mlir::Value dimSize =
                rewriter.create<mlir::tensor::DimOp>(loc, input, dim);
            sizes.push_back(dimSize);
          } else {
            sizes.push_back(rewriter.getIndexAttr(outputType.getDimSize(dim)));
          }
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }

      // Create extract_slice operation using OpBuilder
      // This will properly decompose OpFoldResults into static attrs and
      // dynamic operands
      mlir::OperationState state(
          loc, mlir::tensor::ExtractSliceOp::getOperationName());
      mlir::tensor::ExtractSliceOp::build(rewriter, state, outputType, input,
                                          offsets, sizes, strides);
      auto sliceOp = rewriter.create(state);
      replacements.push_back(sliceOp->getResult(0));

      // Update offset for next slice
      if (i < numOutputs - 1) {
        // IMPORTANT: Use the actual slice size (which may differ from
        // splitLengths[i] when output type has static dimension). This ensures
        // correct offset calculation.
        mlir::Value offsetVal =
            mlir::getValueOrCreateConstantIndexOp(rewriter, loc, currentOffset);
        mlir::Value lengthVal = mlir::getValueOrCreateConstantIndexOp(
            rewriter, loc, actualSliceSize);
        mlir::Value newOffsetVal =
            rewriter.create<mlir::arith::AddIOp>(loc, offsetVal, lengthVal);
        currentOffset = newOffsetVal;
      }
    }

    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }
};

} // namespace

void populateReshapeConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<ReshapeToStdTensor, UnsqueezeToStdTensor, SqueezeToStdTensor,
               SplitToStdTensor>(ctx);
}

} // namespace hip
} // namespace mlir
