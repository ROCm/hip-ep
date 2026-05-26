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

/// True when the defining op of the axes value is compile-time known.
///
/// After constant externalization, small onnx.Constant / arith.constant tensors
/// become memref.get_global + bufferization.to_tensor (see OnnxToHip.cpp);
/// those must still count as constant axes. Arbitrary ToTensor(alloc) is not.
static bool axesDefOpIsCompileTimeKnown(mlir::Operation *axesDefOp) {
  if (mlir::isa<mlir::arith::ConstantOp>(axesDefOp) ||
      axesDefOp->hasAttr("value"))
    return true;
  auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(axesDefOp);
  if (!toTensor)
    return false;
  mlir::Operation *bufDef = toTensor.getBuffer().getDefiningOp();
  return bufDef && mlir::isa<mlir::memref::GetGlobalOp>(bufDef);
}

/// Validate common requirements for Unsqueeze/Squeeze operations.
/// Requires: 2 operands (data, axes), ranked tensors, matching element types,
/// and constant axes (dynamic axes would require runtime shape computation).
mlir::LogicalResult
validateSqueezeUnsqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const char *tensorOpName, mlir::Value &data,
                           mlir::Value &axes, mlir::RankedTensorType &inputType,
                           mlir::RankedTensorType &outputType) {
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2 operands (data, axes)");

  data = op->getOperand(0);
  axes = op->getOperand(1);

  inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  if (inputType.getElementType() != outputType.getElementType())
    return rewriter.notifyMatchFailure(op, "element type mismatch");

  auto axesDefOp = axes.getDefiningOp();
  if (!axesDefOp) {
    std::string msg =
        "axes must be defined by a constant operation (block "
        "argument not supported). Dynamic axes would require "
        "runtime shape computation and cannot use zero-cost tensor.";
    msg += tensorOpName;
    return rewriter.notifyMatchFailure(op, msg);
  }

  if (!axesDefOpIsCompileTimeKnown(axesDefOp)) {
    std::string msg =
        "axes must be compile-time constant (arith.constant, onnx.Constant, or "
        "memref.get_global + bufferization.to_tensor from constant "
        "externalization). Dynamic axes need runtime shape computation and "
        "cannot use zero-cost tensor.";
    msg += tensorOpName;
    msg += " approach";
    return rewriter.notifyMatchFailure(op, msg);
  }

  return mlir::success();
}

/// Try to resolve `output_shape[outDim]` from the ONNX Reshape `shape` operand
/// without `tensor.extract`.
///
/// Motivation: when a single reassociation group contains 2+ dynamic output
/// dims (e.g. an ONNX Reshape from `tensor<?xi64>` to `tensor<?x?xi64>` for the
/// mrope position_ids construction), the input alone only gives the group's
/// TOTAL size — not per-dim sizes.  The previous helper fell back to
/// `tensor.dim(input, srcDim)` for every dyn dim in the group, producing the
/// same SSA value for every output dim.  After CSE that collapses to a single
/// `memref.dim` and the resulting `expand_shape ... output_shape [%n, %n]`
/// claims an `[N, N] = N*N` element layout backed by an `N`-element buffer —
/// downstream `ExpandStridedMetadata` then trips on "There must be at most one
/// dynamic size per group".
///
/// Canonical IR pattern this resolves (Qwen3-style mrope):
///
///   %bs    = onnx.Gather(Shape(input_ids), [0]) // tensor<1xi64>
///   %ss    = onnx.Gather(Shape(input_ids), [1]) // tensor<1xi64>
///   %shape = onnx.Concat(%bs, %ss)              // tensor<2xi64>
///   %tot   = onnx.Mul(%bs, %ss)                 // tensor<1xi64>
///   %r     = onnx.Range(0, %tot, 1)             // tensor<?xi64>, len = bs*ss
///   %pos   = onnx.Reshape(%r, %shape)           // tensor<?x?xi64> = [bs, ss]
///
/// By the time `ReshapeToStdTensor` runs in the greedy driver, the earlier
/// ShapeConversion/GatherShapeFold patterns have rewritten the inner
/// Gather/Shape chain to `tensor.from_elements(arith.index_cast(tensor.dim))`
/// and the Concat into a single `tensor.from_elements(e0, e1)`.  We peek
/// through that and return the per-dim scalar directly so the resulting
/// expand_shape sees `output_shape [bs, ss]` with two DISTINCT SSA values.
///
/// Returns `nullopt` on miss; the caller falls back to the input-dim formula
/// (which is correct when the group has at most one dynamic output dim).
static std::optional<mlir::Value>
tryResolveOutputDimFromReshapeShape(mlir::Value shape, int64_t outDim,
                                    mlir::Location loc,
                                    mlir::PatternRewriter &rewriter) {
  if (!shape)
    return std::nullopt;

  // -- tensor.from_elements(e0, e1, ...)
  if (auto fromElts = shape.getDefiningOp<mlir::tensor::FromElementsOp>()) {
    if (outDim < 0 || outDim >= (int64_t)fromElts.getNumOperands())
      return std::nullopt;
    mlir::Value elem = fromElts.getOperand(outDim);
    // Strip an `arith.index_cast i64 <- index` round-trip so the caller sees
    // the original index-typed dim directly (matches PoolAllocs' hoistable
    // whitelist; an i64-typed value would not be hoistable on its own).
    if (auto castOp = elem.getDefiningOp<mlir::arith::IndexCastOp>()) {
      if (castOp.getIn().getType().isIndex())
        return castOp.getIn();
    }
    if (elem.getType().isIndex())
      return elem;
    return mlir::arith::IndexCastOp::create(rewriter, loc,
                                            rewriter.getIndexType(), elem)
        .getResult();
  }

  // -- compile-time constant tensor (onnx.Constant or any op with `value`)
  if (auto *def = shape.getDefiningOp()) {
    if (auto valueAttr =
            def->getAttrOfType<mlir::DenseIntElementsAttr>("value")) {
      if (outDim < 0 || outDim >= valueAttr.getNumElements())
        return std::nullopt;
      llvm::APInt val = *(valueAttr.value_begin<llvm::APInt>() + outDim);
      int64_t signedVal = val.getSExtValue();
      // The ONNX `-1` infer-this-dim sentinel cannot be turned into a
      // compile-time index value — bail and let the caller's input-dim
      // formula handle it (`-1` resolves to the group's leftover product).
      if (signedVal < 0)
        return std::nullopt;
      return mlir::arith::ConstantIndexOp::create(rewriter, loc, signedVal)
          .getResult();
    }
  }

  // -- onnx.Concat(t0, t1, ...) over rank-1 statically-sized i64 tensors.
  //    Compute which operand `outDim` falls in (relative to the concat axis-0
  //    element index) and recurse into that operand with the local index.
  //    The canonical mrope idiom hits this branch:
  //        shape = Concat(GatherShapeFold(batch), GatherShapeFold(seq))
  //    after `GatherShapeFold` rewrites each Gather(Shape, const) to a
  //    1-element `tensor.from_elements` — the recursive call lands on the
  //    `tensor.from_elements` arm above.
  if (auto *def = shape.getDefiningOp()) {
    if (def->getName().getStringRef() == "onnx.Concat") {
      // ONNX Concat must be along axis 0 here (the shape is a 1-D vector).
      // Read the raw APInt to avoid `IntegerAttr::getInt()`'s
      // signless-only assertion — ONNX exports `axis` as `si64`, so
      // `getInt()` would trip "must be signless integer" before we even
      // reach the value check.
      if (auto axisAttr = def->getAttrOfType<mlir::IntegerAttr>("axis"))
        if (axisAttr.getValue().getSExtValue() != 0)
          return std::nullopt;
      int64_t cum = 0;
      for (mlir::Value operand : def->getOperands()) {
        auto opTy = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
        if (!opTy || opTy.getRank() != 1 || opTy.isDynamicDim(0))
          return std::nullopt;
        int64_t len = opTy.getDimSize(0);
        if (outDim >= cum && outDim < cum + len)
          return tryResolveOutputDimFromReshapeShape(operand, outDim - cum, loc,
                                                     rewriter);
        cum += len;
      }
      return std::nullopt;
    }
  }

  return std::nullopt;
}

/// Build output shape for expand_shape operations.
/// Used by both Reshape and Unsqueeze when expanding dimensions.
///
/// For static dimensions: use compile-time size from outputType.
/// For dynamic dimensions: extract from input via DimOp, dividing out any
/// static dimensions in the same reassociation group.
///
/// When a reassociation group contains 2+ dynamic output dims (typical of an
/// ONNX Reshape from `tensor<?xi64>` to `tensor<?x?xi64>`), the input-dim-
/// based formula is ambiguous — it would assign the same SSA value to every
/// dynamic dim in the group (causing the downstream `[N, N]` failure described
/// on `tryResolveOutputDimFromReshapeShape`).  In that case we read per-dim
/// values from the optional ONNX Reshape `shape` operand instead.  When
/// `shape` is null (`Unsqueeze` does not have one) or resolution misses, we
/// fall through to the input-dim path — safe because Unsqueeze's groups
/// always have at most one dynamic dim, and any leftover Reshape miss surfaces
/// as a structural verifier failure rather than silent data corruption.
llvm::SmallVector<mlir::OpFoldResult>
buildExpandShapeOutputShape(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value data, mlir::RankedTensorType outputType,
                            llvm::ArrayRef<mlir::ReassociationIndices> reassoc,
                            mlir::Value shape) {
  int64_t outputRank = outputType.getRank();

  llvm::SmallVector<int64_t> outDimToInDim(outputRank, -1);
  for (auto [g, group] : llvm::enumerate(reassoc))
    for (int64_t idx : group)
      outDimToInDim[idx] = g;

  // Count dynamic output dims per reassociation group.  A group with 2+ dyn
  // dims is the case where the input alone is not enough information.
  llvm::SmallVector<int64_t> dynPerGroup(reassoc.size(), 0);
  for (int64_t i : llvm::seq<int64_t>(outputRank))
    if (outputType.isDynamicDim(i) && outDimToInDim[i] >= 0)
      dynPerGroup[outDimToInDim[i]]++;

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    int64_t srcDim = outDimToInDim[i];
    const auto &group = reassoc[srcDim];

    // For groups with 2+ dyn dims, attempt to resolve from the shape operand
    // first.  On success we get a DISTINCT per-dim SSA value (e.g. batch_dim
    // for output_shape[0], seq_dim for output_shape[1]) instead of the same
    // total-size value being repeated.
    if (shape && dynPerGroup[srcDim] >= 2) {
      auto resolved =
          tryResolveOutputDimFromReshapeShape(shape, i, loc, rewriter);
      if (resolved) {
        outputShape.push_back(*resolved);
        continue;
      }
      // Fall through; the existing path is structurally wrong here but
      // a downstream verifier ("at most one dynamic size per group") will
      // catch it loudly rather than silently miscompiling.
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

    // Different rank: expand or collapse
    if (outputRank != inputRank) {
      auto reassocOpt =
          mlir::getReassociationIndicesForReshape(inputType, outputType);
      if (!reassocOpt)
        return rewriter.notifyMatchFailure(
            op, "cannot compute reshape reassociation");

      if (outputRank > inputRank) {
        // Expand: use shared helper to build output shape.  Pass the ONNX
        // Reshape `shape` operand so groups with 2+ dyn output dims can be
        // resolved per-dim (see `tryResolveOutputDimFromReshapeShape`).
        mlir::Value shapeOperand =
            op->getNumOperands() >= 2 ? op->getOperand(1) : mlir::Value();
        auto outputShape = buildExpandShapeOutputShape(
            rewriter, loc, data, outputType, *reassocOpt, shapeOperand);
        auto expandOp = mlir::tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt, outputShape);
        rewriter.replaceOp(op, expandOp.getResult());
      } else {
        // Collapse: no dynamic shape computation needed
        auto collapseOp = mlir::tensor::CollapseShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt);
        rewriter.replaceOp(op, collapseOp.getResult());
      }
      return mlir::success();
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
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      // (a) Locate the unique static-different position and validate the rest.
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
        return rewriter.notifyMatchFailure(
            op, "same-rank dynamic reshape: pattern not recognised");

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
        return rewriter.notifyMatchFailure(
            op, "same-rank dynamic reshape: static dims not factor-related");

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
        return rewriter.notifyMatchFailure(
            op, "same-rank dynamic reshape: no adjacent dynamic absorber");

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
      // pool-base hoisting.)  No shape operand is threaded here because the
      // intermediate type is locally constructed (input dim grown into
      // (dyn/K, K) or (K, smaller)) and has at most one dynamic dim per
      // reassoc group by construction — the input-dim formula always works.
      auto intOutShape = buildExpandShapeOutputShape(
          rewriter, loc, data, intType, expandReassoc, /*shape=*/{});

      auto expanded = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, intType, data, expandReassoc, intOutShape);

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
    // Unsqueeze inserts only size-1 dims, so every reassoc group has at most
    // one dynamic output dim — no shape operand is needed.
    auto outputShape = buildExpandShapeOutputShape(
        rewriter, loc, data, outputType, *reassocOpt, /*shape=*/{});
    auto expandOp = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt, outputShape);
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

    auto reassocOpt =
        mlir::getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute squeeze reassociation");

    mlir::Location loc = op->getLoc();
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
        // Custom splits: prefer compile-time constants when available, but
        // also support runtime split-length tensors (e.g. externalized
        // constants loaded via globals).
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
