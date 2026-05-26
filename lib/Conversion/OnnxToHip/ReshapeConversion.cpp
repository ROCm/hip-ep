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

/// Build output shape for expand_shape operations.
/// Used by both Reshape and Unsqueeze when expanding dimensions.
///
/// For static dimensions: use compile-time size from outputType.
/// For dynamic dimensions: extract from input via DimOp, dividing out any
/// static dimensions in the same reassociation group.
///
/// LIMITATION: this helper only handles the "one dynamic output dim per
/// reassociation group" case. When a group has >= 2 dynamic dims (e.g.
/// rank-1 [B*S] -> rank-2 [B, S] with both B and S dynamic), the
/// "inputSize / staticProduct" recovery cannot disambiguate the
/// individual sizes. Use `buildExpandShapeOutputShapeFromShape` instead
/// for that case.
llvm::SmallVector<mlir::OpFoldResult> buildExpandShapeOutputShape(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value data,
    mlir::RankedTensorType outputType,
    llvm::ArrayRef<mlir::ReassociationIndices> reassoc) {
  int64_t outputRank = outputType.getRank();

  llvm::SmallVector<int64_t> outDimToInDim(outputRank, -1);
  for (auto [g, group] : llvm::enumerate(reassoc))
    for (int64_t idx : group)
      outDimToInDim[idx] = g;

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    int64_t srcDim = outDimToInDim[i];
    const auto &group = reassoc[srcDim];

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

/// True iff every dynamic dim in `outputType` can be recovered from
/// `(inputType, reassoc)` alone — i.e. each reassociation group contains
/// at most one dynamic output dim. When this returns false the caller
/// must read sizes from the Reshape shape operand instead (see
/// `buildExpandShapeOutputShapeFromShape`).
static bool
canRecoverDynDimsFromInput(mlir::RankedTensorType outputType,
                           llvm::ArrayRef<mlir::ReassociationIndices> reassoc) {
  for (const auto &group : reassoc) {
    int64_t dynCount = 0;
    for (int64_t idx : group)
      if (outputType.isDynamicDim(idx))
        ++dynCount;
    if (dynCount > 1)
      return false;
  }
  return true;
}

/// If `shapeOperand` is a compile-time-known rank-1 integer tensor, fill
/// `out` with one APInt per element and return success. Recognizes
/// arith.constant + tensor.from_elements (the latter folds in caller via
/// OpFoldResult, but we resolve it here so we can also catch literal -1
/// / 0 dim markers). Returns failure when the shape can only be resolved
/// at runtime (`tensor.from_elements` with non-constant elements, etc.).
static mlir::LogicalResult
tryGetStaticShapeOperandValues(mlir::Value shapeOperand,
                               llvm::SmallVectorImpl<int64_t> &out) {
  mlir::Operation *def = shapeOperand.getDefiningOp();
  if (!def)
    return mlir::failure();

  // Direct arith.constant dense<...>.
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    if (!dense)
      return mlir::failure();
    for (auto v : dense.getValues<mlir::APInt>())
      out.push_back(v.getSExtValue());
    return mlir::success();
  }

  // Externalized constant: memref.get_global -> bufferization.to_tensor
  // would normally carry a "value" attribute on the original onnx.Constant
  // but after lowering it's opaque. Pattern handled at the .from_elements
  // case below instead (since most shape vectors arrive as that op after
  // Concat conversion).
  if (auto fromElems = mlir::dyn_cast<mlir::tensor::FromElementsOp>(def)) {
    for (mlir::Value elem : fromElems.getElements()) {
      mlir::Operation *eDef = elem.getDefiningOp();
      if (!eDef)
        return mlir::failure();
      if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(eDef)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
          out.push_back(intAttr.getInt());
          continue;
        }
      }
      return mlir::failure();
    }
    return mlir::success();
  }

  return mlir::failure();
}

/// Build `tensor.expand_shape`'s output_shape operands by reading the
/// Reshape's shape operand element-by-element. This is the only correct
/// path when a reassociation group has > 1 dynamic output dim — there is
/// no way to recover individual sizes from the input tensor alone, the
/// frontend must have computed them and stored them in the shape input.
///
/// `shapeOperand` must be a rank-1 integer tensor of length == outputRank.
/// Static output dims are pinned to compile-time `index` attrs (so the
/// expand_shape verifier sees them as constants, not dynamic operands).
/// Dynamic output dims are read via `tensor.extract` + `arith.index_cast`.
/// When `shapeOperand` is itself `tensor.from_elements`, the extract
/// folds back to the original SSA value automatically (the same dim
/// value that fed the upstream Concat-and-shape-builder chain).
///
/// Handles the ONNX `shape == 0` / `shape == -1` cases statically when
/// the shape tensor is a compile-time constant; otherwise emits an
/// arith.muli / arith.divui chain to resolve -1 at runtime, which we
/// have not seen in any current model. To keep this fix focused, we
/// reject runtime -1 / 0 markers and let the conversion fall back so
/// the issue is visible at compile time. (No production graph we've
/// audited puts a runtime -1 in a shape vector that also has dynamic
/// EP-input dims; the -1 form is always paired with a fully-static
/// reshape and folds away.)
static mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>>
buildExpandShapeOutputShapeFromShape(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Value shapeOperand,
                                     mlir::RankedTensorType outputType) {
  auto shapeType =
      mlir::dyn_cast<mlir::RankedTensorType>(shapeOperand.getType());
  if (!shapeType || shapeType.getRank() != 1)
    return mlir::failure();
  if (!shapeType.getElementType().isIntOrIndex())
    return mlir::failure();

  int64_t outputRank = outputType.getRank();
  if (shapeType.isDynamicDim(0) || shapeType.getDimSize(0) != outputRank)
    return mlir::failure();

  // Best-effort static fold of the shape operand: lets us detect the
  // -1 / 0 ONNX markers, validate consistency with outputType, and
  // emit pure index attrs (no runtime extract) when everything is
  // compile-time known.
  llvm::SmallVector<int64_t> staticShape;
  bool haveStatic = mlir::succeeded(
      tryGetStaticShapeOperandValues(shapeOperand, staticShape));
  if (haveStatic && static_cast<int64_t>(staticShape.size()) != outputRank)
    haveStatic = false;

  auto indexType = rewriter.getIndexType();

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  outputShape.reserve(outputRank);

  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    if (haveStatic) {
      int64_t v = staticShape[i];
      if (v < 0 || v == 0) {
        // Static -1 or 0 means "infer" / "copy from input": we'd need
        // to compute the value from the input tensor's dims. Defer to
        // the buildExpandShapeOutputShape path -- which can handle a
        // single dynamic dim per group via DimOp + division. If the
        // caller couldn't use that path (multiple dyn dims per group)
        // we conservatively fail here.
        return mlir::failure();
      }
      outputShape.push_back(rewriter.getIndexAttr(v));
      continue;
    }

    // Runtime extract: shape[i] -> index.
    mlir::Value idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
    mlir::Value v = mlir::tensor::ExtractOp::create(rewriter, loc, shapeOperand,
                                                    mlir::ValueRange{idx});
    if (v.getType() != indexType)
      v = mlir::arith::IndexCastOp::create(rewriter, loc, indexType, v);
    outputShape.push_back(v);
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

    // Different rank: expand or collapse.
    if (outputRank != inputRank) {
      auto reassocOpt =
          mlir::getReassociationIndicesForReshape(inputType, outputType);
      if (!reassocOpt)
        return rewriter.notifyMatchFailure(
            op, "cannot compute reshape reassociation");

      if (outputRank > inputRank) {
        // Expand. Two strategies:
        //   (a) Inputs alone are enough to recover every dyn output dim
        //       (each reassoc group has at most one dyn output dim) --
        //       use the input-only `buildExpandShapeOutputShape` so we
        //       don't introduce extra extract ops or a dependency on a
        //       shape operand that may not yet be in a known form.
        //   (b) Otherwise (multiple dyn dims per group, e.g. rank-1
        //       [batch_seq] -> rank-2 [B, S]) read sizes element-by-
        //       element from op->getOperand(1).
        llvm::SmallVector<mlir::OpFoldResult> outputShape;
        if (canRecoverDynDimsFromInput(outputType, *reassocOpt)) {
          outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                    outputType, *reassocOpt);
        } else {
          // Need the Reshape shape operand (2nd ONNX operand).
          if (op->getNumOperands() < 2)
            return rewriter.notifyMatchFailure(
                op, "Reshape needs shape operand for multi-dyn expand");
          auto built = buildExpandShapeOutputShapeFromShape(
              rewriter, loc, op->getOperand(1), outputType);
          if (mlir::failed(built))
            return rewriter.notifyMatchFailure(
                op,
                "cannot derive output_shape for multi-dyn expand from shape "
                "operand (need rank-1 int tensor whose elements resolve at "
                "compile or runtime)");
          outputShape = std::move(*built);
        }
        auto expandOp = mlir::tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt, outputShape);
        rewriter.replaceOp(op, expandOp.getResult());
      } else {
        // Collapse: no dynamic shape computation needed.
        auto collapseOp = mlir::tensor::CollapseShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt);
        rewriter.replaceOp(op, collapseOp.getResult());
      }
      return mlir::success();
    }

    // Same rank, different shape: collapse to 1-D then expand.
    //
    // The static-shape path uses compile-time index attrs for the expand
    // output_shape. The dynamic-shape path reads sizes from the Reshape
    // shape operand the same way the multi-dyn expand above does, so
    // dynamic same-rank reshapes (e.g. text.onnx's [B, S*H_q, 256] ->
    // [B, S, H_q*256] family) lower without any runtime kernel.

    // For the collapse-to-1D step we need the total element count. It is
    // either a compile-time constant (both shapes static, or only the
    // batch dim dyn but a known product on either side) or a runtime
    // product of input dims via tensor.dim + arith.muli.
    int64_t inputElems = inputType.hasStaticShape()
                             ? inputType.getNumElements()
                             : mlir::ShapedType::kDynamic;
    int64_t outputElems = outputType.hasStaticShape()
                              ? outputType.getNumElements()
                              : mlir::ShapedType::kDynamic;
    if (inputElems != mlir::ShapedType::kDynamic &&
        outputElems != mlir::ShapedType::kDynamic && inputElems != outputElems)
      return rewriter.notifyMatchFailure(op, "element count mismatch");

    auto flatType = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                                inputType.getElementType());

    // Static path: keep using a statically-sized flat tensor so downstream
    // canonicalisation / bufferisation has the most information.
    if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
      flatType =
          mlir::RankedTensorType::get({inputElems}, inputType.getElementType());
    }

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
    if (outputType.hasStaticShape()) {
      for (int64_t i : llvm::seq<int64_t>(outputRank))
        flatOutputShape.push_back(
            rewriter.getIndexAttr(outputType.getDimSize(i)));
    } else {
      // Dynamic same-rank reshape: derive each output dim from the
      // Reshape shape operand. The collapse-then-expand pair degenerates
      // into a single buffer reinterpretation after bufferisation, so
      // there's still no runtime kernel.
      if (op->getNumOperands() < 2)
        return rewriter.notifyMatchFailure(
            op, "Reshape needs shape operand for dynamic same-rank reshape");
      auto built = buildExpandShapeOutputShapeFromShape(
          rewriter, loc, op->getOperand(1), outputType);
      if (mlir::failed(built))
        return rewriter.notifyMatchFailure(
            op, "cannot derive output_shape for dynamic same-rank reshape from "
                "shape operand (need rank-1 int tensor whose elements resolve "
                "at compile or runtime)");
      flatOutputShape = std::move(*built);
    }

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
    auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                   outputType, *reassocOpt);
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
