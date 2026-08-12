/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SliceConversion.cpp - Convert onnx.Slice to hipsr.slice -----------===//
//
// onnx.Slice extracts a strided window along a set of axes, which hipsr.slice
// models directly. ONNX carries the window in four operands and states the
// bounds relative to each axis; the hipsr operation carries them as attributes
// already resolved against the axis, so the conversion reads the operands and
// resolves them. The placeholder's shape region is left empty for
// hipsr-populate-shape-region, as for every DPS operation.
//
// Resolving a bound needs the extent it counts against, which is why this
// covers constant windows over statically sized axes and rejects the rest. A
// runtime window, which the operation would have to take as an operand and
// resolve on device, is separate work.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace mlir {
namespace hipsr {
namespace {

// Reads a rank-1 integer constant. The operand may still be the onnx.Constant
// an importer produced or the hipsr.constant the constant pattern already
// built, depending on the order the conversion reaches them, so this keys on
// the `value` attribute both spell rather than on the operation name.
::mlir::FailureOr<::llvm::SmallVector<int64_t>>
readIntVector(::mlir::Value value) {
  ::mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return ::mlir::failure();
  }
  auto dense = definingOp->getAttrOfType<::mlir::DenseIntElementsAttr>("value");
  if (!dense ||
      ::mlir::cast<::mlir::ShapedType>(dense.getType()).getRank() != 1) {
    return ::mlir::failure();
  }
  return ::llvm::map_to_vector(
      dense.getValues<::llvm::APInt>(),
      [](::llvm::APInt entry) -> int64_t { return entry.getSExtValue(); });
}

// Returns operand `index`, or null when the operand list stops short of it
// because ONNX omitted the trailing optional operands.
::mlir::Value optionalOperand(::mlir::Operation *op, unsigned index) {
  return index < op->getNumOperands() ? op->getOperand(index) : ::mlir::Value();
}

// Resolves an ONNX Slice bound against `extent`: a negative bound counts back
// from the end, and one past either end saturates, which is how ONNX spells
// "to the end" as a large sentinel.
int64_t resolveBound(int64_t bound, int64_t extent) {
  if (bound < 0) {
    bound += extent;
  }
  return std::clamp<int64_t>(bound, 0, extent);
}

struct SliceToHipsr : public ::mlir::RewritePattern {
  SliceToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected three to five operands and a single result");
    }

    ::mlir::Value data = op->getOperand(0);
    auto dataType = ::mlir::dyn_cast<::mlir::RankedTensorType>(data.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (dataType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching data and result element types");
    }
    int64_t rank = dataType.getRank();

    ::mlir::FailureOr<::llvm::SmallVector<int64_t>> starts =
        readIntVector(op->getOperand(1));
    ::mlir::FailureOr<::llvm::SmallVector<int64_t>> ends =
        readIntVector(op->getOperand(2));
    if (::mlir::failed(starts) || ::mlir::failed(ends)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected constant starts and ends");
    }

    // ONNX defaults axes to every axis and steps to one per axis.
    ::llvm::SmallVector<int64_t> axes;
    if (::mlir::Value operand = optionalOperand(op, 3)) {
      ::mlir::FailureOr<::llvm::SmallVector<int64_t>> read =
          readIntVector(operand);
      if (::mlir::failed(read)) {
        return rewriter.notifyMatchFailure(op, "expected constant axes");
      }
      axes = std::move(*read);
    } else {
      axes = ::llvm::to_vector(::llvm::seq<int64_t>(0, rank));
    }

    ::llvm::SmallVector<int64_t> steps;
    if (::mlir::Value operand = optionalOperand(op, 4)) {
      ::mlir::FailureOr<::llvm::SmallVector<int64_t>> read =
          readIntVector(operand);
      if (::mlir::failed(read)) {
        return rewriter.notifyMatchFailure(op, "expected constant steps");
      }
      steps = std::move(*read);
    } else {
      steps.assign(axes.size(), 1);
    }

    if (starts->size() != axes.size() || ends->size() != axes.size() ||
        steps.size() != axes.size()) {
      return rewriter.notifyMatchFailure(
          op, "expected one start, end and step per axis");
    }

    // Rewrite the window in place, so the operation carries in-bounds,
    // non-negative bounds against normalized axes.
    ::llvm::SmallVector<int64_t> shape(dataType.getShape());
    ::llvm::SmallDenseSet<int64_t> slicedAxes;
    for (auto [axis, start, end, step] :
         ::llvm::zip_equal(axes, *starts, *ends, steps)) {
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || axis >= rank) {
        return rewriter.notifyMatchFailure(op,
                                           "expected axes in [-rank, rank)");
      }
      if (!slicedAxes.insert(axis).second) {
        return rewriter.notifyMatchFailure(op, "expected distinct axes");
      }
      // A negative step reverses the axis, which is a separate operation.
      if (step <= 0) {
        return rewriter.notifyMatchFailure(op, "expected positive steps");
      }
      if (dataType.isDynamicDim(axis)) {
        return rewriter.notifyMatchFailure(
            op, "expected a static extent on every sliced axis");
      }
      int64_t extent = dataType.getDimSize(axis);
      start = resolveBound(start, extent);
      end = std::max(resolveBound(end, extent), start);
      shape[axis] = ::llvm::divideCeil(end - start, step);
    }
    if (::mlir::failed(
            ::mlir::verifyCompatibleShape(shape, resultType.getShape()))) {
      return rewriter.notifyMatchFailure(
          op, "resolved window does not match the result shape");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init = PlaceholderOp::create(
                             rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                             ::mlir::ValueRange{data}, PlaceholderType::Normal)
                             .getResult(0);
    auto sliceOp =
        SliceOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                        data, init, rewriter.getDenseI64ArrayAttr(*starts),
                        rewriter.getDenseI64ArrayAttr(*ends),
                        rewriter.getDenseI64ArrayAttr(axes),
                        rewriter.getDenseI64ArrayAttr(steps));
    rewriter.replaceOp(op, sliceOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateSliceConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx) {
  patterns.add<SliceToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
