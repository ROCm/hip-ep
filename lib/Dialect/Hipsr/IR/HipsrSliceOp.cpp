/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange SliceOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult SliceOp::verify() {
  auto dataType = cast<ShapedType>(getData().getType());
  auto outputType = cast<ShapedType>(getInit().getType());
  ArrayRef<int64_t> starts = getStarts();
  ArrayRef<int64_t> ends = getEnds();
  ArrayRef<int64_t> axes = getAxes();
  ArrayRef<int64_t> steps = getSteps();
  int64_t rank = dataType.getRank();

  if (starts.size() != axes.size() || ends.size() != axes.size() ||
      steps.size() != axes.size()) {
    return emitOpError("starts, ends and steps must have one entry per axis; ")
           << axes.size() << " axes, but " << starts.size() << " starts, "
           << ends.size() << " ends and " << steps.size() << " steps";
  }
  if (dataType.getElementType() != outputType.getElementType()) {
    return emitOpError("data and output element types must match");
  }

  SmallVector<int64_t> expectedShape(dataType.getShape());
  llvm::SmallDenseSet<int64_t> slicedAxes;
  for (auto [axis, start, end, step] :
       llvm::zip_equal(axes, starts, ends, steps)) {
    if (axis < 0 || axis >= rank) {
      return emitOpError("axes must be in [0, data rank); data rank is ")
             << rank << ", got " << axis;
    }
    if (!slicedAxes.insert(axis).second) {
      return emitOpError("axes must be distinct; got axis ")
             << axis << " twice";
    }
    if (dataType.isDynamicDim(axis)) {
      return emitOpError("a sliced axis must be statically sized; axis ")
             << axis << " is dynamic";
    }
    if (step <= 0) {
      return emitOpError("steps must be positive; got ")
             << step << " on axis " << axis;
    }
    if (start < 0 || end < start || end > dataType.getDimSize(axis)) {
      return emitOpError("the window must lie within the axis; axis ")
             << axis << " has extent " << dataType.getDimSize(axis) << ", got ["
             << start << ", " << end << ")";
    }
    expectedShape[axis] = llvm::divideCeil(end - start, step);
  }

  if (failed(verifyCompatibleShape(expectedShape, outputType.getShape()))) {
    return emitOpError("output shape must be the data shape with each sliced "
                       "axis narrowed to its window");
  }
  return success();
}
