/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- View.cpp - ggml.view -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// A view is the one operation whose extents are not a consequence of
/// anything: GGML records the offset, the four extents, and the four strides
/// its caller asked for, and a sub-region of a source is as valid a view as a
/// rewrite of the whole of it. So the derivation reads the walk rather than
/// deriving it, and recovers only which of those numbers carry a symbol --
/// which the strides say, because a dimension that steps by one of the
/// source's strides is walking that dimension of the source.
LogicalResult ViewOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0))
    return emitOpError() << "expects an operand";
  // The offset is where the view begins in its root's storage, which
  // `deriveStorage` accumulates along the chain. It has to be a number for
  // that to work.
  if (!getStorageOffset(getResult()))
    return emitOpError() << "cannot derive a view whose byte offset is not a "
                            "constant";

  std::optional<Walk> walk = derivation.recoverWalk(
      derivation.operandShape(0), derivation.operandStrides(0),
      derivation.observedResultExtents(), derivation.observedResultStrides());
  if (!walk) {
    return emitOpError() << "cannot recover its extents: "
                         << derivation.arithmetic().getFailure();
  }
  derivation.setResultShape(std::move(walk->extents));
  derivation.setResultStrides(std::move(walk->strides));
  return success();
}

LogicalResult ViewOp::verify() {
  LayoutAttr layout = getGgmlLayout(getResult());
  Attribute byteOffset = getByteOffsetAttr();
  ArrayAttr byteStrides = getByteStridesAttr();
  if (!layout && (!byteOffset || !byteStrides)) {
    return emitOpError(
        "requires view attributes or a result `#ggml.layout` encoding");
  }
  if (byteOffset &&
      failed(verifyAffineExpression(*this, byteOffset, "byte_offset",
                                    /*allowZeroConstant=*/true))) {
    return failure();
  }
  if (byteStrides && byteStrides.size() != 4)
    return emitOpError("requires exactly 4 `byte_strides` entries");
  const bool hasZeroExtent =
      llvm::is_contained(getInput().getType().getShape(), 0) ||
      llvm::is_contained(getResult().getType().getShape(), 0);
  if (byteStrides) {
    for (auto [index, stride] : llvm::enumerate(byteStrides)) {
      std::string name = "byte_strides[" + std::to_string(index) + "]";
      if (failed(verifyAffineExpression(*this, stride, name,
                                        /*allowZeroConstant=*/hasZeroExtent))) {
        return failure();
      }
    }
  }

  // Current modules carry the offset and strides in the result encoding;
  // older modules may carry only operation attributes. Each optional field
  // independently agrees with the layout by value, not attribute spelling.
  if (layout) {
    if ((byteOffset &&
         !haveEquivalentShapeExprs(layout.getOffset(), byteOffset)) ||
        (byteStrides &&
         !llvm::equal(layout.getStrides(), byteStrides.getValue(),
                      haveEquivalentShapeExprs))) {
      return emitOpError(
          "requires view attributes to match the result `#ggml.layout`");
    }
  }

  if (getInput().getType().getElementType() !=
      getResult().getType().getElementType()) {
    return emitOpError("requires input and result tensor element types to "
                       "match");
  }
  return success();
}
