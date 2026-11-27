/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Reshape.cpp - ggml.reshape -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

LogicalResult ReshapeOp::deriveResultShape(ShapeDerivation &derivation) {
  return redrawBoundaries(*this, derivation);
}

LogicalResult ReshapeOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getRank() != 4 || resultType.getRank() != 4) {
    return emitOpError("requires rank-4 tensors in the raw GGML layout");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires input and result element types to match");
  }
  // GGML's reshape reads a contiguous tensor's elements through different
  // extents. The count cannot change, and the layout says the count even when
  // the shape only says `?`: `1024 * token_count` and `128 * 8 * token_count`
  // are the same number of elements, which two anonymous `?` cannot show.
  LayoutAttr input = getGgmlLayout(getInput());
  LayoutAttr result = getGgmlLayout(getResult());
  if (input && result) {
    std::string why;
    FailureOr<Attribute> before = countElements(input, why, *this);
    if (failed(before))
      return emitOpError("has an element count that ") << why;
    FailureOr<Attribute> after = countElements(result, why, *this);
    if (failed(after))
      return emitOpError("has an element count that ") << why;
    if (*before && *after && *before != *after) {
      return emitOpError("requires equal input and result element counts, but "
                         "holds ")
             << *before << " and " << *after;
    }
  } else if (inputType.hasStaticShape() && resultType.hasStaticShape()) {
    // Older modules can omit layout encodings. Their static tensor shapes
    // still determine the count; do not defer this invariant to HIP lowering.
    CardinalitySummary before = summarizeCardinality(inputType);
    CardinalitySummary after = summarizeCardinality(resultType);
    if (before.overflow || after.overflow)
      return emitOpError("has a static element count that overflows int64");
    if (before.staticProduct != after.staticProduct)
      return emitOpError("requires equal input and result element counts, but "
                         "holds ")
             << before.staticProduct << " and " << after.staticProduct;
  }
  if (deriveContiguity(getResult()) == Contiguity::Strided) {
    return emitOpError("requires a dense result: GGML lays a reshape out "
                       "contiguously, so strides it did not choose are not "
                       "a reshape's");
  }
  return success();
}
