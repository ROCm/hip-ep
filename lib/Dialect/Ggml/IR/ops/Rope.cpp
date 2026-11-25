/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Rope.cpp - ggml.rope -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// input[head_dim, heads, tokens, b3], positions[tokens]: a position per
/// token (multi-axis rotations carry several, and are captured unequal).
void RopeOp::constrainOperands(ShapeConstraints &constraints) {
  equateCaptured(getPositions(), /*slot=*/1, 0, getInput(), /*otherSlot=*/0, 2,
                 constraints);
}

GGML_SAME_SHAPE_AS(RopeOp, 0)

LogicalResult RopeOp::verify() {
  const int64_t dimensions = getDimensionsAttr().getInt();
  if (dimensions <= 0 || dimensions % 2 != 0) {
    return emitOpError("requires positive, even `dimensions`");
  }

  if (getOriginalContextAttr().getInt() < 0) {
    return emitOpError("requires non-negative `original_context`");
  }
  if (getSectionsAttr() && getSectionsAttr().size() != 4) {
    return emitOpError("requires exactly four M-RoPE `sections`");
  }

  if (failed(verifyFinitePositiveFloat(*this, getFrequencyBaseAttr(),
                                       "frequency_base")) ||
      failed(verifyFinitePositiveFloat(*this, getFrequencyScaleAttr(),
                                       "frequency_scale")) ||
      failed(verifyFiniteNonNegativeFloat(*this, getExtensionFactorAttr(),
                                          "extension_factor")) ||
      failed(verifyFinitePositiveFloat(*this, getAttentionFactorAttr(),
                                       "attention_factor")) ||
      failed(verifyFiniteNonNegativeFloat(*this, getBetaFastAttr(),
                                          "beta_fast")) ||
      failed(verifyFiniteNonNegativeFloat(*this, getBetaSlowAttr(),
                                          "beta_slow"))) {
    return failure();
  }
  return success();
}
