/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- FlashAttentionExt.cpp - ggml.flash_attention_ext -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// The keys as wide as the queries and as many as the values, the mask as
/// wide as the keys are many and a row per query.
void FlashAttentionExtOp::constrainOperands(ShapeConstraints &constraints) {
  enum Slot : int64_t { kQuery, kKey, kValue, kMask };
  equateCaptured(getKey(), kKey, 0, getQuery(), kQuery, 0, constraints);
  equateCaptured(getValue(), kValue, 1, getKey(), kKey, 1, constraints);
  equateCaptured(getMask(), kMask, 0, getKey(), kKey, 1, constraints);
  equateCaptured(getMask(), kMask, 1, getQuery(), kQuery, 1, constraints);
}

/// query[head_dim, tokens, heads, batch], value[value_dim, ...]
///   -> [value_dim, heads, tokens, batch]
///
/// The width is the value's, as GGML has it: attention over compressed keys
/// (MLA) reads keys wider than the values it returns.
LogicalResult
FlashAttentionExtOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0) || !derivation.hasOperand(2))
    return emitOpError() << "expects query and value operands";
  const Shape &query = derivation.operandShape(0);
  const Shape &value = derivation.operandShape(2);
  derivation.setResultShape({value[0], query[2], query[1], query[3]});
  return success();
}

LogicalResult FlashAttentionExtOp::verify() {
  if (failed(verifyFinitePositiveFloat(*this, getScaleAttr(), "scale")) ||
      failed(
          verifyFiniteNonNegativeFloat(*this, getMaxBiasAttr(), "max_bias")) ||
      failed(verifyFiniteNonNegativeFloat(*this, getLogitSoftcapAttr(),
                                          "logit_softcap"))) {
    return failure();
  }

  if (getMaxBias().convertToDouble() > 0.0 && !getMask()) {
    return emitOpError("positive `max_bias` requires a mask operand");
  }
  return success();
}
