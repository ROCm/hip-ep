/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_QLINEARPLAN_H
#define HIP_DIALECT_TRANSFORMS_QLINEARPLAN_H

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>

#include <optional>

namespace mlir::hip {

// The constants that turn a hip.qlinear_conv into hip.quant_conv and an
// epilogue. With z' = x_zp - offset (offset is getSignedRebiasOffset of the
// input type: 128 for ui8, 0 for i8):
//
//   conv(x - x_zp, w) = conv(x - offset, w) - z' * sum(w over the taps that
//                       read the input, not the padding)
//
// so the epilogue adds `bias` = B - z' * sum(w), then requantizes with
// `multiplier` = x_scale * w_scale / y_scale in f32, the order ORT CPU
// uses. Without padding every output element reads every tap, and `bias`
// holds one value per channel; otherwise it has one value per output element.
// A hip.qlinear_matmul uses the same plan with sums over the columns of B.
struct QLinearPlan {
  // i32, broadcastable to the result: [1, K, 1, 1] or [1, K, OH, OW] for a
  // conv, [1, N] for a matmul. Null when every entry is 0.
  DenseElementsAttr bias;
  // f32, rank 0 or broadcastable to the result along the channel or column.
  DenseElementsAttr multiplier;
  // Rank 0, the output element type.
  DenseElementsAttr outputZeroPoint;
};

// A conv of dequantized operands as an integer conv: hip.quant_conv on the
// quantized input and weights, the zero-point correction, a multiply by
// x_scale * w_scale, and the float bias. The result stays f32, so whatever
// reads the conv (an activation, a quantize) becomes the epilogue.
struct DequantizedConvPlan {
  Value input;
  Value weights;
  // i32, as QLinearPlan::bias without a conv bias. Null when every entry is 0.
  DenseElementsAttr correction;
  // f32, rank 0 or [1, K, 1, 1].
  DenseElementsAttr multiplier;
  // f32 [1, K, 1, 1]. Null without a conv bias.
  DenseElementsAttr bias;
};

// A QLinear op with ui8 weights or a non-zero weight zero point. rocMLIR takes
// i8 weights and has no zero-point operand, so the weights become w - offset
// (128 for ui8, 0 for i8) and the zero point zw' = zw - offset; both fit i8,
// and w - zw does not change. With zw' != 0 the sum also needs zw' times the
// window sums of the input, which a second integer conv or matmul over
// `zeroPointFilter` computes.
struct AsymmetricQLinearPlan {
  // i8, the weights' shape.
  DenseElementsAttr weights;
  // i8, the weights' shape, zw' along the output channel or column. Null
  // when every zw' is 0.
  DenseElementsAttr zeroPointFilter;
  // The bias also holds the constant terms of the weight zero point.
  QLinearPlan requantize;
};

/// The plan of a hip.qlinear_conv that rocMLIR can run as hip.quant_conv and a
/// requantize epilogue, or nullopt when the conv keeps its HIP kernel.
/// A channel whose requantize multiplier is below 2^-33 always writes the
/// output zero point; its multiplier and bias are 0. The plans below and this
/// one are nullopt when every channel is dead (see getDeadQLinearOutput).
std::optional<QLinearPlan> planQLinearConv(QLinearConvOp op);

/// The plan of a hip.qlinear_conv with ui8 weights or a non-zero weight zero
/// point, with static shapes and inline constants; nullopt otherwise, and
/// when an i32 sum could overflow.
std::optional<AsymmetricQLinearPlan>
planAsymmetricQLinearConv(QLinearConvOp op);

/// As planAsymmetricQLinearConv, for a hip.qlinear_matmul with ui8 B or
/// non-zero B zero points.
std::optional<AsymmetricQLinearPlan>
planAsymmetricQLinearMatMul(QLinearMatMulOp op);

/// The plan of a hip.qlinear_matmul as hip.quant_matmul and a requantize
/// epilogue, or nullopt when the matmul keeps its HIP kernel.
std::optional<QLinearPlan> planQLinearMatMul(QLinearMatMulOp op);

/// The plan of a hip.conv of dequantized operands as an integer conv, or
/// nullopt when the conv stays a float conv.
std::optional<DequantizedConvPlan> planDequantizedConv(ConvOp op);

/// True when one of the plans above accepts `op`.
bool isPlannedQLinear(Operation *op);

/// The output of a hip.qlinear_conv or hip.qlinear_matmul whose requantize
/// multiplier is below 2^-33 on every channel or column: every element is the
/// output zero point, for any input and bias. A splat in the static result
/// type, or null for any other op.
DenseElementsAttr getDeadQLinearOutput(Operation *op);

Value emitTensorEmpty(RewriterBase &rewriter, Location loc,
                      RankedTensorType type);

Value emitConstant(RewriterBase &rewriter, Location loc,
                   DenseElementsAttr value);

/// (acc + bias) in i32, then cast to f32 and multiplied by `multiplier`.
Value emitScaledAccumulator(RewriterBase &rewriter, Location loc, Value ctx,
                            Value acc, DenseElementsAttr bias,
                            DenseElementsAttr multiplier);

/// The requantize epilogue of `plan` on the i32 accumulator `acc`, written to
/// `output`.
Value emitRequantizeEpilogue(RewriterBase &rewriter, Location loc, Value ctx,
                             Value acc, const QLinearPlan &plan, Value output);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_QLINEARPLAN_H
