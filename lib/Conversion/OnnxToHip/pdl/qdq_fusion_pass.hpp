/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// qdq_fusion_pass.hpp — PDL fusion pass for QDQ patterns
// Contains common functions required for fusion patterns.
// Note: Design the functions so that they can be reused by
// other patterns whenever possible.
#pragma once

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <optional>

namespace hip {
namespace pdl {

inline mlir::Value tryContextArg(mlir::Operation *op) {
  if (!op)
    return {};
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp || funcOp.getNumArguments() == 0)
    return {};
  return funcOp.getArgument(0);
}

// Element type of the quantized side of a Q/DQ op: the result for
// QuantizeLinear, operand 0 for DequantizeLinear.
inline mlir::IntegerType getQuantizedElementType(mlir::Operation *op) {
  llvm::SmallVector<mlir::Type, 2> candidates;
  if (op->getNumOperands() > 0)
    candidates.push_back(op->getOperand(0).getType());
  if (op->getNumResults() > 0)
    candidates.push_back(op->getResult(0).getType());
  for (mlir::Type type : candidates) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
    if (!shaped)
      continue;
    if (auto intType =
            mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType()))
      return intType;
  }
  return {};
}

// Per-tensor quantization only: a non-splat scale is per-axis and would not
// fold into a single coefficient.
inline std::optional<float> trySplatScale(mlir::Value value) {
  if (!value)
    return std::nullopt;
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  if (!denseAttr || !denseAttr.isSplat())
    return std::nullopt;
  return static_cast<float>(
      denseAttr.getSplatValue<mlir::FloatAttr>().getValueAsDouble());
}

// ONNX makes the Q/DQ zero point optional, so an absent operand contributes
// `absentValue` instead of rejecting the chain.
inline std::optional<int64_t>
trySplatZeropoint(mlir::Operation *op, uint64_t index, int64_t absentValue) {
  if (!op)
    return std::nullopt;
  if (index >= op->getNumOperands())
    return absentValue;
  mlir::Operation *defOp = op->getOperand(index).getDefiningOp();
  if (!defOp)
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  if (!denseAttr || !denseAttr.isSplat())
    return std::nullopt;
  auto quantType = getQuantizedElementType(op);
  if (!quantType)
    return std::nullopt;
  llvm::APInt raw = denseAttr.getSplatValue<llvm::APInt>();
  return quantType.isUnsigned() ? static_cast<int64_t>(raw.getZExtValue())
                                : raw.getSExtValue();
}

// Backing byte count of an onnx.Constant, in either form it can take: an
// inline dense value, or the external location/offset/size triple that every
// multi-megabyte weight arrives as (its bytes are not in the IR at all).
//
// This is the only place the stored value width is observable. ONNX INT4/UINT4
// imports as i8/ui8 at the LOGICAL element count, so the element type cannot
// show the width, and this pass runs BEFORE lowerOnnxConstants stamps
// `packed_int4` -- the marker does not exist yet and cannot be consulted.
inline std::optional<int64_t> tryConstantBackingBytes(mlir::Value value) {
  if (!value)
    return std::nullopt;
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  if (auto sizeAttr = defOp->getAttrOfType<mlir::IntegerAttr>("size"))
    return sizeAttr.getInt();
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  // Splat raw data is a single element, not the buffer, so its length is not
  // comparable to the logical element count.
  if (!denseAttr || denseAttr.isSplat())
    return std::nullopt;
  return static_cast<int64_t>(denseAttr.getRawData().size());
}

// True when an 8-bit-typed constant carries `bits`-wide values: ceil(numel/2)
// bytes for 4-bit nibble pairs (the same relation hip.constant's verifier
// accepts), numel bytes for full-width 8-bit. The byte count is the whole test
// and it is also what keeps the two widths mutually exclusive. The one overlap
// is a single element, which occupies a whole byte either way and is left to
// the 8-bit reading rather than claimed as packed.
inline bool hasQuantStorageBits(mlir::Value value, int64_t bits) {
  if (!value)
    return false;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return false;
  auto elemType =
      mlir::dyn_cast<mlir::IntegerType>(tensorType.getElementType());
  if (!elemType || elemType.getWidth() != 8)
    return false;
  int64_t numel = tensorType.getNumElements();
  int64_t expectedBytes;
  if (bits == 4) {
    if (numel < 2)
      return false;
    expectedBytes = (numel + 1) / 2;
  } else if (bits == 8) {
    expectedBytes = numel;
  } else {
    return false;
  }
  std::optional<int64_t> bytes = tryConstantBackingBytes(value);
  return bytes && *bytes == expectedBytes;
}

// An absent ONNX list attribute means its per-axis default, and every caller
// here asks for exactly that default (stride 1, dilation 1, pad 0), so absence
// passes rather than rejects.
inline bool onnxListAttrAllEqual(mlir::Operation *op, llvm::StringRef name,
                                 int64_t expected) {
  auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!arrayAttr)
    return true;
  for (mlir::Attribute entry : arrayAttr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry);
    if (!intAttr || intAttr.getValue().getSExtValue() != expected)
      return false;
  }
  return true;
}

inline int64_t onnxIntAttrWithDefault(mlir::Operation *op, llvm::StringRef name,
                                      int64_t absentValue) {
  auto intAttr = op->getAttrOfType<mlir::IntegerAttr>(name);
  return intAttr ? intAttr.getValue().getSExtValue() : absentValue;
}

inline bool onnxIntAttrEquals(mlir::Operation *op, llvm::StringRef name,
                              int64_t expected, int64_t absentValue) {
  return onnxIntAttrWithDefault(op, name, absentValue) == expected;
}

//===----------------------------------------------------------------------===//
// Match constraints -- result-free, so several patterns may share them.
//===----------------------------------------------------------------------===//

inline mlir::LogicalResult hasContextArg(mlir::PatternRewriter &,
                                         mlir::PDLResultList &,
                                         llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  return mlir::success(
      static_cast<bool>(tryContextArg(args[0].dyn_cast<mlir::Operation *>())));
}

inline mlir::LogicalResult
isSplatConstantValue(mlir::PatternRewriter &, mlir::PDLResultList &,
                     llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  return mlir::success(
      trySplatScale(args[0].dyn_cast<mlir::Value>()).has_value());
}

inline mlir::LogicalResult
hasExtractableZeropoint(mlir::PatternRewriter &, mlir::PDLResultList &,
                        llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  auto indexAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  if (!indexAttr)
    return mlir::failure();
  return mlir::success(trySplatZeropoint(args[0].dyn_cast<mlir::Operation *>(),
                                         indexAttr.getValue().getZExtValue(), 0)
                           .has_value());
}

// Restrict a Q/DQ op to an 8-bit quantized side.
inline mlir::LogicalResult
isEightBitQuantized(mlir::PatternRewriter &, mlir::PDLResultList &,
                    llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  auto quantType = getQuantizedElementType(op);
  return mlir::success(quantType && quantType.getWidth() == 8);
}

// Restrict a Q/DQ op to an 8- or 16-bit quantized side.
inline mlir::LogicalResult
isEightOrSixteenBitQuantized(mlir::PatternRewriter &, mlir::PDLResultList &,
                             llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  auto quantType = getQuantizedElementType(op);
  if (!quantType)
    return mlir::failure();
  unsigned width = quantType.getWidth();
  return mlir::success(width == 8 || width == 16);
}

// Restrict a Q/DQ op to an 8-, 16- or 32-bit quantized side.
inline mlir::LogicalResult
isEightSixteenOrThirtyTwoBitQuantized(mlir::PatternRewriter &,
                                      mlir::PDLResultList &,
                                      llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  auto quantType = getQuantizedElementType(op);
  if (!quantType)
    return mlir::failure();
  unsigned width = quantType.getWidth();
  return mlir::success(width == 8 || width == 16 || width == 32);
}

// Require every result of `op` to have a static shape.
inline mlir::LogicalResult
hasStaticShapedResults(mlir::PatternRewriter &, mlir::PDLResultList &,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op || op->getNumResults() == 0)
    return mlir::failure();
  for (mlir::Value result : op->getResults()) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(result.getType());
    if (!shaped || !shaped.hasStaticShape())
      return mlir::failure();
  }
  return mlir::success();
}

inline mlir::LogicalResult
hasAttrInt64Equal(mlir::PatternRewriter &, mlir::PDLResultList &,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 4)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  auto nameAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto expected = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  auto absentValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[3].dyn_cast<mlir::Attribute>());
  if (!op || !nameAttr || !expected || !absentValue)
    return mlir::failure();
  return mlir::success(onnxIntAttrEquals(
      op, nameAttr.getValue(), expected.getInt(), absentValue.getInt()));
}

inline mlir::LogicalResult
isPerAxisQuantizedWeight(mlir::PatternRewriter &, mlir::PDLResultList &,
                         llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 4)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  auto rankAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto axisAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  auto bitsAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[3].dyn_cast<mlir::Attribute>());
  // A zero point is required: the fused ops have no way to express its
  // absence.
  if (!op || !rankAttr || !axisAttr || !bitsAttr || op->getNumOperands() != 3)
    return mlir::failure();
  int64_t rank = rankAttr.getInt();
  int64_t quantAxis = axisAttr.getInt();
  int64_t bits = bitsAttr.getInt();

  // block_size > 0 subdivides each slice, a second and finer granularity that
  // one scale per slice cannot represent.
  if (!onnxIntAttrEquals(op, "block_size", 0, /*absentValue=*/0))
    return mlir::failure();

  mlir::Value weights = op->getOperand(0);
  if (!hasQuantStorageBits(weights, bits))
    return mlir::failure();
  auto weightType = mlir::cast<mlir::RankedTensorType>(weights.getType());
  if (weightType.getRank() != rank)
    return mlir::failure();

  int64_t axis = onnxIntAttrWithDefault(op, "axis", /*absentValue=*/1);
  if (axis < 0)
    axis += rank;
  if (axis != quantAxis)
    return mlir::failure();
  int64_t slices = weightType.getDimSize(quantAxis);

  auto scaleType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  if (!scaleType || scaleType.getRank() != 1 ||
      scaleType.getDimSize(0) != slices || !scaleType.getElementType().isF32())
    return mlir::failure();

  // ONNX guarantees zero_point.dtype == x.dtype, so the zero point has both
  // the weight's element type and its value width. That is what lets one
  // `packed_int4` flag describe both operands, as hip.dequantize_linear does.
  mlir::Value zeroPoints = op->getOperand(2);
  auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(zeroPoints.getType());
  if (!zpType || zpType.getRank() != 1 || zpType.getDimSize(0) != slices ||
      zpType.getElementType() != weightType.getElementType())
    return mlir::failure();
  return mlir::success(hasQuantStorageBits(zeroPoints, bits));
}

// The quantized side of a Q/DQ op is UINT16. Pins the qconv fusion to the one
// activation width its kernel implements, so an unsupported model keeps the
// unfused DQ -> Conv -> Q path instead of fusing into a kernel that would
// reject it at runtime.
inline mlir::LogicalResult
isUint16Quantized(mlir::PatternRewriter &, mlir::PDLResultList &,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  auto intType = getQuantizedElementType(op);
  return mlir::success(intType && intType.getWidth() == 16 &&
                       intType.isUnsigned());
}

// onnx.Conv geometry that collapses to a plain per-position dot product down
// the channel axis: a 1x1 kernel over two spatial dims, no grouping, unit
// stride and dilation, no padding. That is the only form the fused kernel
// implements, and it is what the rewrite's hard-coded geometry attributes
// assert -- so this must stay strict enough to justify them.
inline mlir::LogicalResult
isFusableQConvGeometry(mlir::PatternRewriter &, mlir::PDLResultList &,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  auto autoPad = op->getAttrOfType<mlir::StringAttr>("auto_pad");
  if (autoPad && autoPad.getValue() != "NOTSET")
    return mlir::failure();
  // kernel_shape must be present: its absence would make the spatial rank
  // depend on the operand shapes, which the hard-coded attributes cannot cover.
  auto kernelShape = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape");
  if (!kernelShape || kernelShape.size() != 2)
    return mlir::failure();
  return mlir::success(onnxListAttrAllEqual(op, "kernel_shape", 1) &&
                       onnxIntAttrEquals(op, "group", 1, /*absentValue=*/1) &&
                       onnxListAttrAllEqual(op, "strides", 1) &&
                       onnxListAttrAllEqual(op, "dilations", 1) &&
                       onnxListAttrAllEqual(op, "pads", 0));
}

// One scale and one zero point per index along `channelAxis` of `inputType`.
//
// Both are checked for shape only, never for value: a per-channel array
// routinely arrives as an external constant with no bytes in the IR, so unlike
// a per-tensor scale it cannot be folded into an attribute and must reach the
// kernel as an operand to be indexed by channel.
//
// `dequant` must already be known to carry a zero point operand.
inline bool hasPerChannelQuantParams(mlir::Operation *dequant,
                                     mlir::RankedTensorType inputType,
                                     int64_t channelAxis) {
  int64_t channels = inputType.getDimSize(channelAxis);
  auto scaleType =
      mlir::dyn_cast<mlir::RankedTensorType>(dequant->getOperand(1).getType());
  if (!scaleType || scaleType.getRank() != 1 ||
      scaleType.getDimSize(0) != channels)
    return false;
  // ONNX guarantees zero_point.dtype == x.dtype, which is also what lets a
  // single width describe both buffers downstream.
  auto zpType =
      mlir::dyn_cast<mlir::RankedTensorType>(dequant->getOperand(2).getType());
  return zpType && zpType.getRank() == 1 && zpType.getDimSize(0) == channels &&
         zpType.getElementType() == inputType.getElementType();
}

// The weight-side DequantizeLinear carries packed 4-bit weights quantized per
// output channel, which is the whole point of the fusion: those bytes stay
// packed until they are inside the kernel.
inline mlir::LogicalResult
isPackedInt4PerChannelWeight(mlir::PatternRewriter &, mlir::PDLResultList &,
                             llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  // A zero point is required: the fused op has no way to express its absence.
  if (!op || op->getNumOperands() != 3)
    return mlir::failure();
  // ONNX defaults `axis` to 1, so an absent attribute is per-axis on the wrong
  // dimension and must be rejected rather than defaulted.
  if (!onnxIntAttrEquals(op, "axis", 0, /*absentValue=*/1))
    return mlir::failure();
  // block_size > 0 is a second, finer granularity within each channel that the
  // per-channel scale lookup does not model.
  if (!onnxIntAttrEquals(op, "block_size", 0, /*absentValue=*/0))
    return mlir::failure();

  mlir::Value weights = op->getOperand(0);
  if (!hasQuantStorageBits(weights, /*bits=*/4))
    return mlir::failure();
  auto weightType = mlir::cast<mlir::RankedTensorType>(weights.getType());
  // A Conv filter is [Cout, Cin/group, k...], so axis 0 is the output channel.
  if (!hasPerChannelQuantParams(op, weightType, /*channelAxis=*/0))
    return mlir::failure();
  // A 4-bit weight implies a 4-bit zero point. Requiring it to be packed too is
  // what lets one `packed_int4` flag describe both operands.
  return mlir::success(hasQuantStorageBits(op->getOperand(2), /*bits=*/4));
}

inline mlir::LogicalResult
isPerChannelWeight(mlir::PatternRewriter &, mlir::PDLResultList &,
                   llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  mlir::Operation *dequant = args[0].dyn_cast<mlir::Operation *>();
  mlir::Operation *consumer = args[1].dyn_cast<mlir::Operation *>();
  // A zero point is required: the fused per-channel form has no way to spell
  // its absence.
  if (!dequant || !consumer || dequant->getNumOperands() != 3)
    return mlir::failure();
  // block_size > 0 is a second, finer granularity within each channel that a
  // per-channel lookup does not model.
  if (!onnxIntAttrEquals(dequant, "block_size", 0, /*absentValue=*/0))
    return mlir::failure();

  mlir::Value weights = dequant->getOperand(0);
  auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weights.getType());
  if (!weightType || weightType.getRank() < 2 || !weightType.hasStaticShape())
    return mlir::failure();
  int64_t rank = weightType.getRank();
  bool transB = onnxIntAttrEquals(consumer, "transB", 1, /*absentValue=*/0);
  int64_t channelAxis = transB ? rank - 2 : rank - 1;
  if (weightType.getDimSize(channelAxis) < 2)
    return mlir::failure();

  // ONNX defaults `axis` to 1 and allows it negative, so normalize before
  // comparing against the axis the layout puts N on.
  auto axisAttr = dequant->getAttrOfType<mlir::IntegerAttr>("axis");
  int64_t axis = axisAttr ? axisAttr.getValue().getSExtValue() : 1;
  if (axis < 0)
    axis += rank;
  if (axis != channelAxis)
    return mlir::failure();

  if (!hasPerChannelQuantParams(dequant, weightType, channelAxis))
    return mlir::failure();
  // A single width attribute describes both buffers, so their packing has to
  // agree: reading a full-width zero point as nibbles (or the reverse) is
  // silent corruption.
  return mlir::success(hasQuantStorageBits(weights, /*bits=*/4) ==
                       hasQuantStorageBits(dequant->getOperand(2), /*bits=*/4));
}

// onnx.LpNormalization that matches the fused RMS path: p=2 over a static
// trailing axis. Other cases keep the unfused DQ + float LpNorm + Q chain.
inline mlir::LogicalResult
isFusableQLpNormalization(mlir::PatternRewriter &, mlir::PDLResultList &,
                          llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op || op->getNumOperands() != 1)
    return mlir::failure();
  if (!onnxIntAttrEquals(op, "p", 2, /*absentValue=*/2))
    return mlir::failure();

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!inputType || inputType.getRank() == 0)
    return mlir::failure();
  int64_t rank = inputType.getRank();
  int64_t last = inputType.getDimSize(rank - 1);
  if (last == mlir::ShapedType::kDynamic || last <= 0)
    return mlir::failure();

  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  int64_t axis = axisAttr ? axisAttr.getValue().getSExtValue() : -1;
  int64_t normAxis = axis < 0 ? axis + rank : axis;
  return mlir::success(normAxis == rank - 1);
}

// onnx.Sigmoid in a UINT16 QDQ sandwich. Per-tensor scale/zp folding is
// enforced by the splat/zp constraints on the surrounding Q/DQ ops.
inline mlir::LogicalResult
isFusableQSigmoid(mlir::PatternRewriter &, mlir::PDLResultList &,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Operation *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return mlir::failure();
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape() ||
      inputType.getShape() != outputType.getShape())
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Rewrite functions -- reached only after the constraints above accepted.
//===----------------------------------------------------------------------===//

inline mlir::LogicalResult getContextArg(mlir::PatternRewriter &,
                                         mlir::PDLResultList &results,
                                         llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Value ctx = tryContextArg(args[0].dyn_cast<mlir::Operation *>());
  if (!ctx)
    return mlir::failure();
  results.push_back(ctx);
  return mlir::success();
}

inline mlir::LogicalResult
extractScaleValue(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  std::optional<float> scale = trySplatScale(args[0].dyn_cast<mlir::Value>());
  if (!scale)
    return mlir::failure();
  results.push_back(rewriter.getF32FloatAttr(*scale));
  return mlir::success();
}

inline mlir::LogicalResult
extractZeropointValue(mlir::PatternRewriter &rewriter,
                      mlir::PDLResultList &results,
                      llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 3)
    return mlir::failure();
  auto indexAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto defaultValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  if (!indexAttr || !defaultValue)
    return mlir::failure();
  std::optional<int64_t> zeropoint = trySplatZeropoint(
      args[0].dyn_cast<mlir::Operation *>(),
      indexAttr.getValue().getZExtValue(), defaultValue.getInt());
  if (!zeropoint)
    return mlir::failure();
  results.push_back(rewriter.getI64IntegerAttr(*zeropoint));
  return mlir::success();
}

inline mlir::LogicalResult
extractQuantBits(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                 llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  mlir::Value value = args[0].dyn_cast<mlir::Value>();
  if (!value)
    return mlir::failure();
  if (hasQuantStorageBits(value, /*bits=*/4)) {
    results.push_back(rewriter.getI64IntegerAttr(4));
    return mlir::success();
  }
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  auto intType =
      shaped ? mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType())
             : mlir::IntegerType();
  if (!intType)
    return mlir::failure();
  results.push_back(rewriter.getI64IntegerAttr(intType.getWidth()));
  return mlir::success();
}

// args[0] = op, args[1] = attribute name, args[2] = value to use when absent.
// Return failure if the attribute is not a 64-bit integer attribute.
// The result is always signless, because it feeds hip op attributes declared
// as I64Attr while ONNX stores its integer attributes as si64.
inline mlir::LogicalResult
extractAttrInt64(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                 llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 3)
    return mlir::failure();

  auto *op = args[0].dyn_cast<mlir::Operation *>();
  auto nameAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto defaultValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());

  if (!op || !nameAttr || !defaultValue)
    return mlir::failure();

  auto attr = op->getAttrOfType<mlir::IntegerAttr>(nameAttr.getValue());

  if (attr && !attr.getType().isInteger(64))
    return mlir::failure();

  results.push_back(rewriter.getI64IntegerAttr(
      (attr ? attr : defaultValue).getValue().getSExtValue()));
  return mlir::success();
}

// args[0] = op, args[1] = attribute name, args[2] = value to use when absent.
// The result is always f32-typed, because it feeds hip op attributes declared
// as F32Attr while ONNX stores the attribute at the importer's float width.
inline mlir::LogicalResult extractAttrF32(mlir::PatternRewriter &rewriter,
                                          mlir::PDLResultList &results,
                                          llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 3)
    return mlir::failure();

  auto *op = args[0].dyn_cast<mlir::Operation *>();
  auto nameAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto defaultValue = mlir::dyn_cast_or_null<mlir::FloatAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  if (!op || !nameAttr || !defaultValue)
    return mlir::failure();

  auto attr = op->getAttrOfType<mlir::FloatAttr>(nameAttr.getValue());
  results.push_back(rewriter.getF32FloatAttr(
      attr ? attr.getValueAsDouble() : defaultValue.getValueAsDouble()));
  return mlir::success();
}

// Apply PDL patterns
inline bool run(mlir::ModuleOp mlirModule, llvm::MemoryBufferRef pdlBuffer) {
  if (pdlBuffer.getBufferSize() == 0)
    return true;

  mlir::MLIRContext *ctx = mlirModule.getContext();

  mlir::ParserConfig parseConfig(ctx);
  // parseSourceString, not parseSourceFile: the latter's StringRef overload
  // takes a path, so it would try to open the pattern text itself as a file.
  // Both dispatch on the magic bytes, so textual IR and bytecode work either
  // way.
  mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
      mlir::parseSourceString<mlir::ModuleOp>(
          pdlBuffer.getBuffer(), parseConfig, pdlBuffer.getBufferIdentifier());
  if (!pdlModule)
    return false;

  // FrozenRewritePatternSet skips the PDL-to-PDLInterp lowering for a module
  // holding no pdl.pattern, then still asks the bytecode generator for the
  // @matcher function that lowering would have produced. Bail out first so a
  // pattern set that is empty (every pattern disabled) stays a no-op.
  if (pdlModule->getOps<mlir::pdl::PatternOp>().empty())
    return true;

  mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));

  // Register native helpers with low-level signatures
  pdlPatterns.registerConstraintFunction("HasContextArg", hasContextArg);
  pdlPatterns.registerConstraintFunction("IsSplatConstantValue",
                                         isSplatConstantValue);
  pdlPatterns.registerConstraintFunction("HasExtractableZeropoint",
                                         hasExtractableZeropoint);
  pdlPatterns.registerConstraintFunction("IsEightBitQuantized",
                                         isEightBitQuantized);
  pdlPatterns.registerConstraintFunction("IsEightOrSixteenBitQuantized",
                                         isEightOrSixteenBitQuantized);
  pdlPatterns.registerConstraintFunction(
      "IsEightSixteenOrThirtyTwoBitQuantized",
      isEightSixteenOrThirtyTwoBitQuantized);
  pdlPatterns.registerConstraintFunction("HasStaticShapedResults",
                                         hasStaticShapedResults);
  pdlPatterns.registerConstraintFunction("HasAttrInt64Equal",
                                         hasAttrInt64Equal);
  pdlPatterns.registerConstraintFunction("IsPerAxisQuantizedWeight",
                                         isPerAxisQuantizedWeight);
  pdlPatterns.registerConstraintFunction("IsUint16Quantized",
                                         isUint16Quantized);
  pdlPatterns.registerConstraintFunction("IsFusableQConvGeometry",
                                         isFusableQConvGeometry);
  pdlPatterns.registerConstraintFunction("IsPackedInt4PerChannelWeight",
                                         isPackedInt4PerChannelWeight);
  pdlPatterns.registerConstraintFunction("IsPerChannelWeight",
                                         isPerChannelWeight);
  pdlPatterns.registerConstraintFunction("IsFusableQLpNormalization",
                                         isFusableQLpNormalization);
  pdlPatterns.registerConstraintFunction("IsFusableQSigmoid",
                                         isFusableQSigmoid);
  pdlPatterns.registerRewriteFunction("GetContextArg", getContextArg);
  pdlPatterns.registerRewriteFunction("ExtractScaleValue", extractScaleValue);
  pdlPatterns.registerRewriteFunction("ExtractZeropointValue",
                                      extractZeropointValue);
  pdlPatterns.registerRewriteFunction("ExtractQuantBits", extractQuantBits);
  pdlPatterns.registerRewriteFunction("ExtractAttrInt64", extractAttrInt64);
  pdlPatterns.registerRewriteFunction("ExtractAttrF32", extractAttrF32);

  mlir::RewritePatternSet patterns(ctx);
  patterns.add(std::move(pdlPatterns));

  mlir::FrozenRewritePatternSet frozen(std::move(patterns));

  // Walk all FuncOps and apply patterns
  bool ok = true;
  mlirModule.walk([&](mlir::func::FuncOp funcOp) {
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, frozen)))
      ok = false;
  });
  return ok;
}

} // namespace pdl
} // namespace hip
