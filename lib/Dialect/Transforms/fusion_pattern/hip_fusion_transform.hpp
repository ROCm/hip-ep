/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip_fusion_transform.hpp -- native support for the HIP-to-HIP rewrite
// patterns in this directory.
//
// `run()` below is the generic half: it parses the embedded PDL module, binds
// the native helpers, and applies the patterns. It is indifferent to what the
// patterns do, so a one-to-many, many-to-one or many-to-many `hip.*` rewrite
// all go through it unchanged.
//
// The helpers come in two groups:
//
//   * op-agnostic helpers that any pattern can reuse;
//   * readers for a specific op family, currently the Q/DQ quantization
//     parameters.
//
// Adding a pattern that needs a new native helper means adding it below,
// registering it in `registerNativeHelpers`, and declaring it in
// HipFusionTransformPatterns.pdll. Prefer extending the op-agnostic group.
#pragma once

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <optional>

namespace hip {
namespace fusion_transform {

//===----------------------------------------------------------------------===//
// IR readers
//===----------------------------------------------------------------------===//

/// value is const op and return value as DenseElementsAttr, if not return {}
inline mlir::DenseElementsAttr tryHipConstantPayload(mlir::Value value) {
  if (!value)
    return {};
  auto constOp =
      mlir::dyn_cast_or_null<mlir::hip::ConstantOp>(value.getDefiningOp());
  if (!constOp)
    return {};
  return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      constOp->getAttr("value"));
}

/// value is a splat float constant and return it as float, if not return
/// nullopt
inline std::optional<float> tryHipSplatScale(mlir::Value value) {
  mlir::DenseElementsAttr payload = tryHipConstantPayload(value);
  // A non-splat scale is per-axis and has no single scalar equivalent.
  if (!payload || !payload.isSplat())
    return std::nullopt;
  // getSplatValue<FloatAttr> asserts on a non-float payload, so the element
  // type is what guards the read rather than a dyn_cast on its result.
  if (!mlir::isa<mlir::FloatType>(payload.getElementType()))
    return std::nullopt;
  return static_cast<float>(
      payload.getSplatValue<mlir::FloatAttr>().getValueAsDouble());
}

inline mlir::Value getQdqZeroPoint(mlir::Operation *op) {
  // Read through the ODS accessor, not an operand index: the zero point is
  // optional, and an absent one shifts the DPS init operand into its slot.
  if (auto quantOp = mlir::dyn_cast_or_null<mlir::hip::QuantizeLinearOp>(op))
    return quantOp.getZeroPoint();
  if (auto dequantOp =
          mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op))
    return dequantOp.getZeroPoint();
  return {};
}

inline mlir::IntegerType getQdqQuantizedElementType(mlir::Operation *op) {
  mlir::Type type;
  if (auto quantOp = mlir::dyn_cast_or_null<mlir::hip::QuantizeLinearOp>(op))
    type = quantOp.getOutput().getType();
  else if (auto dequantOp =
               mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op))
    type = dequantOp.getInput().getType();
  auto shaped = mlir::dyn_cast_or_null<mlir::ShapedType>(type);
  if (!shaped)
    return {};
  return mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType());
}

/// op is a Q/DQ whose zero point is a splat constant and return it as int64,
/// an absent operand returns absentValue, anything else nullopt
inline std::optional<int64_t> tryHipQdqZeropoint(mlir::Operation *op,
                                                 int64_t absentValue) {
  if (!op)
    return std::nullopt;
  mlir::Value zeroPoint = getQdqZeroPoint(op);
  // ONNX makes the operand optional and defines its absence as zero.
  if (!zeroPoint)
    return absentValue;
  mlir::DenseElementsAttr payload = tryHipConstantPayload(zeroPoint);
  if (!payload || !payload.isSplat())
    return std::nullopt;
  if (!mlir::isa<mlir::IntegerType>(payload.getElementType()))
    return std::nullopt;
  // The payload's own element type carries no signedness for the signless
  // integers the importer produces, so the stored type of the quantized side
  // is what decides how the raw bits are read.
  auto intType = getQdqQuantizedElementType(op);
  if (!intType)
    return std::nullopt;
  llvm::APInt raw = payload.getSplatValue<llvm::APInt>();
  return intType.isUnsigned() ? static_cast<int64_t>(raw.getZExtValue())
                              : raw.getSExtValue();
}

/// op's `name` attribute as a signed int64, absentValue when it is missing,
/// nullopt when it is present but not a 64-bit integer
inline std::optional<int64_t>
tryHipIntAttr(mlir::Operation *op, llvm::StringRef name, int64_t defaultValue) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return defaultValue;
  // Every hip op attribute read this way is declared I64Attr, so a different
  // width means the name does not refer to the attribute the caller meant.
  if (!attr.getType().isInteger(64))
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

/// op's `name` array attribute holds exactly `size` integers, all equal to
/// `expected`
inline bool hipListAttrAllEqual(mlir::Operation *op, llvm::StringRef name,
                                int64_t size, int64_t expected) {
  auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>(name);
  // Unlike their ONNX counterparts these attributes are declared required on
  // the hip op, so an absent one is a malformed op rather than a per-axis
  // default to be filled in. The length is checked because a replacement that
  // hard-codes the geometry is only justified for the rank it was written for.
  if (!arrayAttr || static_cast<int64_t>(arrayAttr.size()) != size)
    return false;
  return llvm::all_of(arrayAttr, [&](mlir::Attribute entry) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry);
    return intAttr && intAttr.getValue().getSExtValue() == expected;
  });
}

//===----------------------------------------------------------------------===//
// Match constraints
//===----------------------------------------------------------------------===//
// Result-free by construction, so several patterns rooted at the same op may
// share them. A constraint that returned a value would make the PDL module
// fail to lower ("operand does not dominate this use"); every extraction
// therefore happens in the rewrite section below.

/// resultType and shapeSource are ranked tensors of the same rank
inline bool isBuildableInit(mlir::Type resultType, mlir::Value shapeSource) {
  auto initType = mlir::dyn_cast_or_null<mlir::RankedTensorType>(resultType);
  if (!initType || !shapeSource)
    return false;
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(shapeSource.getType());
  return sourceType && sourceType.getRank() == initType.getRank();
}

/// resultType and shapeSource are accepted by BuildInit
inline mlir::LogicalResult
canBuildInit(mlir::PatternRewriter &, mlir::PDLResultList &,
             llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  // BuildInit itself has no failure path: a native rewrite that returns
  // failure without pushing its declared result trips an assert in the PDL
  // bytecode, so every precondition it needs is checked here instead.
  return mlir::success(isBuildableInit(args[0].dyn_cast<mlir::Type>(),
                                       args[1].dyn_cast<mlir::Value>()));
}

/// op has exactly one result with exactly one use
inline mlir::LogicalResult
hasSingleUseResult(mlir::PatternRewriter &, mlir::PDLResultList &,
                   llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op || op->getNumResults() != 1)
    return mlir::failure();
  // A second consumer keeps the matched ops alive beside their replacement,
  // so the graph would compute the same thing twice.
  return mlir::success(op->getResult(0).hasOneUse());
}

inline mlir::LogicalResult
isHipSplatScale(mlir::PatternRewriter &, mlir::PDLResultList &,
                llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  return mlir::success(
      tryHipSplatScale(args[0].dyn_cast<mlir::Value>()).has_value());
}

inline mlir::LogicalResult
hasExtractableQdqZeropoint(mlir::PatternRewriter &, mlir::PDLResultList &,
                           llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  return mlir::success(
      tryHipQdqZeropoint(op, /*absentValue=*/0).has_value());
}

/// op is a Q/DQ whose quantized side has one of the element widths listed in
/// the `widths` array attribute
inline mlir::LogicalResult
isHipQdqQuantizedWidth(mlir::PatternRewriter &, mlir::PDLResultList &,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  auto widths = mlir::dyn_cast_or_null<mlir::ArrayAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto intType =
      getQdqQuantizedElementType(args[0].dyn_cast<mlir::Operation *>());
  if (!widths || !intType)
    return mlir::failure();
  // A width the kernel does not implement must leave the unfused chain in
  // place rather than fuse into a kernel that would reject it at runtime.
  return mlir::success(llvm::any_of(widths, [&](mlir::Attribute width) {
    auto widthAttr = mlir::dyn_cast<mlir::IntegerAttr>(width);
    return widthAttr &&
           widthAttr.getInt() == static_cast<int64_t>(intType.getWidth());
  }));
}

/// op is a Q/DQ whose quantized side has an unsigned element type
inline mlir::LogicalResult
isHipQdqUnsignedQuantized(mlir::PatternRewriter &, mlir::PDLResultList &,
                          llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto intType =
      getQdqQuantizedElementType(args[0].dyn_cast<mlir::Operation *>());
  // Signedness selects how a stored code is widened, so it is not a detail a
  // kernel written for one of them can absorb. Kept separate from the width
  // check because the two are independent properties of the same operand.
  return mlir::success(intType && intType.isUnsigned());
}

/// conv is a 1x1 window over two spatial dims with unit stride and dilation,
/// no padding and no grouping
inline mlir::LogicalResult
isHipFusableQConvGeometry(mlir::PatternRewriter &, mlir::PDLResultList &,
                          llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();
  // This is the only geometry that collapses to a per-position dot product
  // down the channel axis, and the replacement restates it as hard-coded
  // attributes, so the check has to be strict enough to justify every one of
  // them. hip.conv resolves ONNX's per-axis defaults during conversion and
  // carries no auto_pad, so the explicit values are the whole story.
  std::optional<int64_t> group = tryHipIntAttr(op, "group", 1);
  return mlir::success(
      group && *group == 1 &&
      hipListAttrAllEqual(op, "kernel_shape", /*size=*/2, /*expected=*/1) &&
      hipListAttrAllEqual(op, "strides", /*size=*/2, /*expected=*/1) &&
      hipListAttrAllEqual(op, "dilations", /*size=*/2, /*expected=*/1) &&
      hipListAttrAllEqual(op, "pads", /*size=*/4, /*expected=*/0));
}

/// dq applies one f32 scale and one zero point per slice along `expectedAxis`
/// of a statically shaped rank-`rank` 8-bit storage weight
inline bool isPerSliceQuantizedWeight(mlir::hip::DequantizeLinearOp dq,
                                      int64_t rank, int64_t expectedAxis) {
  // The per-slice form has no way to express an absent zero point.
  mlir::Value zeroPoints = dq.getZeroPoint();
  if (!zeroPoints)
    return false;

  // block_size > 0 subdivides each slice, a second and finer granularity that
  // one scale per slice cannot represent.
  if (dq.getBlockSize() != 0)
    return false;

  auto weightType =
      mlir::dyn_cast<mlir::RankedTensorType>(dq.getInput().getType());
  if (!weightType || !weightType.hasStaticShape() ||
      weightType.getRank() != rank || !weightType.getElementType().isInteger(8))
    return false;

  // ONNX allows a negative axis, so normalize before comparing it against the
  // axis the consumer's layout puts the slices on.
  int64_t axis = dq.getAxis();
  if (axis < 0)
    axis += weightType.getRank();
  if (axis != expectedAxis)
    return false;
  int64_t slices = weightType.getDimSize(axis);

  auto scaleType =
      mlir::dyn_cast<mlir::RankedTensorType>(dq.getScale().getType());
  if (!scaleType || scaleType.getRank() != 1 ||
      scaleType.getDimSize(0) != slices || !scaleType.getElementType().isF32())
    return false;

  // ONNX guarantees zero_point.dtype == x.dtype, so the zero point carries the
  // weight's element type and its value width. That is what lets one packed
  // 4-bit marker describe both operands.
  auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(zeroPoints.getType());
  return zpType && zpType.getRank() == 1 && zpType.getDimSize(0) == slices &&
         zpType.getElementType() == weightType.getElementType();
}

/// dq is a hip.dequantize_linear quantizing a rank-`rank` weight per slice
/// along `axis`, with packed_int4 set exactly when `packedInt4` is true
inline mlir::LogicalResult
isHipPerAxisQuantizedWeight(mlir::PatternRewriter &, mlir::PDLResultList &,
                            llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 4)
    return mlir::failure();
  auto dq = mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(
      args[0].dyn_cast<mlir::Operation *>());
  auto rankAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto axisAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  auto packedAttr = mlir::dyn_cast_or_null<mlir::BoolAttr>(
      args[3].dyn_cast<mlir::Attribute>());
  if (!dq || !rankAttr || !axisAttr || !packedAttr)
    return mlir::failure();

  // A PDLL `op<>` literal cannot make an attribute conditional, so a fused op
  // that spells the value width as a unit marker needs one pattern per width,
  // and each must reject the other's weight. The element type cannot tell them
  // apart -- a packed 4-bit operand keeps 8-bit storage and its logical
  // element count -- so the marker convert-onnx-to-hip carried over from
  // constant lowering is what decides.
  if (dq.getPackedInt4() != packedAttr.getValue())
    return mlir::failure();
  return mlir::success(
      isPerSliceQuantizedWeight(dq, rankAttr.getInt(), axisAttr.getInt()));
}

/// dq is a hip.dequantize_linear quantizing its weight per output feature of
/// consumer, whose transB decides which of the weight's two axes that feature
/// runs along
inline mlir::LogicalResult
isHipPerChannelQuantizedWeight(mlir::PatternRewriter &, mlir::PDLResultList &,
                               llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  auto dq = mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(
      args[0].dyn_cast<mlir::Operation *>());
  auto *consumer = args[1].dyn_cast<mlir::Operation *>();
  if (!dq || !consumer)
    return mlir::failure();

  // Transposing the weight swaps its two extents and so moves the output
  // feature with them. Any nonzero transB counts as a transpose, which is how
  // the fused op reads the same flag it is handed verbatim.
  std::optional<int64_t> transB = tryHipIntAttr(consumer, "transB", 0);
  if (!transB)
    return mlir::failure();
  int64_t channelAxis = *transB != 0 ? 0 : 1;
  if (!isPerSliceQuantizedWeight(dq, /*rank=*/2, channelAxis))
    return mlir::failure();

  // A lone output feature is per-tensor quantization written as a length-1
  // array. Leaving it unmatched keeps it on the per-tensor path, where the
  // coefficient folds into the instruction stream rather than costing a load
  // per output.
  auto weightType = mlir::cast<mlir::RankedTensorType>(dq.getInput().getType());
  return mlir::success(weightType.getDimSize(channelAxis) >= 2);
}

/// op's `name` attribute equals expected, an absent one being absentValue
inline mlir::LogicalResult
hasHipIntAttrEqual(mlir::PatternRewriter &, mlir::PDLResultList &,
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
  std::optional<int64_t> value =
      tryHipIntAttr(op, nameAttr.getValue(), absentValue.getInt());
  return mlir::success(value && *value == expected.getInt());
}

//===----------------------------------------------------------------------===//
// Rewrite helpers
//===----------------------------------------------------------------------===//

/// a tensor.empty of resultType whose dynamic dims are read off shapeSource
inline mlir::LogicalResult buildInit(mlir::PatternRewriter &rewriter,
                                     mlir::PDLResultList &results,
                                     llvm::ArrayRef<mlir::PDLValue> args) {
  // Guarded by CanBuildInit, so the casts hold.
  auto initType =
      mlir::cast<mlir::RankedTensorType>(args[0].dyn_cast<mlir::Type>());
  mlir::Value shapeSource = args[1].dyn_cast<mlir::Value>();
  mlir::Location loc = shapeSource.getLoc();

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dim : llvm::seq<int64_t>(initType.getRank()))
    if (initType.isDynamicDim(dim))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, shapeSource, dim));

  // getResult(), not the op: an op wrapper converts to both Value and
  // Operation *, which makes push_back ambiguous.
  results.push_back(mlir::tensor::EmptyOp::create(
                        rewriter, loc, initType.getShape(),
                        initType.getElementType(), dynSizes)
                        .getResult());
  return mlir::success();
}

/// op's `name` attribute as a signless i64 attribute, defaultValue when absent
inline mlir::LogicalResult
extractHipIntAttr(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
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
  // No companion constraint: the attributes read here are declared
  // DefaultValuedAttr<I64Attr>, so ODS has already pinned the width and an
  // absent one only means the op is carrying its default. The fallback keeps
  // this total, because a native rewrite that declines still has to push a
  // result or the PDL bytecode asserts.
  std::optional<int64_t> value =
      tryHipIntAttr(op, nameAttr.getValue(), defaultValue.getInt());
  // Signless, because it feeds a hip op attribute declared as I64Attr.
  results.push_back(
      rewriter.getI64IntegerAttr(value.value_or(defaultValue.getInt())));
  return mlir::success();
}

/// op's `name` attribute as an f32 attribute, defaultValue when absent
inline mlir::LogicalResult
extractHipFloatAttr(mlir::PatternRewriter &rewriter,
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
  // No companion constraint, for the same reason as ExtractHipIntAttr: the
  // attributes read here are declared DefaultValuedAttr<F32Attr>, so an absent
  // one only means the op is carrying its default.
  auto attr = op->getAttrOfType<mlir::FloatAttr>(nameAttr.getValue());
  // f32, because it feeds a hip op attribute declared as F32Attr.
  results.push_back(rewriter.getF32FloatAttr(
      static_cast<float>((attr ? attr : defaultValue).getValueAsDouble())));
  return mlir::success();
}

/// dq's quantized value width in bits: 4 when its quantized operands hold two
/// values per byte, otherwise the storage element type's own width
inline mlir::LogicalResult
extractHipQdqValueBits(mlir::PatternRewriter &rewriter,
                       mlir::PDLResultList &results,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  auto dq = mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op);
  mlir::IntegerType intType = getQdqQuantizedElementType(op);
  if (!dq || !intType)
    return mlir::failure();
  // A packed operand keeps its logical element count and an 8-bit element
  // type, so the type cannot report the value width; the marker
  // convert-onnx-to-hip carried over from constant lowering is what can.
  results.push_back(rewriter.getI64IntegerAttr(
      dq.getPackedInt4() ? 4 : static_cast<int64_t>(intType.getWidth())));
  return mlir::success();
}

inline mlir::LogicalResult
extractHipSplatScale(mlir::PatternRewriter &rewriter,
                     mlir::PDLResultList &results,
                     llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 1)
    return mlir::failure();
  std::optional<float> scale =
      tryHipSplatScale(args[0].dyn_cast<mlir::Value>());
  if (!scale)
    return mlir::failure();
  results.push_back(rewriter.getF32FloatAttr(*scale));
  return mlir::success();
}

inline mlir::LogicalResult
extractHipQdqZeropoint(mlir::PatternRewriter &rewriter,
                       mlir::PDLResultList &results,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  auto absentValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  if (!absentValue)
    return mlir::failure();
  std::optional<int64_t> zeropoint = tryHipQdqZeropoint(
      args[0].dyn_cast<mlir::Operation *>(), absentValue.getInt());
  if (!zeropoint)
    return mlir::failure();
  results.push_back(rewriter.getI64IntegerAttr(*zeropoint));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

inline void registerNativeHelpers(mlir::PDLPatternModule &pdlPatterns) {
  // A name declared in HipFusionTransformPatterns.pdll but missing here
  // aborts when the pattern runs.
  pdlPatterns.registerConstraintFunction("HasSingleUseResult",
                                         hasSingleUseResult);
  pdlPatterns.registerConstraintFunction("CanBuildInit", canBuildInit);
  pdlPatterns.registerRewriteFunction("BuildInit", buildInit);
  pdlPatterns.registerConstraintFunction("HasHipIntAttrEqual",
                                         hasHipIntAttrEqual);
  pdlPatterns.registerRewriteFunction("ExtractHipIntAttr", extractHipIntAttr);
  pdlPatterns.registerRewriteFunction("ExtractHipFloatAttr",
                                      extractHipFloatAttr);
  pdlPatterns.registerConstraintFunction("IsHipSplatScale", isHipSplatScale);
  pdlPatterns.registerConstraintFunction("HasExtractableQdqZeropoint",
                                         hasExtractableQdqZeropoint);
  pdlPatterns.registerConstraintFunction("IsHipQdqQuantizedWidth",
                                         isHipQdqQuantizedWidth);
  pdlPatterns.registerConstraintFunction("IsHipQdqUnsignedQuantized",
                                         isHipQdqUnsignedQuantized);
  pdlPatterns.registerConstraintFunction("IsHipFusableQConvGeometry",
                                         isHipFusableQConvGeometry);
  pdlPatterns.registerConstraintFunction("IsHipPerAxisQuantizedWeight",
                                         isHipPerAxisQuantizedWeight);
  pdlPatterns.registerConstraintFunction("IsHipPerChannelQuantizedWeight",
                                         isHipPerChannelQuantizedWeight);
  pdlPatterns.registerRewriteFunction("ExtractHipSplatScale",
                                      extractHipSplatScale);
  pdlPatterns.registerRewriteFunction("ExtractHipQdqZeropoint",
                                      extractHipQdqZeropoint);
  pdlPatterns.registerRewriteFunction("ExtractHipQdqValueBits",
                                      extractHipQdqValueBits);
}

/// apply the patterns in pdlBuffer to every function body in module, an empty
/// buffer is a successful no-op, failure means the patterns will not parse or
/// the driver failed
inline mlir::LogicalResult run(mlir::ModuleOp module,
                               llvm::MemoryBufferRef pdlBuffer) {
  if (pdlBuffer.getBufferSize() == 0)
    return mlir::success();

  mlir::MLIRContext *ctx = module.getContext();
  mlir::ParserConfig parseConfig(ctx);
  // parseSourceString, not parseSourceFile: the latter's StringRef overload
  // takes a path, so it would try to open the pattern text itself as a file.
  // Both dispatch on the magic bytes, so textual IR and bytecode work either
  // way.
  mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
      mlir::parseSourceString<mlir::ModuleOp>(
          pdlBuffer.getBuffer(), parseConfig, pdlBuffer.getBufferIdentifier());
  if (!pdlModule)
    return mlir::failure();

  // FrozenRewritePatternSet skips the PDL-to-PDLInterp lowering for a module
  // holding no pdl.pattern, then still asks the bytecode generator for the
  // @matcher function that lowering would have produced. Bail out first so a
  // pattern set that is empty (every pattern disabled) stays a no-op.
  if (pdlModule->getOps<mlir::pdl::PatternOp>().empty())
    return mlir::success();

  mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));
  registerNativeHelpers(pdlPatterns);

  mlir::RewritePatternSet patterns(ctx);
  patterns.add(std::move(pdlPatterns));
  // Freeze once for the whole module: this is what lowers PDL to PDLInterp
  // and generates the matcher bytecode, which is far too expensive to redo
  // per function.
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));

  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (funcOp.isDeclaration())
      return;
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, frozen)))
      result = mlir::failure();
  });
  return result;
}

} // namespace fusion_transform
} // namespace hip
