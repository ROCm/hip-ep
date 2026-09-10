/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// qdq_fusion_pass.hpp — PDL fusion pass for QDQ patterns

#pragma once

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <limits>
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
// This is the only place 4-bit packing is observable. ONNX INT4/UINT4 imports
// as i8/ui8 at the LOGICAL element count, so the element type cannot show the
// width, and this pass runs BEFORE lowerOnnxConstants stamps `packed_int4` --
// the marker does not exist yet and cannot be consulted.
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

// True when an 8-bit-typed constant really holds ceil(numel/2) packed nibbles,
// which is the same relation hip.constant's verifier accepts. Full-width
// storage is plain int8 and must not be claimed. A single element occupies a
// whole byte either way, so it is indistinguishable from int8 and is rejected.
inline bool isPackedInt4Constant(mlir::Value value) {
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
  if (numel < 2)
    return false;
  std::optional<int64_t> bytes = tryConstantBackingBytes(value);
  return bytes && *bytes == (numel + 1) / 2;
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

// Scalar ONNX attribute equal to `expected`, treating absent as the default
// the caller passes in `absentValue`.
inline bool onnxIntAttrEquals(mlir::Operation *op, llvm::StringRef name,
                              int64_t expected, int64_t absentValue) {
  auto intAttr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!intAttr)
    return absentValue == expected;
  return intAttr.getValue().getSExtValue() == expected;
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

// DQ -> Transpose -> Q may operate directly on the quantized tensor when both
// ends use identical UINT16 per-tensor quantization. Restrict scales to the
// normal finite fp32 range so the eliminated Q(DQ(x)) round-trip is exact for
// every UINT16 value supported by the runtime's fp32 arithmetic.
inline mlir::LogicalResult
hasMatchingTransposeQParams(mlir::PatternRewriter &, mlir::PDLResultList &,
                            llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 2)
    return mlir::failure();
  auto *dq = args[0].dyn_cast<mlir::Operation *>();
  auto *q = args[1].dyn_cast<mlir::Operation *>();
  if (!dq || !q || dq->getNumOperands() < 2 || q->getNumOperands() < 2 ||
      dq->getNumResults() != 1 || q->getNumResults() != 1)
    return mlir::failure();

  auto dqType = getQuantizedElementType(dq);
  auto qType = getQuantizedElementType(q);
  if (!dqType || !qType || dqType != qType || dqType.getWidth() != 16 ||
      !dqType.isUnsigned())
    return mlir::failure();

  auto isScalarTensor = [](mlir::Value value) {
    auto type = mlir::dyn_cast<mlir::ShapedType>(value.getType());
    return type && type.hasStaticShape() && type.getNumElements() == 1;
  };
  if (!isScalarTensor(dq->getOperand(1)) || !isScalarTensor(q->getOperand(1)))
    return mlir::failure();

  std::optional<float> dqScale = trySplatScale(dq->getOperand(1));
  std::optional<float> qScale = trySplatScale(q->getOperand(1));
  if (!dqScale || !qScale || *dqScale != *qScale || !std::isfinite(*dqScale) ||
      *dqScale < std::numeric_limits<float>::min() ||
      *dqScale > std::numeric_limits<float>::max() / 65535.0f)
    return mlir::failure();

  // Scalar scale means axis is immaterial, but blocked quantization is not.
  if (!onnxIntAttrEquals(dq, "block_size", 0, /*absentValue=*/0) ||
      !onnxIntAttrEquals(q, "block_size", 0, /*absentValue=*/0))
    return mlir::failure();

  std::optional<int64_t> dqZp = trySplatZeropoint(dq, 2, 0);
  std::optional<int64_t> qZp = trySplatZeropoint(q, 2, 0);
  return mlir::success(dqZp && qZp && *dqZp == *qZp);
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

// The weight-side DequantizeLinear carries packed 4-bit weights quantized per
// output channel, which is the whole point of the fusion: those bytes stay
// packed until they are inside the kernel.
//
// The scale and zero point are checked for shape only, never for value. They
// routinely arrive as external constants with no bytes in the IR, so unlike the
// activation scale they cannot be folded into attributes and must be passed
// through as operands for the kernel to index by output channel.
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
  if (!isPackedInt4Constant(weights))
    return mlir::failure();
  auto weightType = mlir::cast<mlir::RankedTensorType>(weights.getType());
  // A Conv filter is [Cout, Cin/group, k...], so axis 0 is the output channel.
  int64_t outChannels = weightType.getDimSize(0);

  auto scaleType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  if (!scaleType || scaleType.getRank() != 1 ||
      scaleType.getDimSize(0) != outChannels)
    return mlir::failure();

  // ONNX guarantees zero_point.dtype == x.dtype, so a 4-bit weight implies a
  // 4-bit zero point. Requiring it to be packed too is what lets one
  // `packed_int4` flag describe both operands, as hip.dequantize_linear does.
  mlir::Value zeroPoints = op->getOperand(2);
  auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(zeroPoints.getType());
  if (!zpType || zpType.getRank() != 1 || zpType.getDimSize(0) != outChannels ||
      zpType.getElementType() != weightType.getElementType())
    return mlir::failure();
  return mlir::success(isPackedInt4Constant(zeroPoints));
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

// args[0] = op, args[1] = attribute name, args[2] = value to use when absent.
// Return failure if the attribute is not an i64 integer attribute.
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

  results.push_back(attr ? attr : defaultValue);
  return mlir::success();
}

// Rebuild the Transpose with the quantized input and output type, copying its
// permutation and diagnostic attributes verbatim. The old DQ/Transpose/Q chain
// becomes dead and is removed by the greedy rewrite driver.
inline mlir::LogicalResult
createQuantizedTranspose(mlir::PatternRewriter &rewriter,
                         mlir::PDLResultList &results,
                         llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 3)
    return mlir::failure();
  auto *dq = args[0].dyn_cast<mlir::Operation *>();
  auto *transpose = args[1].dyn_cast<mlir::Operation *>();
  auto *q = args[2].dyn_cast<mlir::Operation *>();
  if (!dq || !transpose || !q || dq->getNumOperands() == 0 ||
      q->getNumResults() != 1)
    return mlir::failure();

  mlir::OperationState state(transpose->getLoc(), "onnx.Transpose");
  state.addOperands(dq->getOperand(0));
  state.addTypes(q->getResult(0).getType());
  state.addAttributes(transpose->getAttrs());
  mlir::Operation *newTranspose = rewriter.create(state);
  results.push_back(newTranspose->getResult(0));
  return mlir::success();
}

// Apply PDL patterns
inline bool run(mlir::ModuleOp mlirModule, llvm::StringRef pdlBytecodeFile) {
  if (pdlBytecodeFile.empty())
    return true;

  mlir::MLIRContext *ctx = mlirModule.getContext();

  mlir::ParserConfig parseConfig(ctx);
  mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
      mlir::parseSourceFile<mlir::ModuleOp>(pdlBytecodeFile, parseConfig);
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
  pdlPatterns.registerConstraintFunction("HasStaticShapedResults",
                                         hasStaticShapedResults);
  pdlPatterns.registerConstraintFunction("IsUint16Quantized",
                                         isUint16Quantized);
  pdlPatterns.registerConstraintFunction("HasMatchingTransposeQParams",
                                         hasMatchingTransposeQParams);
  pdlPatterns.registerConstraintFunction("IsFusableQConvGeometry",
                                         isFusableQConvGeometry);
  pdlPatterns.registerConstraintFunction("IsPackedInt4PerChannelWeight",
                                         isPackedInt4PerChannelWeight);
  pdlPatterns.registerRewriteFunction("GetContextArg", getContextArg);
  pdlPatterns.registerRewriteFunction("ExtractScaleValue", extractScaleValue);
  pdlPatterns.registerRewriteFunction("ExtractZeropointValue",
                                      extractZeropointValue);
  pdlPatterns.registerRewriteFunction("ExtractAttrInt64", extractAttrInt64);
  pdlPatterns.registerRewriteFunction("CreateQuantizedTranspose",
                                      createQuantizedTranspose);

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
