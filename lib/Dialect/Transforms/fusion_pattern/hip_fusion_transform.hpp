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
//   * op-agnostic helpers (`hasSingleUseResult`, `buildInit`) that any
//     pattern can reuse;
//   * readers for a specific op family, currently the Q/DQ quantization
//     parameters that QAddFusion.pdll needs.
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
  pdlPatterns.registerConstraintFunction("IsHipSplatScale", isHipSplatScale);
  pdlPatterns.registerConstraintFunction("HasExtractableQdqZeropoint",
                                         hasExtractableQdqZeropoint);
  pdlPatterns.registerRewriteFunction("ExtractHipSplatScale",
                                      extractHipSplatScale);
  pdlPatterns.registerRewriteFunction("ExtractHipQdqZeropoint",
                                      extractHipQdqZeropoint);
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
