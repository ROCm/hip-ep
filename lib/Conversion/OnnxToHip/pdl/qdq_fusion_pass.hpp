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
  pdlPatterns.registerRewriteFunction("GetContextArg", getContextArg);
  pdlPatterns.registerRewriteFunction("ExtractScaleValue", extractScaleValue);
  pdlPatterns.registerRewriteFunction("ExtractZeropointValue",
                                      extractZeropointValue);
  pdlPatterns.registerRewriteFunction("ExtractAttrInt64", extractAttrInt64);

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
