/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// qdq_fusion_pass.hpp — PDL fusion pass for QDQ MatMul patterns

#pragma once

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hip {
namespace pdl {

// Native constraint: Get context argument from function
// Low-level signature required by MLIR PDL infrastructure
inline mlir::LogicalResult getContextArg(mlir::PatternRewriter &rewriter,
                                         mlir::PDLResultList &results,
                                         llvm::ArrayRef<mlir::PDLValue> args) {
  // args[0] should be the operation
  if (args.size() != 1)
    return mlir::failure();

  auto *op = args[0].dyn_cast<mlir::Operation *>();
  if (!op)
    return mlir::failure();

  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp || funcOp.getNumArguments() == 0)
    return mlir::failure();

  // Return the context (first function argument) as a Value
  results.push_back(funcOp.getArgument(0));
  return mlir::success();
}

// Native constraint: Extract float value from an optional onnx.Constant operand
// Low-level signature required by MLIR PDL infrastructure
inline mlir::LogicalResult
extractScaleValue(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  // args[0] should be the constant value
  if (args.size() != 1)
    return mlir::failure();

  auto constValue = args[0].dyn_cast<mlir::Value>();
  if (!constValue)
    return mlir::failure();

  auto defOp = constValue.getDefiningOp();
  if (!defOp)
    return mlir::failure();

  auto valueAttr = defOp->getAttr("value");
  if (!valueAttr)
    return mlir::failure();

  auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr);
  if (!denseAttr || !denseAttr.isSplat())
    return mlir::failure();

  float scaleValue =
      denseAttr.getSplatValue<mlir::FloatAttr>().getValueAsDouble();

  // Return the extracted float as an f32 attribute
  results.push_back(rewriter.getF32FloatAttr(scaleValue));
  return mlir::success();
}

// Native constraint: Extract integer value from onnx.Constant
// Low-level signature required by MLIR PDL infrastructure
inline mlir::LogicalResult
extractZeropointValue(mlir::PatternRewriter &rewriter,
                      mlir::PDLResultList &results,
                      llvm::ArrayRef<mlir::PDLValue> args) {
  if (args.size() != 3)
    return mlir::failure();
  auto *op = args[0].dyn_cast<mlir::Operation *>();
  auto indexAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[1].dyn_cast<mlir::Attribute>());
  auto defaultValue = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      args[2].dyn_cast<mlir::Attribute>());
  if (!op || !indexAttr || !defaultValue)
    return mlir::failure();

  uint64_t index = indexAttr.getValue().getZExtValue();
  if (index >= op->getNumOperands()) {
    results.push_back(defaultValue);
    return mlir::success();
  }
  auto *defOp = op->getOperand(index).getDefiningOp();
  if (!defOp)
    return mlir::failure();
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(defOp->getAttr("value"));
  if (!denseAttr || !denseAttr.isSplat())
    return mlir::failure();
  auto intType = mlir::dyn_cast<mlir::IntegerType>(denseAttr.getElementType());
  if (!intType)
    return mlir::failure();
  // ONNX zero points are int8/uint8/int32. Sign-extending an unsigned type
  // would read a uint8 zero point of 128 back as -128, so pick the extension
  // that matches the element type. Signless types keep signed semantics.
  llvm::APInt raw = denseAttr.getSplatValue<llvm::APInt>();
  int64_t i64Value = intType.isUnsigned()
                         ? static_cast<int64_t>(raw.getZExtValue())
                         : raw.getSExtValue();
  results.push_back(rewriter.getI64IntegerAttr(i64Value));
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

  mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));

  // Register native constraints with low-level signatures
  pdlPatterns.registerConstraintFunction("GetContextArg", getContextArg);
  pdlPatterns.registerConstraintFunction("ExtractScaleValue",
                                         extractScaleValue);
  pdlPatterns.registerConstraintFunction("ExtractZeropointValue",
                                         extractZeropointValue);

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
