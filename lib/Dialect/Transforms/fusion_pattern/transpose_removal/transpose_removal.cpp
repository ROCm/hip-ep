/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- transpose_removal.cpp - native helpers for TransposeRemoval.pdll ---===//
#include "transpose_removal.hpp"

#include "../fusion_utils.hpp"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <cstdint>

namespace hip {
namespace fusion_transform {
namespace {

/// second undoes first: both are tensor-form transposes whose permutations are
/// inverses and whose shapes close the loop
mlir::LogicalResult
isInverseTransposePair(mlir::PatternRewriter &, mlir::PDLResultList &,
                       llvm::ArrayRef<mlir::PDLValue> args) {
  // The pattern matches both as tensor-form hip.transpose ops.
  auto first =
      mlir::cast<mlir::hip::TransposeOp>(args[0].dyn_cast<mlir::Operation *>());
  auto second =
      mlir::cast<mlir::hip::TransposeOp>(args[1].dyn_cast<mlir::Operation *>());

  // The verifier guarantees both perms are permutations.
  llvm::SmallVector<int64_t> firstPerm =
      mlir::extractFromIntegerArrayAttr<int64_t>(first.getPermAttr());
  llvm::SmallVector<int64_t> secondPerm =
      mlir::extractFromIntegerArrayAttr<int64_t>(second.getPermAttr());
  if (secondPerm != mlir::invertPermutationVector(firstPerm))
    return mlir::failure();

  // The replacement keeps the second transpose's result type, so the type
  // entering the first one has to be exactly that.
  return mlir::success(second->getResult(0).getType() ==
                       first.getInput().getType());
}

/// op computes each output element only from the input element at the same
/// index and has no axis-dependent attributes, so it commutes with any
/// transpose
mlir::LogicalResult isPerElementOp(mlir::PatternRewriter &,
                                   mlir::PDLResultList &,
                                   llvm::ArrayRef<mlir::PDLValue> args) {
  return mlir::success(
      mlir::isa<mlir::hip::GeluOp>(args[0].dyn_cast<mlir::Operation *>()));
}

mlir::LogicalResult rebuildOnInput(mlir::PatternRewriter &rewriter,
                                   mlir::PDLResultList &results,
                                   llvm::ArrayRef<mlir::PDLValue> args) {
  mlir::Value newInit = args[3].dyn_cast<mlir::Value>();
  results.push_back(rebuildOnInputValue(
      rewriter, args[0].dyn_cast<mlir::Operation *>(),
      args[1].dyn_cast<mlir::Value>(), args[2].dyn_cast<mlir::Value>(), newInit,
      newInit.getType()));
  return mlir::success();
}

} // namespace

void registerTransposeRemovalHelpers(mlir::PDLPatternModule &pdlPatterns) {
  pdlPatterns.registerConstraintFunction("IsInverseTransposePair",
                                         isInverseTransposePair);
  pdlPatterns.registerConstraintFunction("IsPerElementOp", isPerElementOp);
  pdlPatterns.registerRewriteFunction("RebuildOnInput", rebuildOnInput);
}

} // namespace fusion_transform
} // namespace hip
