/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsr.cpp - Convert ONNX dialect to the hipsr dialect --------===//
//
// Converts ONNX dialect IR into hipsr dialect IR (tensor DPS). ONNX ops are
// matched by name via the generic MLIR Operation API, so no onnx-mlir headers
// are required. After rewriting, placeholder dependencies are separated from
// data dependencies. A later pass fills shape regions via ShapeRegionInterface.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

// Replace placeholder data dependencies with tied placeholder results.
static ::mlir::LogicalResult
finalizePlaceholderDependencies(::mlir::ModuleOp module) {
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> placeholderForDataResult;
  module.walk([&](::mlir::DestinationStyleOpInterface dpsOp) {
    for (::mlir::OpResult result : dpsOp->getResults()) {
      ::mlir::OpOperand *initOperand = dpsOp.getTiedOpOperand(result);
      if (!initOperand) {
        continue;
      }

      auto placeholder = initOperand->get().getDefiningOp<PlaceholderOp>();
      if (!placeholder || placeholder->getDialect() != dpsOp->getDialect()) {
        continue;
      }
      placeholderForDataResult.try_emplace(result, initOperand->get());
    }
  });

  ::llvm::SmallVector<std::pair<::mlir::OpOperand *, ::mlir::Value>>
      replacements;
  ::mlir::WalkResult walkResult =
      module.walk([&](PlaceholderOp placeholder) -> ::mlir::WalkResult {
        ::mlir::MutableOperandRange mutableInputs =
            placeholder.getInputsMutable();
        for (auto [inputIndex, input] :
             ::llvm::enumerate(placeholder.getInputs())) {
          auto replacement = placeholderForDataResult.find(input);
          ::mlir::Value resolvedInput =
              replacement == placeholderForDataResult.end()
                  ? input
                  : replacement->second;
          if (!PlaceholderOp::isAllowedShapeGraphInput(resolvedInput)) {
            placeholder.emitOpError("input ")
                << inputIndex
                << " must be a block argument or a result of "
                   "hipsr.placeholder, arith.constant, or hipsr.constant; got "
                   "result of '"
                << resolvedInput.getDefiningOp()->getName() << "'";
            return ::mlir::WalkResult::interrupt();
          }
          if (resolvedInput != input) {
            replacements.emplace_back(&mutableInputs[inputIndex],
                                      resolvedInput);
          }
        }
        return ::mlir::WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    return ::mlir::failure();
  }

  for (auto [operand, value] : replacements) {
    operand->set(value);
  }
  return ::mlir::success();
}

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ::mlir::RewritePatternSet patterns(&getContext());
    populateOnnxToHipsrConstantPatterns(patterns);
    populateCastConversionPatterns(patterns, &getContext());
    populateMatMulConversionPatterns(patterns, &getContext());
    populateExpandConversionPatterns(patterns, &getContext());

    // Same driver/config as convert-onnx-to-hip (greedy, ExistingOps): ONNX ops
    // are matched by name and only the ops present on entry are rewritten, so
    // generated hipsr / shape-region IR is left untouched.
    ::mlir::GreedyRewriteConfig config;
    config.setStrictness(::mlir::GreedyRewriteStrictness::ExistingOps);
    if (::mlir::failed(::mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
    // Build the parallel shape graph after all data-graph rewrites finish.
    if (::mlir::failed(finalizePlaceholderDependencies(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
