/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsr.cpp - Convert ONNX dialect to the hipsr dialect --------===//
//
// Converts ONNX dialect IR into hipsr dialect IR (tensor DPS). The conversion
// target keeps helper computations inside hipsr.compute while allowing scalar
// constants as graph roots. After conversion, placeholder dependencies are
// rewired into a parallel shape graph.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

// Shape dependencies flow through placeholders instead of computed data.
static LogicalResult finalizePlaceholderDependencies(ModuleOp module) {
  llvm::DenseMap<Value, Value> placeholderForDataResult;
  module.walk([&](DestinationStyleOpInterface dpsOp) {
    for (OpResult result : dpsOp->getResults()) {
      OpOperand *initOperand = dpsOp.getTiedOpOperand(result);
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

  llvm::SmallVector<std::pair<OpOperand *, Value>> replacements;
  WalkResult walkResult =
      module.walk([&](PlaceholderOp placeholder) -> WalkResult {
        MutableOperandRange mutableInputs = placeholder.getInputsMutable();
        for (auto [inputIndex, input] :
             llvm::enumerate(placeholder.getInputs())) {
          auto replacement = placeholderForDataResult.find(input);
          Value resolvedInput = replacement == placeholderForDataResult.end()
                                    ? input
                                    : replacement->second;
          if (!PlaceholderOp::isAllowedShapeGraphInput(resolvedInput)) {
            placeholder.emitOpError("input ")
                << inputIndex
                << " must be a block argument or a result of "
                   "hipsr.placeholder, arith.constant, or hipsr.constant; got "
                   "result of '"
                << resolvedInput.getDefiningOp()->getName() << "'";
            return WalkResult::interrupt();
          }
          if (resolvedInput != input) {
            replacements.emplace_back(&mutableInputs[inputIndex],
                                      resolvedInput);
          }
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  for (auto [operand, value] : replacements) {
    operand->set(value);
  }
  return success();
}

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    target.addIllegalDialect("onnx");
    target.addLegalDialect<HipsrDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addLegalOp<arith::ConstantOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getParentOfType<ComputeOp>() != nullptr;
    });

    RewritePatternSet patterns(&getContext());
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // Build the parallel shape graph after all data-graph rewrites finish.
    if (failed(finalizePlaceholderDependencies(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
