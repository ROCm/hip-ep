/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsr.cpp - Convert ONNX dialect to the hipsr dialect --------===//
//
// Converts ONNX dialect IR into tensor-form hipsr dialect IR. The conversion
// target keeps helper computations inside hipsr.compute while allowing scalar
// constants as graph roots. After conversion, placeholder inputs are moved onto
// the shape graph.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

// Returns the shape-graph value for a placeholder input. A data result becomes
// the outs operand its producer writes into, which has the same shape.
Value resolveShapeGraphInput(Value input) {
  if (PlaceholderOp::isAllowedShapeGraphInput(input)) {
    return input;
  }
  // Block arguments are allowed, so anything left is a result.
  auto result = cast<OpResult>(input);
  OperandRange destinations = getHipsrDestinationOperands(result.getOwner());
  if (result.getResultNumber() >= destinations.size()) {
    return input;
  }
  return destinations[result.getResultNumber()];
}

// onnx.NoValue stands in for an omitted optional operand, so it only becomes
// dead once the conversion of its consumer drops that operand. Sweeping it up
// therefore belongs after the conversion.
void eraseDeadNoValue(ModuleOp module) {
  SmallVector<onnx::NoValueOp> dead;
  module.walk([&](onnx::NoValueOp op) {
    if (op->use_empty()) {
      dead.push_back(op);
    }
  });
  for (onnx::NoValueOp op : dead) {
    op.erase();
  }
}

// Placeholder inputs form the shape graph, not the data graph.
// PlaceholderOp::verify reports any input this cannot fix.
void rewirePlaceholderInputs(ModuleOp module) {
  module.walk([](PlaceholderOp placeholder) {
    SmallVector<Value> resolvedInputs =
        llvm::map_to_vector(placeholder.getInputs(), resolveShapeGraphInput);
    placeholder.getInputsMutable().assign(resolvedInputs);
  });
}

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](RankedTensorType type) -> Type {
      // Leave rank-0 scalars (compile-time host roots lowered to
      // arith.constant) and tensors that already name a space (e.g. host
      // shapes) untouched.
      if (type.getRank() == 0 || type.getEncoding()) {
        return type;
      }
      return tensorTypeInSpace(type, MemorySpace::Device);
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<onnx::OnnxDialect>();
    // The consumer's conversion drops the operand this stands for, so the
    // placeholder has to survive until then.
    target.addLegalOp<onnx::NoValueOp>();
    target.addLegalDialect<HipsrDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<arith::ConstantOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });
    // Helper operations are legal in the hipsr region that owns them: a compute
    // body or a placeholder's shape region.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getParentOfType<ComputeOp>() != nullptr ||
             op->getParentOfType<PlaceholderOp>() != nullptr;
    });

    RewritePatternSet patterns(&getContext());
    populateOnnxToHipsrConstantPatterns(converter, patterns);
    populateCastConversionPatterns(converter, patterns, &getContext());
    populateMatMulConversionPatterns(converter, patterns, &getContext());
    populateExpandConversionPatterns(converter, patterns, &getContext());
    populateMinConversionPatterns(converter, patterns, &getContext());
    populateShapeConversionPatterns(converter, patterns, &getContext());
    populateReshapeConversionPatterns(converter, patterns, &getContext());
    populateUnsqueezeConversionPatterns(converter, patterns, &getContext());
    populateEqualConversionPatterns(converter, patterns, &getContext());
    populateTransposeConversionPatterns(converter, patterns, &getContext());
    populateGatherConversionPatterns(converter, patterns, &getContext());
    populateSliceConversionPatterns(converter, patterns, &getContext());
    populateScatterNDConversionPatterns(converter, patterns, &getContext());
    populateNonZeroConversionPatterns(converter, patterns, &getContext());
    populateReturnConversionPatterns(converter, patterns, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    eraseDeadNoValue(module);
    // Runs after conversion so every producer has its outs operand.
    rewirePlaceholderInputs(module);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
