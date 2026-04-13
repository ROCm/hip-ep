/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TorchToHip.cpp - Convert Torch dialect to HIP dialect (tensor DPS) -===//
//
// Converts Torch dialect IR (from torch-mlir) into HIP dialect IR using
// destination-passing style (DPS) with tensor types.  Torch ops are matched by
// name via the generic MLIR Operation API, so no torch-mlir headers or
// libraries are required.  Bufferization to memref is handled by a separate
// downstream pass.
//
//===----------------------------------------------------------------------===//

#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTTORCHTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx) {
  mlir::RewritePatternSet patterns(ctx);
  populateTorchMatMulConversionPatterns(patterns, ctx);
  populateTorchConvConversionPatterns(patterns, ctx);
  populateTorchElementwiseConversionPatterns(patterns, ctx);
  populateTorchActivationConversionPatterns(patterns, ctx);
  populateTorchCastConversionPatterns(patterns, ctx);
  populateTorchGatherConversionPatterns(patterns, ctx);
  populateTorchReduceConversionPatterns(patterns, ctx);
  populateTorchReshapeConversionPatterns(patterns, ctx);
  populateTorchTransposeConversionPatterns(patterns, ctx);
  populateTorchNormConversionPatterns(patterns, ctx);
  populateTorchSliceCatConversionPatterns(patterns, ctx);
  populateTorchGqaConversionPatterns(patterns, ctx);
  populateTorchMatMulNBitsConversionPatterns(patterns, ctx);
  populateTorchQMoEConversionPatterns(patterns, ctx);
  populateTorchMiscConversionPatterns(patterns, ctx);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
  if (mlir::failed(
          mlir::applyPatternsGreedily(funcOp, std::move(patterns), config)))
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Module metadata generation
//===----------------------------------------------------------------------===//

/// Generate module metadata attributes required by GenerateInterfacePass.
/// Must be called BEFORE patterns transform function signatures.
///
/// Looks for @forward first (torch-mlir convention), then @main_graph
/// (ONNX convention), then falls back to any single public func.
static mlir::LogicalResult generateModuleMetadata(mlir::ModuleOp module) {
  mlir::func::FuncOp targetFunc;

  // Try @forward first (torch-mlir convention).
  targetFunc = module.lookupSymbol<mlir::func::FuncOp>("forward");
  // Then try @main_graph (ONNX convention).
  if (!targetFunc)
    targetFunc = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
  // Fall back to any single public function.
  if (!targetFunc) {
    mlir::func::FuncOp candidate;
    int publicCount = 0;
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.isPublic()) {
        candidate = funcOp;
        ++publicCount;
      }
    });
    if (publicCount == 1)
      targetFunc = candidate;
  }

  if (!targetFunc) {
    // No single entry function found -- skip metadata generation.
    // This is expected for unit conversion tests with multiple test functions.
    // Metadata is only required for E2E compilation (GenerateInterface pass).
    return mlir::success();
  }

  auto originalFuncType = targetFunc.getFunctionType();
  mlir::OpBuilder builder(module.getContext());

  int64_t inputCount = originalFuncType.getNumInputs();
  llvm::SmallVector<mlir::Attribute> inputShapes;
  llvm::SmallVector<int64_t> inputElementSizes;

  for (mlir::Type inputType : originalFuncType.getInputs()) {
    if (mlir::isa<mlir::hip::ContextType>(inputType)) {
      --inputCount;
      continue;
    }
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        targetFunc.emitError("unsupported element type in input: ") << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      inputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      inputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      targetFunc.emitError("non-tensor input type: ") << inputType;
      return mlir::failure();
    }
  }

  int64_t outputCount = originalFuncType.getNumResults();
  llvm::SmallVector<mlir::Attribute> outputShapes;
  llvm::SmallVector<int64_t> outputElementSizes;

  for (mlir::Type resultType : originalFuncType.getResults()) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        targetFunc.emitError("unsupported element type in output: ")
            << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      outputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      outputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      targetFunc.emitError("non-tensor output type: ") << resultType;
      return mlir::failure();
    }
  }

  module->setAttr("hipdnn.input_count", builder.getI64IntegerAttr(inputCount));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputElementSizes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(outputCount));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputShapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputElementSizes));

  LLVM_DEBUG({
    llvm::dbgs() << "[convert-torch-to-hip] module metadata:"
                 << " input_count=" << inputCount
                 << " input_shapes=" << builder.getArrayAttr(inputShapes)
                 << " input_element_sizes="
                 << builder.getDenseI64ArrayAttr(inputElementSizes)
                 << " output_count=" << outputCount
                 << " output_shapes=" << builder.getArrayAttr(outputShapes)
                 << " output_element_sizes="
                 << builder.getDenseI64ArrayAttr(outputElementSizes) << "\n";
  });

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Torch op cleanup
//===----------------------------------------------------------------------===//

/// Erase dead torch.constant.none, torch.constant.int, torch.constant.bool,
/// and torch.prim.ListConstruct ops that have no remaining uses.
static void cleanupTorchOps(mlir::ModuleOp module) {
  // Iterate until fixed point: erasing a ListConstruct may make its
  // constant.int operands dead, which need a second pass to clean up.
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> toErase;
    module.walk([&](mlir::Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if ((name == "torch.constant.none" || name == "torch.constant.int" ||
           name == "torch.constant.bool" || name == "torch.constant.float" ||
           name == "torch.prim.ListConstruct") &&
          op->use_empty())
        toErase.push_back(op);
    });
    for (auto *op : toErase) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// ConvertTorchToHip Pass
//===----------------------------------------------------------------------===//

struct ConvertTorchToHipPass
    : public impl::ConvertTorchToHipPassBase<ConvertTorchToHipPass> {
  using ConvertTorchToHipPassBase::ConvertTorchToHipPassBase;

  void runOnOperation() override;
};

void ConvertTorchToHipPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module.getContext();

  // Capture original function signatures as module metadata before lowering.
  if (mlir::failed(generateModuleMetadata(module)))
    return signalPassFailure();

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    if (mlir::failed(convertComputeOps(funcOp, ctx)))
      return signalPassFailure();
  }

  // Clean up dead torch utility ops.
  cleanupTorchOps(module);
}

} // namespace

} // namespace hip
} // namespace mlir
