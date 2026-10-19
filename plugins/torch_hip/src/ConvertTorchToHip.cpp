/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ConvertTorchToHip.cpp - Torch-MLIR -> HIP dialect ------------------===//
//
// Skeleton (PR A): the pass is registered and wired into the pipeline, but it
// only contains the guard. Modules without `torch` ops (e.g. every ONNX model
// compiled by a plugin-enabled build) are left untouched; modules with `torch`
// ops fail with a clear message until the conversion lands.
//
//===----------------------------------------------------------------------===//

#include "torch_hip/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::hip {
namespace {

/// Dialect namespace of Torch-MLIR ops.
constexpr llvm::StringLiteral kTorchDialectNamespace = "torch";

/// True if `op` belongs to the Torch dialect. Matched by namespace so it works
/// whether or not the dialect is registered in the context.
bool isTorchOp(Operation *op) {
  return op->getName().getDialectNamespace() == kTorchDialectNamespace;
}

/// True if any op nested in `module` is a Torch op.
bool moduleHasTorchOps(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    return isTorchOp(op) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Emits a user-facing error with the `[torch_hip]` prefix.
void emitTorchHipError(Operation *op, llvm::StringRef message) {
  op->emitError() << "[torch_hip] " << message;
}

struct ConvertTorchToHipPass
    : public PassWrapper<ConvertTorchToHipPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTorchToHipPass)

  llvm::StringRef getArgument() const final {
    return kConvertTorchToHipPassName;
  }

  llvm::StringRef getDescription() const final {
    return "Lower Torch-MLIR (torch dialect) ops to the HIP dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Guard: the plugin is linked into every host binary, so non-Torch
    // modules also reach this pass and must stay unchanged.
    if (!moduleHasTorchOps(module))
      return;

    emitTorchHipError(module, "Torch input is not supported yet: the "
                              "Torch -> HIP conversion is not implemented in "
                              "this build.");
    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTorchToHipPass() {
  return std::make_unique<ConvertTorchToHipPass>();
}

void registerConvertTorchToHipPass() {
  PassRegistration<ConvertTorchToHipPass>();
}

} // namespace mlir::hip
