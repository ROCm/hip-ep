/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Dialect/Hip/TestHipPasses.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

/// Probe the HipDpsOp shared default on a zero-result memref-mode operation.
struct TestHipDpsDefaultReifyPass final
    : public mlir::PassWrapper<TestHipDpsDefaultReifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHipDpsDefaultReifyPass)

  llvm::StringRef getArgument() const final {
    return "test-hip-dps-default-reify";
  }
  llvm::StringRef getDescription() const final {
    return "Test the shared HIP DPS default reification contract";
  }

  void runOnOperation() final {
    mlir::IRRewriter rewriter(&getContext());
    mlir::WalkResult walkResult =
        getOperation().walk([&](mlir::hip::HipDpsOp op) -> mlir::WalkResult {
          mlir::Operation *operation = op.getOperation();
          if (!operation->hasAttr("test.default_reify"))
            return mlir::WalkResult::advance();
          rewriter.setInsertionPoint(operation);
          mlir::ReifiedRankedShapedTypeDims reified;
          if (mlir::failed(op.reifyResultShapes(rewriter, reified))) {
            operation->emitOpError(
                "shared default reification unexpectedly failed");
            return mlir::WalkResult::interrupt();
          }
          if (!reified.empty()) {
            operation->emitOpError()
                << "shared default reification returned " << reified.size()
                << " vector(s) for a zero-result memref-mode op";
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

void mlir::hip::test::registerHipTestPasses(DialectRegistry &registry) {
  (void)registry;
  PassRegistration<TestHipDpsDefaultReifyPass>();
}
