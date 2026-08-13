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
          if (operation->hasAttr("test.reify_rank_zero")) {
            auto reify =
                mlir::cast<mlir::ReifyRankedShapedTypeOpInterface>(operation);
            rewriter.setInsertionPoint(operation);
            mlir::ReifiedRankedShapedTypeDims reified;
            if (mlir::failed(reify.reifyResultShapes(rewriter, reified)) ||
                reified.size() != 1 || !reified.front().empty()) {
              operation->emitOpError(
                  "rank-zero reification must return one empty shape");
              return mlir::WalkResult::interrupt();
            }
            operation->setAttr("test.rank_zero_reified",
                               rewriter.getUnitAttr());
            return mlir::WalkResult::advance();
          }

          if (operation->hasAttr("test.default_reify_failure_atomic")) {
            auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(operation);
            mlir::Operation::operand_range inits = dps.getDpsInits();
            if (operation->getNumResults() < 2 || inits.size() < 2) {
              operation->emitOpError(
                  "default-reify atomicity fixture requires two results");
              return mlir::WalkResult::interrupt();
            }

            mlir::Value secondInit = *std::next(inits.begin());
            mlir::Value secondResult = operation->getResult(1);
            mlir::Type initType = secondInit.getType();
            mlir::Type resultType = secondResult.getType();
            mlir::Type invalidType = rewriter.getI32Type();
            secondInit.setType(invalidType);
            secondResult.setType(invalidType);

            mlir::Block *block = operation->getBlock();
            size_t before = std::distance(block->begin(), block->end());
            rewriter.setInsertionPoint(operation);
            mlir::ReifiedRankedShapedTypeDims reified;
            mlir::LogicalResult status =
                op.reifyResultShapes(rewriter, reified);
            size_t after = std::distance(block->begin(), block->end());

            secondInit.setType(initType);
            secondResult.setType(resultType);
            if (mlir::succeeded(status)) {
              operation->emitOpError(
                  "malformed second result unexpectedly reified");
              return mlir::WalkResult::interrupt();
            }
            if (after != before || !reified.empty()) {
              operation->emitOpError("failed default reification mutated IR");
              return mlir::WalkResult::interrupt();
            }
            operation->setAttr("test.default_reify_failure_atomic_passed",
                               rewriter.getUnitAttr());
            return mlir::WalkResult::advance();
          }

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
