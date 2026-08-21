/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Dialect/Hip/TestHipPasses.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iterator>
#include <optional>

namespace {

class TestMalformedReifyOp final
    : public mlir::Op<TestMalformedReifyOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::VariadicResults,
                      mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() {
    return "hip.test_malformed_reify";
  }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static const llvm::StringRef names[] = {"kind"};
    return names;
  }

  mlir::LogicalResult
  reifyResultShapes(mlir::OpBuilder &builder,
                    mlir::ReifiedRankedShapedTypeDims &reified) {
    auto kind = (*this)->getAttrOfType<mlir::StringAttr>("kind");
    if (!kind)
      return emitOpError("requires a string 'kind' attribute");

    reified.clear();
    if (kind.getValue() == "result_count")
      return mlir::success();
    if (kind.getValue() == "rank") {
      reified.push_back({builder.getIndexAttr(1)});
      return mlir::success();
    }
    if (kind.getValue() == "static_contradiction") {
      reified.push_back({builder.getIndexAttr(3), builder.getIndexAttr(4)});
      return mlir::success();
    }
    if (kind.getValue() == "negative_extent") {
      reified.push_back({builder.getIndexAttr(-2)});
      return mlir::success();
    }
    return emitOpError("unknown malformed-reifier fixture kind '")
           << kind.getValue() << "'";
  }
};

/// Resolve tensor.dim through the whole-shape interface implementation. This
/// test-only path keeps semantic reifier tests independent from the production
/// direct-dimension override.
struct ResolveDimFromWholeShapeReify final
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setHasBoundedRewriteRecursion(); }

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp dimOp,
                  mlir::PatternRewriter &rewriter) const final {
    auto result = mlir::dyn_cast<mlir::OpResult>(dimOp.getSource());
    if (!result || !mlir::isa<mlir::hip::HipDpsOp>(result.getOwner()))
      return mlir::failure();
    std::optional<int64_t> dim = dimOp.getConstantIndex();
    if (!dim)
      return mlir::failure();

    auto reify =
        mlir::cast<mlir::ReifyRankedShapedTypeOpInterface>(result.getOwner());
    mlir::ReifiedRankedShapedTypeDims shapes;
    if (mlir::failed(reify.reifyResultShapes(rewriter, shapes)) ||
        result.getResultNumber() >= shapes.size() || *dim < 0 ||
        *dim >= static_cast<int64_t>(shapes[result.getResultNumber()].size()))
      return mlir::failure();

    mlir::OpFoldResult extent = shapes[result.getResultNumber()][*dim];
    if (!extent)
      return mlir::failure();
    mlir::Value value =
        mlir::getValueOrCreateConstantIndexOp(rewriter, dimOp.getLoc(), extent);
    rewriter.replaceOp(dimOp, value);
    return mlir::success();
  }
};

struct TestHipWholeShapeDimReifyPass final
    : public mlir::PassWrapper<TestHipWholeShapeDimReifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHipWholeShapeDimReifyPass)

  llvm::StringRef getArgument() const final {
    return "test-hip-whole-shape-dim-reify";
  }
  llvm::StringRef getDescription() const final {
    return "Resolve HIP result dims through whole-shape semantic reification";
  }

  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ResolveDimFromWholeShapeReify>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

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
          if (operation->hasAttr("test.default_reify_dim_failure_atomic")) {
            if (operation->getNumResults() == 0) {
              operation->emitOpError(
                  "direct-dim atomicity fixture requires a tensor result");
              return mlir::WalkResult::interrupt();
            }

            auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(operation);
            mlir::OpResult result = operation->getResult(0);
            mlir::OpOperand *tiedOperand = dps.getTiedOpOperand(result);
            auto initType = mlir::cast<mlir::RankedTensorType>(
                tiedOperand->get().getType());
            if (initType.getRank() == 0) {
              operation->emitOpError(
                  "direct-dim atomicity fixture requires positive rank");
              return mlir::WalkResult::interrupt();
            }

            llvm::SmallVector<int64_t> invalidShape(initType.getShape());
            invalidShape[0] =
                initType.isDynamicDim(0) ? 7 : initType.getDimSize(0) + 1;
            mlir::Type invalidType = mlir::RankedTensorType::get(
                invalidShape, initType.getElementType(),
                initType.getEncoding());
            tiedOperand->get().setType(invalidType);

            mlir::Block *block = operation->getBlock();
            size_t before = std::distance(block->begin(), block->end());
            rewriter.setInsertionPoint(operation);
            auto reify =
                mlir::cast<mlir::ReifyRankedShapedTypeOpInterface>(operation);
            bool rejected =
                mlir::failed(reify.reifyDimOfResult(rewriter, -1, 0)) &&
                mlir::failed(reify.reifyDimOfResult(rewriter, 0, -1)) &&
                mlir::failed(
                    reify.reifyDimOfResult(rewriter, 0, initType.getRank())) &&
                mlir::failed(reify.reifyDimOfResult(rewriter, 0, 0));
            size_t after = std::distance(block->begin(), block->end());
            tiedOperand->get().setType(initType);

            if (!rejected || after != before) {
              operation->emitOpError(
                  "invalid direct-dim reification mutated IR or succeeded");
              return mlir::WalkResult::interrupt();
            }
            operation->setAttr("test.default_reify_dim_failure_atomic_passed",
                               rewriter.getUnitAttr());
            return mlir::WalkResult::advance();
          }

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
            mlir::Type initType = secondInit.getType();
            auto rankedInit = mlir::cast<mlir::RankedTensorType>(initType);
            llvm::SmallVector<int64_t> invalidShape(rankedInit.getShape());
            invalidShape.push_back(mlir::ShapedType::kDynamic);
            mlir::Type invalidType = mlir::RankedTensorType::get(
                invalidShape, rankedInit.getElementType(),
                rankedInit.getEncoding());
            secondInit.setType(invalidType);

            mlir::Block *block = operation->getBlock();
            size_t before = std::distance(block->begin(), block->end());
            rewriter.setInsertionPoint(operation);
            mlir::ReifiedRankedShapedTypeDims reified;
            mlir::LogicalResult status =
                op.reifyResultShapes(rewriter, reified);
            size_t after = std::distance(block->begin(), block->end());

            secondInit.setType(initType);
            if (mlir::succeeded(status)) {
              operation->emitOpError(
                  "rank-mismatched second init unexpectedly reified");
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
  registry.addExtension(+[](MLIRContext *, HipDialect *dialect) {
    RegisteredOperationName::insert<TestMalformedReifyOp>(*dialect);
  });
  PassRegistration<TestHipWholeShapeDimReifyPass>();
  PassRegistration<TestHipDpsDefaultReifyPass>();
}
