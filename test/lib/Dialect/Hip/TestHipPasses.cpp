/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Dialect/Hip/TestHipPasses.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInference.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

#include <iterator>

namespace {

/// Shared body of the malformed-reifier fixtures: a successful reification
/// that breaks the contract in the way named by the op's `kind` attribute.
mlir::LogicalResult
reifyMalformedShapes(mlir::Operation *op, mlir::OpBuilder &builder,
                     mlir::ReifiedRankedShapedTypeDims &reified) {
  auto kind = op->getAttrOfType<mlir::StringAttr>("kind");
  if (!kind)
    return op->emitOpError("requires a string 'kind' attribute");

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
  return op->emitOpError("unknown malformed-reifier fixture kind '")
         << kind.getValue() << "'";
}

/// Reifies but is not destination-style, so --hip-infer-shapes must skip it.
/// Its reifier is malformed: visiting it at all would report an error.
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
    return reifyMalformedShapes(*this, builder, reified);
  }
};

/// The destination-style form of the fixture above: one destination per
/// result, so --hip-infer-shapes visits it and must reject its reification.
class TestMalformedReifyDpsOp final
    : public mlir::Op<TestMalformedReifyDpsOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::VariadicResults,
                      mlir::DestinationStyleOpInterface::Trait,
                      mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "hip.test_malformed_reify_dps";
  }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static const llvm::StringRef names[] = {"kind"};
    return names;
  }
  mlir::MutableOperandRange getDpsInitsMutable() {
    return mlir::MutableOperandRange(*this);
  }
  mlir::LogicalResult
  reifyResultShapes(mlir::OpBuilder &builder,
                    mlir::ReifiedRankedShapedTypeDims &reified) {
    return reifyMalformedShapes(*this, builder, reified);
  }
};

/// `%y = <dialect>.like(%x, %init)`: a destination-style op whose single
/// result has the shape of `%x`. Its reifier reads `%x`'s static extents, so
/// --hip-infer-shapes can refine `?` result dims whenever `%x` is static.
template <typename ConcreteOp>
class TestLikeOpBase
    : public mlir::Op<ConcreteOp, mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::OneResult,
                      mlir::DestinationStyleOpInterface::Trait,
                      mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using mlir::Op<ConcreteOp, mlir::OpTrait::NOperands<2>::Impl,
                 mlir::OpTrait::OneResult,
                 mlir::DestinationStyleOpInterface::Trait,
                 mlir::ReifyRankedShapedTypeOpInterface::Trait>::Op;
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
  mlir::MutableOperandRange getDpsInitsMutable() {
    return mlir::MutableOperandRange(this->getOperation(), 1, 1);
  }
  mlir::LogicalResult
  reifyResultShapes(mlir::OpBuilder &builder,
                    mlir::ReifiedRankedShapedTypeDims &reified) {
    mlir::Value input = this->getOperation()->getOperand(0);
    reified.assign(1,
                   mlir::tensor::getMixedSizes(builder, this->getLoc(), input));
    return mlir::success();
  }
};

class TestOptInLikeOp final : public TestLikeOpBase<TestOptInLikeOp> {
public:
  using TestLikeOpBase::TestLikeOpBase;
  static llvm::StringRef getOperationName() { return "test_shape_opt_in.like"; }
};

class TestUnmarkedLikeOp final : public TestLikeOpBase<TestUnmarkedLikeOp> {
public:
  using TestLikeOpBase::TestLikeOpBase;
  static llvm::StringRef getOperationName() {
    return "test_shape_unmarked.like";
  }
};

/// A non-HIP dialect that opts into --hip-infer-shapes.
class TestShapeOptInDialect final : public mlir::Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShapeOptInDialect)

  explicit TestShapeOptInDialect(mlir::MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                mlir::TypeID::get<TestShapeOptInDialect>()) {
    getContext()->loadDialect<mlir::tensor::TensorDialect>();
    addOperations<TestOptInLikeOp>();
    addInterfaces<mlir::hip::ShapeInferenceDialectInterface>();
  }
  static llvm::StringRef getDialectNamespace() { return "test_shape_opt_in"; }
};

/// The same operation in a dialect that does not opt in.
class TestShapeUnmarkedDialect final : public mlir::Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShapeUnmarkedDialect)

  explicit TestShapeUnmarkedDialect(mlir::MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                mlir::TypeID::get<TestShapeUnmarkedDialect>()) {
    getContext()->loadDialect<mlir::tensor::TensorDialect>();
    addOperations<TestUnmarkedLikeOp>();
  }
  static llvm::StringRef getDialectNamespace() { return "test_shape_unmarked"; }
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
    RegisteredOperationName::insert<TestMalformedReifyDpsOp>(*dialect);
  });
  registry.insert<TestShapeOptInDialect, TestShapeUnmarkedDialect>();
  PassRegistration<TestHipDpsDefaultReifyPass>();
}
