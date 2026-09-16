/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ConvertShapeToExtentPass.cpp - Shape types to extent tensors -------===//
//
// Retypes shape values so the upstream shape passes can lower them.
// !shape.shape becomes tensor<Nxindex>, and !shape.size becomes index.
//
// --convert-shape-to-std only matches those new types, so it does nothing
// until this pass runs. This pass lowers four ops itself, because upstream has
// no pattern for them: shape.from_extents, shape.concat, shape.size_to_index,
// and shape.const_size.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTSHAPETOEXTENTPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Rank inference
//===----------------------------------------------------------------------===//

// Returns how many extents each shape result of `op` holds, or kDynamic when
// the already converted `operands` do not say.
//
// An op missing below gets tensor<?xindex>. shape.split_at is one: upstream
// lowers it to tensor.extract_slice, which returns tensor<?xindex> whatever
// type we ask for.
int64_t inferExtentCount(Operation *op, ValueRange operands) {
  return llvm::TypeSwitch<Operation *, int64_t>(op)
      .Case([&](shape::ShapeOfOp) {
        auto argType = dyn_cast<ShapedType>(operands.front().getType());
        return argType && argType.hasRank() ? argType.getRank()
                                            : ShapedType::kDynamic;
      })
      .Case([](shape::ConstShapeOp constShape) {
        return static_cast<int64_t>(constShape.getShape().size());
      })
      .Case([&](shape::BroadcastOp) {
        auto extentCount = [](Value shape) {
          if (!shape::isExtentTensorType(shape.getType())) {
            return ShapedType::kDynamic;
          }
          return cast<RankedTensorType>(shape.getType()).getDimSize(0);
        };
        // Broadcasting left-pads shorter shapes with ones, so the result is as
        // long as the longest operand.
        SmallVector<int64_t> counts =
            llvm::map_to_vector(operands, extentCount);
        if (ShapedType::isDynamicShape(counts)) {
          return ShapedType::kDynamic;
        }
        return *llvm::max_element(counts);
      })
      .Case([&](shape::FromExtentsOp) {
        return static_cast<int64_t>(operands.size());
      })
      .Default([](Operation *) { return ShapedType::kDynamic; });
}

//===----------------------------------------------------------------------===//
// Retyping the operations upstream can already lower
//===----------------------------------------------------------------------===//

// Retypes an op and leaves the real lowering to --convert-shape-to-std.
//
// Operands and results have to change together. shape's verifySizeOrIndexOp
// keeps a result opaque while any operand can still carry an error.
//
// Before: %0 = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
// After:  %0 = shape.shape_of %a : tensor<?x4xf16> -> tensor<2xindex>
template <typename OpTy>
struct RetypeShapeOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    int64_t extentCount = inferExtentCount(op, operands);

    // !shape.shape becomes an extent tensor of `extentCount` extents.
    // !shape.size becomes index. Every other type stays as it is.
    SmallVector<Type> resultTypes =
        llvm::map_to_vector(op->getResultTypes(), [extentCount](Type type) {
          return llvm::TypeSwitch<Type, Type>(type)
              .Case([&](shape::ShapeType shapeType) {
                return shape::getExtentTensorType(shapeType.getContext(),
                                                  extentCount);
              })
              .Case([](shape::SizeType sizeType) {
                return IndexType::get(sizeType.getContext());
              })
              .Default(type);
        });
    Operation *retyped =
        rewriter.create(op.getLoc(), op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
    rewriter.replaceOp(op, retyped->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowering the operations upstream has no pattern for
//===----------------------------------------------------------------------===//

// Before: %0 = shape.from_extents %a, %b : index, index
// After:  %0 = tensor.from_elements %a, %b : tensor<2xindex>
struct LowerFromExtents : public OpConversionPattern<shape::FromExtentsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::FromExtentsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange extents = adaptor.getExtents();
    if (!llvm::all_of(extents.getTypes(), llvm::IsaPred<IndexType>)) {
      return rewriter.notifyMatchFailure(op, "extents are not indices yet");
    }
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op,
        shape::getExtentTensorType(getContext(), inferExtentCount(op, extents)),
        extents);
    return success();
  }
};

// Before: %0 = shape.concat %a, %b : tensor<2xindex>, tensor<1xindex>
//                                    -> !shape.shape
// After:  %0 = tensor.concat dim(0) %a, %b
//             : (tensor<2xindex>, tensor<1xindex>) -> tensor<3xindex>
struct LowerConcat : public OpConversionPattern<shape::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getOperands();
    if (!llvm::all_of(inputs.getTypes(), shape::isExtentTensorType)) {
      return rewriter.notifyMatchFailure(op, "operands are not extent tensors "
                                             "yet");
    }
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(
        op, tensor::ConcatOp::inferResultType(/*dim=*/0, inputs.getTypes()),
        /*dim=*/0, inputs);
    return success();
  }
};

// Before: %0 = shape.size_to_index %e : index
// After:  every use reads %e directly
struct LowerSizeToIndex : public OpConversionPattern<shape::SizeToIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::SizeToIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<IndexType>(adaptor.getArg().getType())) {
      return rewriter.notifyMatchFailure(op, "operand is not an index yet");
    }
    rewriter.replaceOp(op, adaptor.getArg());
    return success();
  }
};

// ODS pins the result of shape.const_size to !shape.size, so it cannot be
// retyped. Leaving it alone does not work either: verifySizeOrIndexOp turns
// every consumer's result back into a shape type.
//
// Before: %0 = shape.const_size 2
// After:  %0 = arith.constant 2 : index
struct LowerConstSize : public OpConversionPattern<shape::ConstSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::ConstSizeOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
        op, op.getValue().getSExtValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertShapeToExtentPass
    : impl::ConvertShapeToExtentPassBase<ConvertShapeToExtentPass> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    // Inline the shape regions so the conversion below sees straight-line
    // code. Upstream has no pattern for scf.execute_region.
    //
    // This runs fewer patterns than --canonicalize, with folding off, because
    // two of them break shape-typed IR:
    // - ShapeOfOpToConstShapeOp builds a tensor.cast to !shape.shape.
    // - BroadcastOp::fold returns an index tensor attribute for any result
    //   type.
    //
    // Folding off also leaves the hipsr.compute bodies alone.
    RewritePatternSet inliners(context);
    scf::ExecuteRegionOp::getCanonicalizationPatterns(inliners, context);
    shape::AssumingOp::getCanonicalizationPatterns(inliners, context);
    if (failed(applyPatternsGreedily(
            function, std::move(inliners),
            GreedyRewriteConfig().enableFolding(false)))) {
      return signalPassFailure();
    }

    // True when `op` no longer uses !shape.shape or !shape.size.
    auto isRetyped = [](Operation *op) {
      constexpr auto isOpaque =
          llvm::IsaPred<shape::ShapeType, shape::SizeType>;
      return llvm::none_of(op->getOperandTypes(), isOpaque) &&
             llvm::none_of(op->getResultTypes(), isOpaque);
    };

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalDialect<shape::ShapeDialect>(isRetyped);
    target.addDynamicallyLegalOp<PreserveShapeOp>(isRetyped);
    target.addIllegalOp<shape::ConcatOp, shape::ConstSizeOp,
                        shape::FromExtentsOp, shape::SizeToIndexOp>();

    RewritePatternSet patterns(context);
    patterns
        .add<LowerConcat, LowerConstSize, LowerFromExtents, LowerSizeToIndex>(
            context);
    patterns.add<
        RetypeShapeOp<shape::BroadcastOp>, RetypeShapeOp<shape::ConstShapeOp>,
        RetypeShapeOp<shape::CstrBroadcastableOp>,
        RetypeShapeOp<shape::CstrEqOp>, RetypeShapeOp<shape::GetExtentOp>,
        RetypeShapeOp<shape::NumElementsOp>, RetypeShapeOp<shape::ShapeOfOp>,
        RetypeShapeOp<shape::SplitAtOp>, RetypeShapeOp<PreserveShapeOp>>(
        context);

    if (failed(applyPartialConversion(function, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
