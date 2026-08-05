/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct MatMulPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getA() const { return in(0); }
  Value getB() const { return in(1); }
};
} // namespace

MutableOperandRange MatMulOp::getDpsInitsMutable() { return getInitMutable(); }

// A and B must be at least 1-D: matmul needs a contraction dim, and the shape
// region reads the last one or two dims of each.
LogicalResult MatMulOp::verify() {
  if (cast<ShapedType>(getA().getType()).getRank() < 1) {
    return emitOpError("operand A must be at least 1-D");
  }
  if (cast<ShapedType>(getB().getType()).getRank() < 1) {
    return emitOpError("operand B must be at least 1-D");
  }
  return success();
}

void MatMulOp::populateShapeRegion(OpBuilder &, Block &) {}

namespace mlir {
namespace hipsr {

LogicalResult populateMatMulShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                        MatMulOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  MatMulPlaceholderShapeArgs args{shapeBlock};
  Value aShape = args.getA();
  Value bShape = args.getB();
  int64_t aRank = cast<ShapedType>(op.getA().getType()).getRank();
  int64_t bRank = cast<ShapedType>(op.getB().getType()).getRank();
  bool aIs1D = aRank == 1;
  bool bIs1D = bRank == 1;

  auto extent = [&](OpBuilder &b, Value inputShape, int64_t index) -> Value {
    return shape::GetExtentOp::create(b, loc, inputShape, index);
  };

  // K is A's last dim; B's last dim when B is 1-D, else its second-to-last.
  int64_t kAIndex = aRank - 1;
  int64_t kBIndex = bIs1D ? bRank - 1 : bRank - 2;
  Type shapeType = shape::ShapeType::get(ctx);
  Type witnessType = shape::WitnessType::get(ctx);
  Value aKShape = shape::FromExtentsOp::create(
      builder, loc, shapeType, ValueRange{extent(builder, aShape, kAIndex)});
  Value bKShape = shape::FromExtentsOp::create(
      builder, loc, shapeType, ValueRange{extent(builder, bShape, kBIndex)});
  Value kWitness = shape::CstrEqOp::create(builder, loc, witnessType,
                                           ValueRange{aKShape, bKShape});

  // Only leading dimensions participate in MatMul batch broadcasting.
  auto getBatchShape = [&](Value inputShape, int64_t rank) -> Value {
    int64_t batchRank = std::max<int64_t>(rank - 2, 0);
    Value splitIndex = shape::ConstSizeOp::create(builder, loc, batchRank);
    return shape::SplitAtOp::create(builder, loc,
                                    TypeRange{shapeType, shapeType}, inputShape,
                                    splitIndex)
        .getHead();
  };
  Value aBatchShape = getBatchShape(aShape, aRank);
  Value bBatchShape = getBatchShape(bShape, bRank);
  Value batchWitness = shape::CstrBroadcastableOp::create(
      builder, loc, aBatchShape, bBatchShape);
  Value witness = shape::AssumingAllOp::create(
      builder, loc, witnessType, ValueRange{kWitness, batchWitness});

  auto assuming = shape::AssumingOp::create(
      builder, loc, witness,
      [&](OpBuilder &b, Location) -> SmallVector<Value, 2> {
        Value batchShape = shape::BroadcastOp::create(
            b, loc, shapeType, aBatchShape, bBatchShape, StringAttr{});

        SmallVector<Value, 2> matrixExtents;
        if (!aIs1D) {
          matrixExtents.push_back(extent(b, aShape, aRank - 2));
        }
        if (!bIs1D) {
          matrixExtents.push_back(extent(b, bShape, bRank - 1));
        }
        Value matrixShape = shape::FromExtentsOp::create(
            b, loc, shapeType, ValueRange{matrixExtents});
        Value resultShape =
            shape::ConcatOp::create(b, loc, shapeType, batchShape, matrixShape);
        return {resultShape};
      });

  ShapeYield2Op::create(builder, loc, ValueRange{assuming.getResult(0)});
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapHipblasLtMatmul = "wrap_hipblasLtMatmul";

struct MatMulLowering : ConvertOpToLLVMPattern<MatMulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto aType = dyn_cast<MemRefType>(op.getA().getType());
    auto bType = dyn_cast<MemRefType>(op.getB().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!aType || !bType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }

    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    MemRefDescriptor aDesc(adaptor.getA());
    MemRefDescriptor bDesc(adaptor.getB());
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();

    Value m =
        aRank == 1 ? createI64Const(1) : aDesc.size(rewriter, loc, aRank - 2);
    Value k = aDesc.size(rewriter, loc, aRank - 1);
    Value n =
        bRank == 1 ? createI64Const(1) : bDesc.size(rewriter, loc, bRank - 1);

    Value batchCount = createI64Const(1);
    for (int64_t i : llvm::seq<int64_t>(0, std::max<int64_t>(aRank - 2, 0))) {
      batchCount = LLVM::MulOp::create(rewriter, loc, batchCount,
                                       aDesc.size(rewriter, loc, i));
    }

    Value bBatchStride;
    if (bRank <= 2) {
      bBatchStride = createI64Const(0);
    } else {
      bool allLeadingStatic = true;
      int64_t staticLeadingProduct = 1;
      for (int64_t i : llvm::seq<int64_t>(0, bRank - 2)) {
        if (bType.isDynamicDim(i)) {
          allLeadingStatic = false;
          break;
        }
        staticLeadingProduct *= bType.getDimSize(i);
      }
      if (allLeadingStatic) {
        bBatchStride =
            staticLeadingProduct <= 1
                ? createI64Const(0)
                : LLVM::MulOp::create(rewriter, loc, k, n).getResult();
      } else {
        Value one = createI64Const(1);
        Value zero = createI64Const(0);
        Value leadingProduct = one;
        for (int64_t i : llvm::seq<int64_t>(0, bRank - 2)) {
          leadingProduct = LLVM::MulOp::create(rewriter, loc, leadingProduct,
                                               bDesc.size(rewriter, loc, i));
        }
        Value isBroadcast = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::sle, leadingProduct, one);
        Value kn = LLVM::MulOp::create(rewriter, loc, k, n);
        bBatchStride =
            LLVM::SelectOp::create(rewriter, loc, isBroadcast, zero, kn);
      }
    }

    int64_t elementSize = aType.getElementType().getIntOrFloatBitWidth() / 8;
    using MatMulCall =
        RuntimeFunc<i32, hostPtr, slotIndex, devicePtr, devicePtr, devicePtr,
                    i64, i64, i64, i64, i64, i64>;
    auto matMulFunc = MatMulCall::lookupOrCreateFn(rewriter, loc, module,
                                                   kWrapHipblasLtMatmul);
    if (failed(matMulFunc)) {
      return failure();
    }
    if (failed(matMulFunc->call(adaptor.getCtx(), SlotIndex{op.getOperation()},
                                adaptor.getA(), adaptor.getB(),
                                adaptor.getInit(), m, n, k, batchCount,
                                elementSize, bBatchStride))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrMatMulLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MatMulLowering>(converter);
}
