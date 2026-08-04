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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct MatMulShapeArgs : ShapeRegionArgs {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getA() const { return *in(0); }
  Value getB() const { return *in(1); }
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

void MatMulOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  MLIRContext *ctx = builder.getContext();
  MatMulShapeArgs args{shapeBlock};
  Value aArg = args.getA();
  Value bArg = args.getB();
  Value shA = builder.create<shape::ShapeOfOp>(loc, aArg);
  Value shB = builder.create<shape::ShapeOfOp>(loc, bArg);
  int64_t aRank = cast<ShapedType>(aArg.getType()).getRank();
  int64_t bRank = cast<ShapedType>(bArg.getType()).getRank();
  bool aIs1D = aRank == 1;
  bool bIs1D = bRank == 1;

  auto extent = [&](OpBuilder &b, Value shape, int64_t idx) -> Value {
    Value c = b.create<arith::ConstantIndexOp>(loc, idx);
    return b.create<shape::GetExtentOp>(loc, shape, c);
  };

  // K is A's last dim; B's last dim when B is 1-D, else its second-to-last.
  int64_t kAIdx = aRank - 1;
  int64_t kBIdx = bIs1D ? bRank - 1 : bRank - 2;
  auto shapeTy = shape::ShapeType::get(ctx);
  Value sa = builder.create<shape::FromExtentsOp>(
      loc, shapeTy, ValueRange{extent(builder, shA, kAIdx)});
  Value sb = builder.create<shape::FromExtentsOp>(
      loc, shapeTy, ValueRange{extent(builder, shB, kBIdx)});
  Value witness = builder.create<shape::CstrEqOp>(
      loc, shape::WitnessType::get(ctx), ValueRange{sa, sb});

  // Output dims go inside the region so they hold under the K-equal assumption.
  auto assuming = builder.create<shape::AssumingOp>(
      loc, witness, [&](OpBuilder &b, Location) -> SmallVector<Value, 2> {
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        // Broadcast two batch dims: a dim of 1 takes the other side.
        auto broadcastDim = [&](Value da, Value db) -> Value {
          Value aIs1 =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, da, one);
          return b.create<arith::SelectOp>(loc, aIs1, db, da);
        };

        // A 1-D operand acts as rank 2 for the multiply (A: (1,K), B: (K,1));
        // the effective rank drives the batch-dim count below.
        int64_t aEffRank = aIs1D ? 2 : aRank;
        int64_t bEffRank = bIs1D ? 2 : bRank;

        // M from A, N from B. A 1-D operand contributes neither (its promoted
        // unit dim is stripped from the result).
        Value m = aIs1D ? Value() : extent(b, shA, aRank - 2);
        Value n = bIs1D ? Value() : extent(b, shB, bRank - 1);

        // Batch dims: the leading dims of each operand, right-aligned and
        // broadcast. The result's batch rank is the larger of the two.
        int64_t aBatch = aEffRank - 2;
        int64_t bBatch = bEffRank - 2;
        int64_t batchRank = std::max(aBatch, bBatch);
        SmallVector<Value, 2> dims;
        dims.reserve(batchRank + 2);
        for (int64_t i : llvm::seq<int64_t>(0, batchRank)) {
          // Right-align: a shorter operand has an implicit 1 at its missing
          // leading axes, so a negative index means "that side is 1, take the
          // other".
          int64_t aAxis = i - (batchRank - aBatch);
          int64_t bAxis = i - (batchRank - bBatch);
          if (aAxis < 0) {
            dims.push_back(extent(b, shB, bAxis));
          } else if (bAxis < 0) {
            dims.push_back(extent(b, shA, aAxis));
          } else {
            dims.push_back(
                broadcastDim(extent(b, shA, aAxis), extent(b, shB, bAxis)));
          }
        }
        // Append M then N, skipping the promoted (later-stripped) unit dims.
        if (m) {
          dims.push_back(m);
        }
        if (n) {
          dims.push_back(n);
        }
        return dims;
      });

  Type elemTy = cast<ShapedType>(getResult(0).getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{assuming.getResults()},
                               TypeRange{elemTy});
}

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
