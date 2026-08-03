/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange MatMulOp::getDpsInitsMutable() { return getInitMutable(); }

// A and B must be at least 1-D because matmul needs a contraction dimension.
LogicalResult MatMulOp::verify() {
  if (cast<ShapedType>(getA().getType()).getRank() < 1) {
    return emitOpError("operand A must be at least 1-D");
  }
  if (cast<ShapedType>(getB().getType()).getRank() < 1) {
    return emitOpError("operand B must be at least 1-D");
  }
  return success();
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
