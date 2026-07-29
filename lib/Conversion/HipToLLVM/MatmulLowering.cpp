/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "hip/Dialect/IR/HipShapeUtils.h"

namespace mlir {
namespace hip {
namespace {

// Lower `hip.matmul` to the hipBLASLt runtime ABI. Each operand gets an
// independent batch stride so either whole matrix may be broadcast.
//
// Before:
//   hip.matmul ins(%a, %b : memref<128x4096xf16>,
//                           memref<2x4096x1024xf16>) outs(%out : ...)
// After:
//   %a_stride = llvm.mlir.constant(0 : i64) : i64
//   %b_stride = llvm.mul %k, %n : i64
//   llvm.call @wrap_hipblasLtMatmul(..., %a_stride, %b_stride)
struct MatmulOpLowering : public ConvertOpToLLVMPattern<MatmulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value APtr = extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc);
    Value BPtr = extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t ARank = AType.getRank();
    int64_t BRank = BType.getRank();
    int64_t outputRank = outputType.getRank();
    if (failed(verifyStridedBatchMatmul(AType.getShape(), BType.getShape(),
                                        [&]() { return op.emitOpError(); })))
      return failure();

    MemRefDescriptor ADesc(adaptor.getA());
    MemRefDescriptor BDesc(adaptor.getB());
    MemRefDescriptor outputDesc(adaptor.getOutput());

    Value M = ADesc.size(rewriter, loc, ARank - 2);
    Value K = ADesc.size(rewriter, loc, ARank - 1);
    Value N = BDesc.size(rewriter, loc, BRank - 1);

    // The output shape already contains the broadcasted leading dimensions.
    Value batchCount = createI64Const(1);
    for (int64_t i : llvm::seq<int64_t>(0, outputRank - 2)) {
      Value dim = outputDesc.size(rewriter, loc, i);
      batchCount = LLVM::MulOp::create(rewriter, loc, batchCount, dim);
    }

    unsigned elemBits = AType.getElementType().getIntOrFloatBitWidth();
    Value elemSize = createI64Const(elemBits / 8);

    // Each operand independently uses stride 0 when it contains one matrix,
    // or its matrix size when it contains one matrix per output batch.
    auto computeBatchStride = [&](MemRefType type, MemRefDescriptor desc,
                                  Value matrixElements) -> Value {
      int64_t rank = type.getRank();
      if (rank == 2)
        return createI64Const(0);

      bool allLeadingStatic = true;
      int64_t staticLeadingProduct = 1;
      for (int64_t i : llvm::seq<int64_t>(0, rank - 2)) {
        if (type.isDynamicDim(i)) {
          allLeadingStatic = false;
          break;
        }
        staticLeadingProduct *= type.getDimSize(i);
      }
      if (allLeadingStatic)
        return staticLeadingProduct <= 1 ? createI64Const(0) : matrixElements;

      Value one = createI64Const(1);
      Value leadingProduct = one;
      for (int64_t i : llvm::seq<int64_t>(0, rank - 2)) {
        Value dim = desc.size(rewriter, loc, i);
        leadingProduct =
            LLVM::MulOp::create(rewriter, loc, leadingProduct, dim).getRes();
      }
      Value isBroadcast = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::sle, leadingProduct, one);
      return LLVM::SelectOp::create(rewriter, loc, isBroadcast,
                                    createI64Const(0), matrixElements)
          .getRes();
    };
    Value aMatrixElements = LLVM::MulOp::create(rewriter, loc, M, K);
    Value bMatrixElements = LLVM::MulOp::create(rewriter, loc, K, N);
    Value aBatchStride = computeBatchStride(AType, ADesc, aMatrixElements);
    Value bBatchStride = computeBatchStride(BType, BDesc, bMatrixElements);

    // Runtime signature:
    // int wrap_hipblasLtMatmul(RuntimeState* state,
    //                          const void* A, const void* B, void* output,
    //                          int64_t M, int64_t N, int64_t K,
    //                          int64_t batch_count, int64_t elem_size,
    //                          int64_t a_batch_stride,
    //                          int64_t b_batch_stride)
    SmallVector<Type, 12> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // A
        ptrType, // B
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // batch_count
        i64Type, // elem_size
        i64Type, // a_batch_stride
        i64Type  // b_batch_stride
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipblasltMatmul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {statePtr,
                                   getOpStateSlotValue(op, rewriter, loc),
                                   APtr,
                                   BPtr,
                                   outputPtr,
                                   M,
                                   N,
                                   K,
                                   batchCount,
                                   elemSize,
                                   aBatchStride,
                                   bBatchStride};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMatmulLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<MatmulOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
