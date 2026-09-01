/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== hipBLASLt ops =========================================================

// hip.hipblaslt.matmul(handle) ins(A, B) outs(C)
//   -> hip_hipblaslt_matmul(handle, A, B, C, rankA, rankB, batch, M, K, N)
// Rank-generic: batch from A if 3D, B broadcast if rankB < rankA.
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

    // Helper: create i64 constant
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Use extractMemRefDataPtr (alignedPtr + GEP(offset)) so that subview
    // operands with non-zero descriptor offsets are correctly adjusted.
    // extractContiguousMemRefPtr returns alignedPtr only and silently reads
    // the parent buffer base when offset != 0.
    Value statePtr = adaptor.getCtx();
    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    auto OutType = cast<MemRefType>(op.getOutput().getType());
    Value APtr = extractMemRefDataPtr(adaptor.getA(), AType, getTypeConverter(),
                                      rewriter, loc);
    Value BPtr = extractMemRefDataPtr(adaptor.getB(), BType, getTypeConverter(),
                                      rewriter, loc);
    Value outputPtr = extractMemRefDataPtr(adaptor.getOutput(), OutType,
                                           getTypeConverter(), rewriter, loc);

    // Get memref types and shapes
    int64_t ARank = AType.getRank();
    int64_t BRank = BType.getRank();
    int64_t transA = op.getTransA();
    int64_t transB = op.getTransB();

    // === DYNAMIC SHAPE SUPPORT ===
    // For dynamic shapes, we compute dimensions at runtime
    MemRefDescriptor ADesc(adaptor.getA());
    MemRefDescriptor BDesc(adaptor.getB());

    // Compute logical M, K, N from runtime dimensions.
    // A: [..., M, K] or [..., K, M] when transA; B: [..., K, N] or [..., N, K]
    // when transB.
    Value M = (ARank >= 2) ? (transA ? ADesc.size(rewriter, loc, ARank - 1)
                                     : ADesc.size(rewriter, loc, ARank - 2))
                           : createI64Const(1);
    Value K = transA ? ADesc.size(rewriter, loc, ARank - 2)
                     : ADesc.size(rewriter, loc, ARank - 1);
    Value N = transB ? BDesc.size(rewriter, loc, BRank - 2)
                     : BDesc.size(rewriter, loc, BRank - 1);

    // Compute batch count from leading dimensions of A
    Value batchCount;
    if (ARank == 2) {
      batchCount = createI64Const(1);
    } else {
      batchCount = ADesc.size(rewriter, loc, 0);
      for (int64_t i = 1; i < ARank - 2; ++i) {
        Value dim = ADesc.size(rewriter, loc, i);
        batchCount = LLVM::MulOp::create(rewriter, loc, batchCount, dim);
      }
    }

    // Compute element size in bytes
    unsigned elemBits = AType.getElementType().getIntOrFloatBitWidth();
    Value elemSize = createI64Const(elemBits / 8);

    // b_batch_stride: the per-batch stride (in elements) for hipBLASLt's
    // STRIDED_BATCH_OFFSET on layA when batch_count > 1. Two distinct B
    // memref shapes both reach this site with batch_count > 1 and need
    // DIFFERENT strides:
    //   * Rank-2 broadcast weight `[K, N]` → one matrix shared across all
    //     batches → stride = 0.
    //   * Rank-N per-batch weight `[d_0, ..., d_{N-3}, K, N]` whose leading
    //     dims hold more than one matrix → stride = K * N.
    //   * Rank-N weight whose leading dims multiply to 1 (e.g. `[1, K, N]`,
    //     `[1, 1, K, N]`) → still ONE matrix; stride = 0.
    // Previously encoded as a `b_batched` bool keyed on `BRank > 2`. That
    // misclassified the `[1, K, N]` leading-one case as per-batch, causing
    // OOB reads of `K*N` elements past the end of the weight on every batch
    // beyond the first.
    //
    // Leading-dim product is computed at compile time when all leading dims
    // are static (the common case for shipping models). When any leading dim
    // is dynamic we emit a runtime `select` over the leading-product so the
    // descriptor cache picks the right pre-built layout per call.
    //
    // Before:
    //   hip.matmul ins(%A, %B : memref<2x128x4096xf16>,
    //   memref<1x4096x1024xf16>)
    // After:
    //   %stride = llvm.mlir.constant(0 : i64) : i64   // leading product is 1
    //   llvm.call @wrap_hipblasLtMatmul(..., %stride) : (...) -> i32
    Value bBatchStride;
    if (BRank == 2) {
      bBatchStride = createI64Const(0);
    } else {
      bool allLeadingStatic = true;
      int64_t staticLeadingProduct = 1;
      for (int64_t i : llvm::seq<int64_t>(0, BRank - 2)) {
        if (BType.isDynamicDim(i)) {
          allLeadingStatic = false;
          break;
        }
        staticLeadingProduct *= BType.getDimSize(i);
      }
      if (allLeadingStatic) {
        bBatchStride = (staticLeadingProduct <= 1)
                           ? createI64Const(0)
                           : LLVM::MulOp::create(rewriter, loc, K, N).getRes();
      } else {
        Value one = createI64Const(1);
        Value zero = createI64Const(0);
        Value leadingProduct = one;
        for (int64_t i : llvm::seq<int64_t>(0, BRank - 2)) {
          Value dim = BDesc.size(rewriter, loc, i);
          leadingProduct =
              LLVM::MulOp::create(rewriter, loc, leadingProduct, dim).getRes();
        }
        Value isBroadcast = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::sle, leadingProduct, one);
        Value kn = LLVM::MulOp::create(rewriter, loc, K, N).getRes();
        bBatchStride =
            LLVM::SelectOp::create(rewriter, loc, isBroadcast, zero, kn)
                .getRes();
      }
    }

    // Runtime signature:
    // int wrap_hipblasLtMatmul(RuntimeState* state,
    //                          const void* A, const void* B, void* output,
    //                          int64_t M, int64_t N, int64_t K,
    //                          int64_t batch_count, int64_t elem_size,
    //                          int64_t b_batch_stride, int64_t transA,
    //                          int64_t transB, int op_state_slot)
    SmallVector<Type, 13> paramTypes = {
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
        i64Type, // b_batch_stride
        i64Type, // transA
        i64Type  // transB
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipblasltMatmul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 13> args = {statePtr,
                                   getOpStateSlotValue(op, rewriter, loc),
                                   APtr,
                                   BPtr,
                                   outputPtr,
                                   M,
                                   N,
                                   K,
                                   batchCount,
                                   elemSize,
                                   bBatchStride,
                                   createI64Const(transA),
                                   createI64Const(transB)};

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
