/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Dense custom GEMM ops =================================================

// hip.matmul lowers to the custom dense GEMM wrappers. The historical
// flat-batch symbol is retained for compatible serialized runtime calls.
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

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value APtr = extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc);
    Value BPtr = extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Get memref types and shapes
    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
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
    Value K = transA && ARank >= 2 ? ADesc.size(rewriter, loc, ARank - 2)
                                   : ADesc.size(rewriter, loc, ARank - 1);
    Value N = BRank == 1 ? createI64Const(1)
              : transB   ? BDesc.size(rewriter, loc, BRank - 2)
                         : BDesc.size(rewriter, loc, BRank - 1);

    // Compute batch count from leading dimensions of A
    Value batchCount;
    if (ARank <= 2) {
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
    Value elemSize = createI64Const(AType.getElementType().isBF16()
                                        ? -2
                                        : static_cast<int64_t>(elemBits / 8));

    // b_batch_stride: the per-batch stride in elements for the retained flat
    // runtime ABI when batch_count > 1. Two distinct B
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
    // flat wrapper uses the right source matrix per call.
    //
    // Before:
    //   hip.matmul ins(%A, %B : memref<2x128x4096xf16>,
    //   memref<1x4096x1024xf16>)
    // After:
    //   %stride = llvm.mlir.constant(0 : i64) : i64   // leading product is 1
    //   llvm.call @wrap_hipblasLtMatmul(..., %stride) : (...) -> i32
    Value bBatchStride;
    if (BRank <= 2) {
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

    // hipBLASLt reads a 4-byte element as f32. An i32 matmul has to take the
    // integer kernel instead, or the bits are interpreted as floats. The
    // kernel still needs B's batch stride: 0 reuses one matrix, K*N steps
    // to the next. Omitting it made a rank-2 B look batch-contiguous.
    if (AType.getElementType().isInteger(32)) {
      if (transA || transB)
        return op.emitError("hip.matmul of i32 does not support transpose");
      SmallVector<Type, 9> i32Params = {ptrType, ptrType, ptrType,
                                        ptrType, i64Type, i64Type,
                                        i64Type, i64Type, i64Type};
      FailureOr<LLVM::LLVMFuncOp> i32Fn = LLVM::lookupOrCreateFn(
          rewriter, module, kWrapMatmulI32, i32Params, i32Type);
      if (failed(i32Fn))
        return failure();
      SmallVector<Value, 9> i32Args = {
          statePtr, APtr, BPtr, outputPtr, M, N, K, batchCount, bBatchStride};
      LLVM::CallOp::create(rewriter, loc, *i32Fn, i32Args);
      rewriter.eraseOp(op);
      return success();
    }

    // Runtime signature:
    if (!isa<FloatType>(AType.getElementType()))
      return op.emitError("custom MatMul requires a floating dtype");

    // Keep the compact historical ABI for a tight A batch and a B batch that
    // is either identical or fully broadcast. General leading-axis broadcast
    // needs the actual per-axis dimensions; a single flat stride loses them.
    const int64_t aBatchRank = std::max<int64_t>(ARank - 2, 0);
    const int64_t bBatchRank = std::max<int64_t>(BRank - 2, 0);
    const int64_t batchRank = std::max(aBatchRank, bBatchRank);
    bool bBroadcast = true, identicalBatches = true;
    for (int64_t axis = 0; axis < batchRank; ++axis) {
      const int64_t ai = axis - (batchRank - aBatchRank);
      const int64_t bi = axis - (batchRank - bBatchRank);
      const int64_t ad = ai < 0 ? 1 : AType.getDimSize(ai);
      const int64_t bd = bi < 0 ? 1 : BType.getDimSize(bi);
      bBroadcast &= bd == 1;
      identicalBatches &=
          ad != ShapedType::kDynamic && bd != ShapedType::kDynamic && ad == bd;
    }
    const bool generalBroadcast = !bBroadcast && !identicalBatches;
    if (generalBroadcast) {
      auto emitBatchDims = [&](MemRefDescriptor &desc, int64_t rank) -> Value {
        auto arrayType =
            LLVM::LLVMArrayType::get(i64Type, std::max<int64_t>(rank, 1));
        Value array = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType,
                                             createI64Const(1), 8);
        for (int64_t i = 0; i < rank; ++i) {
          Value address = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                              array, createI64Const(i));
          LLVM::StoreOp::create(rewriter, loc, desc.size(rewriter, loc, i),
                                address);
        }
        return array;
      };
      SmallVector<Type, 15> types = {ptrType, i32Type, ptrType, ptrType,
                                     ptrType, i64Type, i64Type, i64Type,
                                     i64Type, i64Type, i64Type, i64Type,
                                     ptrType, i64Type, ptrType};
      auto function = LLVM::lookupOrCreateFn(
          rewriter, module, "wrap_matmul_broadcast", types, i32Type);
      if (failed(function))
        return failure();
      SmallVector<Value, 15> values = {statePtr,
                                       getOpStateSlotValue(op, rewriter, loc),
                                       APtr,
                                       BPtr,
                                       outputPtr,
                                       M,
                                       N,
                                       K,
                                       elemSize,
                                       createI64Const(transA),
                                       createI64Const(transB),
                                       createI64Const(aBatchRank),
                                       emitBatchDims(ADesc, aBatchRank),
                                       createI64Const(bBatchRank),
                                       emitBatchDims(BDesc, bBatchRank)};
      LLVM::CallOp::create(rewriter, loc, *function, values);
      rewriter.eraseOp(op);
      return success();
    }

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
