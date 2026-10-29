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
// Rank-generic: batch from the larger leading product, the other operand
// broadcast when its leading dims multiply to 1.
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
    Value K = transA ? ADesc.size(rewriter, loc, ARank - 2)
                     : ADesc.size(rewriter, loc, ARank - 1);
    Value N = transB ? BDesc.size(rewriter, loc, BRank - 2)
                     : BDesc.size(rewriter, loc, BRank - 1);

    // Batch layout. One strided-batched GEMM covers three shapes:
    //   * equal batch dims: A and B both per-batch;
    //   * B's leading dims multiply to 1 (`[K, N]`, `[1, K, N]`, ...): one
    //     weight shared by every batch of A, B stride 0;
    //   * A's leading dims multiply to 1 (e.g. `[1, 1, T, K] x [1, H, K, N]`):
    //     one activation against per-batch weights, A stride 0.
    // The batch count is the larger leading product. Any other broadcast
    // (`[2, 1, M, K] x [1, 3, K, N]`) needs one stride per batch dim and is
    // rejected when the shapes show it. A per-batch operand advances M*K
    // (A) or K*N (B); stepping a single matrix by that reads past its end on
    // every batch after the first, and stride 0 on a per-batch operand makes
    // every batch read matrix 0.
    //
    // Leading-dim products fold at compile time when all leading dims are
    // static (the common case for shipping models); otherwise the stride and
    // count are runtime `select`s over the products, and the descriptor
    // cache picks the matching layout per call.
    //
    // Before:
    //   hip.matmul ins(%A, %B : memref<2x128x4096xf16>,
    //   memref<1x4096x1024xf16>)
    // After:
    //   %stride = llvm.mlir.constant(0 : i64) : i64   // leading product is 1
    //   llvm.call @wrap_hipblasLtMatmul(..., %stride, ...) : (...) -> i32
    struct Leading {
      std::optional<int64_t> staticProduct;
      Value product; // set only when a leading dim is dynamic
    };
    auto leadingProduct = [&](MemRefType type, MemRefDescriptor &desc,
                              int64_t rank) -> Leading {
      std::optional<int64_t> staticProduct = 1;
      for (int64_t i : llvm::seq<int64_t>(0, rank - 2)) {
        if (type.isDynamicDim(i)) {
          staticProduct.reset();
          break;
        }
        *staticProduct *= type.getDimSize(i);
      }
      if (staticProduct)
        return {staticProduct, Value()};
      Value product = createI64Const(1);
      for (int64_t i : llvm::seq<int64_t>(0, rank - 2))
        product = LLVM::MulOp::create(rewriter, loc, product,
                                      desc.size(rewriter, loc, i))
                      .getRes();
      return {std::nullopt, product};
    };
    auto productValue = [&](const Leading &lead) -> Value {
      return lead.product ? lead.product : createI64Const(*lead.staticProduct);
    };
    Leading aLead = leadingProduct(AType, ADesc, ARank);
    Leading bLead = leadingProduct(BType, BDesc, BRank);

    if (aLead.staticProduct && bLead.staticProduct &&
        *aLead.staticProduct > 1 && *bLead.staticProduct > 1) {
      ArrayRef<int64_t> aBatch = AType.getShape().drop_back(2);
      ArrayRef<int64_t> bBatch = BType.getShape().drop_back(2);
      size_t batchRank = std::max(aBatch.size(), bBatch.size());
      auto aligned = [&](ArrayRef<int64_t> batch, size_t i) -> int64_t {
        size_t pad = batchRank - batch.size();
        return i < pad ? 1 : batch[i - pad];
      };
      for (size_t i = 0; i < batchRank; ++i) {
        if (aligned(aBatch, i) != aligned(bBatch, i))
          return op.emitOpError(
              "batch broadcast between two multi-matrix operands is not "
              "supported by the strided-batched GEMM");
      }
    }

    auto batchStride = [&](const Leading &lead, Value rows,
                           Value cols) -> Value {
      if (lead.staticProduct && *lead.staticProduct <= 1)
        return createI64Const(0);
      Value matrixElems =
          LLVM::MulOp::create(rewriter, loc, rows, cols).getRes();
      if (lead.staticProduct)
        return matrixElems;
      Value isBroadcast =
          LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sle,
                               lead.product, createI64Const(1));
      return LLVM::SelectOp::create(rewriter, loc, isBroadcast,
                                    createI64Const(0), matrixElems)
          .getRes();
    };
    Value batchCount;
    if (aLead.staticProduct && bLead.staticProduct) {
      batchCount =
          createI64Const(std::max(*aLead.staticProduct, *bLead.staticProduct));
    } else if (bLead.staticProduct && *bLead.staticProduct <= 1) {
      batchCount = aLead.product;
    } else if (aLead.staticProduct && *aLead.staticProduct <= 1) {
      batchCount = bLead.product;
    } else {
      Value aProduct = productValue(aLead), bProduct = productValue(bLead);
      Value aLarger = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::sge, aProduct, bProduct);
      batchCount =
          LLVM::SelectOp::create(rewriter, loc, aLarger, aProduct, bProduct)
              .getRes();
    }
    Value bBatchStride = batchStride(bLead, K, N);
    Value aBatchStride = batchStride(aLead, M, K);

    // Compute element size in bytes
    unsigned elemBits = AType.getElementType().getIntOrFloatBitWidth();
    Value elemSize = createI64Const(elemBits / 8);

    // Runtime signature:
    // int wrap_hipblasLtMatmul(RuntimeState* state,
    //                          const void* A, const void* B, void* output,
    //                          int64_t M, int64_t N, int64_t K,
    //                          int64_t batch_count, int64_t elem_size,
    //                          int64_t b_batch_stride, int64_t a_batch_stride,
    //                          int64_t transA, int64_t transB,
    //                          int op_state_slot)
    SmallVector<Type, 14> paramTypes = {
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
        i64Type, // a_batch_stride
        i64Type, // transA
        i64Type  // transB
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipblasltMatmul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {statePtr,
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
                                   aBatchStride,
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
