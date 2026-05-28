/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y, count_buf) {input_data_type}
//   -> wrap_nonzero(state, input_ptr, output_ptr, count_ptr,
//                   input_num_elements, input_rank,
//                   input_dims_ptr, output_capacity, input_data_type)
//
// `count_ptr` points to a single i32 in the GPU pool. The kernel writes
// the actual nonzero count via atomicAdd. Downstream ScatterND reads it
// directly on the GPU without D2H.
//
// `input_dims_ptr` is a host-side array of the input shape dims (needed
// by the kernel to decompose flat index into multi-dim coordinates).
struct NonZeroOpLowering : public ConvertOpToLLVMPattern<NonZeroOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NonZeroOp op, OpAdaptor adaptor,
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

    auto inputType = dyn_cast<MemRefType>(op.getX().getType());
    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    auto countType = dyn_cast<MemRefType>(op.getCountBuf().getType());
    if (!inputType || !outputType || !countType)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero lowering expects ranked memref operands");
    if (outputType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero result memref must be rank-2");

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);
    Value countPtr =
        extractContiguousMemRefPtr(adaptor.getCountBuf(), rewriter, loc);

    // Total input element count (handles static + dynamic dims).
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    // R is a static property of the input memref type.
    int64_t rank = inputType.getRank();
    Value inputRank = createI64Const(rank);

    // Build host-side array of input dims for coordinate decomposition.
    Value one = createI64Const(1);
    auto arrType =
        LLVM::LLVMArrayType::get(i64Type, std::max(rank, (int64_t)1));
    Value dimsArr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
    for (int64_t i = 0; i < rank; ++i) {
      Value dim = getMemRefDimSize(inputType, i, adaptor.getX(), rewriter, loc);
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, dimsArr, idx);
      LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
    }

    // Output capacity = N dim of the output memref.
    Value outputCapacity = getMemRefDimSize(outputType, /*dimIdx=*/1,
                                            adaptor.getY(), rewriter, loc);

    Value inputDataTypeVal = createI64Const(op.getInputDataType());

    // int wrap_nonzero(RuntimeState* state, void* input, void* output,
    //                  int32_t* count_ptr, int64_t input_num_elements,
    //                  int64_t input_rank, const int64_t* input_dims,
    //                  int64_t output_capacity, int64_t input_data_type)
    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type,
                                       ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNonZero, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, inputPtr,         outputPtr,
                                  countPtr, inputNumElements, inputRank,
                                  dimsArr,  outputCapacity,   inputDataTypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateNonZeroLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<NonZeroOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
