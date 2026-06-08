/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "llvm/ADT/Sequence.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y, count_buf) {input_data_type}
//   -> wrap_nonzero(state, input_ptr, output_ptr, count_ptr,
//                   input_num_elements, input_rank,
//                   input_dims_ptr, output_capacity, input_data_type)
//
// `count_ptr` points to the single-i32 count_buf in the GPU pool. The kernel
// writes the actual nonzero count there via atomicAdd; a downstream ScatterND
// reads it directly on the GPU (no D2H) to bound the rows it processes.
//
// `input_dims_ptr` is a stack array of the input shape dims, used by the kernel
// to decompose a flat non-zero index into multi-dim coordinates. Static dims
// are compile-time constants; dynamic dims are read from the memref descriptor.
//
// `input_num_elements` is the product of input dims (static + dynamic via the
// MemRefDescriptor). `input_rank` is the static rank R. `output_capacity` is
// the N dim of the output (the upper-bound nonzero count the conversion pinned
// to numel(X)), read from the output descriptor so the path stays correct even
// if a future conversion uses a smaller upper bound.
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

    // Stack array of input dims for the kernel's flat->multi-dim decomposition.
    Value one = createI64Const(1);
    auto arrType =
        LLVM::LLVMArrayType::get(i64Type, std::max(rank, (int64_t)1));
    Value dimsArr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
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
