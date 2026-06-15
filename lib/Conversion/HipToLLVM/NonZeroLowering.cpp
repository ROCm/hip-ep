/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y, count) {input_data_type}
//   -> wrap_nonzero(state, input_ptr, output_ptr, count_ptr,
//                   input_num_elements, input_rank, input_dims,
//                   output_capacity, input_data_type)
//
// `count_ptr` is the device i32 scalar buffer (the `count` DPS init) that the
// kernel writes with the true number of non-zero elements.
// `input_num_elements` is the total count of input elements (product of
// input dims, supports static + dynamic shapes via MemRefDescriptor).
// `input_rank` is the static rank of the input (R), passed as a compile-
// time i64 constant.
// `input_dims` is a host i64[R] array of the input extents (the kernel copies
// it to the device to decompose flat indices into multi-dim coordinates). It
// is materialised here as a stack alloca filled with per-dim sizes.
// `output_capacity` is the size of the dynamic N dim of the output (i.e.,
// the upper-bound number of nonzero entries materialised by the conversion
// pass). For NonZero today this equals input_num_elements, but lowering
// computes it directly from the output descriptor so the path is correct
// even if the conversion ever uses a smaller upper bound.
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
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero lowering expects ranked memref operands");
    if (outputType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero result memref must be rank-2");

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);
    Value countPtr =
        extractContiguousMemRefPtr(adaptor.getCount(), rewriter, loc);

    // Total input element count (handles static + dynamic dims).
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    // R is a static property of the input memref type.
    int64_t rank = inputType.getRank();
    Value inputRank = createI64Const(rank);

    // Host i64[R] array of input extents. Stack-allocated in the generated
    // (host) inference_compute; the runtime copies it H2D.
    Value dimsPtr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, inputRank);
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      Value dimVal = getMemRefDimSize(inputType, /*dimIdx=*/i, adaptor.getX(),
                                      rewriter, loc);
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, dimsPtr,
                                      ValueRange{createI64Const(i)});
      LLVM::StoreOp::create(rewriter, loc, dimVal, gep);
    }

    // Output capacity = N dim (data-dependent at the source level; the
    // tensor.empty in OnnxToHip pins it to numel(X) upper-bound, but read
    // from the descriptor here for correctness under any future re-size).
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
                                  dimsPtr,  outputCapacity,   inputDataTypeVal};
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
