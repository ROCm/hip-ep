/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y) {input_data_type}
//   -> wrap_nonzero(state, input_ptr, output_ptr, /*count_ptr=*/null,
//                   input_num_elements, input_rank, input_dims_ptr,
//                   output_capacity, input_data_type)
//
// `count_ptr` is passed as a null pointer: the runtime owns a per-state
// single-int32 scratch for the data-dependent count (it is consumed GPU-side
// and never needs to be an explicit IR value).
// `input_dims_ptr` is a host-stack array (alloca) of the input shape dims; the
// kernel uses it to decompose a flat index into multi-dim coordinates.
// `input_num_elements` is the total input element count (product of input
// dims; supports static + dynamic via the MemRef descriptor).
// `input_rank` is the static input rank R (compile-time i64 constant).
// `output_capacity` is the dynamic N dim of the output, read from its
// descriptor (today == input_num_elements, but read directly so a smaller
// future upper bound still lowers correctly).
//
// Before:
//   hip.nonzero(%ctx) ins(%x : memref<3x4xi1,1>) outs(%y : memref<2x?xi64,1>)
// After:
//   %dims = llvm.alloca : !llvm.ptr           // [3, 4]
//   llvm.call @wrap_nonzero(%state, %xp, %yp, %null, %numel, %rank, %dims,
//                           %cap, %dtype)
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

    // count_ptr = null: the runtime owns the per-state count scratch.
    Value countPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Total input element count (handles static + dynamic dims).
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    // R is a static property of the input memref type.
    int64_t rank = inputType.getRank();
    Value inputRank = createI64Const(rank);

    // Host-stack array of the input dims (one i64 per dim), so the kernel can
    // decompose a flat index into row-major multi-dim coordinates. Static dims
    // fold to constants; dynamic dims are read from the descriptor.
    Value one = createI64Const(1);
    auto arrType =
        LLVM::LLVMArrayType::get(i64Type, std::max(rank, (int64_t)1));
    Value dimsArr = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one,
                                           /*align=*/8);
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      Value dim = getMemRefDimSize(inputType, i, adaptor.getX(), rewriter, loc);
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, dimsArr, idx);
      LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
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
