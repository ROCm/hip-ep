/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y) {input_data_type}
//   -> wrap_nonzero(state, input_ptr, output_ptr,
//                   input_num_elements, input_rank,
//                   output_capacity, input_data_type, input_shape)
//
// `input_num_elements` is the total count of input elements (product of
// input dims, supports static + dynamic shapes via MemRefDescriptor).
// `input_rank` is the static rank of the input (R), passed as a compile-
// time i64 constant.
// `output_capacity` is the size of the dynamic N dim of the output (i.e.,
// the upper-bound number of nonzero entries materialised by the conversion
// pass). For NonZero today this equals input_num_elements, but lowering
// computes it directly from the output descriptor so the path is correct
// even if the conversion ever uses a smaller upper bound.
// `input_shape` is a host-side `i64[R]` array (one entry per input dim,
// static dims fold to constants, dynamic dims read from the descriptor)
// stored on the stack via `llvm.alloca`. The runtime forwards it to the
// kernel, which derives row-major strides and unravels each non-zero flat
// index into its R coordinates. This mirrors the `input_shape` arg of
// wrap_transpose / wrap_tile / wrap_slice etc.
//
// Before (rank-2 dynamic input):
//   %y = hip.nonzero(%ctx) ins(%x : memref<?x4xi64>)
//                          outs(%out : memref<2x?xi64>)
// After (LLVM):
//   %shape = llvm.alloca %1 x !llvm.array<2 x i64>
//   llvm.store %dim0, %shape[0]   ; dynamic dim from descriptor sizes[0]
//   llvm.store %c4,   %shape[1]   ; static dim folded to constant
//   llvm.call @wrap_nonzero(%state, %inPtr, %outPtr, %numel, %c2,
//                           %cap, %dtype, %shape)
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

    // Total input element count (handles static + dynamic dims).
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    // R is a static property of the input memref type.
    Value inputRank = createI64Const(inputType.getRank());

    // Output capacity = N dim (data-dependent at the source level; the
    // tensor.empty in OnnxToHip pins it to numel(X) upper-bound, but read
    // from the descriptor here for correctness under any future re-size).
    Value outputCapacity = getMemRefDimSize(outputType, /*dimIdx=*/1,
                                            adaptor.getY(), rewriter, loc);

    Value inputDataTypeVal = createI64Const(op.getInputDataType());

    // Build the host-side input_shape array: i64[R] on the stack, one entry
    // per input dim (static dims fold to constants; dynamic dims read the
    // descriptor sizes array). Same construction as TransposeLowering's
    // input_shape arg. The kernel needs these dims to unravel a flat
    // row-major index into R coordinates.
    int64_t rank = inputType.getRank();
    Value oneI64 = createI64Const(1);
    auto shapeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Value shapeAlloca = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, shapeArrayType, oneI64, /*alignment=*/8);
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      Value dimVal = getMemRefDimSize(inputType, static_cast<unsigned>(i),
                                      adaptor.getX(), rewriter, loc);
      Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(i));
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                      shapeAlloca, idxVal);
      LLVM::StoreOp::create(rewriter, loc, dimVal, gep);
    }

    // int wrap_nonzero(RuntimeState* state, void* input, void* output,
    //                  int64_t input_num_elements, int64_t input_rank,
    //                  int64_t output_capacity, int64_t input_data_type,
    //                  const int64_t* input_shape)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, i64Type, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNonZero, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,         inputPtr,   outputPtr,
                                  inputNumElements, inputRank,  outputCapacity,
                                  inputDataTypeVal, shapeAlloca};
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
