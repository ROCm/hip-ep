/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.transpose(%ctx) ins(%input) outs(%output) {perm = [...]}
//   -> wrap_transpose(state, input, output, rank, input_shape, perm,
//                     num_elements, element_size_bytes)
//
// input_shape[rank] and perm[rank] are stored on the stack via LLVM::AllocaOp;
// the runtime treats them as host-side metadata and forwards them to the
// transpose kernel after computing strides.
struct TransposeOpLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t rank = inputType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "transpose requires non-scalar input");

    ArrayAttr permAttr = op.getPerm();
    if (static_cast<int64_t>(permAttr.size()) != rank)
      return op.emitOpError("perm length must match input rank");

    // element_size_bytes
    Type elemType = inputType.getElementType();
    if (!elemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for transpose");
    int64_t elemSizeBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (elemSizeBytes <= 0)
      return rewriter.notifyMatchFailure(op, "unsupported element bit width");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(rank));
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elemSizeBytes));
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1));

    // Allocate input_shape[rank] and perm[rank] on the stack.
    auto shapeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    auto permArrayType = LLVM::LLVMArrayType::get(i64Type, rank);

    Value shapeAlloca = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, shapeArrayType, oneI64, /*alignment=*/8);
    Value permAlloca = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, permArrayType, oneI64, /*alignment=*/8);

    // Populate input_shape via per-dim load from the memref descriptor (static
    // dims fold to constants; dynamic dims hit the descriptor sizes array).
    for (int64_t i = 0; i < rank; ++i) {
      Value dimVal = getMemRefDimSize(inputType, static_cast<unsigned>(i),
                                      adaptor.getInput(), rewriter, loc);
      Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(i));
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                      shapeAlloca, idxVal);
      LLVM::StoreOp::create(rewriter, loc, dimVal, gep);
    }

    // Populate perm from the compile-time attribute.
    for (int64_t i = 0; i < rank; ++i) {
      int64_t permVal =
          cast<IntegerAttr>(permAttr[i]).getValue().getSExtValue();
      Value permConst = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(permVal));
      Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(i));
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                      permAlloca, idxVal);
      LLVM::StoreOp::create(rewriter, loc, permConst, gep);
    }

    Value numElems =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    // int wrap_transpose(RuntimeState* state, void* input, void* output,
    //                    int64_t rank, const int64_t* input_shape,
    //                    const int64_t* perm, int64_t num_elements,
    //                    int64_t element_size_bytes)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       ptrType, ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapTranspose, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr, inputPtr,    outputPtr,
                                  rankVal,  shapeAlloca, permAlloca,
                                  numElems, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns) {
  patterns.add<TransposeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
