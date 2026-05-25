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
    // Operand 1 is `input`; operand 2 is `output`. When `hip-annotate-
    // input-dim-slots` has tagged the input as consuming a Cat-C slot,
    // the descriptor's `alignedPtr` points at the upper-bound DPS init
    // buffer (uninitialised — the producer wrote into a separately
    // allocated exact-size buffer published via the slot table). Read
    // the published pointer instead so the transpose kernel sees the
    // actual data.
    Value inputPtr = extractContiguousMemRefPtrWithSlot(
        op, /*operandIdx=*/1, adaptor.getInput(), statePtr, rewriter, loc);
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
    // For dynamic dims annotated by `hip-annotate-input-dim-slots` we read
    // the runtime-published value from the slot table instead of the
    // descriptor — the descriptor encodes the upper-bound pool size, not
    // the post-Category-C actual count. Operand 1 is the `input` (operand 0
    // is the `!hip.context`).
    constexpr unsigned kInputOperandIdx = 1;
    for (int64_t i = 0; i < rank; ++i) {
      Value dimVal = getMemRefDimSizeWithSlot(
          op, kInputOperandIdx, inputType, static_cast<unsigned>(i),
          adaptor.getInput(), statePtr, rewriter, loc);
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

    // num_elements: product of input_shape, honouring the slot-aware dim
    // reads above so the kernel writes only the runtime-published count of
    // elements (not the upper-bound pool allocation).
    Value numElems = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(1));
    for (int64_t i = 0; i < rank; ++i) {
      Value dimVal = getMemRefDimSizeWithSlot(
          op, kInputOperandIdx, inputType, static_cast<unsigned>(i),
          adaptor.getInput(), statePtr, rewriter, loc);
      numElems = LLVM::MulOp::create(rewriter, loc, numElems, dimVal);
    }

    // Phase 2.4 of slot-buffer-coalesce: publish any output slots
    // reserved by `hip-reserve-propagator-slots` BEFORE the kernel
    // call. For Transpose, output dim `d` resolves to input dim
    // `perm[d]`; the value has already been materialised above and is
    // either folded to a constant (static input dim) or loaded via
    // slot-aware getMemRefDimSizeWithSlot (dynamic input dim that is
    // itself slot-bound). We re-compute through the same helper so the
    // SSA edges land in the right block.
    auto dimSizeProvider = [&](unsigned resultIdx, unsigned outDim) -> Value {
      if (resultIdx != 0)
        return Value();
      int64_t srcDim =
          cast<IntegerAttr>(permAttr[outDim]).getValue().getSExtValue();
      return getMemRefDimSizeWithSlot(
          op, kInputOperandIdx, inputType, static_cast<unsigned>(srcDim),
          adaptor.getInput(), statePtr, rewriter, loc);
    };
    emitPropagatorSlotPublishes(op, statePtr, dimSizeProvider, rewriter, loc);

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
