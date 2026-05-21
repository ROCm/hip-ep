/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.nonzero(ctx, x, y) {input_data_type, slot_id, ...}
//   -> wrap_nonzero(state, input_ptr,
//                   input_num_elements, input_rank,
//                   input_data_type, slot_id, input_shape_dev)
//
// `input_num_elements` is the total count of input elements (product of
// input dims, supports static + dynamic shapes via MemRefDescriptor).
// `input_rank` is the static rank of the input (R), passed as a compile-
// time i64 constant.
// `input_data_type` is the HIPDNN_EP_DATATYPE_* of the input.
// `slot_id` is the per-module unique slot index this op publishes the
// dynamic dim (`N`) into. NonZero is strictly Category-C, so slot_id is
// REQUIRED (set during OnnxToHip conversion).
// `input_shape_dev` is a GPU-resident int64 array of length `input_rank`
// holding the input shape. NonZero needs it for the per-axis coordinate
// decomposition in the fill pass; we pack the shape into a stack-alloca
// int64 array (constants for static dims, MemRefDescriptor sizes[] for
// dynamic dims) and let the runtime hipMemcpyAsync H2D it once per call
// into a tiny dyn-pool slab.
//
// The DPS `outs(%y)` operand is intentionally IGNORED by the runtime —
// the upper-bound buffer it points at is never written. The runtime
// publishes the actual exact-sized GPU buffer via
// hipdnn_ep_state_publish_buffer and the EP reads it after compute.
// We still keep the operand in the IR because bufferization requires it
// for the DPS pattern; the IR-level buffer effect is preserved for
// alias analysis even though no kernel writes to it.
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
    auto slotAttr = op.getSlotIdAttr();
    if (!slotAttr) {
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero must carry a slot_id attribute (assigned during "
              "ONNX->HIP conversion)");
    }

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);

    Value inputNumElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    Value inputRankConst = createI64Const(inputType.getRank());
    Value inputDataTypeVal = createI64Const(op.getInputDataType());
    Value slotIdConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr((int32_t)slotAttr.getInt()));

    // Build a per-call int64 shape array on the stack. Static dims become
    // constants; dynamic dims come from the MemRefDescriptor's sizes[] (the
    // same field computeNumElements would have read). The runtime
    // hipMemcpyAsync's this single small allocation H2D into a dyn-pool
    // slab and uses it during the fill kernel.
    const int64_t rank = inputType.getRank();
    Value shapeArrayLen = createI64Const(rank);
    Value shapeArr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, shapeArrayLen,
                               /*alignment=*/8);
    auto inputDesc = MemRefDescriptor(adaptor.getX());
    for (int64_t d = 0; d < rank; ++d) {
      Value dimVal;
      if (inputType.isDynamicDim(d)) {
        dimVal = inputDesc.size(rewriter, loc, d);
      } else {
        dimVal = createI64Const(inputType.getDimSize(d));
      }
      Value idx = createI64Const(d);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                       shapeArr, ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, dimVal, slot);
    }

    // int wrap_nonzero(RuntimeState*, const void* input,
    //                  int64_t input_num_elements, int64_t input_rank,
    //                  int64_t input_data_type, int32_t slot_id,
    //                  const int64_t* input_shape_host)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, i64Type, i64Type,
                                       i64Type, i32Type, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapNonZero, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,        inputPtr,
                                  inputNumElements, inputRankConst,
                                  inputDataTypeVal, slotIdConst,
                                  shapeArr};
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
