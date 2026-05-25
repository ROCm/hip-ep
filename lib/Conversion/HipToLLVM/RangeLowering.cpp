/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Maps MLIR element type to hip_dtype_t for hip_range() in range_kernel.hip.
// ONNX Range only defines int32/int64/float/double; the kernel implements those
// plus int16. F16/BF16 are not ONNX Range types and are not implemented in
// hip_range — reject here so lowering fails instead of a runtime default case.
static int64_t getHipCustomKernelDType(Type elemType) {
  if (elemType.isF32())
    return 0; // HIP_DTYPE_FLOAT32
  if (elemType.isInteger(64))
    return 2; // HIP_DTYPE_INT64
  if (elemType.isInteger(32))
    return 3; // HIP_DTYPE_INT32
  if (elemType.isF64())
    return 4; // HIP_DTYPE_FLOAT64
  if (elemType.isInteger(16))
    return 6; // HIP_DTYPE_INT16
  return -1;
}

// hip.range(ctx, start, limit, delta, output)
//   -> wrap_range(state, start, limit, delta, output, output_num_elements,
//                 hip_dtype)
struct RangeOpLowering : public ConvertOpToLLVMPattern<RangeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t hipDType = getHipCustomKernelDType(outputType.getElementType());
    if (hipDType < 0)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported element type for range");

    Value statePtr = adaptor.getCtx();
    Value startPtr =
        extractContiguousMemRefPtr(adaptor.getStart(), rewriter, loc);
    Value limitPtr =
        extractContiguousMemRefPtr(adaptor.getLimit(), rewriter, loc);
    Value deltaPtr =
        extractContiguousMemRefPtr(adaptor.getDelta(), rewriter, loc);

    Value hipDTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(hipDType));

    // Category-C dispatch: when the conversion attached a `slot_id`
    // attribute (intermediate operand provenance), the wrapper must own
    // the output buffer allocation -- it cannot read the EP-marshalled
    // output buffer because the EP could not yet resolve the dim. Emit
    // a call to wrap_range_dyn(state, start, limit, delta, hip_dtype,
    // slot_id) instead of the static wrap_range. The dyn variant
    // publishes the resolved dim AND the GPU buffer to the slot table;
    // the EP host-side resolver reads them post-compute.
    if (auto slotAttr = op->getAttrOfType<IntegerAttr>("slot_id")) {
      Value slotIdVal = LLVM::ConstantOp::create(
          rewriter, loc, i32Type,
          rewriter.getI32IntegerAttr(static_cast<int32_t>(slotAttr.getInt())));
      SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                         ptrType, i64Type, i32Type};
      FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
          rewriter, module, "wrap_range_dyn", paramTypes, i32Type);
      if (failed(funcOp))
        return failure();
      SmallVector<Value, 6> args = {statePtr, startPtr,    limitPtr,
                                    deltaPtr, hipDTypeVal, slotIdVal};
      LLVM::CallOp::create(rewriter, loc, *funcOp, args);
      rewriter.eraseOp(op);
      return success();
    }

    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value outputNumElems =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    // int wrap_range(RuntimeState* state, void* start, void* limit,
    //                void* delta, void* output, int64_t output_num_elements,
    //                int64_t hip_dtype)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRange, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,   startPtr,  limitPtr,
                                  deltaPtr,   outputPtr, outputNumElems,
                                  hipDTypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRangeLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<RangeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
