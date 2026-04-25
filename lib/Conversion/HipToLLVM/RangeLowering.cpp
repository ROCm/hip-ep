/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// hip.range -> hip_range_i64 / hip_range_f32 (in transpose_kernel.hip
// sibling: range_kernel.hip).
//
// CAVEAT: the runtime size of the output is computed by the kernel as
// `n = (limit - start) / delta`.  The DPS init buffer is allocated by
// the conversion pattern with placeholder size 1; we therefore use the
// memref descriptor's first stride * size to figure out the buffer
// capacity (TODO: fix this once hip.range gets a real dynamic-alloc
// path).

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

inline constexpr const char *kHipRangeI64 = "hip_range_i64";
inline constexpr const char *kHipRangeF32 = "hip_range_f32";

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
    Type f32Type = rewriter.getF32Type();

    auto outMemref = cast<MemRefType>(op.getOutput().getType());
    Type elemType = outMemref.getElementType();
    bool isI64 = elemType.isInteger(64);
    bool isI32 = elemType.isInteger(32);
    bool isF32 = elemType.isF32();
    if (!isI64 && !isI32 && !isF32)
      return op.emitOpError(
                 "hip.range supports only i32/i64/f32 element types");
    int hipDtype = isI64 ? 2 : (isI32 ? 3 : 0);

    // Pass start/limit/delta as device pointers; the runtime helper
    // memcpy's them back to host before computing n.  Avoids the host
    // CPU loading from a device pointer.
    Value startPtr = extractMemRefPtr(adaptor.getStart(), rewriter, loc);
    Value limitPtr = extractMemRefPtr(adaptor.getLimit(), rewriter, loc);
    Value deltaPtr = extractMemRefPtr(adaptor.getDelta(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute output capacity = product of output dims (mostly 1
    // dynamic dim for ONNX Range).
    Value capacity = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
    {
      MemRefDescriptor desc(adaptor.getOutput());
      for (int64_t i = 0; i < outMemref.getRank(); ++i) {
        Value d;
        if (outMemref.isDynamicDim(i))
          d = desc.size(rewriter, loc, i);
        else
          d = LLVM::ConstantOp::create(
              rewriter, loc, i64Type,
              rewriter.getI64IntegerAttr(outMemref.getDimSize(i)));
        capacity = LLVM::MulOp::create(rewriter, loc, capacity, d);
      }
    }
    Value dtypeI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(hipDtype));

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                     ptrType, i64Type, i32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, "hip_range_dyn", paramTypes,
                                i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {adaptor.getCtx(), startPtr, limitPtr, deltaPtr,
                                outPtr,           capacity, dtypeI32};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    (void)f32Type; (void)isF32;
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateRangeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<RangeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
