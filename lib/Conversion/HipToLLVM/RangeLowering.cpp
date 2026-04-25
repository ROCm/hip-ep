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
    bool isF32 = elemType.isF32();
    if (!isI64 && !isF32)
      return op.emitOpError(
                 "hip.range supports only i64 and f32 element types");

    // Pull start / limit / delta scalars (memrefs of rank-0).
    auto extractScalar = [&](Value memref, Type wantType) -> Value {
      Value ptr = extractMemRefPtr(memref, rewriter, loc);
      return LLVM::LoadOp::create(rewriter, loc, wantType, ptr);
    };
    Type wantScalar = isI64 ? i64Type : f32Type;
    Value startVal = extractScalar(adaptor.getStart(), wantScalar);
    Value limitVal = extractScalar(adaptor.getLimit(), wantScalar);
    Value deltaVal = extractScalar(adaptor.getDelta(), wantScalar);

    // n = max(ceil((limit - start) / delta), 0)
    Value n;
    if (isI64) {
      Value diff =
          LLVM::SubOp::create(rewriter, loc, limitVal, startVal).getResult();
      n = LLVM::SDivOp::create(rewriter, loc, diff, deltaVal).getResult();
    } else {
      Value diff =
          LLVM::FSubOp::create(rewriter, loc, limitVal, startVal).getResult();
      Value div =
          LLVM::FDivOp::create(rewriter, loc, diff, deltaVal).getResult();
      n = LLVM::FPToSIOp::create(rewriter, loc, i64Type, div).getResult();
    }

    // Clamp negative -> 0.
    Value zeroI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(0));
    Value isPos = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sgt,
                                        n, zeroI64);
    n = LLVM::SelectOp::create(rewriter, loc, isPos, n, zeroI64);

    // Output device pointer.
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Type retType = i32Type;
    SmallVector<Type> paramTypes;
    if (isI64) {
      paramTypes = {ptrType, i64Type, i64Type, i64Type, ptrType};
    } else {
      paramTypes = {ptrType, f32Type, f32Type, i64Type, ptrType};
    }
    StringRef name = isI64 ? kHipRangeI64 : kHipRangeF32;
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, name, paramTypes, retType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {adaptor.getCtx(), startVal, deltaVal, n, outPtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
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
