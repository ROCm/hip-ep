/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange CopyD2HOp::getDpsInitsMutable() { return getInitMutable(); }

namespace {

constexpr const char *kWrapCopyD2H = "wrap_copy_d2h";

// `wrap_copy_d2h` starts the copy on the context's stream and takes a byte
// count.
struct CopyD2HLowering : ConvertOpToLLVMPattern<CopyD2HOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CopyD2HOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto initType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!initType) {
      return rewriter.notifyMatchFailure(op, "operands must be memrefs");
    }

    Value sizeBytes = getSizeInBytes(loc, initType.getElementType(), rewriter);
    for (Value extent : extractShape(initType, adaptor.getInit(), rewriter, loc,
                                     getIndexType())) {
      sizeBytes = LLVM::MulOp::create(rewriter, loc, sizeBytes, extent);
    }

    // hostPtr passes its value through, so unwrap the destination here. The
    // devicePtr tag unwraps the source itself.
    Value dstPtr = extractContiguousMemRefPtr(adaptor.getInit(), rewriter, loc);

    using CopyCall = RuntimeFunc<i32, hostPtr, hostPtr, devicePtr, i64>;
    auto copyFunc = CopyCall::lookupOrCreateFn(
        rewriter, loc, op->getParentOfType<ModuleOp>(), kWrapCopyD2H);
    if (failed(copyFunc)) {
      return failure();
    }
    if (failed(copyFunc->call(adaptor.getCtx(), dstPtr, adaptor.getSrc(),
                              sizeBytes))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrCopyD2HLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CopyD2HLowering>(converter);
}
