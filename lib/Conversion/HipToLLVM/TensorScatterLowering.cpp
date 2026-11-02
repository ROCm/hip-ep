/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.tensor_scatter(%ctx) ins(%past, %update) [write_indices(%wi)]
//     outs(%out) {axis, circular}
// -> wrap_tensor_scatter(state, past, update, write_indices, out, elem_bytes,
//                        batch, outer, max_seq, seq, inner, circular)
//
// The buffers are viewed as [batch, outer, max_seq | seq, inner]: outer is
// the product of the dims between batch and axis, inner of the dims after it.
struct TensorScatterOpLowering
    : public ConvertOpToLLVMPattern<TensorScatterOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TensorScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto pastType = cast<MemRefType>(op.getPast().getType());
    auto updateType = cast<MemRefType>(op.getUpdate().getType());
    const unsigned bits = pastType.getElementTypeBitWidth();
    if (bits % 8 != 0)
      return rewriter.notifyMatchFailure(op, "sub-byte element type");

    auto i64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto dimProduct = [&](MemRefType type, Value desc, int64_t lo,
                          int64_t hi) -> Value {
      Value product = i64Const(1);
      for (int64_t d = lo; d < hi; ++d)
        product = LLVM::MulOp::create(
            rewriter, loc, product,
            getMemRefDimSize(type, d, desc, rewriter, loc));
      return product;
    };

    const int64_t axis = op.getAxis();
    const int64_t rank = pastType.getRank();
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    Value writeIndices =
        adaptor.getWriteIndices()
            ? extractContiguousMemRefPtr(adaptor.getWriteIndices(), rewriter,
                                         loc)
            : nullPtr;

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getPast(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getUpdate(), rewriter, loc),
        writeIndices,
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        i64Const(bits / 8),
        getMemRefDimSize(pastType, 0, adaptor.getPast(), rewriter, loc),
        dimProduct(pastType, adaptor.getPast(), 1, axis),
        getMemRefDimSize(pastType, axis, adaptor.getPast(), rewriter, loc),
        getMemRefDimSize(updateType, axis, adaptor.getUpdate(), rewriter, loc),
        dimProduct(pastType, adaptor.getPast(), axis + 1, rank),
        i64Const(op.getCircular() ? 1 : 0)};

    SmallVector<Type> paramTypes(5, ptrType);
    paramTypes.append(7, i64Type);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapTensorScatter, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTensorScatterLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<TensorScatterOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
