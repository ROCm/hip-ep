/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.expand(ctx, input, shape, output)
//   -> wrap_expand(state, in_ptr, shape_ptr, out_ptr,
//                  in_shape_ptr, in_rank, out_shape_ptr, out_rank, data_type)
struct ExpandOpLowering : public ConvertOpToLLVMPattern<ExpandOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inType = cast<MemRefType>(op.getInput().getType());
    auto outType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataType = getHipdnnDataType(inType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value shapePtr =
        extractContiguousMemRefPtr(adaptor.getShape(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value inShape = emitShapeArray(inType, adaptor.getInput());
    Value outShape = emitShapeArray(outType, adaptor.getOutput());
    Value inRank = createI64Const(inType.getRank());
    Value outRank = createI64Const(outType.getRank());
    Value dataTypeVal = createI64Const(dataType);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, // state, in, shape, out
                                       ptrType, i64Type, // in_shape, in_rank
                                       ptrType, i64Type, // out_shape, out_rank
                                       i64Type};         // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapExpand, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, inPtr,   shapePtr,
                                  outPtr,   inShape, inRank,
                                  outShape, outRank, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateExpandLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<ExpandOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
