/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.where(ctx, condition, x, y, output)
//   -> wrap_where(state, cond_ptr, x_ptr, y_ptr, out_ptr,
//                 cond_shape_ptr, cond_rank,
//                 x_shape_ptr,    x_rank,
//                 y_shape_ptr,    y_rank,
//                 out_shape_ptr,  out_rank,
//                 data_type)
//
// Where has no fixed layout: each operand may be any rank and ONNX
// multidirectional (NumPy-style) broadcasting applies. For each operand we
// alloca a stack array of its real dim sizes (compile-time constants for
// static dims, descriptor.size() for dynamic dims), pass the pointer plus
// rank to the runtime, and let the runtime left-pad each operand shape to
// the output rank before launching the HIP kernel.
//
// data_type identifies the X/Y/output element type (HIPDNN_EP_DATATYPE_*);
// the condition is always 1-byte bool.
struct WhereOpLowering : public ConvertOpToLLVMPattern<WhereOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto condType = cast<MemRefType>(op.getCondition().getType());
    auto xType = cast<MemRefType>(op.getX().getType());
    auto yType = cast<MemRefType>(op.getY().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.where: unsupported output element type");

    // alloca a stack [max(rank, 1) x i64] array, store each dim size, return
    // the base pointer. Uses max(rank, 1) so rank-0 (scalar) operands still
    // have a valid allocation; the runtime ignores the buffer when rank == 0.
    Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                         rewriter.getI64IntegerAttr(1));
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
        Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                            arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value condShape = emitShapeArray(condType, adaptor.getCondition());
    Value xShape = emitShapeArray(xType, adaptor.getX());
    Value yShape = emitShapeArray(yType, adaptor.getY());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value condRank = createI64Const(condType.getRank());
    Value xRank = createI64Const(xType.getRank());
    Value yRank = createI64Const(yType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(dataType);

    // Signature: state, 4 data ptrs, 4 (shape_ptr, rank) pairs, data_type.
    //            = 5 ptrs + 4 ptrs + 4 i64 + 1 i64 = 14 params
    SmallVector<Type, 14> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state + 4 data ptrs
        ptrType, i64Type,                            // cond_shape, cond_rank
        ptrType, i64Type,                            // x_shape,    x_rank
        ptrType, i64Type,                            // y_shape,    y_rank
        ptrType, i64Type,                            // out_shape,  out_rank
        i64Type};                                    // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapWhere, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getCondition(), rewriter, loc),
        extractMemRefPtr(adaptor.getX(), rewriter, loc),
        extractMemRefPtr(adaptor.getY(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        condShape, condRank,
        xShape,    xRank,
        yShape,    yRank,
        outShape,  outRank,
        dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateWhereLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<WhereOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
