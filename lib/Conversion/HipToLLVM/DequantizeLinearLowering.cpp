/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.dequantize_linear lowering
//===----------------------------------------------------------------------===//
//
// Before:
//   hip.dequantize_linear(%ctx)
//       ins(%x, %scale : memref<4x8xi8, 1>, memref<8xf32, 1>)
//       zero_points(%zp : memref<8xui8, 1>)
//       outs(%out : memref<4x8xf32, 1>)
//       {axis = 1, block_size = 0, output_dtype = 0}
//
// After:
//   llvm.call @wrap_dequantize_linear(
//       %state, %x_p, %scale_p, %zp_p, %out_p,
//       %x_shape, %x_rank, %scale_shape, %scale_rank, %out_shape, %out_rank,
//       %axis, %block_size, %output_dtype,
//       %x_dtype, %scale_dtype, %out_dtype)

struct DequantizeLinearOpLowering
    : public ConvertOpToLLVMPattern<DequantizeLinearOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DequantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto xType = cast<MemRefType>(op.getX().getType());
    auto scaleType = cast<MemRefType>(op.getXScale().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t xDtype = getHipdnnDataType(xType.getElementType());
    int64_t scaleDtype = getHipdnnDataType(scaleType.getElementType());
    int64_t outDtype = getHipdnnDataType(outputType.getElementType());
    if (xDtype < 0 || scaleDtype < 0 || outDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    Value statePtr = adaptor.getCtx();
    Value xPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value scalePtr =
        extractContiguousMemRefPtr(adaptor.getXScale(), rewriter, loc);
    Value zpPtr =
        extractOptionalMemRefPtr(adaptor.getXZeroPoint(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) -> Value {
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
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        Value dim = getMemRefDimSize(type, static_cast<unsigned>(i), descriptor,
                                     rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value xShape = emitShapeArray(xType, adaptor.getX());
    Value scaleShape = emitShapeArray(scaleType, adaptor.getXScale());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value xRank = createI64Const(xType.getRank());
    Value scaleRank = createI64Const(scaleType.getRank());
    Value outRank = createI64Const(outputType.getRank());

    Value axis = createI64Const(op.getAxis());
    Value blockSize = createI64Const(op.getBlockSize());
    Value outputDtype = createI64Const(op.getOutputDtype());

    Value xDtypeVal = createI64Const(xDtype);
    Value scaleDtypeVal = createI64Const(scaleDtype);
    Value outDtypeVal = createI64Const(outDtype);

    SmallVector<Type, 22> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType,
        ptrType, i64Type, ptrType, i64Type, ptrType, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDequantizeLinear, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 22> args = {
        statePtr,     xPtr,         scalePtr,      zpPtr,
        outPtr,       xShape,       xRank,         scaleShape,
        scaleRank,    outShape,     outRank,       axis,
        blockSize,    outputDtype,  xDtypeVal,     scaleDtypeVal,
        outDtypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateDequantizeLinearLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<DequantizeLinearOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
