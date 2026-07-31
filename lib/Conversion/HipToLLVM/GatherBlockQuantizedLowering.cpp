/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.gather_block_quantized lowering
//===----------------------------------------------------------------------===//
//
// Before:
//   hip.gather_block_quantized(%ctx)
//       ins(%data, %indices, %scales :
//           memref<2048x96xui8, 1>, memref<8xi64, 1>,
//           memref<2048x12xf16, 1>)
//       zero_points(%zp : memref<2048x12xui8, 1>)
//       outs(%out : memref<8x96xf16, 1>)
//       {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
//
// After:
//   %data_shape = llvm.alloca <i64 x 2>     ; { 2048, 96 }
//   %idx_shape  = llvm.alloca <i64 x 1>     ; { 8 }
//   %scl_shape  = llvm.alloca <i64 x 2>     ; { 2048, 12 }
//   %out_shape  = llvm.alloca <i64 x 2>     ; { 8, 96 }
//   llvm.call @wrap_gather_block_quantized(
//       %state, %data_p, %idx_p, %scl_p, %zp_p, %out_p,
//       %data_shape, %data_rank=2,
//       %idx_shape,  %idx_rank=1,
//       %scl_shape,  %scl_rank=2,
//       %out_shape,  %out_rank=2,
//       %bits=4, %block_size=16, %gather_axis=0, %quantize_axis=1,
//       %data_dtype=7,    ; HIPDNN_EP_DATATYPE_UINT8 packed nibbles
//       %indices_dtype=4, ; HIPDNN_EP_DATATYPE_INT64
//       %scales_dtype=1)  ; HIPDNN_EP_DATATYPE_HALF
//
// Each shape array is materialised on the LLVM stack; dynamic dims are
// extracted from the MemRef descriptor at call time, static dims are folded
// to LLVM constants. `zero_points` is nullable — absent operand lowers to
// the LLVM null pointer (extractOptionalMemRefPtr).

struct GatherBlockQuantizedOpLowering
    : public ConvertOpToLLVMPattern<GatherBlockQuantizedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherBlockQuantizedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto scalesType = cast<MemRefType>(op.getScales().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataDtype = getHipdnnDataType(dataType.getElementType());
    if (dataType.getElementType().isUnsignedInteger(8) || op.getBits() == 8 ||
        op->hasAttr("unsigned_quant_storage"))
      dataDtype = 7; // HIPDNN_EP_DATATYPE_UINT8
    int64_t indicesDtype = getHipdnnDataType(indicesType.getElementType());
    int64_t scalesDtype = getHipdnnDataType(scalesType.getElementType());
    if (dataDtype < 0 || indicesDtype < 0 || scalesDtype < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type on data/indices/scales");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value scalesPtr =
        extractContiguousMemRefPtr(adaptor.getScales(), rewriter, loc);
    Value zpPtr =
        extractOptionalMemRefPtr(adaptor.getZeroPoints(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);

    // Materialise an i64[rank] shape array on the stack. For each dim, use
    // a runtime extractvalue when dynamic and an LLVM constant when static
    // (kDynamic sentinel is a large negative number — never feed it to the
    // runtime as a real size).
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

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value indicesShape = emitShapeArray(indicesType, adaptor.getIndices());
    Value scalesShape = emitShapeArray(scalesType, adaptor.getScales());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value dataRank = createI64Const(dataType.getRank());
    Value indicesRank = createI64Const(indicesType.getRank());
    Value scalesRank = createI64Const(scalesType.getRank());
    Value outRank = createI64Const(outputType.getRank());

    Value bits = createI64Const(op.getBits());
    Value blockSize = createI64Const(op.getBlockSize());
    Value gatherAxis = createI64Const(op.getGatherAxis());
    Value quantAxis = createI64Const(op.getQuantizeAxis());

    Value dataDtypeVal = createI64Const(dataDtype);
    Value indicesDtypeVal = createI64Const(indicesDtype);
    Value scalesDtypeVal = createI64Const(scalesDtype);

    Value quantStorageBitsVal =
        createI64Const(op.getQuantStorageBits().value_or(op.getBits()));

    SmallVector<Type, 24> paramTypes = {
        ptrType,                            // state
        ptrType, ptrType, ptrType,          // data, indices, scales
        ptrType,                            // zero_points (nullable)
        ptrType,                            // output
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // indices_shape, indices_rank
        ptrType, i64Type,                   // scales_shape, scales_rank
        ptrType, i64Type,                   // output_shape, output_rank
        i64Type, i64Type, i64Type, i64Type, // bits, block_size, gather_axis,
                                            // quantize_axis
        i64Type, i64Type, i64Type,          // data_dtype, indices_dtype,
                                            // scales_dtype
        i64Type                             // quant_storage_bits
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGatherBlockQuantized, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 24> args = {
        statePtr,           dataPtr,         indicesPtr,
        scalesPtr,          zpPtr,           outPtr,
        dataShape,          dataRank,        indicesShape,
        indicesRank,        scalesShape,     scalesRank,
        outShape,           outRank,         bits,
        blockSize,          gatherAxis,      quantAxis,
        dataDtypeVal,       indicesDtypeVal, scalesDtypeVal,
        quantStorageBitsVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatherBlockQuantizedLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GatherBlockQuantizedOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
