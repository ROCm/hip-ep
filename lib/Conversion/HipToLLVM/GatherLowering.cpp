/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.gather(state, data, indices, output) — operands follow ONNX Gather:
// data = operand 0, indices = operand 1 (see HipOps.td).
struct GatherOpLowering : public ConvertOpToLLVMPattern<GatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute data_num_elements
    auto dataType = cast<MemRefType>(op.getData().getType());
    Value dataNumElementsVal =
        computeNumElements(dataType, adaptor.getData(), rewriter, loc);

    // Compute indices_num_elements
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    Value indicesNumElementsVal =
        computeNumElements(indicesType, adaptor.getIndices(), rewriter, loc);

    // Compute output_num_elements
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value outputNumElementsVal =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    // element_size_bytes
    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // axis attribute (normalize negative axis before emitting the constant so
    // runtime and axis_size use the same index as ONNX Gather).
    int64_t axisAttr = op.getAxis();
    int64_t dataRank = dataType.getRank();
    if (axisAttr < 0)
      axisAttr += dataRank;
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(axisAttr));

    // axis_size = data.shape[axis]; inner_size = product(data.shape[axis+1:]).
    Value axisSizeVal =
        getMemRefDimSize(dataType, static_cast<unsigned>(axisAttr),
                         adaptor.getData(), rewriter, loc);
    Value innerSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
    for (int64_t d : llvm::seq<int64_t>(axisAttr + 1, dataRank)) {
      Value ds = getMemRefDimSize(dataType, static_cast<unsigned>(d),
                                  adaptor.getData(), rewriter, loc);
      innerSizeVal = LLVM::MulOp::create(rewriter, loc, innerSizeVal, ds);
    }

    // ONNX indices may be int32 or int64; pass the byte width so the runtime
    // kernel reads them with the correct stride (reading int32 as int64
    // otherwise fuses adjacent indices into out-of-range values).
    unsigned indicesElemBytes =
        indicesType.getElementType().getIntOrFloatBitWidth() / 8;
    Value indicesElemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(indicesElemBytes));

    int64_t dataHipDtype = getHipdnnDataType(dataType.getElementType());
    int64_t indicesHipDtype = getHipdnnDataType(indicesType.getElementType());
    if (dataHipDtype < 0 || indicesHipDtype < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported gather data or indices element type for lowering");

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

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value indicesShape = emitShapeArray(indicesType, adaptor.getIndices());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());
    Value dataRankVal = createI64Const(dataType.getRank());
    Value indicesRankVal = createI64Const(indicesType.getRank());
    Value outRankVal = createI64Const(outputType.getRank());
    Value dataDtypeVal = createI64Const(dataHipDtype);
    Value indicesDtypeVal = createI64Const(indicesHipDtype);

    // wrap_gather(..., indices_element_size_bytes,
    //               data_shape*, data_rank, indices_shape*, indices_rank,
    //               output_shape*, output_rank,
    //               data_hip_dtype, indices_hip_dtype)
    SmallVector<Type, 20> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, ptrType, i64Type, ptrType, i64Type,
        ptrType, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGather, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        statePtr,           dataPtr,          indicesPtr,       outputPtr,
        axisVal,            dataNumElementsVal, indicesNumElementsVal,
        outputNumElementsVal, axisSizeVal,    innerSizeVal,     elemSizeVal,
        indicesElemSizeVal, dataShape,        dataRankVal,      indicesShape,
        indicesRankVal,     outShape,         outRankVal,       dataDtypeVal,
        indicesDtypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatherLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<GatherOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
