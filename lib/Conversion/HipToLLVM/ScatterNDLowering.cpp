/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.scatter_nd(ctx, data, indices, updates, output) {reduction}
//   -> wrap_scatter_nd(state, data_ptr, indices_ptr, updates_ptr, out_ptr,
//                      data_shape_ptr, data_rank,
//                      indices_shape_ptr, indices_rank,
//                      updates_shape_ptr, updates_rank,
//                      output_shape_ptr, output_rank,
//                      reduction_id, data_type)
//
// `reduction_id` encodes the ONNX `reduction` string attribute as a small
// integer enum (see ScatterNDOpLowering::reductionIdFromString below). The
// runtime side is a stub today that only logs its parameters, but the
// signature is shaped to match the future kernel implementation.
struct ScatterNDOpLowering : public ConvertOpToLLVMPattern<ScatterNDOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static int64_t reductionIdFromString(StringRef r) {
    if (r == "add")
      return 1;
    if (r == "mul")
      return 2;
    if (r == "min")
      return 3;
    if (r == "max")
      return 4;
    return 0; // "none" / default
  }

  LogicalResult
  matchAndRewrite(ScatterNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto updatesType = cast<MemRefType>(op.getUpdates().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    Value statePtr = adaptor.getCtx();
    // Operand layout: 0=ctx, 1=data, 2=indices, 3=updates, 4=output.
    // When `hip-annotate-input-dim-slots` has tagged an operand as
    // consuming a Cat-C slot, the descriptor's `alignedPtr` points
    // at the upper-bound DPS init buffer (uninitialised — the
    // producer wrote into a separately allocated exact-size buffer
    // published via the slot table). Read the published pointer
    // instead so the scatter_nd kernel sees the actual data. In the
    // Qwen embedding graph both `indices` (from Transpose of NonZero)
    // and `updates` (from Slice of the Reshape of image_features)
    // would otherwise be garbage.
    Value dataPtr = extractContiguousMemRefPtrWithSlot(
        op, /*operandIdx=*/1, adaptor.getData(), statePtr, rewriter, loc);
    Value indicesPtr = extractContiguousMemRefPtrWithSlot(
        op, /*operandIdx=*/2, adaptor.getIndices(), statePtr, rewriter, loc);
    Value updatesPtr = extractContiguousMemRefPtrWithSlot(
        op, /*operandIdx=*/3, adaptor.getUpdates(), statePtr, rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    // Slot-aware variant: for dynamic dims annotated by
    // `hip-annotate-input-dim-slots` we read the runtime-published
    // value from the slot table instead of the descriptor (which
    // encodes the upper-bound pool size). This is the critical path
    // for the Qwen embedding model — indices (operand 2) and updates
    // (operand 3) both arrive as upper-bound buffers whose true row
    // count is the published NonZero slot.
    auto emitShapeArray = [&](MemRefType type, Value descriptor,
                              unsigned operandIdx) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSizeWithSlot(
            op, operandIdx, type, static_cast<unsigned>(i), descriptor,
            statePtr, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    // Operand layout per Hip_ScatterNDOp: 0=ctx, 1=data, 2=indices,
    // 3=updates, 4=output.
    Value dataShape = emitShapeArray(dataType, adaptor.getData(), 1);
    Value indicesShape = emitShapeArray(indicesType, adaptor.getIndices(), 2);
    Value updatesShape = emitShapeArray(updatesType, adaptor.getUpdates(), 3);
    Value outShape = emitShapeArray(outputType, adaptor.getOutput(), 4);

    Value dataRank = createI64Const(dataType.getRank());
    Value indicesRank = createI64Const(indicesType.getRank());
    Value updatesRank = createI64Const(updatesType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value reductionVal =
        createI64Const(reductionIdFromString(op.getReduction()));
    Value dataTypeVal = createI64Const(hipDtype);

    SmallVector<Type, 16> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state, data, idx,
                                                     // updates, out
        ptrType, i64Type,                            // data_shape, data_rank
        ptrType, i64Type,                            // idx_shape, idx_rank
        ptrType, i64Type,                            // upd_shape, upd_rank
        ptrType, i64Type,                            // out_shape, out_rank
        i64Type, i64Type};                           // reduction_id, data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapScatterND, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {
        statePtr,    dataPtr,  indicesPtr,   updatesPtr,   outPtr,
        dataShape,   dataRank, indicesShape, indicesRank,  updatesShape,
        updatesRank, outShape, outRank,      reductionVal, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateScatterNDLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns) {
  patterns.add<ScatterNDOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
