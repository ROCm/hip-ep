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
//                      count_ptr,
//                      data_shape_ptr, data_rank,
//                      indices_shape_ptr, indices_rank,
//                      updates_shape_ptr, updates_rank,
//                      output_shape_ptr, output_rank,
//                      reduction_id, data_type)
//
// `reduction_id` encodes the ONNX `reduction` string attribute as a small
// integer enum (see ScatterNDOpLowering::reductionIdFromString below).
//
// `count_ptr` is a device pointer to a dynamic "valid index-slice count" that
// the NonZero->ScatterND path uses to skip padded rows. hip.scatter_nd has no
// such operand: its `indices` buffer is already truncated to the valid extent
// upstream (readback_dim -> subview -> transpose), so `indices_shape` already
// equals the valid count and we pass null. The runtime treats a null count_ptr
// as "process all num_slices rows".
//
// The argument is part of the ABI, not optional: wrap_scatter_nd is a 16-param
// C function whose 6th parameter is count_ptr. Omitting it shifts every later
// argument by one slot, so the callee reads the `indices_shape` pointer as
// `data_rank` and dereferences a small-integer-as-pointer (SIGSEGV). Keep the
// param list here in lockstep with the wrap_scatter_nd declaration in
// hipdnn_ep_runtime.h.
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
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value updatesPtr =
        extractContiguousMemRefPtr(adaptor.getUpdates(), rewriter, loc);
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

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value indicesShape = emitShapeArray(indicesType, adaptor.getIndices());
    Value updatesShape = emitShapeArray(updatesType, adaptor.getUpdates());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value dataRank = createI64Const(dataType.getRank());
    Value indicesRank = createI64Const(indicesType.getRank());
    Value updatesRank = createI64Const(updatesType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value reductionVal =
        createI64Const(reductionIdFromString(op.getReduction()));
    Value dataTypeVal = createI64Const(hipDtype);

    // The op has no valid-count operand; pass null (see file header).
    Value nullCountPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    SmallVector<Type, 16> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state, data, idx,
                                                     // updates, out
        ptrType,                                     // count_ptr (nullable)
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
        statePtr,     dataPtr,      indicesPtr,   updatesPtr,
        outPtr,       nullCountPtr, dataShape,    dataRank,
        indicesShape, indicesRank,  updatesShape, updatesRank,
        outShape,     outRank,      reductionVal, dataTypeVal};

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
