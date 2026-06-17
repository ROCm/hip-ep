/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Region ops: inline body and erase =====================================

template <typename OpTy>
struct GraphRegionOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    rewriter.inlineBlockBefore(&body, op);
    rewriter.eraseOp(op);
    return success();
  }
};

using MiopenGraphOpLowering = GraphRegionOpLowering<MiopenGraphOp>;
using HipblasltGraphOpLowering = GraphRegionOpLowering<HipblasltGraphOp>;

// --- HipDNNGraphOp: hip.hipdnn_graph -> hipdnn_graph_execute(state,
//     graph_id, num_io, uids, ptrs)
struct HipDNNGraphOpLowering : public ConvertOpToLLVMPattern<HipDNNGraphOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HipDNNGraphOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    auto ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();

    int32_t graphId = op.getGraphId();
    Value graphIdConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(graphId));

    auto inputUidsAttr = op.getInputUids();
    auto outputUidsAttr = op.getOutputUids();
    int32_t numInputs = adaptor.getInputs().size();
    int32_t numOutputs = adaptor.getOutputs().size();
    int32_t numIo = numInputs + numOutputs;

    if (static_cast<int32_t>(inputUidsAttr.size()) != numInputs)
      return op.emitError("input_uids length (")
             << inputUidsAttr.size() << ") != number of inputs (" << numInputs
             << ")";
    if (static_cast<int32_t>(outputUidsAttr.size()) != numOutputs)
      return op.emitError("output_uids length (")
             << outputUidsAttr.size() << ") != number of outputs ("
             << numOutputs << ")";

    Value numIoConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(numIo));
    Value oneConst = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(1));

    // Allocate stack arrays: int64_t uids[numIo], void* ptrs[numIo]
    auto arrayI64Type = LLVM::LLVMArrayType::get(i64Type, numIo);
    auto arrayPtrType = LLVM::LLVMArrayType::get(ptrType, numIo);

    Value uidsArr = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayI64Type,
                                           oneConst, 8);
    Value ptrsArr = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayPtrType,
                                           oneConst, 8);

    // Store UIDs (compile-time constants from attributes)
    for (int32_t i : llvm::seq<int32_t>(numInputs)) {
      int64_t uid =
          cast<IntegerAttr>(inputUidsAttr[i]).getValue().getSExtValue();
      Value uidVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(uid));
      Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(i));
      Value gepUid =
          LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, uidsArr, idxVal);
      LLVM::StoreOp::create(rewriter, loc, uidVal, gepUid);
    }
    for (int32_t i : llvm::seq<int32_t>(numOutputs)) {
      int64_t uid =
          cast<IntegerAttr>(outputUidsAttr[i]).getValue().getSExtValue();
      Value uidVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(uid));
      Value idxVal = LLVM::ConstantOp::create(
          rewriter, loc, i32Type, rewriter.getI32IntegerAttr(numInputs + i));
      Value gepUid =
          LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, uidsArr, idxVal);
      LLVM::StoreOp::create(rewriter, loc, uidVal, gepUid);
    }

    // Store aligned_ptr from each memref descriptor
    for (int32_t i : llvm::seq<int32_t>(numInputs)) {
      Value ptr =
          extractContiguousMemRefPtr(adaptor.getInputs()[i], rewriter, loc);
      Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(i));
      Value gepPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, ptrsArr, idxVal);
      LLVM::StoreOp::create(rewriter, loc, ptr, gepPtr);
    }
    for (int32_t i : llvm::seq<int32_t>(numOutputs)) {
      Value ptr =
          extractContiguousMemRefPtr(adaptor.getOutputs()[i], rewriter, loc);
      Value idxVal = LLVM::ConstantOp::create(
          rewriter, loc, i32Type, rewriter.getI32IntegerAttr(numInputs + i));
      Value gepPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, ptrsArr, idxVal);
      LLVM::StoreOp::create(rewriter, loc, ptr, gepPtr);
    }

    // Signature: int32_t hipdnn_graph_execute(void*, int32_t, int32_t,
    //                                         int64_t*, void**)
    SmallVector<Type, 5> paramTypes = {ptrType, i32Type, i32Type, ptrType,
                                       ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipDNNGraphExecute, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {statePtr, graphIdConst, numIoConst, uidsArr,
                                  ptrsArr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGraphLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MiopenGraphOpLowering, HipblasltGraphOpLowering,
               HipDNNGraphOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
