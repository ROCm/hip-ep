/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

/// hip.reduce_max -> wrap_reduce_max runtime call.
/// Identical calling convention to ReduceSum lowering.
struct ReduceMaxOpLowering : public ConvertOpToLLVMPattern<ReduceMaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value axesPtr =
        extractContiguousMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto axesType = cast<MemRefType>(op.getAxes().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataTypeEnum = getHipdnnDataType(dataType.getElementType());
    if (dataTypeEnum < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.reduce_max");

    Value dataNumElements = createI64Const(1);
    MemRefDescriptor dataDesc(adaptor.getData());
    for (auto dimIdx : llvm::seq<int64_t>(dataType.getRank())) {
      Value dimSize = dataType.isDynamicDim(dimIdx)
                          ? dataDesc.size(rewriter, loc, dimIdx)
                          : createI64Const(dataType.getDimSize(dimIdx));
      dataNumElements =
          LLVM::MulOp::create(rewriter, loc, dataNumElements, dimSize);
    }

    Value outputNumElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());
    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize = outputType.isDynamicDim(dimIdx)
                          ? outputDesc.size(rewriter, loc, dimIdx)
                          : createI64Const(outputType.getDimSize(dimIdx));
      outputNumElements =
          LLVM::MulOp::create(rewriter, loc, outputNumElements, dimSize);
    }

    Value axesNumElements = createI64Const(1);
    MemRefDescriptor axesDesc(adaptor.getAxes());
    for (auto dimIdx : llvm::seq<int64_t>(axesType.getRank())) {
      Value dimSize = axesType.isDynamicDim(dimIdx)
                          ? axesDesc.size(rewriter, loc, dimIdx)
                          : createI64Const(axesType.getDimSize(dimIdx));
      axesNumElements =
          LLVM::MulOp::create(rewriter, loc, axesNumElements, dimSize);
    }

    Value dataTypeVal = createI64Const(dataTypeEnum);
    Value keepdimsVal = createI64Const(op.getKeepdims());
    Value noopWithEmptyAxesVal = createI64Const(op.getNoopWithEmptyAxes());

    // int wrap_reduce_max(RuntimeState* state, void* data, void* axes,
    //                     void* output, int64_t data_num_elements,
    //                     int64_t output_num_elements,
    //                     int64_t axes_num_elements, int64_t data_type,
    //                     int64_t keepdims, int64_t noop_with_empty_axes)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReduceMax, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {statePtr,        dataPtr,
                                   axesPtr,         outputPtr,
                                   dataNumElements, outputNumElements,
                                   axesNumElements, dataTypeVal,
                                   keepdimsVal,     noopWithEmptyAxesVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateReduceMaxLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReduceMaxOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
