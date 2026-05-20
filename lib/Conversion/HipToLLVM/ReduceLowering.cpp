/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Generic template for reduction-style ops with the shared
//   (data, axes, output, keepdims, noop_with_empty_axes)
// signature, lowered to wrap_{op}.
//
// Handles: hip.reduce_sum, hip.reduce_max, hip.reduce_prod.
//
// All three runtime functions share the exact same calling convention:
//   int wrap_reduce_*(RuntimeState* state, void* data, void* axes,
//                     void* output, int64_t data_num_elements,
//                     int64_t output_num_elements,
//                     int64_t axes_num_elements, int64_t data_type,
//                     int64_t keepdims, int64_t noop_with_empty_axes)
//
// Supports both static and dynamic shapes (computes num_elements at runtime).
template <typename OpTy>
struct ReduceOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;
  const char *opName;

  ReduceOpLowering(const LLVMTypeConverter &converter, const char *func,
                   const char *op)
      : ConvertOpToLLVMPattern<OpTy>(converter), funcName(func), opName(op) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
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
    if (dataTypeEnum < 0) {
      std::string msg = "unsupported element type for hip.";
      msg += opName;
      return rewriter.notifyMatchFailure(op, msg);
    }

    Value dataNumElements =
        computeNumElements(dataType, adaptor.getData(), rewriter, loc);
    Value outputNumElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);
    Value axesNumElements =
        computeNumElements(axesType, adaptor.getAxes(), rewriter, loc);

    Value dataTypeVal = createI64Const(dataTypeEnum);
    Value keepdimsVal = createI64Const(op.getKeepdims());
    Value noopWithEmptyAxesVal = createI64Const(op.getNoopWithEmptyAxes());

    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes, i32Type);
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

void populateReduceLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.insert<ReduceOpLowering<ReduceSumOp>>(converter, kWrapReduceSum,
                                                 "reduce_sum");
  patterns.insert<ReduceOpLowering<ReduceMaxOp>>(converter, kWrapReduceMax,
                                                 "reduce_max");
  patterns.insert<ReduceOpLowering<ReduceProdOp>>(converter, kWrapReduceProd,
                                                  "reduce_prod");
}

} // namespace hip
} // namespace mlir
