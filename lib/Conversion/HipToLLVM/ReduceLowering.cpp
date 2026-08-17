/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

namespace mlir {
namespace hip {
namespace {

// Generic template for reduction-style ops with the shared
//   (data, axes, output, keepdims, noop_with_empty_axes)
// signature, lowered to wrap_{op}.
//
// Handles: hip.reduce_sum, hip.reduce_mean, hip.reduce_max, hip.reduce_min,
// hip.reduce_prod.
//
// All six runtime functions share the exact same calling convention:
//   int wrap_reduce_*(RuntimeState* state, void* data, void* axes,
//                     void* output, int64_t data_num_elements,
//                     int64_t output_num_elements,
//                     int64_t axes_num_elements, int64_t data_type,
//                     int64_t keepdims, int64_t noop_with_empty_axes,
//                     int64_t inner_size)
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
    auto dataType = cast<MemRefType>(op.getData().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    StringRef operationName = op->getName().getStringRef();
    if (!isSupportedReductionElementType(operationName,
                                         dataType.getElementType()))
      return op.emitOpError()
             << "lowering does not support element type "
             << dataType.getElementType() << "; supported types: "
             << getSupportedReductionElementTypes(operationName);
    // The HIP op verifier compares normalized_axes against the structural
    // constant source before dialect conversion starts. By the time this
    // pattern runs, memref.get_global may already be rewritten to LLVM, so the
    // durable normalized attribute is the semantic input here.
    FailureOr<SmallVector<int64_t>> normalizedAxes =
        normalizeReductionAxes(dataType.getRank(), op.getNormalizedAxes());
    if (failed(normalizedAxes) ||
        !llvm::equal(op.getNormalizedAxes(), *normalizedAxes))
      return op.emitOpError(
          "lowering requires normalized_axes to be a normalized contiguous "
          "span");

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
    Value axesNumElements = createI64Const(normalizedAxes->size());

    Value dataTypeVal = createI64Const(dataTypeEnum);
    Value keepdimsVal = createI64Const(op.getKeepdims());
    Value noopWithEmptyAxesVal = createI64Const(op.getNoopWithEmptyAxes());

    // The runtime kernel treats a contiguous reduced axis span as one flattened
    // reduction dimension. `inner_size` is therefore the product of input
    // dimensions strictly after the span's final axis. Derive it from the
    // validated axes source, never by comparing input/output extents: equal
    // extents can make axis 0 and axis 1 shape-indistinguishable.
    int64_t staticInnerSize = 1;
    SmallVector<int64_t> dynamicInnerDims;
    if (!normalizedAxes->empty()) {
      int64_t lastReducedAxis = normalizedAxes->back();
      for (int64_t dimIdx :
           llvm::seq<int64_t>(lastReducedAxis + 1, dataType.getRank())) {
        if (dataType.isDynamicDim(dimIdx))
          dynamicInnerDims.push_back(dimIdx);
        else
          staticInnerSize *= dataType.getDimSize(dimIdx);
      }
    }
    Value innerSizeVal = createI64Const(staticInnerSize);
    for (int64_t dimIdx : dynamicInnerDims)
      innerSizeVal = LLVM::MulOp::create(
          rewriter, loc, innerSizeVal,
          getMemRefDimSize(dataType, dimIdx, adaptor.getData(), rewriter, loc));

    SmallVector<Type, 11> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    FailureOr<LLVM::LLVMFuncOp> recordStatusFunc = LLVM::lookupOrCreateFn(
        rewriter, module, kHipRecordStatus, {ptrType, i32Type}, i32Type);
    if (failed(recordStatusFunc))
      return failure();

    SmallVector<Value, 11> args = {statePtr,        dataPtr,
                                   axesPtr,         outputPtr,
                                   dataNumElements, outputNumElements,
                                   axesNumElements, dataTypeVal,
                                   keepdimsVal,     noopWithEmptyAxesVal,
                                   innerSizeVal};

    Value status =
        LLVM::CallOp::create(rewriter, loc, *funcOp, args).getResult();
    LLVM::CallOp::create(rewriter, loc, *recordStatusFunc,
                         ValueRange{statePtr, status});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateReduceLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.insert<ReduceOpLowering<ReduceSumOp>>(converter, kWrapReduceSum,
                                                 "reduce_sum");
  patterns.insert<ReduceOpLowering<ReduceMeanOp>>(converter, kWrapReduceMean,
                                                  "reduce_mean");
  patterns.insert<ReduceOpLowering<ReduceL2Op>>(converter, kWrapReduceL2,
                                                "reduce_l2");
  patterns.insert<ReduceOpLowering<ReduceMaxOp>>(converter, kWrapReduceMax,
                                                 "reduce_max");
  patterns.insert<ReduceOpLowering<ReduceMinOp>>(converter, kWrapReduceMin,
                                                 "reduce_min");
  patterns.insert<ReduceOpLowering<ReduceProdOp>>(converter, kWrapReduceProd,
                                                  "reduce_prod");
}

} // namespace hip
} // namespace mlir
