/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.reduce_sum(ctx, input, output) {axes = [...], keepdims = ...}
//   -> wrap_miopenReduceSum(state, input, output, num_elements,
//                           axes_ptr, num_axes, keepdims, data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct ReduceSumOpLowering : public ConvertOpToLLVMPattern<ReduceSumOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers using alignedPtr
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

    // Compute data_num_elements (supports dynamic shapes)
    Value dataNumElements = createI64Const(1);
    MemRefDescriptor dataDesc(adaptor.getData());

    for (auto dimIdx : llvm::seq<int64_t>(dataType.getRank())) {
      Value dimSize;
      if (dataType.isDynamicDim(dimIdx)) {
        dimSize = dataDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(dataType.getDimSize(dimIdx));
      }
      dataNumElements =
          LLVM::MulOp::create(rewriter, loc, dataNumElements, dimSize);
    }

    // Compute output_num_elements (supports dynamic shapes)
    Value outputNumElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      outputNumElements =
          LLVM::MulOp::create(rewriter, loc, outputNumElements, dimSize);
    }

    // Compute axes_num_elements to detect empty axes
    Value axesNumElements = createI64Const(1);
    MemRefDescriptor axesDesc(adaptor.getAxes());
    for (auto dimIdx : llvm::seq<int64_t>(axesType.getRank())) {
      Value dimSize;
      if (axesType.isDynamicDim(dimIdx)) {
        dimSize = axesDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(axesType.getDimSize(dimIdx));
      }
      axesNumElements =
          LLVM::MulOp::create(rewriter, loc, axesNumElements, dimSize);
    }

    // Element size in bytes
    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    Value keepdimsVal = createI64Const(op.getKeepdims());
    Value noopWithEmptyAxesVal = createI64Const(op.getNoopWithEmptyAxes());

    // int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
    //                     void* output, int64_t data_num_elements,
    //                     int64_t output_num_elements, int64_t
    //                     axes_num_elements, int64_t element_size_bytes,
    //                     int64_t keepdims, int64_t noop_with_empty_axes)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReduceSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {statePtr,        dataPtr,
                                   axesPtr,         outputPtr,
                                   dataNumElements, outputNumElements,
                                   axesNumElements, elemSizeVal,
                                   keepdimsVal,     noopWithEmptyAxesVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateReduceSumLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReduceSumOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
