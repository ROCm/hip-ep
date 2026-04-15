/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.reciprocal(ctx, input, output)
//   -> wrap_reciprocal(state, input, output, num_elements, data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
//
// Runtime implementation uses miopenActivationPOWER with gamma=-1.0
// Formula: (0 + 1*x)^(-1) = x^(-1) = 1/x
struct ReciprocalOpLowering : public ConvertOpToLLVMPattern<ReciprocalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReciprocalOp op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getY(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getY().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getY());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Get data type enum (f32=0, f16=1, bf16=2)
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.reciprocal");

    Value dataTypeVal = createI64Const(dataType);

    // int wrap_reciprocal(RuntimeState* state, void* input, void* output,
    //                     int64_t num_elements, int64_t data_type)
    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReciprocal, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {statePtr, inputPtr, outputPtr, numElements,
                                  dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.sqrt(ctx, input, output)
//   -> wrap_sqrt(state, input, output, num_elements, data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
//
// Runtime implementation uses miopenActivationPOWER with gamma=0.5
// Formula: (0 + 1*x)^(0.5) = x^(0.5) = √x
struct SqrtOpLowering : public ConvertOpToLLVMPattern<SqrtOp> {
  using ConvertOpToLLVMPattern<SqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, SqrtOp::Adaptor adaptor,
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

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getY(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getY().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getY());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Get data type enum (f32=0, f16=1, bf16=2)
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.sqrt");

    Value dataTypeVal = createI64Const(dataType);

    // int wrap_sqrt(RuntimeState* state, void* input, void* output,
    //               int64_t num_elements, int64_t data_type)
    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSqrt, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {statePtr, inputPtr, outputPtr, numElements,
                                  dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populatePowerLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReciprocalOpLowering, SqrtOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
