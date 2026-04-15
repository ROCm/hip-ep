/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Generic template for unary power operations lowering
// Handles: hip.reciprocal, hip.sqrt (and future: hip.square, hip.cube, etc.)
//
// All power ops lower to: wrap_<op>(state, input, output, num_elements, data_type)
// Runtime uses miopenActivationPOWER with different gamma values:
//   - Reciprocal: gamma=-1.0  → x^(-1) = 1/x
//   - Sqrt:       gamma=0.5   → x^(0.5) = √x
//   - Square:     gamma=2.0   → x^2
//   - Cube:       gamma=3.0   → x^3
template <typename OpTy>
struct PowerOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;
  const char *opName;

  PowerOpLowering(const LLVMTypeConverter &converter, const char *func,
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
    if (dataType < 0) {
      std::string msg = "unsupported element type for hip.";
      msg += opName;
      return rewriter.notifyMatchFailure(op, msg);
    }

    Value dataTypeVal = createI64Const(dataType);

    // int wrap_<op>(RuntimeState* state, void* input, void* output,
    //               int64_t num_elements, int64_t data_type)
    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes, i32Type);
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
  patterns.insert<PowerOpLowering<ReciprocalOp>>(converter, kWrapReciprocal,
                                                  "reciprocal");
  patterns.insert<PowerOpLowering<SqrtOp>>(converter, kWrapSqrt, "sqrt");
}

} // namespace hip
} // namespace mlir
