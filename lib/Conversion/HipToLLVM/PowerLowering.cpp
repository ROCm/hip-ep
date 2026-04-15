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
// All power ops lower to unified: wrap_power(state, input, output, num_elements, data_type, gamma)
// Runtime uses miopenActivationPOWER with formula: y = x^gamma
//   - Reciprocal: gamma=-1.0  → x^(-1) = 1/x
//   - Sqrt:       gamma=0.5   → x^(0.5) = √x
//   - Square:     gamma=2.0   → x^2
//   - Cube:       gamma=3.0   → x^3
template <typename OpTy>
struct PowerOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  double gamma;
  const char *opName;

  PowerOpLowering(const LLVMTypeConverter &converter, double g, const char *op)
      : ConvertOpToLLVMPattern<OpTy>(converter), gamma(g), opName(op) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    // Helper to create constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF64Const = [&](double value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f64Type,
                                      rewriter.getF64FloatAttr(value));
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
    Value gammaVal = createF64Const(gamma);

    // int wrap_power(RuntimeState* state, void* input, void* output,
    //                int64_t num_elements, int64_t data_type, double gamma)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, f64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapPower, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr, inputPtr, outputPtr, numElements,
                                  dataTypeVal, gammaVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populatePowerLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<PowerOpLowering<ReciprocalOp>>(converter, -1.0, "reciprocal");
  patterns.insert<PowerOpLowering<SqrtOp>>(converter, 0.5, "sqrt");
}

} // namespace hip
} // namespace mlir
