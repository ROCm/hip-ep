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
// Ops lower to wrap_power(state, input, output, num_elements, data_type,
// alpha, beta, gamma). Reciprocal (gamma=-1) and Sqrt (gamma=0.5) use HIP
// elementwise kernels at runtime; other powers use MIOpen POWER: y=(α+βx)^γ.
//   - Reciprocal: alpha=0, beta=1, gamma=-1.0 → HIP elementwise 1/x
//   - Sqrt:       alpha=0, beta=1, gamma=0.5   → HIP elementwise sqrt (ONNX)
//   - Square:     alpha=0, beta=1, gamma=2.0   → (0 + 1*x)^2 = x^2
//   - Cube:       alpha=0, beta=1, gamma=3.0   → (0 + 1*x)^3 = x^3
template <typename OpTy>
struct PowerOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  double alpha, beta, gamma;
  const char *opName;

  PowerOpLowering(const LLVMTypeConverter &converter, double a, double b,
                  double g, const char *op)
      : ConvertOpToLLVMPattern<OpTy>(converter), alpha(a), beta(b), gamma(g),
        opName(op) {}

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
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "hip power lowering expects ranked memref outs operand");

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
    Value alphaVal = createF64Const(alpha);
    Value betaVal = createF64Const(beta);
    Value gammaVal = createF64Const(gamma);

    // int wrap_power(RuntimeState* state, void* input, void* output,
    //                int64_t num_elements, int64_t data_type,
    //                double alpha, double beta, double gamma)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, f64Type, f64Type, f64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapPower, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,    inputPtr, outputPtr, numElements,
                                  dataTypeVal, alphaVal, betaVal,   gammaVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePowerLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Reciprocal: (0 + 1*x)^(-1) = 1/x; Sqrt: (0 + 1*x)^(0.5) = √x
  // Same LLVM callee @wrap_power; reciprocal/sqrt use HIP kernels in power.cpp.
  patterns.insert<PowerOpLowering<ReciprocalOp>>(converter, 0.0, 1.0, -1.0,
                                                 "reciprocal");
  patterns.insert<PowerOpLowering<SqrtOp>>(converter, 0.0, 1.0, 0.5, "sqrt");
}

} // namespace hip
} // namespace mlir
