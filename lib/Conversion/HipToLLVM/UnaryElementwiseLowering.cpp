/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Generic template for unary elementwise operations lowering.
// Handles: hip.neg, hip.not, hip.cos, hip.sin, hip.sign (and future unary
// elementwise ops).
//
// Ops lower to wrap_{op}(state, input, output, num_elements, data_type).
template <typename OpTy>
struct UnaryElementwiseOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;
  const char *opName;

  UnaryElementwiseOpLowering(const LLVMTypeConverter &converter,
                             const char *func, const char *op)
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
    Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outputType) {
      std::string msg = "hip.";
      msg += opName;
      msg += " lowering expects ranked memref outs operand";
      return rewriter.notifyMatchFailure(op, msg);
    }

    Value numElements =
        computeNumElements(outputType, adaptor.getY(), rewriter, loc);

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    // For bool (i1) types (e.g. hip.not), data_type is not in the standard
    // enum. Pass 0 since the runtime stub is empty and doesn't use it.
    if (dataType < 0 && outputType.getElementType().isInteger(1))
      dataType = 0;
    if (dataType < 0) {
      std::string msg = "unsupported element type for hip.";
      msg += opName;
      return rewriter.notifyMatchFailure(op, msg);
    }

    Value dataTypeVal = createI64Const(dataType);

    // int wrap_{op}(RuntimeState* state, void* input, void* output,
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

void populateUnaryElementwiseLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<UnaryElementwiseOpLowering<AbsOp>>(converter, kWrapAbs,
                                                     "abs");
  patterns.insert<UnaryElementwiseOpLowering<NegOp>>(converter, kWrapNeg,
                                                     "neg");
  patterns.insert<UnaryElementwiseOpLowering<NotOp>>(converter, kWrapNot,
                                                     "not");
  patterns.insert<UnaryElementwiseOpLowering<CosOp>>(converter, kWrapCos,
                                                     "cos");
  patterns.insert<UnaryElementwiseOpLowering<SinOp>>(converter, kWrapSin,
                                                     "sin");
  patterns.insert<UnaryElementwiseOpLowering<CeilOp>>(converter, kWrapCeil,
                                                      "ceil");
  patterns.insert<UnaryElementwiseOpLowering<ExpOp>>(converter, kWrapExp,
                                                     "exp");
  patterns.insert<UnaryElementwiseOpLowering<SignOp>>(converter, kWrapSign,
                                                      "sign");
}

} // namespace hip
} // namespace mlir
