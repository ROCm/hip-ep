/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Generic template for unary elementwise operations lowering.
// Handles: hip.neg, hip.not, hip.cos, hip.erf, hip.sin, hip.round, hip.atan,
// hip.floor, hip.sign, hip.exp, hip.sigmoid, hip.tanh (and other unary
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

// hip.isnan writes a 1-byte boolean. data_type is the INPUT float type;
// the output element type is not a HIPDNN float enum value.
struct IsNaNOpLowering : public ConvertOpToLLVMPattern<IsNaNOp> {
  using ConvertOpToLLVMPattern<IsNaNOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IsNaNOp op, IsNaNOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = dyn_cast<MemRefType>(op.getX().getType());
    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.isnan lowering expects ranked memref operands");

    Type outputElem = outputType.getElementType();
    if (!outputElem.isInteger(1) && !outputElem.isInteger(8))
      return rewriter.notifyMatchFailure(
          op, "hip.isnan output must be a 1-byte boolean");

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.isnan");

    Value numElements =
        computeNumElements(outputType, adaptor.getY(), rewriter, loc);
    Value dataTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dataType));

    // int wrap_isnan(RuntimeState* state, void* input, void* output,
    //                int64_t num_elements, int64_t data_type)
    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapIsNaN, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc), numElements,
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
  patterns.add<IsNaNOpLowering>(converter);
  patterns.insert<UnaryElementwiseOpLowering<CosOp>>(converter, kWrapCos,
                                                     "cos");
  patterns.insert<UnaryElementwiseOpLowering<ErfOp>>(converter, kWrapErf,
                                                     "erf");
  patterns.insert<UnaryElementwiseOpLowering<SinOp>>(converter, kWrapSin,
                                                     "sin");
  patterns.insert<UnaryElementwiseOpLowering<CeilOp>>(converter, kWrapCeil,
                                                      "ceil");
  patterns.insert<UnaryElementwiseOpLowering<RoundOp>>(converter, kWrapRound,
                                                       "round");
  patterns.insert<UnaryElementwiseOpLowering<AtanOp>>(converter, kWrapAtan,
                                                      "atan");
  patterns.insert<UnaryElementwiseOpLowering<FloorOp>>(converter, kWrapFloor,
                                                       "floor");
  patterns.insert<UnaryElementwiseOpLowering<ExpOp>>(converter, kWrapExp,
                                                     "exp");
  patterns.insert<UnaryElementwiseOpLowering<SigmoidOp>>(
      converter, kWrapSigmoid, "sigmoid");
  patterns.insert<UnaryElementwiseOpLowering<TanhOp>>(converter, kWrapTanh,
                                                      "tanh");
  patterns.insert<UnaryElementwiseOpLowering<LogOp>>(converter, kWrapLog,
                                                     "log");
  patterns.insert<UnaryElementwiseOpLowering<SignOp>>(converter, kWrapSign,
                                                      "sign");
}

} // namespace hip
} // namespace mlir
