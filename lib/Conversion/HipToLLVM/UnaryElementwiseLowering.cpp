/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.unary_elementwise to wrap_elementwise_unary in the runtime
// bitcode.  Identical pattern to PowerLowering.cpp; the kind discriminator
// + (alpha, beta) attribute pair flow straight through.

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct UnaryElementwiseLowering
    : public ConvertOpToLLVMPattern<UnaryElementwiseOp> {
  using ConvertOpToLLVMPattern<UnaryElementwiseOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnaryElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getX(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getY(), rewriter, loc);

    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.unary_elementwise expects a ranked memref output");

    Value numElements =
        computeNumElements(outputType, adaptor.getY(), rewriter, loc);

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.unary_elementwise unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto f64Const = [&](double v) {
      return LLVM::ConstantOp::create(rewriter, loc, f64Type,
                                      rewriter.getF64FloatAttr(v));
    };

    Value dtype = i64Const(dataType);
    Value kind = i64Const(static_cast<int64_t>(op.getKind()));
    Value alpha = f64Const(op.getAlpha().convertToDouble());
    Value beta = f64Const(op.getBeta().convertToDouble());

    // int wrap_elementwise_unary(RuntimeState*, void* in, void* out,
    //                            int64 num_elements, int64 data_type,
    //                            int64 kind, double alpha, double beta)
    SmallVector<Type, 8> paramTypes = {ptrType,  ptrType, ptrType, i64Type,
                                       i64Type,  i64Type, f64Type, f64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapElementwiseUnary, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr, inputPtr, outputPtr, numElements,
                                  dtype,    kind,     alpha,     beta};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateUnaryElementwiseLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<UnaryElementwiseLowering>(converter);
}

} // namespace hip
} // namespace mlir
