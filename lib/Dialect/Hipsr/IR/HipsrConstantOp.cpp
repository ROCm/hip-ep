/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult ConstantOp::verify() {
  bool hasValue = getValueAttr() != nullptr;
  bool hasSource = getSourceAttr() != nullptr;
  bool hasOffset = getOffsetAttr() != nullptr;
  bool hasSize = getSizeAttr() != nullptr;
  bool hasIndex = getIndexAttr() != nullptr;

  if (hasValue == hasSource) {
    return emitOpError("expected exactly one of {value} or {source}");
  }

  if (hasOffset != hasSize || hasOffset != hasIndex) {
    return emitOpError("`offset`, `size` and `index` must be set together");
  }

  return success();
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.cpp.inc"

namespace {

constexpr const char *kHipsrGetConstant = "hipdnn_ep_constant_get";

struct ConstantLowering : public ConvertOpToLLVMPattern<ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isExternalized()) {
      op.emitError(
          "hipsr.constant reached LLVM lowering without externalization; "
          "run -hipsr-externalize-constants first");
      return failure();
    }

    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i64Type = IntegerType::get(ctx, 64);

    auto memRefType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memRefType) {
      op.emitError("hipsr.constant: externalized constant must have a memref "
                   "result to lower to a runtime call");
      return failure();
    }

    auto llvmFn = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFn) {
      return rewriter.notifyMatchFailure(op, "expected enclosing llvm.func");
    }
    if (llvmFn.getNumArguments() == 0) {
      op.emitError("hipsr.constant: enclosing function has no arguments; "
                   "expected the runtime context at arg 0");
      return failure();
    }
    Value ctxArg = llvmFn.getArgument(0);

    Value indexVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getIndexAttr().getInt()));

    SmallVector<Type, 2> paramTypes = {ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipsrGetConstant, paramTypes, ptrType);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 2> args = {ctxArg, indexVal};
    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace)) {
      return failure();
    }

    Value dataPtr = callOp.getResult();
    // TODO: hipdnn_ep_constant_get should return !llvm.ptr<1> so this cast is
    // unnecessary.
    if (*addrSpace != 0) {
      dataPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, *addrSpace), dataPtr);
    }

    auto shape = memRefType.getShape();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    for (int64_t dim : shape) {
      sizes.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dim)));
    }
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), LLVM::ConstantOp::create(
                                          rewriter, loc, i64Type,
                                          rewriter.getI64IntegerAttr(stride)));
      stride *= shape[i];
    }

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, dataPtr, dataPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrConstantLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConstantLowering>(converter);
}
