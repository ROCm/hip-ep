/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReturnConversion.cpp - Convert onnx.Return to func.return ----------===//
//
// onnx.Return carries graph structure rather than a computation, so it lowers
// to func.return instead of to a hipsr operation.
//
// The conversion also retypes the enclosing function's results. A pattern does
// not carry the pass converter, so an operand keeps the memory space its
// producer chose and the signature follows it.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hipsr {
namespace {

struct ReturnToFunc
    : public ::mlir::OpConversionPattern<::mlir::onnx::ReturnOp> {
  ReturnToFunc(const ::mlir::TypeConverter &typeConverter,
               ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::ReturnOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    ::mlir::ValueRange operands = adaptor.getOperands();
    // onnx.Return carries HasParent<func::FuncOp>.
    auto funcOp = ::mlir::cast<::mlir::func::FuncOp>(op->getParentOp());
    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setFunctionType(rewriter.getFunctionType(
          funcOp.getFunctionType().getInputs(), operands.getTypes()));
    });
    rewriter.replaceOpWithNewOp<::mlir::func::ReturnOp>(op, operands);
    return ::mlir::success();
  }
};

} // namespace

void populateReturnConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx) {
  patterns.add<ReturnToFunc>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
