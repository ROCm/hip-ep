/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- EqualConversion.cpp - Convert onnx.Equal to hipsr.equal ------------===//
//
// The mask becomes ui8, the byte the runtime writes per element. The runtime
// reads both operands from device memory, so a host constant operand gets a
// device constant of its own.
//
// The placeholder's shape region is left empty: hipsr.equal is DPS, so
// hipsr-populate-shape-region fills it in later, as for every DPS op.
//
// Before, comparing token ids against one id held as a host scalar:
//   %id = arith.constant dense<248056> : tensor<i64>
//   %m = "onnx.Equal"(%ids, %id)
//       : (tensor<?x?xi64, #hipsr.mem<device>>, tensor<i64>)
//       -> tensor<?x?xi1>
//
// After, with the ins types left out:
//   %token = hipsr.constant {value = dense<248056> : tensor<i64>}
//       : tensor<i64, #hipsr.mem<device>>
//   %init = hipsr.placeholder(%ctx) ins(%ids, %token)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?x?xui8, #hipsr.mem<device>>
//   %m = hipsr.equal(%ctx) ins(%ids, %token)
//       outs(%init : tensor<?x?xui8, #hipsr.mem<device>>)
//       : tensor<?x?xui8, #hipsr.mem<device>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {
namespace {

bool onDevice(Value operand) { return isDeviceRankedTensor(operand.getType()); }

// Builds a device constant with the same value and shape as a host one.
//
// The constant conversion leaves a rank-0 constant on the host, which is what
// an axis or an index needs, but this operand is data to compare. The rank is
// kept as it is: the runtime pads each shape to 4D with leading ones, so a
// rank-0 operand already broadcasts.
FailureOr<Value> constantOnDevice(ConversionPatternRewriter &rewriter,
                                  Location loc, Value operand) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  DenseElementsAttr value;
  if (!type || !matchPattern(operand, m_Constant(&value))) {
    return failure();
  }
  return ConstantOp::create(rewriter, loc,
                            tensorTypeInSpace(type, MemorySpace::Device), value,
                            /*index=*/IntegerAttr(), /*offset=*/IntegerAttr(),
                            /*size=*/IntegerAttr())
      .getResult();
}

struct EqualToHipsr : public OpConversionPattern<onnx::EqualOp> {
  // The type converter is unused: this pattern builds the mask type itself, and
  // converting the operands would overwrite the memory space their producers
  // chose. It stays in the signature so every pattern is built the same way.
  EqualToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getC().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The comparison runs on the device and this adds no copy, so a host
    // operand is only taken as a constant, which gets a device one.
    // TODO: Compare two host values on the host instead.
    Location loc = op.getLoc();
    SmallVector<Value, 2> operands{adaptor.getA(), adaptor.getB()};
    if (llvm::none_of(operands, onDevice)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a device-resident operand");
    }
    // TODO: Copy a host operand that holds no compile-time value.
    for (Value &operand : operands) {
      if (onDevice(operand)) {
        continue;
      }
      FailureOr<Value> deviceOperand = constantOnDevice(rewriter, loc, operand);
      if (failed(deviceOperand)) {
        return rewriter.notifyMatchFailure(
            op, "expected a device-resident operand or a constant");
      }
      operand = *deviceOperand;
    }

    // ONNX declares a bool mask; hipsr uses the byte the runtime writes.
    RankedTensorType maskType = tensorTypeInSpace(
        RankedTensorType::get(resultType.getShape(),
                              rewriter.getIntegerType(8, /*isSigned=*/false)),
        MemorySpace::Device);
    Value init = PlaceholderOp::create(rewriter, loc, TypeRange{maskType}, *ctx,
                                       operands, PlaceholderType::Normal)
                     .getResult(0);
    auto equalOp = EqualOp::create(rewriter, loc, TypeRange{maskType}, *ctx,
                                   operands[0], operands[1], init);
    rewriter.replaceOp(op, equalOp.getResult(0));
    return success();
  }
};

} // namespace

void populateEqualConversionPatterns(const TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<EqualToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
