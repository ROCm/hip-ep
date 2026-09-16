/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TransposeConversion.cpp - Convert onnx.Transpose to hipsr.transpose =//
//
// onnx.Transpose permutes axes, which hipsr.transpose models directly. ONNX
// leaves `perm` optional and defaults it to the reverse permutation, so the
// conversion spells that default out.
//
// The placeholder's shape region is left empty: hipsr.transpose is DPS, so
// hipsr-populate-shape-region fills it in later, as for every DPS op.
//
// Before, the index matrix NonZero produces:
//   %t = "onnx.Transpose"(%idx) {perm = [1, 0]}
//       : (tensor<3x?xi64, #hipsr.mem<device>>) -> tensor<?x3xi64>
//
// After:
//   %init = hipsr.placeholder(%ctx) ins(%idx)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?x3xi64, #hipsr.mem<device>>
//   %t = hipsr.transpose(%ctx) ins(%idx : tensor<3x?xi64, #hipsr.mem<device>>)
//       outs(%init : tensor<?x3xi64, #hipsr.mem<device>>)
//       {perm = array<i64: 1, 0>} : tensor<?x3xi64, #hipsr.mem<device>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {
namespace hipsr {
namespace {

// An absent `perm` means the reverse permutation.
SmallVector<int64_t> resolvePerm(onnx::TransposeOp op, int64_t rank) {
  ArrayAttr perm = op.getPermAttr();
  if (!perm) {
    return llvm::to_vector(llvm::reverse(llvm::seq<int64_t>(0, rank)));
  }
  return llvm::map_to_vector(
      perm.getAsValueRange<IntegerAttr>(),
      [](const APInt &axis) { return axis.getSExtValue(); });
}

struct TransposeToHipsr : public OpConversionPattern<onnx::TransposeOp> {
  // The type converter is unused: converting the operand would overwrite the
  // memory space its producer chose. It stays in the signature so every pattern
  // is built the same way.
  TransposeToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getTransposed().getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    // The permutation runs on the device and this adds no copy, so a host
    // input is rejected.
    if (!isDeviceRankedTensor(inputType)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a device-resident input");
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<int64_t> perm = resolvePerm(op, inputType.getRank());
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);
    Value init =
        PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal)
            .getResult(0);
    auto transposeOp =
        TransposeOp::create(rewriter, loc, TypeRange{resultType}, *ctx, input,
                            init, rewriter.getDenseI64ArrayAttr(perm));
    rewriter.replaceOp(op, transposeOp.getResult(0));
    return success();
  }
};

} // namespace

void populateTransposeConversionPatterns(const TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<TransposeToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
