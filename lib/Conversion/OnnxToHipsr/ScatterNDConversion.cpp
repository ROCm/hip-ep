/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ScatterNDConversion.cpp - onnx.ScatterND -> hipsr.scatter_nd ------===//
//
// onnx.ScatterND overwrites the slices its indices address, which
// hipsr.scatter_nd models directly. ONNX also reduces a duplicate index into
// its destination through `reduction`; that is a read-modify-write rather than
// an overwrite, so this rejects the four reducing modes.
//
// The placeholder's shape region is left empty: hipsr.scatter_nd is DPS, so
// hipsr-populate-shape-region fills it in later.
//
// Before, writing image features into token embeddings:
//   %o = "onnx.ScatterND"(%embeds, %positions, %features) {reduction = "none"}
//       : (tensor<?x?x4096xf16, #hipsr.mem<device>>,
//          tensor<?x3xi64, #hipsr.mem<device>>,
//          tensor<?xf16, #hipsr.mem<device>>) -> tensor<?x?x4096xf16>
//
// After, with the ins types left out:
//   %init = hipsr.placeholder(%ctx) ins(%embeds)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?x?x4096xf16, #hipsr.mem<device>>
//   %o = hipsr.scatter_nd(%ctx) ins(%embeds, %positions, %features)
//       outs(%init : tensor<?x?x4096xf16, #hipsr.mem<device>>)
//       : tensor<?x?x4096xf16, #hipsr.mem<device>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hipsr {
namespace {

struct ScatterNDToHipsr : public OpConversionPattern<onnx::ScatterNDOp> {
  // The type converter is unused: converting the operands would overwrite the
  // memory space their producers chose. It stays in the signature so every
  // pattern is built the same way.
  ScatterNDToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::ScatterNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getReduction() != "none") {
      return rewriter.notifyMatchFailure(op, "expected reduction = none");
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor result");
    }
    // The scatter runs on the device and this emits no copy, so a host operand
    // is rejected. That also rules out an unranked one, which carries no space.
    // TODO: Copy a host operand to the device instead.
    if (!llvm::all_of(adaptor.getOperands(), [](Value operand) {
          return isDeviceRankedTensor(operand.getType());
        })) {
      return rewriter.notifyMatchFailure(op,
                                         "expected device-resident operands");
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The output takes the data's shape, so the data alone drives the
    // placeholder.
    Location loc = op.getLoc();
    Value data = adaptor.getData();
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);
    Value init =
        PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                              ValueRange{data}, PlaceholderType::Normal)
            .getResult(0);
    auto scatterOp =
        ScatterNDOp::create(rewriter, loc, TypeRange{resultType}, *ctx, data,
                            adaptor.getIndices(), adaptor.getUpdates(), init);
    rewriter.replaceOp(op, scatterOp.getResult(0));
    return success();
  }
};

} // namespace

void populateScatterNDConversionPatterns(const TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ScatterNDToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
