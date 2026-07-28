/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX com.microsoft.GatherBlockQuantized -> HIP gather_block_quantized
//===----------------------------------------------------------------------===//
//
// Before:
//   %out = "onnx.Custom"(%data, %indices, %scales, %zero_points)
//       {function_name = "GatherBlockQuantized",
//        domain_name = "com.microsoft",
//        bits = 4 : si64, block_size = 16 : si64,
//        gather_axis = 0 : si64, quantize_axis = 1 : si64}
//       : (tensor<2048x96xui8>, tensor<8xi64>,
//          tensor<2048x12xf16>, tensor<2048x12xui8>) -> tensor<8x96xf16>
//
// After:
//   %init = tensor.empty() : tensor<8x96xf16>
//   %out = hip.gather_block_quantized(%ctx)
//       ins(%data, %indices, %scales :
//           tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>)
//       zero_points(%zp : tensor<2048x12xui8>)
//       outs(%init : tensor<8x96xf16>)
//       {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
//       : tensor<8x96xf16>
//
// Output shape derivation (mirrors plain Gather):
//   out.shape = data.shape[0:gather_axis]
//             ++ indices.shape
//             ++ data.shape[gather_axis+1:]
// The dequant block axis lives entirely inside `data`, so the output has
// no extra "blocks" dim — the runtime fans out per-element on the gathered
// rows during the dequantize step.

struct GatherBlockQuantizedToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherBlockQuantizedToHip)
  GatherBlockQuantizedToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult GatherBlockQuantizedToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GatherBlockQuantized")
    return rewriter.notifyMatchFailure(op,
                                       "not a GatherBlockQuantized custom op");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 3)
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs (data, indices, scales)");
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected exactly 1 result (output)");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Value data = op->getOperand(0);
  mlir::Value indices = op->getOperand(1);
  mlir::Value scales = op->getOperand(2);

  // ONNX models that omit `zero_points` may either drop the operand entirely
  // (only 3 operands present) or pass an `onnx.NoValue` of !NoneType — match
  // MatMulNBitsConversion's handling so both forms produce a null Value.
  mlir::Value zeroPoints;
  if (op->getNumOperands() >= 4) {
    mlir::Value v = op->getOperand(3);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoints = v;
  }

  // Attribute extraction. Spec defaults:
  //   bits           — required, 4 or 8
  //   block_size     — required, power of 2 >= 16
  //   gather_axis    — optional, default 0
  //   quantize_axis  — optional, default 0
  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  if (!bitsIntAttr)
    return rewriter.notifyMatchFailure(op, "missing required `bits` attribute");
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  if (!blockSizeIntAttr)
    return rewriter.notifyMatchFailure(
        op, "missing required `block_size` attribute");
  auto gatherAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("gather_axis");
  auto quantAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("quantize_axis");

  auto bitsAttr = rewriter.getI64IntegerAttr(bitsIntAttr.getSInt());
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSizeIntAttr.getSInt());
  auto gatherAxisAttr = rewriter.getI64IntegerAttr(
      gatherAxisIntAttr ? gatherAxisIntAttr.getSInt() : 0);
  auto quantAxisAttr = rewriter.getI64IntegerAttr(
      quantAxisIntAttr ? quantAxisIntAttr.getSInt() : 0);

  // Output shape: [data[0:gather_axis], indices.shape, data[gather_axis+1:]].
  // Mirror GatherConversion's dim-mapping: walk output dims, source each
  // dynamic dim from either data or indices according to its position.
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

  int64_t gatherAxis = gatherAxisAttr.getInt();
  int64_t normalizedGatherAxis =
      gatherAxis < 0 ? gatherAxis + dataType.getRank() : gatherAxis;

  llvm::SmallVector<mlir::Value> dynSizes;
  int64_t outDimIdx = 0;
  for (auto i : llvm::seq<int64_t>(0, normalizedGatherAxis)) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    outDimIdx++;
  }
  for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, indices, i));
    outDimIdx++;
  }
  for (auto i :
       llvm::seq<int64_t>(normalizedGatherAxis + 1, dataType.getRank())) {
    if (outDimIdx < resultType.getRank() && resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    outDimIdx++;
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto hipOp = mlir::hip::GatherBlockQuantizedOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, context, data, indices,
      scales, zeroPoints, init, bitsAttr, blockSizeAttr, gatherAxisAttr,
      quantAxisAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateGatherBlockQuantizedConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.add<GatherBlockQuantizedToHip>(ctx);
}

} // namespace hip
} // namespace mlir
