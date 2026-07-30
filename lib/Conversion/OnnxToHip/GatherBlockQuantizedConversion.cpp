/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GatherBlockQuantizedUtils.h"
#include "OnnxToHipUtils.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX com.microsoft.GatherBlockQuantized -> HIP gather_block_quantized
//===----------------------------------------------------------------------===//
//
// Storage semantics (unsigned vs signed) come from ONNX T1 + bits, not from
// signless MLIR integer types alone. `unsigned_quant_storage` is set when:
//   - MorphiZen import legalized UINT4 or annotated UINT8 weights, or
//   - the data tensor element type is ui8 at conversion time.
//
// quantize_axis is taken from the ONNX attribute when present; otherwise it is
// inferred from (data, scales, block_size, bits) shape invariants.

struct GatherBlockQuantizedToHip : public mlir::RewritePattern {
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

  mlir::Value zeroPoints;
  if (op->getNumOperands() >= 4) {
    mlir::Value v = op->getOperand(3);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoints = v;
  }

  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  if (!bitsIntAttr)
    return rewriter.notifyMatchFailure(op, "missing required `bits` attribute");
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  if (!blockSizeIntAttr)
    return rewriter.notifyMatchFailure(
        op, "missing required `block_size` attribute");
  auto gatherAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("gather_axis");
  auto quantAxisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("quantize_axis");

  const int64_t bits = bitsIntAttr.getSInt();
  const int64_t blockSize = blockSizeIntAttr.getSInt();
  if (bits != 4 && bits != 8)
    return rewriter.notifyMatchFailure(op, "GBQ `bits` must be 4 or 8");
  if (blockSize <= 0)
    return rewriter.notifyMatchFailure(op, "invalid `block_size`");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  auto scalesType = mlir::cast<mlir::RankedTensorType>(scales.getType());

  std::optional<int64_t> explicitQuantAxis;
  if (quantAxisIntAttr)
    explicitQuantAxis = quantAxisIntAttr.getSInt();
  auto quantAxisOr = gbq::resolveQuantizeAxis(dataType, scalesType, blockSize,
                                              bits, explicitQuantAxis);
  if (mlir::failed(quantAxisOr))
    return rewriter.notifyMatchFailure(
        op, "could not resolve `quantize_axis` from GBQ data/scales shapes");

  bool unsignedQuantStorage = gbq::resolveUnsignedQuantStorage(
      bits, op->hasAttr("unsigned_quant_storage"), dataType.getElementType(),
      data);

  auto bitsAttr = rewriter.getI64IntegerAttr(bits);
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSize);
  auto gatherAxisAttr = rewriter.getI64IntegerAttr(
      gatherAxisIntAttr ? gatherAxisIntAttr.getSInt() : 0);
  auto quantAxisAttr = rewriter.getI64IntegerAttr(*quantAxisOr);

  int64_t gatherAxis = gatherAxisAttr.getInt();
  int64_t normalizedGatherAxis =
      gatherAxis < 0 ? gatherAxis + dataType.getRank() : gatherAxis;

  auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());
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
  if (unsignedQuantStorage)
    hipOp->setAttr("unsigned_quant_storage", rewriter.getUnitAttr());
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
