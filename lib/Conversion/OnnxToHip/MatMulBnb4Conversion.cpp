/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

// com.microsoft.MatMulBnb4 is a bitsandbytes FP4/NF4 matmul. The weight blob
// dequantizes to row-major [N, K] (high nibble first, times a per-block
// absmax) and the GEMM is always A @ B^T. tosa.matmul_t_block_scaled is MXFP
// and does not describe this codebook, so the weight becomes hip.dequant_bnb4
// and the product becomes hip.matmul with transB = 1.
//
// hip-fuse-rocmlir can pull a single-use dequant into the matmul's rock
// kernel when every shape is static; HipToTosa then lowers both. A dynamic
// activation is rejected by that fusion, leaving the dequant on the HIP
// kernel and the GEMM on hipBLASLt.
struct MatMulBnb4ToHip : public mlir::RewritePattern {
  MatMulBnb4ToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

bool isBnb4BlockSize(int64_t blockSize) {
  return blockSize == 16 || blockSize == 32 || blockSize == 64 ||
         blockSize == 128 || blockSize == 256;
}

mlir::LogicalResult
MatMulBnb4ToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MatMulBnb4")
    return rewriter.notifyMatchFailure(op, "not a MatMulBnb4 custom op");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "MatMulBnb4 expects 3 inputs");

  auto readI64 = [&](mlir::StringRef name,
                     int64_t &out) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr)
      return rewriter.notifyMatchFailure(op, "missing MatMulBnb4 attribute");
    out = attr.getValue().getSExtValue();
    return mlir::success();
  };

  int64_t k = 0;
  int64_t n = 0;
  int64_t blockSize = 0;
  int64_t quantType = 0;
  if (failed(readI64("K", k)) || failed(readI64("N", n)) ||
      failed(readI64("block_size", blockSize)) ||
      failed(readI64("quant_type", quantType)))
    return mlir::failure();

  int64_t transB = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transB"))
    transB = attr.getValue().getSExtValue();
  int64_t trainingMode = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("training_mode"))
    trainingMode = attr.getValue().getSExtValue();

  if (transB != 1)
    return rewriter.notifyMatchFailure(
        op, "MatMulBnb4 transB must be 1; the dequant buffer is [N, K]");
  if (trainingMode != 0)
    return rewriter.notifyMatchFailure(
        op, "MatMulBnb4 training_mode nests the absmax and is not supported");
  if (quantType != 0 && quantType != 1)
    return rewriter.notifyMatchFailure(op, "quant_type must be FP4 or NF4");
  if (k <= 0 || n <= 0 || !isBnb4BlockSize(blockSize))
    return rewriter.notifyMatchFailure(op, "unsupported MatMulBnb4 shape");
  if (n > std::numeric_limits<int64_t>::max() / k)
    return rewriter.notifyMatchFailure(op, "N * K overflows");

  const int64_t numel = n * k;
  if (numel % 2 != 0 || numel % blockSize != 0)
    return rewriter.notifyMatchFailure(
        op, "N * K must fill nibble pairs and whole blocks");

  mlir::Value activation = op->getOperand(0);
  mlir::Value packed = op->getOperand(1);
  mlir::Value absmax = op->getOperand(2);
  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(activation.getType());
  auto packedType = mlir::dyn_cast<mlir::RankedTensorType>(packed.getType());
  auto absmaxType = mlir::dyn_cast<mlir::RankedTensorType>(absmax.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!aType || !packedType || !absmaxType || !resultType)
    return rewriter.notifyMatchFailure(op, "MatMulBnb4 tensors must be ranked");
  if (!aType.getElementType().isF32() || !absmaxType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return rewriter.notifyMatchFailure(op, "MatMulBnb4 is f32");
  if (!packedType.getElementType().isUnsignedInteger(8))
    return rewriter.notifyMatchFailure(op, "packed weights must be ui8");
  if (!packedType.hasStaticShape() || !absmaxType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "weight tensors must be static");
  if (packedType.getNumElements() != numel / 2 ||
      absmaxType.getNumElements() != numel / blockSize)
    return rewriter.notifyMatchFailure(op, "packed or absmax length");
  if (aType.getRank() < 2 || resultType.getRank() != aType.getRank())
    return rewriter.notifyMatchFailure(op, "activation rank");
  if (aType.isDynamicDim(aType.getRank() - 1) ||
      aType.getDimSize(aType.getRank() - 1) != k)
    return rewriter.notifyMatchFailure(op, "activation K does not match");
  if (resultType.isDynamicDim(resultType.getRank() - 1) ||
      resultType.getDimSize(resultType.getRank() - 1) != n)
    return rewriter.notifyMatchFailure(op, "result N does not match");
  for (int64_t dim : llvm::seq<int64_t>(resultType.getRank() - 1)) {
    if (resultType.isDynamicDim(dim) != aType.isDynamicDim(dim))
      return rewriter.notifyMatchFailure(op, "batch shape mismatch");
    if (!resultType.isDynamicDim(dim) &&
        resultType.getDimSize(dim) != aType.getDimSize(dim))
      return rewriter.notifyMatchFailure(op, "batch shape mismatch");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  auto weightType = mlir::RankedTensorType::get({n, k}, rewriter.getF32Type());
  mlir::Value weightInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, weightType.getShape(), weightType.getElementType());
  auto dequant = mlir::hip::DequantBnb4Op::create(
      rewriter, loc, context, packed, absmax, weightInit,
      rewriter.getI64IntegerAttr(k), rewriter.getI64IntegerAttr(n),
      rewriter.getI64IntegerAttr(blockSize),
      rewriter.getI64IntegerAttr(quantType));

  llvm::SmallVector<mlir::Value> dynSizes;
  const int64_t rank = resultType.getRank();
  for (int64_t dimIdx : llvm::seq<int64_t>(rank - 1)) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, activation, dimIdx));
  }
  mlir::Value matmulInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  llvm::SmallVector<mlir::NamedAttribute> attrs = {
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(1))};
  llvm::SmallVector<mlir::Value> operands = {context, activation,
                                             dequant.getResult(0), matmulInit};
  auto matmul = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  rewriter.replaceOp(op, matmul.getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulBnb4ConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<MatMulBnb4ToHip>(ctx);
}

} // namespace hip
} // namespace mlir
