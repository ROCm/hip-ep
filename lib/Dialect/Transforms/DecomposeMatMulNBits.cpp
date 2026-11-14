/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "DecomposeMatMulNBits.h"

#include "hip/Dialect/IR/HipDialect.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace mlir::hip {

namespace {

// hip.matmul_nbits -> unpack_int4, blocked dequantize_linear, hip.matmul
// {transB = 1}, and a bias add when one is present. The anchor pass then
// outlines that chain. Anything this rejects stays on the custom kernel.
class DecomposeMatMulNBits : public OpRewritePattern<MatMulNBitsOp> {
public:
  using OpRewritePattern<MatMulNBitsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulNBitsOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    if (op.getBits() != 4)
      return rewriter.notifyMatchFailure(op, "only bits=4 is split");
    if (op.getGIdx())
      return rewriter.notifyMatchFailure(op, "g_idx stays on the HIP kernel");

    int64_t k = op.getK();
    int64_t n = op.getN();
    int64_t blockSize = op.getBlockSize();
    if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0)
      return rewriter.notifyMatchFailure(
          op, "block_size must be a power of 2 and >= 16");
    if (k <= 0 || n <= 0 || k % blockSize != 0)
      return rewriter.notifyMatchFailure(
          op, "K must be positive and divisible by block_size");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(op.getB().getType());
    auto scaleType = dyn_cast<RankedTensorType>(op.getScales().getType());
    if (!resultType || !resultType.hasStaticShape() || !aType ||
        !aType.hasStaticShape() || !bType || !bType.hasStaticShape() ||
        !scaleType || !scaleType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static shapes are split");
    Type elemType = resultType.getElementType();
    if ((!elemType.isF16() && !elemType.isF32()) ||
        aType.getElementType() != elemType)
      return rewriter.notifyMatchFailure(
          op, "activation and result must be f16 or f32");
    if (!scaleType.getElementType().isF16() &&
        !scaleType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "scales must be f16 or f32");
    if (aType.getRank() < 2 || aType.getShape().back() != k)
      return rewriter.notifyMatchFailure(op, "A trailing dim must be K");
    if (resultType.getRank() != aType.getRank() ||
        resultType.getShape().drop_back() != aType.getShape().drop_back() ||
        resultType.getShape().back() != n)
      return rewriter.notifyMatchFailure(op, "result shape must be A with N");

    int64_t kBlocks = k / blockSize;
    int64_t blob = blockSize / 2;
    if (bType.getRank() != 3 || bType.getDimSize(0) != n ||
        bType.getDimSize(1) != kBlocks || bType.getDimSize(2) != blob ||
        !bType.getElementType().isUnsignedInteger(8))
      return rewriter.notifyMatchFailure(
          op, "B must be packed ui8 [N, k_blocks, block_size/2]");
    if (scaleType.getNumElements() != n * kBlocks)
      return rewriter.notifyMatchFailure(op, "scales must have N * k_blocks");

    Value zp = op.getZeroPoints();
    if (zp) {
      // zp_elem_size 1 is a packed nibble stream, which this split does not
      // unpack. One zero point per block is passed through.
      if (op.getZpElemSize() == 1)
        return rewriter.notifyMatchFailure(
            op, "packed zero points stay on the HIP kernel");
      auto zpType = dyn_cast<RankedTensorType>(zp.getType());
      if (!zpType || !zpType.hasStaticShape() ||
          zpType.getNumElements() != n * kBlocks)
        return rewriter.notifyMatchFailure(
            op, "zero point must be static with N * k_blocks values");
      Type zpElem = zpType.getElementType();
      if (!isa<FloatType>(zpElem) && !isa<IntegerType>(zpElem))
        return rewriter.notifyMatchFailure(op, "unsupported zero point type");
    }
    if (Value bias = op.getBias()) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape() ||
          biasType.getElementType() != elemType ||
          biasType.getRank() > resultType.getRank())
        return rewriter.notifyMatchFailure(op, "bias is not broadcastable");
      int64_t offset = resultType.getRank() - biasType.getRank();
      for (auto [idx, dim] : llvm::enumerate(biasType.getShape())) {
        int64_t resultDim = resultType.getDimSize(offset + idx);
        if (dim != 1 && dim != resultDim)
          return rewriter.notifyMatchFailure(op, "bias is not broadcastable");
      }
    }

    Location loc = op.getLoc();
    Value ctx = op.getHandle();
    auto ui8 =
        IntegerType::get(rewriter.getContext(), 8, IntegerType::Unsigned);
    auto packedType = RankedTensorType::get({n, kBlocks * blob}, ui8);
    SmallVector<ReassociationIndices, 2> reassoc(2);
    reassoc[0].push_back(0);
    reassoc[1].push_back(1);
    reassoc[1].push_back(2);
    Value flat = tensor::CollapseShapeOp::create(rewriter, loc, packedType,
                                                 op.getB(), reassoc);

    auto unpackedType = RankedTensorType::get({n, k}, ui8);
    Value unpackInit =
        tensor::EmptyOp::create(rewriter, loc, unpackedType.getShape(), ui8);
    Value unpacked =
        UnpackInt4Op::create(rewriter, loc, TypeRange{unpackedType}, ctx, flat,
                             unpackInit)
            .getResult(0);

    if (!zp) {
      auto zpType = RankedTensorType::get({n, kBlocks}, elemType);
      zp = ConstantOp::create(
          rewriter, loc, zpType,
          DenseElementsAttr::get(zpType, rewriter.getFloatAttr(elemType, 8.0)));
    }

    auto weightType = RankedTensorType::get({n, k}, elemType);
    Value weightInit =
        tensor::EmptyOp::create(rewriter, loc, weightType.getShape(), elemType);
    Value weight =
        DequantizeLinearOp::create(rewriter, loc, TypeRange{weightType}, ctx,
                                   unpacked, op.getScales(), zp, weightInit,
                                   rewriter.getI64IntegerAttr(/*axis=*/1),
                                   rewriter.getI64IntegerAttr(blockSize))
            .getResult(0);

    Value matmulInit =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), elemType);
    SmallVector<NamedAttribute> attrs = {
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(1))};
    SmallVector<Value> matmulOperands = {ctx, op.getA(), weight, matmulInit};
    Value y =
        MatmulOp::create(rewriter, loc, matmulOperands, attrs).getResult(0);

    Value result = y;
    if (Value bias = op.getBias()) {
      Value addInit = tensor::EmptyOp::create(rewriter, loc,
                                              resultType.getShape(), elemType);
      result = AddOp::create(rewriter, loc, resultType, ctx, y, bias, addInit)
                   .getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void decomposeMatMulNBits(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<DecomposeMatMulNBits>(funcOp.getContext());
  walkAndApplyPatterns(funcOp, std::move(patterns));
}

} // namespace mlir::hip
