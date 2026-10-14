/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_shared_act.cpp - One int8 activation per W4A8 reader --===//
//
// Why this exists is in the header; what follows is the IR it emits.
//
//   Before:
//     %q = hip.matmul_nbits(%ctx) ins(%x, %Bq, %Sq) outs(%eq)
//     %k = hip.matmul_nbits(%ctx) ins(%x, %Bk, %Sk) outs(%ek)
//     %v = hip.matmul_nbits(%ctx) ins(%x, %Bv, %Sv) outs(%ev)
//   After:
//     %xq, %xs = hip.quantize_act(%ctx) ins(%x) outs(%e8, %ef)
//         {block_size = 32}
//     %q = hip.matmul_nbits(%ctx) ins(%x, %Bq, %Sq) quantized(%xq, %xs)
//         outs(%eq)
//     ... (same for %k, %v)
//
// The quantize op is placed right after the activation's definition, so it
// dominates every reader. Projections with a different block_size get their
// own quantize op when two or more of them share it.
//===----------------------------------------------------------------------===//

#include "matmul_nbits_shared_act.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace hip {
namespace fusion_transform {
namespace {

/// Whether `op` can read a hip.quantize_act result on the W4A8 dp4a path.
bool readsInt8Activation(mlir::hip::MatMulNBitsOp op) {
  if (op.getAQuant() || op.getGIdx() || op->getNumResults() != 1)
    return false;
  if (op.getBits() != 4 || (op.getZeroPoints() && op.getZpElemSize() != 1))
    return false;
  int64_t k = op.getK();
  int64_t bs = op.getBlockSize();
  if (k % 32 != 0 || bs < 32 || (bs & (bs - 1)) != 0 || k % bs != 0)
    return false;
  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(op.getA().getType());
  return aType && aType.getElementType().isF16() && aType.getRank() >= 1 &&
         aType.getShape().back() == k;
}

struct MatMulNBitsSharedActivation final
    : public mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp> {
  using mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MatMulNBitsOp root,
                  mlir::PatternRewriter &rewriter) const override {
    if (!readsInt8Activation(root))
      return rewriter.notifyMatchFailure(root, "not a W4A8 dp4a reader");

    mlir::Value a = root.getA();
    llvm::SmallVector<mlir::hip::MatMulNBitsOp> readers;
    for (mlir::Operation *user : a.getUsers()) {
      auto mm = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(user);
      if (mm && mm.getA() == a && readsInt8Activation(mm) &&
          mm.getBlockSize() == root.getBlockSize() &&
          !llvm::is_contained(readers, mm))
        readers.push_back(mm);
    }
    if (readers.size() < 2)
      return rewriter.notifyMatchFailure(root, "activation has one reader");

    auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
    int64_t bs = root.getBlockSize();
    mlir::Location loc = a.getLoc();
    rewriter.setInsertionPointAfterValue(a);

    llvm::SmallVector<mlir::Value> dynSizes;
    for (auto [i, d] : llvm::enumerate(aType.getShape()))
      if (mlir::ShapedType::isDynamic(d))
        dynSizes.push_back(mlir::tensor::DimOp::create(
            rewriter, loc, a, static_cast<int64_t>(i)));
    auto qType =
        mlir::RankedTensorType::get(aType.getShape(), rewriter.getI8Type());
    llvm::SmallVector<int64_t> scaleShape(aType.getShape());
    scaleShape.back() /= bs;
    auto sType = mlir::RankedTensorType::get(scaleShape, rewriter.getF32Type());
    mlir::Value qInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, qType.getShape(), qType.getElementType(), dynSizes);
    mlir::Value sInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, sType.getShape(), sType.getElementType(), dynSizes);

    auto quant = mlir::hip::QuantizeActOp::create(
        rewriter, loc, mlir::TypeRange{qType, sType}, root.getHandle(), a,
        qInit, sInit, rewriter.getI64IntegerAttr(bs));

    for (mlir::hip::MatMulNBitsOp mm : readers)
      rewriter.modifyOpInPlace(mm, [&] {
        mm.getAQuantMutable().assign(quant->getResult(0));
        mm.getAScaleMutable().assign(quant->getResult(1));
      });
    return mlir::success();
  }
};

} // namespace

void populateMatMulNBitsSharedActivationPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit) {
  patterns.add<MatMulNBitsSharedActivation>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
