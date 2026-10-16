/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- rope_layout_fusion.cpp - fp32 BNSH rope round trip to fp16 BSNH ----===//
//
// Attention exports that keep q / k in fp16 [B, S, N, D] often rotate them in
// the fp32 BNSH layout of com.microsoft.RotaryEmbedding: four extra launches
// around every rope. hip.rope rotates an fp16 BSNH tensor in fp32 directly
// when its caches are fp32, so the round trip folds away:
//
//   Before:
//     %t = hip.transpose(%x : [B, S, N, D] f16) {perm = [0, 2, 1, 3]}
//     %f = hip.cast<f32>(%t)
//     %r = hip.rope(%f, %pos, %cos, %sin)            (f32 caches)
//     %h = hip.cast<f16>(%r)
//     %y = hip.transpose(%h) {perm = [0, 2, 1, 3]}   : [B, S, N, D]
//   After:
//     %r = hip.rope(collapse(%x) : [B, S, N * D], %pos, %cos, %sin)
//     %y = expand(%r) : [B, S, N, D]
//
// Both forms round once from the same fp32 rotation, so the rewrite is
// exact. Shape queries of the BNSH intermediates move to the matching axis
// of %x.
//
//===----------------------------------------------------------------------===//

#include "rope_layout_fusion.hpp"

#include "fusion_utils.hpp"

namespace hip {
namespace fusion_transform {
namespace {

using namespace utils;
using mlir::hip::CastOp;
using mlir::hip::RopeOp;
using mlir::hip::TransposeOp;

// BSNH <-> BNSH; the permutation is its own inverse.
constexpr int64_t kHeadSeqSwap[] = {0, 2, 1, 3};

bool isHeadSeqSwap(TransposeOp op) {
  auto perm = op.getPerm().getAsValueRange<mlir::IntegerAttr>();
  return llvm::equal(llvm::map_range(perm, [](const llvm::APInt &v) {
                       return v.getSExtValue();
                     }),
                     kHeadSeqSwap);
}

// tensor.dim of a BNSH value -> tensor.dim of the BSNH source.
void retargetSwappedShapeQueries(mlir::PatternRewriter &rewriter,
                                 mlir::Value bnsh, mlir::Value bsnh) {
  auto type = mlir::cast<mlir::RankedTensorType>(bnsh.getType());
  for (mlir::OpOperand &use : llvm::make_early_inc_range(bnsh.getUses())) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
    if (!dim)
      continue;
    const int64_t index = *dim.getConstantIndex();
    rewriter.setInsertionPoint(dim);
    if (type.isDynamicDim(index))
      rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(dim, bsnh,
                                                      kHeadSeqSwap[index]);
    else
      rewriter.replaceOp(dim, mlir::arith::ConstantIndexOp::create(
                                  rewriter, dim.getLoc(),
                                  type.getDimSize(index)));
  }
}

bool shapeQueriesAreConstant(mlir::Value v) {
  return llvm::all_of(v.getUsers(), [](mlir::Operation *user) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user);
    return !dim || (dim.getConstantIndex() && *dim.getConstantIndex() < 4);
  });
}

struct RopeLayoutFusion : public mlir::OpRewritePattern<RopeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RopeOp rope,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value ctx = rope.getCtx();
    auto toF32 = rope.getInput().getDefiningOp<CastOp>();
    if (!toF32 || toF32.getCtx() != ctx || toF32.getTo() != kOnnxFloat ||
        !hasSingleValueUse(rope.getInput(), rope))
      return rewriter.notifyMatchFailure(rope, "input is not a cast to f32");
    auto toBnsh = toF32.getInput().getDefiningOp<TransposeOp>();
    if (!toBnsh || toBnsh.getCtx() != ctx || !isHeadSeqSwap(toBnsh) ||
        !hasSingleValueUse(toF32.getInput(), toF32))
      return rewriter.notifyMatchFailure(rope, "no BSNH -> BNSH transpose");

    if (rope->getNumResults() != 1)
      return rewriter.notifyMatchFailure(rope, "not in tensor form");
    auto toF16 =
        mlir::dyn_cast_if_present<CastOp>(soleValueUser(rope->getResult(0)));
    if (!toF16 || toF16.getCtx() != ctx || toF16.getTo() != kOnnxFloat16 ||
        toF16.getInput() != rope->getResult(0))
      return rewriter.notifyMatchFailure(rope, "result is not cast to f16");
    auto toBsnh = mlir::dyn_cast_if_present<TransposeOp>(
        soleValueUser(toF16->getResult(0)));
    if (!toBsnh || toBsnh.getCtx() != ctx || !isHeadSeqSwap(toBsnh) ||
        toBsnh.getInput() != toF16->getResult(0))
      return rewriter.notifyMatchFailure(rope, "no BNSH -> BSNH transpose");

    mlir::Value x = toBnsh.getInput();
    auto xType = rankedType(x);
    auto cosType = rankedType(rope.getCosCache());
    auto sinType = rankedType(rope.getSinCache());
    if (!xType || xType.getRank() != 4 || !isF16(x) ||
        toBsnh->getResult(0).getType() != xType ||
        xType.isDynamicDim(2) || xType.isDynamicDim(3) || !cosType ||
        !sinType || !cosType.getElementType().isF32() ||
        !sinType.getElementType().isF32())
      return rewriter.notifyMatchFailure(rope, "unsupported types");
    const int64_t heads = xType.getDimSize(2);
    const int64_t headDim = xType.getDimSize(3);
    if (rope.getNumHeads() != 0 && rope.getNumHeads() != heads)
      return rewriter.notifyMatchFailure(rope, "num_heads disagrees");
    for (mlir::Value v : {toBnsh->getResult(0), toF32->getResult(0),
                          rope->getResult(0), toF16->getResult(0)})
      if (!shapeQueriesAreConstant(v))
        return rewriter.notifyMatchFailure(rope, "unexpected shape query");

    for (mlir::Value v : {toBnsh->getResult(0), toF32->getResult(0),
                          rope->getResult(0), toF16->getResult(0)})
      retargetSwappedShapeQueries(rewriter, v, x);

    mlir::Location loc = rope.getLoc();
    rewriter.setInsertionPoint(toBsnh);
    llvm::SmallVector<mlir::ReassociationIndices> heads3d = {{0}, {1}, {2, 3}};
    auto flatType = mlir::RankedTensorType::get(
        {xType.getDimSize(0), xType.getDimSize(1), heads * headDim},
        xType.getElementType());
    mlir::Value flat = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, flatType, x, heads3d);
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i : {0, 1})
      if (flatType.isDynamicDim(i))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, x, i));
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, flatType.getShape(), flatType.getElementType(),
        dynSizes);
    auto fused = RopeOp::create(
        rewriter, loc, mlir::TypeRange{flatType}, ctx, flat,
        rope.getPositionIds(), rope.getCosCache(), rope.getSinCache(), init,
        rope.getInterleavedAttr(), rewriter.getI64IntegerAttr(heads),
        rope.getRotaryEmbeddingDimAttr());
    mlir::Value y = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, xType, fused->getResult(0), heads3d);
    rewriter.replaceOp(toBsnh, y);
    mlir::Operation *chain[] = {toF16, rope, toF32, toBnsh};
    eraseChain(rewriter, chain);
    return mlir::success();
  }
};

} // namespace

void populateRopeLayoutFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit) {
  patterns.add<RopeLayoutFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
