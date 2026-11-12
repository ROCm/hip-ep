/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- mla_attention_fusion.cpp - Expanded latent cache to MLA attention --===//
//
// Keys and values that are linear up-projections of one cached row
// [c (L), k_r (R)] per position carry H x (dn + R + dv) values per row that
// the L + R cached values determine. When the graph expands such a cache
// right before hip.kv_attention, this pattern replaces the attention by
// hip.mla_attention, which evaluates the same function on the cache itself.
//
//   Before (H heads; dn, R, dv per-head dims; L latent width):
//     %lat  = tensor.collapse_shape %cache [[0], [1, 2], [3]]  : [B, T, L + R]
//     %c    = tensor.extract_slice %lat[0, 0, 0] [B, T, L]
//     %kr   = tensor.extract_slice %lat[0, 0, L] [B, T, R]
//     %kn   = hip.matmul ins(%c, %W_k) | hip.matmul_nbits ins(%c, %W_k, %s_k)
//     %kn4  = tensor.expand_shape %kn [[0], [1], [2, 3]]     : [B, T, H, dn]
//     %kr4  = tensor.expand_shape %kr [[0], [1], [2, 3]]     : [B, T, 1, R]
//     %krh  = hip.expand ins(%kr4, [1, 1, H, 1])             : [B, T, H, R]
//     %k0   = tensor.insert_slice %kn4 into %empty[0, 0, 0, 0]
//     %k4   = tensor.insert_slice %krh into %k0[0, 0, 0, dn] : [B, T, H, dn +
//     R] %v    = hip.matmul ins(%c, %W_v) | hip.matmul_nbits ins(%c, %W_v,
//     %s_v) %v4   = tensor.expand_shape %v [[0], [1], [2, 3]]      : [B, T, H,
//     dv] %k, %vt = hip.transpose %k4, %v4 {perm = [0, 2, 1, 3]} %y    =
//     hip.kv_attention ins(%q, %k, %vt) [kv_seqlen(%len)]
//                {num_heads = kv_num_heads = H}
//
//   After:
//     %y = hip.mla_attention ins(%q, %cache, %W_k, %W_v)
//            [k_b_scales(%s_k) v_b_scales(%s_v)] [kv_seqlen(%len)]
//
// A reshape round trip (collapse_shape then expand_shape with the same
// [[0], [1], [2, 3]] grouping) between the key and its transpose is looked
// through. Both orders of the two key parts are accepted. Each slice must
// cover the full batch and row axes of its source or destination, proven by
// extent tracing; the weights must be hip.constant values, which the
// rewritten op's state reads on the device. Every check runs before the
// first IR change, so a declined match leaves the IR untouched; the
// expansion ops become dead and are removed by the driver.
//
//===----------------------------------------------------------------------===//

#include "mla_attention_fusion.hpp"

#include "extent_trace.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "mla-attention-fusion"

STATISTIC(NumMlaFused, "Number of hip.kv_attention ops over an expanded "
                       "latent cache replaced by hip.mla_attention");

namespace hip {
namespace fusion_transform {
namespace {

using mlir::RankedTensorType;
using mlir::Value;

RankedTensorType rankedType(Value v) {
  return v ? mlir::dyn_cast<RankedTensorType>(v.getType()) : RankedTensorType();
}

bool isHeadSplit(llvm::ArrayRef<mlir::ReassociationIndices> groups) {
  return groups.size() == 3 && groups[0] == mlir::ReassociationIndices{0} &&
         groups[1] == mlir::ReassociationIndices{1} &&
         groups[2] == mlir::ReassociationIndices{2, 3};
}

// [B, T, X] -> [B, T, heads, d] with static heads and d.
Value headSplitSource(Value v, int64_t heads, int64_t d) {
  auto expand = v ? v.getDefiningOp<mlir::tensor::ExpandShapeOp>() : nullptr;
  RankedTensorType t = rankedType(v);
  if (!expand || !isHeadSplit(expand.getReassociationIndices()) || !t ||
      t.getRank() != 4 || t.getDimSize(2) != heads || t.getDimSize(3) != d)
    return {};
  return expand.getSrc();
}

// The [B, T, H, d] operand of the BSHD -> BNSH transpose feeding `bnsh`.
Value bshdOf(Value bnsh) {
  auto transpose = bnsh.getDefiningOp<mlir::hip::TransposeOp>();
  if (!transpose)
    return {};
  llvm::SmallVector<int64_t> perm;
  for (mlir::Attribute a : transpose.getPerm())
    perm.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
  if (perm != llvm::SmallVector<int64_t>{0, 2, 1, 3})
    return {};
  Value x = transpose.getInput();
  // The singleton groups pin the batch and row axes, and the static head
  // axes must match, so this collapse/expand pair is the identity.
  if (auto expand = x.getDefiningOp<mlir::tensor::ExpandShapeOp>())
    if (auto collapse =
            expand.getSrc().getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
      RankedTensorType in = rankedType(collapse.getSrc());
      RankedTensorType out = rankedType(x);
      if (isHeadSplit(expand.getReassociationIndices()) &&
          isHeadSplit(collapse.getReassociationIndices()) && in && out &&
          in.getRank() == 4 && !in.isDynamicDim(2) && !in.isDynamicDim(3) &&
          in.getDimSize(2) == out.getDimSize(2) &&
          in.getDimSize(3) == out.getDimSize(3))
        x = collapse.getSrc();
    }
  return x;
}

bool isUnitStrides(llvm::ArrayRef<mlir::OpFoldResult> strides) {
  return llvm::all_of(strides, [](mlir::OpFoldResult s) {
    return mlir::isConstantIntValue(s, 1);
  });
}

bool offsetsAre(llvm::ArrayRef<mlir::OpFoldResult> offsets,
                llvm::ArrayRef<int64_t> want) {
  if (offsets.size() != want.size())
    return false;
  for (auto [offset, w] : llvm::zip(offsets, want))
    if (!mlir::isConstantIntValue(offset, w))
      return false;
  return true;
}

// [B, T, L + R] -> [B, T, width] at column `column`, over every row.
bool isColumnSlice(mlir::tensor::ExtractSliceOp slice, Value lat,
                   int64_t column, int64_t width) {
  if (!slice || slice.getSource() != lat ||
      !isUnitStrides(slice.getMixedStrides()) ||
      !offsetsAre(slice.getMixedOffsets(), {0, 0, column}))
    return false;
  llvm::SmallVector<mlir::OpFoldResult> sizes = slice.getMixedSizes();
  return sizes.size() == 3 && mlir::isConstantIntValue(sizes[2], width) &&
         sizeIsExtent(sizes[0], lat, 0) && sizeIsExtent(sizes[1], lat, 1);
}

// `slice` writes [B, T, heads, width] at column `column` of every row of its
// destination.
bool isColumnInsert(mlir::tensor::InsertSliceOp slice, int64_t column,
                    int64_t heads, int64_t width) {
  if (!slice || !isUnitStrides(slice.getMixedStrides()) ||
      !offsetsAre(slice.getMixedOffsets(), {0, 0, 0, column}))
    return false;
  llvm::SmallVector<mlir::OpFoldResult> sizes = slice.getMixedSizes();
  Value dest = slice.getDest();
  return sizes.size() == 4 && mlir::isConstantIntValue(sizes[2], heads) &&
         mlir::isConstantIntValue(sizes[3], width) &&
         sizeIsExtent(sizes[0], dest, 0) && sizeIsExtent(sizes[1], dest, 1);
}

std::optional<llvm::SmallVector<int64_t>> intConstant(Value v) {
  auto constant = v ? v.getDefiningOp<mlir::hip::ConstantOp>() : nullptr;
  auto dense = constant ? mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                              constant.getValueAttr())
                        : mlir::DenseIntElementsAttr();
  if (!dense)
    return std::nullopt;
  llvm::SmallVector<int64_t> out;
  for (const llvm::APInt &x : dense)
    out.push_back(x.getSExtValue());
  return out;
}

bool isConstant(Value v) {
  return v && v.getDefiningOp<mlir::hip::ConstantOp>();
}

struct Projection {
  Value weight, scales;
  int64_t blockSize = 0;
};

// out = c @ W with W an fp16 [L, n] hip.matmul weight or an int4
// hip.matmul_nbits weight (K = L, N = n, fp16 scales, implicit zero point,
// no bias).
std::optional<Projection> matchProjection(Value out, Value c, int64_t lat,
                                          int64_t n) {
  RankedTensorType outType = rankedType(out);
  if (!outType || outType.getRank() != 3 || outType.isDynamicDim(2) ||
      outType.getDimSize(2) != n || !outType.getElementType().isF16())
    return std::nullopt;

  if (auto matmul = out.getDefiningOp<mlir::hip::MatmulOp>()) {
    RankedTensorType w = rankedType(matmul.getB());
    if (matmul.getA() != c || matmul.getTransA() != 0 ||
        matmul.getTransB() != 0 || !isConstant(matmul.getB()) || !w ||
        !w.hasStaticShape() || w.getRank() != 2 || w.getDimSize(0) != lat ||
        w.getDimSize(1) != n || !w.getElementType().isF16())
      return std::nullopt;
    return Projection{matmul.getB(), Value(), 0};
  }

  auto nbits = out.getDefiningOp<mlir::hip::MatMulNBitsOp>();
  if (!nbits || nbits.getA() != c || nbits.getZeroPoints() || nbits.getGIdx() ||
      nbits.getBias())
    return std::nullopt;
  const auto block = static_cast<int64_t>(nbits.getBlockSize());
  if (static_cast<int64_t>(nbits.getK()) != lat ||
      static_cast<int64_t>(nbits.getN()) != n || nbits.getBits() != 4 ||
      nbits.getScaleElemSize() != 2 || block < 16 || lat % block != 0)
    return std::nullopt;
  Value w = nbits.getB(), s = nbits.getScales();
  RankedTensorType wt = rankedType(w), st = rankedType(s);
  if (!isConstant(w) || !isConstant(s) || !wt || !wt.hasStaticShape() ||
      wt.getRank() != 3 || wt.getDimSize(0) != n ||
      wt.getDimSize(1) != lat / block || wt.getDimSize(2) != block / 2 ||
      !wt.getElementType().isInteger(8) || !st || !st.hasStaticShape() ||
      st.getNumElements() != n * (lat / block) || !st.getElementType().isF16())
    return std::nullopt;
  return Projection{w, s, block};
}

struct MlaAttentionFusion
    : public mlir::OpRewritePattern<mlir::hip::KvAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::KvAttentionOp attn,
                  mlir::PatternRewriter &rewriter) const override {
    const auto heads = static_cast<int64_t>(attn.getNumHeads());
    if (static_cast<int64_t>(attn.getKvNumHeads()) != heads)
      return rewriter.notifyMatchFailure(attn, "needs num_heads == "
                                               "kv_num_heads");
    RankedTensorType qType = rankedType(attn.getQuery());
    RankedTensorType kType = rankedType(attn.getKey());
    RankedTensorType vType = rankedType(attn.getValue());
    RankedTensorType yType = rankedType(attn.getOutput());
    if (!qType || !kType || !vType || !yType ||
        !qType.getElementType().isF16() || !yType.getElementType().isF16())
      return rewriter.notifyMatchFailure(attn, "query/output not fp16");
    const int64_t dq = kType.getDimSize(3);
    const int64_t dv = vType.getDimSize(3);

    // K = insert_slice(rope part, insert_slice(nope part, tensor.empty)).
    Value k4 = bshdOf(attn.getKey());
    Value v4 = bshdOf(attn.getValue());
    if (!k4 || !v4)
      return rewriter.notifyMatchFailure(attn, "key/value are not transposed "
                                               "[B, T, H, d] tensors");
    auto outer = k4.getDefiningOp<mlir::tensor::InsertSliceOp>();
    auto inner =
        outer ? outer.getDest().getDefiningOp<mlir::tensor::InsertSliceOp>()
              : nullptr;
    if (!inner || !inner.getDest().getDefiningOp<mlir::tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(attn, "key is not two parts written "
                                               "into an empty tensor");
    auto lastOffset = [](mlir::tensor::InsertSliceOp s) {
      return mlir::getConstantIntValue(s.getMixedOffsets().back());
    };
    mlir::tensor::InsertSliceOp nopeInsert = inner, ropeInsert = outer;
    if (lastOffset(inner) != std::optional<int64_t>(0))
      std::swap(nopeInsert, ropeInsert);
    RankedTensorType nopeType = rankedType(nopeInsert.getSource());
    RankedTensorType ropeType = rankedType(ropeInsert.getSource());
    if (!nopeType || !ropeType || nopeType.getRank() != 4 ||
        ropeType.getRank() != 4 || nopeType.isDynamicDim(3) ||
        ropeType.isDynamicDim(3))
      return rewriter.notifyMatchFailure(attn, "key parts are not rank 4");
    const int64_t nope = nopeType.getDimSize(3);
    const int64_t rope = ropeType.getDimSize(3);
    if (nope + rope != dq || !isColumnInsert(nopeInsert, 0, heads, nope) ||
        !isColumnInsert(ropeInsert, nope, heads, rope))
      return rewriter.notifyMatchFailure(attn, "key parts do not tile "
                                               "[B, T, H, nope + rope]");

    // Rope part: hip.expand of [B, T, 1, R] over the heads.
    auto expand = ropeInsert.getSource().getDefiningOp<mlir::hip::ExpandOp>();
    auto shape = expand ? intConstant(expand.getShape()) : std::nullopt;
    if (!shape || *shape != llvm::SmallVector<int64_t>{1, 1, heads, 1})
      return rewriter.notifyMatchFailure(attn, "rope key is not broadcast "
                                               "over heads");
    Value kr = headSplitSource(expand.getInput(), 1, rope);
    Value kn = headSplitSource(nopeInsert.getSource(), heads, nope);
    Value v = headSplitSource(v4, heads, dv);
    if (!kr || !kn || !v)
      return rewriter.notifyMatchFailure(attn, "key/value parts are not "
                                               "split into heads");

    // Both parts come from one latent row [c | k_r].
    auto krSlice = kr.getDefiningOp<mlir::tensor::ExtractSliceOp>();
    Value lat = krSlice ? krSlice.getSource() : Value();
    RankedTensorType latType = rankedType(lat);
    if (!latType || latType.getRank() != 3 || latType.isDynamicDim(2) ||
        latType.getDimSize(2) <= rope)
      return rewriter.notifyMatchFailure(attn, "rope key is not a latent "
                                               "slice");
    const int64_t latent = latType.getDimSize(2) - rope;
    Value c;
    if (auto matmul = kn.getDefiningOp<mlir::hip::MatmulOp>())
      c = matmul.getA();
    else if (auto nbits = kn.getDefiningOp<mlir::hip::MatMulNBitsOp>())
      c = nbits.getA();
    auto cSlice = c ? c.getDefiningOp<mlir::tensor::ExtractSliceOp>() : nullptr;
    if (!isColumnSlice(cSlice, lat, 0, latent) ||
        !isColumnSlice(krSlice, lat, latent, rope))
      return rewriter.notifyMatchFailure(attn, "latent and rope key are not "
                                               "full slices of one row");
    auto collapse = lat.getDefiningOp<mlir::tensor::CollapseShapeOp>();
    Value cache = collapse ? collapse.getSrc() : Value();
    RankedTensorType cacheType = rankedType(cache);
    llvm::SmallVector<mlir::ReassociationIndices, 4> groups;
    if (collapse)
      groups = collapse.getReassociationIndices();
    if (!cacheType || cacheType.getRank() != 4 || groups.size() != 3 ||
        groups[1] != mlir::ReassociationIndices{1, 2} ||
        cacheType.isDynamicDim(1) || cacheType.getDimSize(1) != 1 ||
        !cacheType.getElementType().isF16())
      return rewriter.notifyMatchFailure(attn, "latent is not an fp16 "
                                               "[B, 1, T, L + R] cache");

    auto kProj = matchProjection(kn, c, latent, heads * nope);
    auto vProj = matchProjection(v, c, latent, heads * dv);
    if (!kProj || !vProj || kProj->blockSize != vProj->blockSize)
      return rewriter.notifyMatchFailure(attn, "up-projections are not "
                                               "matching constant "
                                               "matmul/matmul_nbits");

    auto i64 = [&](int64_t x) { return rewriter.getI64IntegerAttr(x); };
    rewriter.setInsertionPoint(attn);
    auto mla = mlir::hip::MlaAttentionOp::create(
        rewriter, attn.getLoc(), attn.getCtx(), attn.getQuery(), cache,
        kProj->weight, vProj->weight, kProj->scales, vProj->scales,
        attn.getKvSeqlen(), attn.getOutput(), i64(heads), i64(latent),
        i64(nope), i64(rope), i64(dv), attn.getScaleAttr(),
        i64(kProj->blockSize));
    rewriter.replaceOp(attn, mla->getResults());
    ++NumMlaFused;
    return mlir::success();
  }
};

} // namespace

void populateMlaAttentionFusionPattern(mlir::RewritePatternSet &patterns,
                                       mlir::PatternBenefit benefit) {
  patterns.add<MlaAttentionFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
