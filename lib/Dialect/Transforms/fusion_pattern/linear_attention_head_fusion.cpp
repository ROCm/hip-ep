// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "linear_attention_head_fusion.hpp"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace hip::fusion_transform {
namespace {
using namespace mlir;

bool headGroups(ArrayRef<ReassociationIndices> groups) {
  return groups.size() == 3 && groups[0] == ReassociationIndices{0} &&
         groups[1] == ReassociationIndices{1} &&
         groups[2] == ReassociationIndices({2, 3, 4});
}

RankedTensorType plainF32(Value value, int rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == rank && !type.getEncoding() &&
                 type.getElementType().isF32()
             ? type
             : RankedTensorType{};
}

struct HeadInput {
  Value base;
  int64_t heads, repeatedHeads, width;
};

std::optional<HeadInput> matchHeads(Value value, Value ctx) {
  auto flatten = value.getDefiningOp<tensor::CollapseShapeOp>();
  if (!flatten || !headGroups(flatten.getReassociationIndices()))
    return {};
  auto expand = flatten.getSrc().getDefiningOp<mlir::hip::ExpandOp>();
  if (!expand || expand.getCtx() != ctx)
    return {};
  auto split = expand.getInput().getDefiningOp<tensor::ExpandShapeOp>();
  if (!split || !headGroups(split.getReassociationIndices()))
    return {};
  auto base = plainF32(split.getSrc(), 3);
  auto splitType = plainF32(split.getResult(), 5);
  auto repeated = plainF32(flatten.getSrc(), 5);
  auto result = plainF32(value, 3);
  if (!base || !splitType || !repeated || !result)
    return {};
  const int64_t heads = splitType.getDimSize(2),
                width = splitType.getDimSize(4);
  const int64_t repeat = repeated.getDimSize(3), hidden = result.getDimSize(2);
  if (heads <= 0 || width <= 0 || repeat <= 1 || hidden <= 0 ||
      splitType.getDimSize(3) != 1 || repeated.getDimSize(2) != heads ||
      repeated.getDimSize(4) != width || base.getDimSize(2) <= 0 ||
      base.getDimSize(2) % width || base.getDimSize(2) / width != heads ||
      hidden % width || (hidden / width) % heads ||
      (hidden / width) / heads != repeat)
    return {};
  for (int i : {0, 1})
    if (base.getDimSize(i) != splitType.getDimSize(i) ||
        base.getDimSize(i) != repeated.getDimSize(i) ||
        base.getDimSize(i) != result.getDimSize(i))
      return {};

  // Match only a literal repeat of the singleton axis. Do not read payloads
  // from external files or process memory to prove this transformation.
  auto shape = expand.getShape().getDefiningOp<mlir::hip::ConstantOp>();
  auto dims = shape
                  ? dyn_cast_or_null<DenseIntElementsAttr>(shape.getValueAttr())
                  : DenseIntElementsAttr{};
  if (!dims || dims.getNumElements() != 5)
    return {};
  int index = 0;
  for (const APInt &dim : dims.getValues<APInt>()) {
    if (dim.getSExtValue() != (index == 3 ? repeat : 1))
      return {};
    ++index;
  }

  // Prove that the destination keeps the input batch and sequence extents.
  // Matching two dynamic type dimensions alone does not prove equality.
  auto empty = expand.getOutput().getDefiningOp<tensor::EmptyOp>();
  if (!empty || empty.getType() != repeated)
    return {};
  unsigned dynamic = 0;
  for (int i : {0, 1}) {
    if (!repeated.isDynamicDim(i))
      continue;
    Value size = empty.getDynamicSizes()[dynamic++];
    auto dim = size.getDefiningOp<tensor::DimOp>();
    if (!dim || dim.getConstantIndex() != i ||
        (dim.getSource() != split.getResult() &&
         dim.getSource() != split.getSrc()))
      return {};
  }
  return HeadInput{split.getSrc(), heads, hidden / width, width};
}

void redirectLeadingDims(Value from, Value to, PatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(from.getUsers())) {
    auto dim = dyn_cast<tensor::DimOp>(user);
    if (!dim)
      continue;
    auto axis = dim.getConstantIndex();
    // Only batch and sequence are unchanged. The hidden extent is smaller.
    if (axis && (*axis == 0 || *axis == 1))
      rewriter.modifyOpInPlace(dim, [&] { dim.getSourceMutable().assign(to); });
  }
}

struct CompactHeads : OpRewritePattern<mlir::hip::LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::hip::LinearAttentionOp op,
                                PatternRewriter &rewriter) const override {
    auto query = matchHeads(op.getQuery(), op.getCtx());
    auto key = matchHeads(op.getKey(), op.getCtx());
    if (!query || !key || query->width != key->width ||
        op.getQNumHeads() != op.getKvNumHeads() ||
        query->repeatedHeads != op.getKvNumHeads() ||
        key->repeatedHeads != op.getKvNumHeads())
      return failure();
    // Both mappings select floor(state_head * source_heads / state_heads).
    // The output still has kv_num_heads heads. No arithmetic changes.
    redirectLeadingDims(op.getQuery(), query->base, rewriter);
    redirectLeadingDims(op.getKey(), key->base, rewriter);
    rewriter.modifyOpInPlace(op, [&] {
      op.getQueryMutable().assign(query->base);
      op.getKeyMutable().assign(key->base);
      op.setQNumHeadsAttr(rewriter.getI64IntegerAttr(query->heads));
    });
    return success();
  }
};
} // namespace
void populateLinearAttentionHeadPattern(mlir::RewritePatternSet &patterns) {
  patterns.add<CompactHeads>(patterns.getContext());
}
} // namespace hip::fusion_transform
