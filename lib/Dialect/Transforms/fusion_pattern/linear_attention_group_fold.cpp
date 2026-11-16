/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- linear_attention_group_fold.cpp - drop grouped q/k broadcasts ------===//
//
// Mamba2 exports repeat each of G B/C groups across the R = H_kv / G heads
// that share it before the SSD recurrence:
//
//   Before:
//     %e = tensor.expand_shape %x [[0], [1], [2, 3, 4]]
//            : tensor<?x?x1024xf32> into tensor<?x?x8x1x128xf32>
//     %r = hip.expand(%ctx) ins(%e, %shape) outs(%init)
//            : tensor<?x?x8x16x128xf32>          // %shape = [1, 1, 1, 16, 1]
//     %c = tensor.collapse_shape %r [[0], [1], [2, 3, 4]]
//            : tensor<?x?x8x16x128xf32> into tensor<?x?x16384xf32>
//     %y:2 = hip.linear_attention(%ctx) ins(%c, %c2, %v ...)
//            {q_num_heads = 128, kv_num_heads = 128}
//   After:
//     %y:2 = hip.linear_attention(%ctx) ins(%x, %x2, %v ...)
//            {q_num_heads = 8, kv_num_heads = 128}
//
// The broadcast must be on the axis right after the group axis. A [1, G, D]
// to [R, G, D] tile lays the copies out group-minor, which is not the h/R
// head mapping the kernel applies, and is left alone.
//
//===----------------------------------------------------------------------===//

#include "linear_attention_group_fold.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

struct GroupedOperand {
  mlir::Value source;
  int64_t groups = 0;
  mlir::tensor::CollapseShapeOp collapse;
};

bool isLastThreeMerged(llvm::ArrayRef<mlir::ReassociationIndices> groups) {
  return groups.size() == 3 && groups[0] == mlir::ReassociationIndices{0} &&
         groups[1] == mlir::ReassociationIndices{1} &&
         groups[2] == mlir::ReassociationIndices{2, 3, 4};
}

// `value` repeat-interleaves [B, T, G*D] to [B, T, heads*D]; returns the
// grouped source and G.
std::optional<GroupedOperand> matchGroupBroadcast(mlir::Value value,
                                                  int64_t heads) {
  auto collapse = value.getDefiningOp<mlir::tensor::CollapseShapeOp>();
  if (!collapse || !isLastThreeMerged(collapse.getReassociationIndices())) {
    return std::nullopt;
  }
  auto expand = collapse.getSrc().getDefiningOp<mlir::hip::ExpandOp>();
  if (!expand) {
    return std::nullopt;
  }
  auto split = expand.getInput().getDefiningOp<mlir::tensor::ExpandShapeOp>();
  if (!split || !isLastThreeMerged(split.getReassociationIndices())) {
    return std::nullopt;
  }

  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(split.getType());
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(collapse.getSrc().getType());
  if (!inType || !outType || inType.getRank() != 5 || outType.getRank() != 5) {
    return std::nullopt;
  }
  const int64_t groups = inType.getDimSize(2);
  const int64_t repeat = outType.getDimSize(3);
  const int64_t width = inType.getDimSize(4);
  if (mlir::ShapedType::isDynamic(groups) ||
      mlir::ShapedType::isDynamic(repeat) ||
      mlir::ShapedType::isDynamic(width) || inType.getDimSize(3) != 1 ||
      outType.getDimSize(2) != groups || outType.getDimSize(4) != width ||
      groups * repeat != heads) {
    return std::nullopt;
  }

  // Only the repeat axis may broadcast; every other target extent is 1 and
  // so keeps the input's extent, including the dynamic batch and sequence.
  auto shape = expand.getShape().getDefiningOp<mlir::hip::ConstantOp>();
  auto dims = shape ? mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                          shape.getValueAttr())
                    : nullptr;
  if (!dims || dims.getNumElements() != 5) {
    return std::nullopt;
  }
  int64_t axis = 0;
  for (const llvm::APInt &d : dims.getValues<llvm::APInt>()) {
    const int64_t want = axis == 3 ? repeat : 1;
    if (d.getSExtValue() != want) {
      return std::nullopt;
    }
    ++axis;
  }
  return GroupedOperand{split.getSrc(), groups, collapse};
}

struct LinearAttentionGroupFold
    : public mlir::OpRewritePattern<mlir::hip::LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::LinearAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const int64_t heads = static_cast<int64_t>(op.getKvNumHeads());
    std::optional<GroupedOperand> query =
        static_cast<int64_t>(op.getQNumHeads()) == heads
            ? matchGroupBroadcast(op.getQuery(), heads)
            : std::nullopt;
    std::optional<GroupedOperand> key = matchGroupBroadcast(op.getKey(), heads);
    if (!query && !key) {
      return rewriter.notifyMatchFailure(op, "no grouped q/k broadcast");
    }

    rewriter.modifyOpInPlace(op, [&] {
      if (query) {
        op.getQueryMutable().assign(query->source);
        op.setQNumHeads(static_cast<uint64_t>(query->groups));
      }
      if (key) {
        op.getKeyMutable().assign(key->source);
      }
    });

    // Batch and sequence queries of the dropped copy read the same extents
    // from the source; the copy itself is erased once nothing else reads it.
    for (std::optional<GroupedOperand> g : {query, key}) {
      if (!g) {
        continue;
      }
      mlir::Value copy = g->collapse.getResult();
      rewriter.replaceUsesWithIf(copy, g->source, [](mlir::OpOperand &use) {
        auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
        if (!dim) {
          return false;
        }
        std::optional<int64_t> index = dim.getConstantIndex();
        return index && *index < 2;
      });
    }
    return mlir::success();
  }
};

} // namespace

void populateLinearAttentionGroupFoldPattern(mlir::RewritePatternSet &patterns,
                                             mlir::PatternBenefit benefit) {
  patterns.add<LinearAttentionGroupFold>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
