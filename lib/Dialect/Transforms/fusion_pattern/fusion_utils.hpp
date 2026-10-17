/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- fusion_utils.hpp - Helpers shared by the C++ fusion patterns -------===//
//
// Matching and cleanup helpers for patterns that fold a chain of primitive
// hip ops into one op. Shape queries (tensor.dim) of the folded values are
// not value uses: they only read a type and are moved to the surviving
// operand.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_TRANSFORMS_FUSION_PATTERN_FUSION_UTILS_HPP
#define HIP_DIALECT_TRANSFORMS_FUSION_PATTERN_FUSION_UTILS_HPP

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace utils {

// ONNX TensorProto element types used by hip.cast's `to`.
constexpr int64_t kOnnxFloat = 1;
constexpr int64_t kOnnxFloat16 = 10;

inline mlir::RankedTensorType rankedType(mlir::Value v) {
  return mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
}

inline bool isF16(mlir::Value v) {
  auto t = rankedType(v);
  return t && t.getElementType().isF16();
}

// True when every use is either `consumer` or a tensor.dim.
inline bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

// The one op other than tensor.dim using `value`, if there is exactly one.
inline mlir::Operation *soleValueUser(mlir::Value value) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen && seen != use.getOwner())
      return nullptr;
    seen = use.getOwner();
  }
  return seen;
}

// Binds `op` to whichever of lhs / rhs an OpT defines and `other` to the
// remaining operand.
template <typename OpT>
bool matchEither(mlir::Value lhs, mlir::Value rhs, OpT &op,
                 mlir::Value &other) {
  if ((op = lhs.getDefiningOp<OpT>())) {
    other = rhs;
    return true;
  }
  if ((op = rhs.getDefiningOp<OpT>())) {
    other = lhs;
    return true;
  }
  return false;
}

// Shape queries of `old` ask only for axes it shares with `replacement`:
// static axes, or dynamic ones ahead of the replacement's last axis.
inline bool shapeQueriesRetargetable(mlir::Value old,
                                     mlir::Value replacement) {
  auto oldType = rankedType(old);
  auto newType = rankedType(replacement);
  if (!oldType || !newType)
    return false;
  for (mlir::Operation *user : old.getUsers()) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user);
    if (!dim)
      continue;
    std::optional<int64_t> index = dim.getConstantIndex();
    if (!index || *index >= oldType.getRank())
      return false;
    if (oldType.isDynamicDim(*index) && *index >= newType.getRank() - 1)
      return false;
  }
  return true;
}

inline void retargetShapeQueries(mlir::PatternRewriter &rewriter,
                                 mlir::Value old, mlir::Value replacement) {
  auto oldType = mlir::cast<mlir::RankedTensorType>(old.getType());
  for (mlir::OpOperand &use : llvm::make_early_inc_range(old.getUses())) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
    if (!dim)
      continue;
    int64_t index = *dim.getConstantIndex();
    rewriter.setInsertionPoint(dim);
    if (oldType.isDynamicDim(index))
      rewriter.modifyOpInPlace(dim, [&] { use.set(replacement); });
    else
      rewriter.replaceOp(dim, mlir::arith::ConstantIndexOp::create(
                                  rewriter, dim.getLoc(),
                                  oldType.getDimSize(index)));
  }
}

// Erase `chain` (consumers before producers, each left with no uses) and
// the destinations only they used.
inline void eraseChain(mlir::PatternRewriter &rewriter,
                       llvm::ArrayRef<mlir::Operation *> chain) {
  llvm::SmallVector<mlir::Operation *> inits;
  for (mlir::Operation *op : chain) {
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op))
      for (mlir::Value init : dps.getDpsInits())
        if (mlir::Operation *def = init.getDefiningOp())
          inits.push_back(def);
    rewriter.eraseOp(op);
  }
  llvm::SmallPtrSet<mlir::Operation *, 8> dropped;
  for (mlir::Operation *def : inits)
    if (!dropped.contains(def) && mlir::isOpTriviallyDead(def)) {
      dropped.insert(def);
      rewriter.eraseOp(def);
    }
}

inline bool
splitsLastAxisOnly(llvm::ArrayRef<mlir::ReassociationIndices> groups,
                   int64_t rank) {
  if ((int64_t)groups.size() != rank)
    return false;
  for (int64_t i = 0; i + 1 < rank; ++i)
    if (groups[i].size() != 1 || groups[i][0] != i)
      return false;
  return groups.back().size() == 2 && groups.back()[0] == rank - 1;
}

// collapse(hip.rms_norm(expand(x) : [..., H, d], weight [d])) over the last
// axis of f16 x, the expand splitting and the collapse merging only the
// last axis: a per-head norm of x.
struct HeadNorm {
  mlir::Value input;
  mlir::Value weight;
  int64_t headSize;
  float epsilon;
  llvm::SmallVector<mlir::Operation *, 3> chain; // collapse, norm, expand
};

// `v` is a per-head norm whose only value use is `consumer`.
inline std::optional<HeadNorm> matchHeadNorm(mlir::Value v,
                                             mlir::Operation *consumer,
                                             mlir::Value ctx) {
  auto collapse = v.getDefiningOp<mlir::tensor::CollapseShapeOp>();
  if (!collapse || !hasSingleValueUse(v, consumer))
    return std::nullopt;
  auto norm = collapse.getSrc().getDefiningOp<mlir::hip::RmsNormOp>();
  if (!norm || norm.getCtx() != ctx ||
      !hasSingleValueUse(norm->getResult(0), collapse))
    return std::nullopt;
  auto expand = norm.getInput().getDefiningOp<mlir::tensor::ExpandShapeOp>();
  if (!expand || !hasSingleValueUse(expand.getResult(), norm))
    return std::nullopt;

  auto inType = rankedType(expand.getSrc());
  auto headType = rankedType(expand.getResult());
  if (!inType || !headType || !isF16(expand.getSrc()) ||
      collapse.getResult().getType() != inType ||
      headType.isDynamicDim(headType.getRank() - 1) ||
      !splitsLastAxisOnly(collapse.getReassociationIndices(),
                          inType.getRank()) ||
      !splitsLastAxisOnly(expand.getReassociationIndices(), inType.getRank()))
    return std::nullopt;
  const int64_t axis = norm.getAxis();
  if (axis != -1 && axis != headType.getRank() - 1)
    return std::nullopt;
  const int64_t headSize = headType.getDimSize(headType.getRank() - 1);
  auto weightType = rankedType(norm.getScale());
  if (!weightType || weightType.getRank() != 1 ||
      weightType.getDimSize(0) != headSize || !isF16(norm.getScale()))
    return std::nullopt;

  HeadNorm match{expand.getSrc(), norm.getScale(), headSize,
                 norm.getEpsilon().convertToFloat(), {}};
  match.chain = {collapse, norm, expand};
  return match;
}

} // namespace utils
} // namespace fusion_transform
} // namespace hip

#endif // HIP_DIALECT_TRANSFORMS_FUSION_PATTERN_FUSION_UTILS_HPP
