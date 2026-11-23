/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- runtime_shape.cpp - prove two tensors share their runtime extents --===//

#include "runtime_shape.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/Sequence.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

// One dynamic axis after projection producers have been followed.
struct Extent {
  bool isStatic = false;
  int64_t staticSize = 0;
  mlir::Value value;
  int64_t dim = 0;

  bool operator==(const Extent &other) const {
    if (isStatic || other.isStatic)
      return isStatic && other.isStatic && staticSize == other.staticSize;
    return value == other.value && dim == other.dim;
  }
};

// Follow one matmul-like result axis to the operand axis that supplies it.
// A static result is returned directly. std::nullopt means this op does not
// reveal the axis, so the caller keeps the op result as the extent.
std::optional<Extent> stepProjection(mlir::Operation *op, int64_t outDim) {
  if (op->getNumResults() == 0)
    return std::nullopt;
  auto mapMatmul = [&](mlir::Value a, mlir::Value b, int64_t transA,
                       int64_t transB,
                       int64_t resultRank) -> std::optional<Extent> {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
    if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2)
      return std::nullopt;
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();
    int64_t batchRank = std::max(aRank - 2, bRank - 2);
    int64_t outRank = batchRank + 2;
    if (outDim < 0 || outDim >= outRank || outRank != resultRank)
      return std::nullopt;

    if (outDim + 2 == outRank) {
      int64_t aDim = transA ? aRank - 1 : aRank - 2;
      return Extent{false, 0, a, aDim};
    }
    if (outDim + 1 == outRank) {
      int64_t bDim = transB ? bRank - 2 : bRank - 1;
      return Extent{false, 0, b, bDim};
    }

    int64_t aPad = batchRank - (aRank - 2);
    int64_t bPad = batchRank - (bRank - 2);
    int64_t aExtent =
        outDim < aPad ? 1
                      : aType.getDimSize(static_cast<unsigned>(outDim - aPad));
    int64_t bExtent =
        outDim < bPad ? 1
                      : bType.getDimSize(static_cast<unsigned>(outDim - bPad));
    bool aCanonical = outDim >= aPad && aExtent != 1;
    bool bCanonical = outDim >= bPad && bExtent != 1;
    bool pickA = aCanonical || (!bCanonical && outDim >= aPad);
    if (pickA && outDim >= aPad)
      return Extent{false, 0, a, outDim - aPad};
    if (!pickA && outDim >= bPad)
      return Extent{false, 0, b, outDim - bPad};
    return std::nullopt;
  };

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return std::nullopt;
  int64_t resultRank = resultType.getRank();

  if (auto matmul = mlir::dyn_cast<mlir::hip::MatmulOp>(op))
    return mapMatmul(matmul.getA(), matmul.getB(), matmul.getTransA(),
                     matmul.getTransB(), resultRank);
  if (auto qmatmul = mlir::dyn_cast<mlir::hip::QMatMulOp>(op))
    return mapMatmul(qmatmul.getA(), qmatmul.getB(), qmatmul.getTransA(),
                     qmatmul.getTransB(), resultRank);
  if (auto nbits = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(op)) {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(nbits.getA().getType());
    if (!aType || aType.getRank() != resultRank || outDim < 0 ||
        outDim >= resultRank)
      return std::nullopt;
    // The trailing axis is the N attribute, not A's contraction axis.
    if (outDim + 1 == resultRank)
      return Extent{true, static_cast<int64_t>(nbits.getN()), {}, 0};
    return Extent{false, 0, nbits.getA(), outDim};
  }
  return std::nullopt;
}

// Follow casts, projections and explicit destination extents through the
// gated-normalization graph. Keep the walk bounded; unsupported or cyclic
// provenance must decline rather than assume equal dynamic dimensions.
constexpr int kMaxExtentHops = 32;

// A declared size is either a constant, an existing tensor dimension, or an
// identical scalar SSA value. No arithmetic equivalence is assumed.
std::optional<Extent> shapeSizeExtent(mlir::OpFoldResult size) {
  if (auto constant = mlir::getConstantIntValue(size))
    return Extent{true, *constant, {}, 0};
  auto value = mlir::dyn_cast<mlir::Value>(size);
  if (!value)
    return std::nullopt;
  for (int hop = 0; hop < kMaxExtentHops; ++hop) {
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
      // Only peel the 64-bit index/integer bridge used by shape reification.
      // A narrower integer can truncate an extent and is not equivalent.
      auto integer = mlir::dyn_cast<mlir::IntegerType>(
          cast.getIn().getType().isIndex() ? cast.getType()
                                           : cast.getIn().getType());
      if (integer && integer.getWidth() == 64) {
        value = cast.getIn();
        continue;
      }
    }
    if (auto query = value.getDefiningOp<mlir::tensor::DimOp>())
      if (auto index = query.getConstantIndex())
        return Extent{false, 0, query.getSource(), *index};
    return Extent{false, 0, value, -1};
  }
  return std::nullopt;
}

// Keep every proven anchor, so two branches can meet at a local shared input
// without having to reach the beginning of an arbitrarily deep residual graph.
llvm::SmallVector<Extent> traceExtents(mlir::Value value, int64_t dim) {
  llvm::SmallVector<Extent> extents;
  mlir::Value current = value;
  int64_t axis = dim;
  for (int hop = 0; hop < kMaxExtentHops; ++hop) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(current.getType());
    if (!type || axis < 0 || axis >= type.getRank())
      break;
    if (!type.isDynamicDim(static_cast<unsigned>(axis))) {
      extents.push_back(Extent{true, type.getDimSize(axis), {}, 0});
      break;
    }
    extents.push_back(Extent{false, 0, current, axis});
    mlir::Operation *def = current.getDefiningOp();
    if (!def)
      break;
    if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(def)) {
      current = cast.getSource();
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::hip::CastOp>(def)) {
      current = cast.getInput();
      continue;
    }
    if (std::optional<Extent> stepped = stepProjection(def, axis)) {
      if (stepped->isStatic) {
        extents.push_back(*stepped);
        break;
      }
      current = stepped->value;
      axis = stepped->dim;
      continue;
    }
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(def)) {
      if (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
        current = dps.getTiedOpOperand(result)->get();
        continue;
      }
    }
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(def)) {
      auto groups = collapse.getReassociationIndices();
      // A singleton reassociation preserves this axis. Merged dynamic axes
      // require a product proof and are deliberately left unsupported.
      if (groups[axis].size() != 1)
        break;
      current = collapse.getSrc();
      axis = groups[axis].front();
      continue;
    }
    std::optional<Extent> sizeExtent;
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
      sizeExtent = shapeSizeExtent(expand.getMixedOutputShape()[axis]);
    } else if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(def)) {
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(slice.getSource().getType());
      // Restrict to rank-preserving slices; their explicit sizes describe
      // each output axis directly, independently of offsets and data values.
      if (!sourceType || sourceType.getRank() != type.getRank())
        break;
      sizeExtent = shapeSizeExtent(slice.getMixedSizes()[axis]);
    } else if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(def)) {
      unsigned dynamicIndex = 0;
      for (int64_t i = 0; i < axis; ++i)
        dynamicIndex += type.isDynamicDim(i);
      sizeExtent = shapeSizeExtent(empty.getDynamicSizes()[dynamicIndex]);
    }
    if (!sizeExtent)
      break;
    if (sizeExtent->isStatic || sizeExtent->dim == -1) {
      extents.push_back(*sizeExtent);
      break;
    }
    current = sizeExtent->value;
    axis = sizeExtent->dim;
  }
  return extents;
}

} // namespace

// True when each dynamic axis has a common, proven runtime extent. Static
// axes are settled by type equality; equal type spelling alone proves no
// equality between independent dynamic arguments.
bool sameRuntimeShape(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || lhsType != rhsType)
    return false;
  for (int64_t dim : llvm::seq<int64_t>(0, lhsType.getRank())) {
    if (!lhsType.isDynamicDim(static_cast<unsigned>(dim)))
      continue;
    auto left = traceExtents(lhs, dim);
    auto right = traceExtents(rhs, dim);
    bool equal = false;
    for (const auto &l : left)
      for (const auto &r : right)
        equal |= l == r;
    if (!equal)
      return false;
  }
  return true;
}

} // namespace fusion_transform
} // namespace hip
