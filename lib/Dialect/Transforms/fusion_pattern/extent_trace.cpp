/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- extent_trace.cpp - Prove two dynamic extents equal -----------------===//
//
// hip.matmul and hip.qmatmul contribute batch and M from the activation (N
// from the weight); hip.matmul_nbits contributes every axis but the last
// from the activation and the last from its N attribute. That is the same
// source choice as reifyMatmulLikeShape and MatMulNBitsOp::reifyResultShapes.
//
//===----------------------------------------------------------------------===//

#include "extent_trace.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include <algorithm>

namespace hip {
namespace fusion_transform {
namespace {

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
// graph. Keep the walk bounded; unsupported or cyclic provenance must decline
// rather than assume equal dynamic dimensions.
constexpr int kMaxExtentHops = 32;

} // namespace

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

bool extentsMeet(llvm::ArrayRef<Extent> lhs, llvm::ArrayRef<Extent> rhs) {
  for (const Extent &l : lhs)
    for (const Extent &r : rhs)
      if (l == r)
        return true;
  return false;
}

bool sameRuntimeExtent(mlir::Value lhs, int64_t lhsDim, mlir::Value rhs,
                       int64_t rhsDim) {
  return extentsMeet(traceExtents(lhs, lhsDim), traceExtents(rhs, rhsDim));
}

bool sizeIsExtent(mlir::OpFoldResult size, mlir::Value value, int64_t dim) {
  std::optional<Extent> declared = shapeSizeExtent(size);
  if (!declared)
    return false;
  llvm::SmallVector<Extent> lhs =
      declared->isStatic || declared->dim == -1
          ? llvm::SmallVector<Extent>{*declared}
          : traceExtents(declared->value, declared->dim);
  return extentsMeet(lhs, traceExtents(value, dim));
}

} // namespace fusion_transform
} // namespace hip
