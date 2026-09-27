/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HostIndex.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {

/// Producer-chain depth walked by `resolveHostIndex`. Shape arithmetic in
/// exported graphs is a handful of ops deep; the bound only stops a
/// pathological walk.
static constexpr int kHostIndexMaxDepth = 8;

mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  if (!value)
    return nullptr;
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defOp)) {
    auto bufDef =
        toTensor.getBuffer().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!bufDef)
      return nullptr;
    auto module = bufDef->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return nullptr;
    auto global =
        module.lookupSymbol<mlir::memref::GlobalOp>(bufDef.getNameAttr());
    if (!global)
      return nullptr;
    return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        global.getInitialValueAttr());
  }
  return nullptr;
}

// Both the pre- and post-conversion spelling of ONNX shape arithmetic are
// accepted, because the greedy driver gives no ordering guarantee between the
// calling pattern and the ones rewriting the producers: `onnx.Shape` may still
// be present, or may already be
// `tensor.from_elements(arith.index_cast(tensor.dim))`; `onnx.Sub` may already
// be `hip.sub`; `onnx.Concat` may already be a `tensor.insert_slice` chain into
// a `tensor.empty`. On Gemma-4 the Pad amounts arrive in the converted Concat
// spelling and the Slice bounds as `hip.sub` over `tensor.from_elements`.
mlir::Value resolveHostIndex(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value v, int64_t idx, int depth) {
  if (!v || idx < 0 || depth > kHostIndexMaxDepth)
    return {};
  auto vType = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!vType || !vType.getElementType().isSignlessInteger())
    return {};
  // Bound the request against the value's own extent once, here, rather than in
  // each producer branch: an out-of-range element must fail rather than resolve
  // to something plausible. Without this an `onnx.Shape` narrowed by its `end`
  // attribute would hand back dim(start + idx) for an element it does not have.
  if (vType.hasStaticShape() && idx >= vType.getNumElements())
    return {};

  if (mlir::DenseElementsAttr dense = getCompileTimeConstantTensor(v)) {
    if (idx >= dense.getNumElements())
      return {};
    return mlir::arith::ConstantIndexOp::create(
        b, loc,
        (*(dense.getValues<mlir::APInt>().begin() + idx)).getSExtValue());
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return {};

  if (auto fromElems = mlir::dyn_cast<mlir::tensor::FromElementsOp>(def)) {
    if (idx >= static_cast<int64_t>(fromElems.getElements().size()))
      return {};
    mlir::Value elem = fromElems.getElements()[idx];
    // ShapeToTensorDims index_casts every tensor.dim to i64 to pack it; take
    // the index back rather than casting a second time.
    if (auto cast = elem.getDefiningOp<mlir::arith::IndexCastOp>())
      if (cast.getIn().getType().isIndex())
        return cast.getIn();
    return mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), elem);
  }

  if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(def))
    return resolveHostIndex(b, loc, cast.getSource(), idx, depth + 1);

  // Reshapes keep the row-major element order, so the linear index carries
  // through unchanged.
  if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(def))
    return resolveHostIndex(b, loc, collapse.getSrc(), idx, depth + 1);
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(def))
    return resolveHostIndex(b, loc, expand.getSrc(), idx, depth + 1);

  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(def)) {
    if (extract.getSourceType().getRank() != 1)
      return {};
    int64_t off = extract.getStaticOffsets()[0];
    int64_t stride = extract.getStaticStrides()[0];
    if (mlir::ShapedType::isDynamic(off) || mlir::ShapedType::isDynamic(stride))
      return {};
    return resolveHostIndex(b, loc, extract.getSource(), off + idx * stride,
                            depth + 1);
  }

  // A converted 1-D Concat: each insert covers [off, off + size) of the
  // destination and defers every other element to the destination chain. The
  // chain is as long as the Concat has operands, and it is acyclic, so walking
  // it does not count against the depth bound.
  if (auto insert = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(def)) {
    if (vType.getRank() != 1)
      return {};
    int64_t off = insert.getStaticOffsets()[0];
    int64_t size = insert.getStaticSizes()[0];
    if (mlir::ShapedType::isDynamic(off) || mlir::ShapedType::isDynamic(size) ||
        insert.getStaticStrides()[0] != 1)
      return {};
    if (idx >= off && idx < off + size)
      return resolveHostIndex(b, loc, insert.getSource(), idx - off, depth + 1);
    return resolveHostIndex(b, loc, insert.getDest(), idx, depth);
  }

  llvm::StringRef opName = def->getName().getStringRef();

  // Unconverted 1-D Concat: pick the operand covering idx.
  if (opName == "onnx.Concat") {
    auto axisAttr = def->getAttrOfType<mlir::IntegerAttr>("axis");
    if (vType.getRank() != 1 || !axisAttr)
      return {};
    int64_t axis = axisAttr.getValue().getSExtValue();
    if (axis != 0 && axis != -1)
      return {};
    int64_t offset = 0;
    for (mlir::Value in : def->getOperands()) {
      auto inType = mlir::dyn_cast<mlir::RankedTensorType>(in.getType());
      if (!inType || inType.getRank() != 1 || inType.isDynamicDim(0))
        return {};
      int64_t n = inType.getDimSize(0);
      if (idx < offset + n)
        return resolveHostIndex(b, loc, in, idx - offset, depth + 1);
      offset += n;
    }
    return {};
  }

  // Unconverted onnx.Shape: element idx is dim (start + idx) of the operand.
  // `start` is normalized as ShapeToTensorDims normalizes it; `end` only bounds
  // how many elements exist, which the extent check above covers.
  if (opName == "onnx.Shape") {
    mlir::Value input = def->getOperand(0);
    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inType)
      return {};
    int64_t rank = inType.getRank();
    int64_t start = 0;
    if (auto startAttr = def->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getSInt();
    if (start < 0)
      start += rank;
    start = std::max(start, int64_t(0));
    int64_t dimIdx = start + idx;
    if (dimIdx < 0 || dimIdx >= rank)
      return {};
    if (!inType.isDynamicDim(dimIdx))
      return mlir::arith::ConstantIndexOp::create(b, loc,
                                                  inType.getDimSize(dimIdx));
    return mlir::tensor::DimOp::create(b, loc, input, dimIdx);
  }

  // Binary shape arithmetic. hip elementwise ops are DPS, so their operands are
  // (ctx, lhs, rhs, out); the ONNX forms are plain (lhs, rhs).
  unsigned lhsPos = 0;
  bool isHip =
      mlir::isa<mlir::hip::SubOp, mlir::hip::AddOp, mlir::hip::MulOp>(def);
  if (isHip)
    lhsPos = 1;
  else if (opName != "onnx.Sub" && opName != "onnx.Add" && opName != "onnx.Mul")
    return {};

  mlir::Value lhs = def->getOperand(lhsPos);
  mlir::Value rhs = def->getOperand(lhsPos + 1);
  // A rank-0 or single-element operand is broadcast against the other.
  auto elemIdx = [&](mlir::Value operand) -> int64_t {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    return (t && t.hasStaticShape() && t.getNumElements() == 1) ? 0 : idx;
  };
  mlir::Value l = resolveHostIndex(b, loc, lhs, elemIdx(lhs), depth + 1);
  mlir::Value r = resolveHostIndex(b, loc, rhs, elemIdx(rhs), depth + 1);
  if (!l || !r)
    return {};

  bool isSub = isHip ? mlir::isa<mlir::hip::SubOp>(def) : opName == "onnx.Sub";
  bool isAdd = isHip ? mlir::isa<mlir::hip::AddOp>(def) : opName == "onnx.Add";
  if (isSub)
    return mlir::arith::SubIOp::create(b, loc, l, r);
  if (isAdd)
    return mlir::arith::AddIOp::create(b, loc, l, r);
  return mlir::arith::MulIOp::create(b, loc, l, r);
}

bool resolveHostIndices(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v,
                        llvm::SmallVectorImpl<mlir::Value> &out) {
  out.clear();
  if (!v)
    return false;
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape())
    return false;
  for (int64_t i = 0; i < t.getNumElements(); ++i) {
    mlir::Value e = resolveHostIndex(b, loc, v, i);
    if (!e) {
      out.clear();
      return false;
    }
    out.push_back(e);
  }
  return true;
}

} // namespace hip
} // namespace mlir
