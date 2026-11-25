/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Rewrites that offer main_graph to the rocMLIR anchor fusion in a form it can
// take. The pass runs after hip-fuse-rocmlir-elementwise, so a pattern can
// tell a pointwise reader that will run in a rocMLIR kernel, inside a
// hip.rocmlir_elementwise module, from one that stays a HIP op.
//
// The anchor fusion plans only the top-level block of main_graph, so the pass
// rewrites only the ops of that block.

#include "KernelOutline.h"
#include "RocMlirFusionPredicates.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>

namespace mlir::hip {
#define GEN_PASS_DEF_ROCMLIRPREFUSIONPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Constant single-index gather -> slice
//===----------------------------------------------------------------------===//

// The single in-range index of a constant scalar or one-element `indices`,
// normalized against `axisSize`, if it is that.
std::optional<int64_t> singleConstantIndex(Value indices, int64_t axisSize) {
  Attribute attr;
  if (auto carrier = indices.getDefiningOp<ConstantOp>())
    attr = carrier.getValueAttr();
  else
    matchPattern(indices, m_Constant(&attr));
  auto ints = dyn_cast_or_null<DenseIntElementsAttr>(attr);
  if (!ints || ints.getNumElements() != 1)
    return std::nullopt;
  int64_t index = (*ints.getValues<APInt>().begin()).getSExtValue();
  if (index < 0)
    index += axisSize;
  if (index < 0 || index >= axisSize)
    return std::nullopt;
  return index;
}

// Whether the anchor fusion will likely take `module` into a kernel: as the
// prologue or epilogue of an anchor it touches, or as a kernel of its own,
// which the fusion makes only from a module of two or more compute ops.
bool likelyInKernel(RocMlirElementwiseOp module) {
  for (Value input : module.getInputs())
    if (Operation *def = input.getDefiningOp(); def && isAnchorOp(def))
      return true;
  for (Value result : module.getResults())
    for (Operation *user : result.getUsers())
      if (isAnchorOp(user))
        return true;
  int64_t computeOps = 0;
  for (Operation &op : module.getBody().front().without_terminator())
    if (!isa<tensor::EmptyOp>(op) && !getFusableViewKind(&op))
      ++computeOps;
  return computeOps >= 2;
}

// A module takes each input once, however many of its ops read it. Inlined
// back, each of those ops would copy the slice, so the module's block
// argument for `use` must have one reader, through fusable views.
bool hasOneReaderInside(RocMlirElementwiseOp module, OpOperand &use) {
  Value arg = module.getBody().front().getArgument(use.getOperandNumber());
  while (arg.hasOneUse()) {
    Operation *reader = *arg.getUsers().begin();
    if (!getFusableViewKind(reader) || reader->getNumResults() != 1)
      return true;
    arg = reader->getResult(0);
  }
  return false;
}

// Whether `value` has one reader and that reader will likely run in a rocMLIR
// kernel: an anchor, a module the fusion takes, or a fusable view whose
// result in turn qualifies. A slice the fusion still declines then costs one
// copy for its one reader, about what the gather cost.
bool feedsFusionCandidate(Value value) {
  if (!value.hasOneUse())
    return false;
  OpOperand &use = *value.use_begin();
  Operation *reader = use.getOwner();
  if (isAnchorOp(reader))
    return true;
  if (auto module = dyn_cast<RocMlirElementwiseOp>(reader))
    return likelyInKernel(module) && hasOneReaderInside(module, use);
  return getFusableViewKind(reader) && reader->getNumResults() == 1 &&
         feedsFusionCandidate(reader->getResult(0));
}

// A gather of one constant index reads a size-1 slice of the gathered axis,
// which a kernel that reads it absorbs as a view:
//
//   %y = hip.gather(%ctx) ins(%x, %c1 : tensor<2x3x5x2xf32>, tensor<i64>)
//                         outs(%e : tensor<2x3x5xf32>) {axis = 3}
//
// becomes
//
//   %s = tensor.extract_slice %x[0, 0, 0, 1] [2, 3, 5, 1] [1, 1, 1, 1]
//   %y = tensor.collapse_shape %s [[0], [1], [2, 3]]  // tensor<2x3x5xf32>
//
// A one-element 1-D index keeps the unit dim, so only the slice is emitted.
// Integer data stays a gather: rocMLIR kernels take floating-point data. An
// identical gather is skipped too: CSE would merge the two slices into one
// value with two readers.
struct GatherAsSlice : OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    Value data = gather.getData();
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    auto indicesType =
        dyn_cast<RankedTensorType>(gather.getIndices().getType());
    auto resultType =
        dyn_cast<RankedTensorType>(gather->getResult(0).getType());
    if (!dataType || !indicesType || !resultType ||
        !dataType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        !isa<FloatType>(dataType.getElementType()) ||
        resultType.getElementType() != dataType.getElementType() ||
        indicesType.getNumElements() != 1 || indicesType.getRank() > 1)
      return rewriter.notifyMatchFailure(gather, "not a static float gather "
                                                 "of one index");
    int64_t rank = dataType.getRank();
    int64_t axis = gather.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(gather, "axis out of range");
    int64_t axisSize = dataType.getDimSize(axis);
    std::optional<int64_t> index =
        singleConstantIndex(gather.getIndices(), axisSize);
    if (!index)
      return rewriter.notifyMatchFailure(gather, "no constant in-range index");
    if (!feedsFusionCandidate(gather->getResult(0)))
      return rewriter.notifyMatchFailure(gather, "no single fusable reader");
    for (Operation *user : data.getUsers()) {
      auto other = dyn_cast<GatherOp>(user);
      if (other && other != gather && other.getData() == data &&
          other.getAxis() == gather.getAxis() &&
          other.getIndices().getType() == indicesType &&
          singleConstantIndex(other.getIndices(), axisSize) == index)
        return rewriter.notifyMatchFailure(gather, "identical gather");
    }

    SmallVector<int64_t> sliceShape(dataType.getShape());
    sliceShape[axis] = 1;
    bool keepsUnitDim = indicesType.getRank() == 1;
    SmallVector<int64_t> expectedShape(sliceShape);
    if (!keepsUnitDim)
      expectedShape.erase(expectedShape.begin() + axis);
    if (ArrayRef<int64_t>(expectedShape) != resultType.getShape())
      return rewriter.notifyMatchFailure(gather, "unexpected result shape");

    Location loc = gather.getLoc();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    offsets[axis] = rewriter.getIndexAttr(*index);
    SmallVector<OpFoldResult> sizes;
    for (int64_t d : sliceShape)
      sizes.push_back(rewriter.getIndexAttr(d));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    auto sliceType =
        RankedTensorType::get(sliceShape, dataType.getElementType());
    Value result = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, data, offsets, sizes, strides);
    if (!keepsUnitDim) {
      // Fold the unit dim into a neighbour; a rank-1 slice collapses to
      // rank 0.
      SmallVector<ReassociationIndices> groups;
      if (rank > 1) {
        for (int64_t d = 0; d < rank; ++d)
          if (d != axis)
            groups.push_back({d});
        if (axis == rank - 1)
          groups.back().push_back(axis);
        else
          groups[axis].insert(groups[axis].begin(), axis);
      }
      result = tensor::CollapseShapeOp::create(rewriter, loc, resultType,
                                               result, groups);
    }
    Value init = gather.getOutput();
    rewriter.replaceOp(gather, result);
    if (auto empty = init.getDefiningOp<tensor::EmptyOp>();
        empty && empty->use_empty())
      rewriter.eraseOp(empty);
    return success();
  }
};

struct RocMlirPreFusionPass
    : public impl::RocMlirPreFusionPassBase<RocMlirPreFusionPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph" || funcOp.isExternal())
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<GatherAsSlice>(&getContext());
    // Only the existing ops of the top-level block, without folding, so the
    // pass changes nothing that its patterns do not match.
    SmallVector<Operation *> ops = llvm::map_to_vector(
        funcOp.getBody().front(), [](Operation &op) { return &op; });
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    config.enableFolding(false);
    (void)applyOpPatternsGreedily(ops, std::move(patterns), config);
  }
};

} // namespace
} // namespace mlir::hip
