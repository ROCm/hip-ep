/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::hip {
#define GEN_PASS_DEF_SIMPLIFYALGEBRAPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Horizontal merge of sibling convolutions, the `find_conv_dot_horiz_fusion`
// matcher from MIGraphX simplify_algebra. Further algebra matchers belong in
// this pass; this is the only one.
FailureOr<DenseElementsAttr> inlineWeight(ConvOp conv) {
  auto type = dyn_cast<RankedTensorType>(conv.getWeights().getType());
  // The merge concatenates the filters as APFloat.
  if (!type || !type.hasStaticShape() || type.getRank() != 4 ||
      !isa<FloatType>(type.getElementType()))
    return failure();
  auto constant = conv.getWeights().getDefiningOp<ConstantOp>();
  if (!constant || constant.getSourceKind() != ConstantOp::SourceKind::Inline)
    return failure();
  auto value = dyn_cast<DenseElementsAttr>(constant.getValueAttr());
  if (!value || value.getType() != type)
    return failure();
  return value;
}

// Same convolution operator and output shape. Weight shape is compared by the
// caller. A different convolution on the same activation, such as a 3x3 next
// to two 2x2 siblings, returns false and stays in its own cluster.
bool sameHorizontal(ConvOp conv, ConvOp first) {
  if (conv.getGroup() != 1 || conv.getBias())
    return false;
  if (conv.getKernelShape() != first.getKernelShape() ||
      conv.getStrides() != first.getStrides() ||
      conv.getPads() != first.getPads() ||
      conv.getDilations() != first.getDilations())
    return false;
  auto outType = dyn_cast<RankedTensorType>(conv.getResult(0).getType());
  auto firstOut = dyn_cast<RankedTensorType>(first.getResult(0).getType());
  return outType && firstOut && outType.getRank() == 4 &&
         outType.hasStaticShape() && outType.getShape() == firstOut.getShape();
}

void mergeHorizontal(OpBuilder &builder, ArrayRef<ConvOp> convs,
                     ArrayRef<DenseElementsAttr> weights) {
  ConvOp first = convs.front();
  auto weightType = cast<RankedTensorType>(weights.front().getType());
  auto outType = cast<RankedTensorType>(first.getResult(0).getType());
  const int64_t count = convs.size();
  SmallVector<int64_t> fusedWeightShape(weightType.getShape());
  fusedWeightShape[0] *= count;
  auto fusedWeightType =
      RankedTensorType::get(fusedWeightShape, weightType.getElementType());
  SmallVector<APFloat> fusedValues;
  fusedValues.reserve(fusedWeightType.getNumElements());
  for (DenseElementsAttr weight : weights)
    fusedValues.append(weight.getValues<APFloat>().begin(),
                       weight.getValues<APFloat>().end());

  // Before the first sibling, so the slices dominate every use of the convs
  // they replace, including uses that sit between two siblings.
  builder.setInsertionPoint(first);
  Location loc = first.getLoc();
  Value fusedWeight =
      ConstantOp::create(builder, loc, fusedWeightType,
                         DenseElementsAttr::get(fusedWeightType, fusedValues))
          .getResult();
  SmallVector<int64_t> fusedOutShape(outType.getShape());
  fusedOutShape[1] *= count;
  auto fusedOutType =
      RankedTensorType::get(fusedOutShape, outType.getElementType());
  Value fusedInit =
      tensor::EmptyOp::create(builder, loc, fusedOutType.getShape(),
                              fusedOutType.getElementType())
          .getResult();
  Value fused =
      ConvOp::create(
          builder, loc, TypeRange{fusedOutType},
          ValueRange{first.getCtx(), first.getInput(), fusedWeight, fusedInit},
          ArrayRef<NamedAttribute>{
              builder.getNamedAttr("kernel_shape", first.getKernelShape()),
              builder.getNamedAttr("strides", first.getStrides()),
              builder.getNamedAttr("pads", first.getPads()),
              builder.getNamedAttr("dilations", first.getDilations()),
              builder.getNamedAttr("group", first.getGroupAttr())})
          .getResult(0);

  int64_t channel = 0;
  const int64_t channels = outType.getDimSize(1);
  for (ConvOp conv : convs) {
    SmallVector<OpFoldResult> offsets{
        builder.getIndexAttr(0), builder.getIndexAttr(channel),
        builder.getIndexAttr(0), builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{
        builder.getIndexAttr(outType.getDimSize(0)),
        builder.getIndexAttr(channels),
        builder.getIndexAttr(outType.getDimSize(2)),
        builder.getIndexAttr(outType.getDimSize(3))};
    SmallVector<OpFoldResult> strides(4, builder.getIndexAttr(1));
    Value slice = tensor::ExtractSliceOp::create(builder, loc, outType, fused,
                                                 offsets, sizes, strides)
                      .getResult();
    conv.getResult(0).replaceAllUsesWith(slice);
    channel += channels;
  }
  for (ConvOp conv : llvm::reverse(convs)) {
    Value weight = conv.getWeights();
    Value init = conv.getOutput();
    conv.erase();
    // The init first: it can be the same value as the weight, which must not
    // be inspected once its constant is erased.
    if (auto empty = init.getDefiningOp<tensor::EmptyOp>();
        empty && empty->use_empty())
      empty.erase();
    if (auto constant = weight.getDefiningOp<ConstantOp>();
        constant && constant->use_empty())
      constant.erase();
  }
}

struct SimplifyAlgebraPass
    : public impl::SimplifyAlgebraPassBase<SimplifyAlgebraPass> {
  void runOnOperation() override {
    // Siblings merge only within one block: the merged conv is placed before
    // the first of them, which does not dominate a sibling in another region.
    // MapVector keeps the merge order, and so the output, deterministic.
    llvm::MapVector<std::pair<Block *, Value>, SmallVector<ConvOp>> groups;
    getOperation().walk([&](ConvOp conv) {
      groups[{conv->getBlock(), conv.getInput()}].push_back(conv);
    });

    OpBuilder builder(&getContext());
    for (auto &entry : groups) {
      // Partition like MIGraphX group_by: each cluster is the users that match
      // the seed. A cluster of one is left unchanged.
      SmallVector<ConvOp> pending = std::move(entry.second);
      while (!pending.empty()) {
        ConvOp seed = pending.front();
        FailureOr<DenseElementsAttr> seedWeight = inlineWeight(seed);
        if (failed(seedWeight) || !sameHorizontal(seed, seed)) {
          pending.erase(pending.begin());
          continue;
        }
        ArrayRef<int64_t> seedShape =
            cast<RankedTensorType>(seedWeight->getType()).getShape();
        SmallVector<ConvOp> cluster{seed};
        SmallVector<DenseElementsAttr> weights{*seedWeight};
        SmallVector<ConvOp> rest;
        for (ConvOp conv : ArrayRef<ConvOp>(pending).drop_front()) {
          FailureOr<DenseElementsAttr> weight = inlineWeight(conv);
          if (succeeded(weight) && sameHorizontal(conv, seed) &&
              cast<RankedTensorType>(weight->getType()).getShape() ==
                  seedShape) {
            cluster.push_back(conv);
            weights.push_back(*weight);
          } else {
            rest.push_back(conv);
          }
        }
        if (cluster.size() >= 2)
          mergeHorizontal(builder, cluster, weights);
        pending = std::move(rest);
      }
    }
  }
};

} // namespace
} // namespace mlir::hip
