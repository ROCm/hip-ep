/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- pointwise_conv_to_matmul.cpp - 1x1 hip.conv as a batched matmul ----===//
//
// Why this is a matmul and what is left alone are in the header; what follows
// is the IR it emits.
//
//   Before:
//     %y = hip.conv(%ctx) ins(%x, %w, %b : tensor<NxCxHxW>, tensor<MxCx1x1>,
//                                          tensor<M>)
//                         outs(%init : tensor<NxMxHxW>)
//                         {kernel_shape = [1, 1], strides = [1, 1],
//                          pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
//   After:
//     %xf = tensor.collapse_shape %x [[0], [1], [2, 3]]    : tensor<NxCx(H*W)>
//     %wf = tensor.collapse_shape %w [[0], [1, 2, 3]]      : tensor<MxC>
//     %e  = tensor.empty(...)                              : tensor<NxMx(H*W)>
//     %p  = hip.matmul(%ctx) ins(%wf, %xf) outs(%e)        : tensor<NxMx(H*W)>
//     %b3 = tensor.expand_shape %b [[0, 1, 2]]             : tensor<1xMx1>
//     %if = tensor.collapse_shape %init [[0], [1], [2, 3]] : tensor<NxMx(H*W)>
//     %s  = hip.add(%ctx) ins(%p, %b3) outs(%if)           : tensor<NxMx(H*W)>
//     %y  = tensor.expand_shape %s [[0], [1], [2, 3]]      : tensor<NxMxHxW>
//
// `hip.matmul` broadcasts its rank-2 left operand over the right operand's
// batch, so the filter is read once per GEMM rather than materialised per
// image. Without a bias the expanded matmul result replaces the conv directly.
// A rank-3 conv has a single spatial axis, so its input and result need no
// reshape at all; a rank-5 conv collapses three spatial axes into one, and
// its bias add stays rank 3 like every other.
//
// Any batch or spatial extent may be dynamic: the reshapes only regroup them,
// and the bias add writes into the conv's own destination, so no extents are
// derived beyond the matmul's init. The channel counts must be static, being
// the GEMM's M and K.
//===----------------------------------------------------------------------===//

#include "pointwise_conv_to_matmul.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace hip {
namespace fusion_transform {
namespace {

llvm::SmallVector<int64_t> readI64Array(mlir::ArrayAttr attr) {
  llvm::SmallVector<int64_t> values;
  if (!attr)
    return values;
  for (mlir::Attribute a : attr)
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
      values.push_back(ia.getValue().getSExtValue());
  return values;
}

/// hip.matmul lowers to hipBLASLt for these element types only.
bool isMatmulElementType(mlir::Type t) {
  return t.isF16() || t.isBF16() || t.isF32();
}

/// Whether `conv` is a pointwise convolution that `hip.matmul` can run as is.
/// On failure, `reason` names the guard that declined.
bool isPointwiseConv(mlir::hip::ConvOp conv, mlir::RankedTensorType xType,
                     mlir::RankedTensorType wType, mlir::RankedTensorType yType,
                     llvm::StringRef &reason) {
  auto decline = [&](llvm::StringRef why) {
    reason = why;
    return false;
  };

  const int64_t rank = xType.getRank();
  if (rank < 3 || rank > 5 || wType.getRank() != rank ||
      yType.getRank() != rank)
    return decline("pwconv.rank");
  const int64_t nSpatial = rank - 2;

  if (conv.getGroup() != 1)
    return decline("pwconv.group_ne_1");

  // hip.conv carries every geometry attribute explicitly (convert-onnx-to-hip
  // fills in ONNX's defaults), so an absent or short array is not a default
  // to assume but an op this pattern does not understand.
  llvm::SmallVector<int64_t> kernel = readI64Array(conv.getKernelShape());
  llvm::SmallVector<int64_t> strides = readI64Array(conv.getStrides());
  llvm::SmallVector<int64_t> pads = readI64Array(conv.getPads());
  llvm::SmallVector<int64_t> dilations = readI64Array(conv.getDilations());
  if (static_cast<int64_t>(kernel.size()) != nSpatial ||
      static_cast<int64_t>(strides.size()) != nSpatial ||
      static_cast<int64_t>(pads.size()) != 2 * nSpatial ||
      static_cast<int64_t>(dilations.size()) != nSpatial)
    return decline("pwconv.geometry_arity");
  auto isOne = [](int64_t v) { return v == 1; };
  if (!llvm::all_of(kernel, isOne))
    return decline("pwconv.kernel_ne_1");
  // A strided 1x1 reads a strided subset of the pixels, which is not a plain
  // matrix operand: the input would need a gather first.
  if (!llvm::all_of(strides, isOne))
    return decline("pwconv.stride_ne_1");
  if (llvm::any_of(pads, [](int64_t p) { return p != 0; }))
    return decline("pwconv.has_padding");
  if (!llvm::all_of(dilations, isOne))
    return decline("pwconv.has_dilation");

  mlir::Type elemType = xType.getElementType();
  if (wType.getElementType() != elemType || yType.getElementType() != elemType)
    return decline("pwconv.mixed_element_types");
  if (!isMatmulElementType(elemType))
    return decline("pwconv.element_type");

  // The channel counts are the GEMM's M and K; the filter must agree with the
  // input on K, and be a unit kernel by type as well as by attribute.
  const int64_t cin = xType.getDimSize(1);
  const int64_t cout = wType.getDimSize(0);
  if (mlir::ShapedType::isDynamic(cin) || mlir::ShapedType::isDynamic(cout))
    return decline("pwconv.channels_dynamic");
  if (wType.getDimSize(1) != cin || yType.getDimSize(1) != cout)
    return decline("pwconv.channels_mismatch");
  for (int64_t i : llvm::seq<int64_t>(2, rank))
    if (wType.getDimSize(i) != 1)
      return decline("pwconv.weight_not_unit");

  // A pointwise conv keeps the batch and every spatial extent. Requiring the
  // two types to state them identically keeps every reshape below a pure
  // regrouping, with nothing to reconcile between static and dynamic.
  if (xType.getDimSize(0) != yType.getDimSize(0))
    return decline("pwconv.batch_mismatch");
  bool singlePixel = true;
  for (int64_t i : llvm::seq<int64_t>(2, rank)) {
    if (xType.getDimSize(i) != yType.getDimSize(i))
      return decline("pwconv.spatial_mismatch");
    singlePixel = singlePixel && xType.getDimSize(i) == 1;
  }
  // One pixel per image is patch-embed's single-patch case: a 2-D hip.gemm
  // with the bias in its epilogue, and no separate add.
  if (rank >= 4 && singlePixel)
    return decline("pwconv.single_pixel");

  if (mlir::Value bias = conv.getBias()) {
    auto bType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
    if (!bType || bType.getRank() != 1 || bType.getDimSize(0) != cout ||
        bType.getElementType() != elemType)
      return decline("pwconv.bias_shape");
  }
  return true;
}

struct PointwiseConvToMatmul
    : public mlir::OpRewritePattern<mlir::hip::ConvOp> {
  PointwiseConvToMatmul(mlir::MLIRContext *ctx, mlir::PatternBenefit benefit)
      : mlir::OpRewritePattern<mlir::hip::ConvOp>(ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::ConvOp conv,
                  mlir::PatternRewriter &rewriter) const override {
    // Runs pre-bufferization, where a HIP DPS op carries its result as a
    // tensor. In memref mode there is no result to replace.
    if (conv->getNumResults() != 1)
      return rewriter.notifyMatchFailure(conv, "pwconv.no_tensor_result");

    mlir::Value x = conv.getInput();
    mlir::Value w = conv.getWeights();
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto wType = mlir::dyn_cast<mlir::RankedTensorType>(w.getType());
    auto yType =
        mlir::dyn_cast<mlir::RankedTensorType>(conv->getResult(0).getType());
    if (!xType || !wType || !yType)
      return rewriter.notifyMatchFailure(conv, "pwconv.not_ranked");

    llvm::StringRef reason;
    if (!isPointwiseConv(conv, xType, wType, yType, reason))
      return rewriter.notifyMatchFailure(conv, reason);

    // Every check is done; nothing below can fail.
    mlir::Location loc = conv.getLoc();
    mlir::Value ctx = conv.getCtx();
    mlir::Type elemType = xType.getElementType();
    const int64_t rank = xType.getRank();
    const int64_t cin = xType.getDimSize(1);
    const int64_t cout = wType.getDimSize(0);
    const int64_t batch = xType.getDimSize(0);

    // [N, C, S_0, ..., S_{d-1}] <-> [N, C, prod(S)]; an identity for d == 1,
    // so a rank-3 conv is used as is.
    const bool reshape = rank != 3;
    llvm::SmallVector<mlir::ReassociationIndices> pixelReassoc{{0}, {1}, {}};
    int64_t pixels = 1;
    for (int64_t i : llvm::seq<int64_t>(2, rank)) {
      pixelReassoc[2].push_back(i);
      pixels = (mlir::ShapedType::isDynamic(pixels) || xType.isDynamicDim(i))
                   ? mlir::ShapedType::kDynamic
                   : pixels * xType.getDimSize(i);
    }

    mlir::Value xFlat = x;
    if (reshape)
      xFlat = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc,
          mlir::RankedTensorType::get({batch, cin, pixels}, elemType), x,
          pixelReassoc);

    // [M, C, 1, ...] -> [M, C]: the filter as the GEMM's left operand.
    llvm::SmallVector<mlir::ReassociationIndices> filterReassoc{{0}, {}};
    for (int64_t i : llvm::seq<int64_t>(1, rank))
      filterReassoc[1].push_back(i);
    mlir::Value wFlat = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, mlir::RankedTensorType::get({cout, cin}, elemType), w,
        filterReassoc);

    // The GEMM result [N, M, P] takes its dynamic extents from the reshaped
    // input, which already carries them.
    auto mmType = mlir::RankedTensorType::get({batch, cout, pixels}, elemType);
    llvm::SmallVector<mlir::Value> mmDynSizes;
    if (mmType.isDynamicDim(0))
      mmDynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, xFlat, 0).getResult());
    if (mmType.isDynamicDim(2))
      mmDynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, xFlat, 2).getResult());
    mlir::Value mmInit =
        mlir::tensor::EmptyOp::create(rewriter, loc, mmType.getShape(),
                                      elemType, mmDynSizes)
            .getResult();
    mlir::Value product =
        mlir::hip::MatmulOp::create(rewriter, loc,
                                    mlir::ValueRange{ctx, wFlat, xFlat, mmInit},
                                    llvm::ArrayRef<mlir::NamedAttribute>{})
            ->getResult(0);

    // The bias add runs on the matmul's [N, M, P] result, before the expand:
    // a rank-3 broadcast whatever the conv's rank (the elementwise lowering
    // stops at rank 4). It writes into the conv's own destination, regrouped
    // the same way: both ops overwrite all of it.
    mlir::Value result = product;
    if (mlir::Value bias = conv.getBias()) {
      // [M] -> [1, M, 1]: broadcast over the batch and every pixel.
      mlir::Value shapedBias = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, mlir::RankedTensorType::get({1, cout, 1}, elemType),
          bias, llvm::SmallVector<mlir::ReassociationIndices>{{0, 1, 2}});
      mlir::Value addInit = conv.getOutput();
      if (reshape)
        addInit = mlir::tensor::CollapseShapeOp::create(rewriter, loc, mmType,
                                                        addInit, pixelReassoc);
      result = mlir::hip::AddOp::create(rewriter, loc, mmType, ctx, result,
                                        shapedBias, addInit)
                   .getResult(0);
    }

    if (reshape) {
      // Back to the conv's own extents, read off its input: stride 1 and no
      // padding keep every one of them.
      llvm::SmallVector<mlir::OpFoldResult> outSizes;
      for (int64_t i : llvm::seq<int64_t>(0, rank)) {
        if (i == 1)
          outSizes.push_back(rewriter.getIndexAttr(cout));
        else if (xType.isDynamicDim(i))
          outSizes.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, x, i).getResult());
        else
          outSizes.push_back(rewriter.getIndexAttr(xType.getDimSize(i)));
      }
      result = mlir::tensor::ExpandShapeOp::create(rewriter, loc, yType, result,
                                                   pixelReassoc, outSizes);
    }

    rewriter.replaceOp(conv, result);
    return mlir::success();
  }
};

} // namespace

void populatePointwiseConvToMatmulPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit) {
  patterns.add<PointwiseConvToMatmul>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
