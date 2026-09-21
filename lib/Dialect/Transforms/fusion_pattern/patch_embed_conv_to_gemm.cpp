/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- patch_embed_conv_to_gemm.cpp - hip.conv that is really a GEMM ------===//
//
// A Conv with `stride == kernel`, no padding and no dilation partitions the
// input spatial volume into disjoint patches and contracts each patch against
// every filter. That is a GEMM: one row per (batch, patch), one column per
// output channel, contracting over C * prod(kernel).
//
//   out[n, m, o_0..o_{d-1}]
//     = bias[m] + sum_{c, j_0..j_{d-1}}
//         input[n, c, o_0*k_0 + j_0, ...] * W[m, c, j_0, ...]
//
// Why this is a native C++ pattern and what declining costs are in the header;
// what follows is the IR it emits.
//
// Single patch per batch element (kernel == input spatial). Both permutations
// are identities on the linear layout, so neither is emitted:
//
//   Before:
//     %y = hip.conv(%ctx) ins(%x, %w, %b) outs(%init : tensor<Nx M x1x1>)
//   After:
//     %xf = tensor.collapse_shape %x [[0], [1, 2, 3]]  : tensor<Nx K>
//     %wf = tensor.collapse_shape %w [[0], [1, 2, 3]]  : tensor<Mx K>
//     %g  = hip.gemm(%ctx) ins(%xf, %wf, %b) outs(...) {transB = 1}
//     %y  = tensor.expand_shape %g [[0], [1, 2, 3]]    : tensor<Nx M x1x1>
//
// Multiple patches. Gathering each patch into a contiguous GEMM row needs a
// permutation, and delivering the result in the channels-first layout the Conv
// declares needs another:
//
//   After:
//     %s  = tensor.expand_shape %x       -> [N, C, O_0, k_0, ..., O_d, k_d]
//     %t  = hip.transpose %s {perm = [0, 2,4,..., 1, 3,5,...]}
//                                        -> [N, O..., C, k...]
//     %xf = tensor.collapse_shape %t     -> [N*prod(O), K]
//     %wf = tensor.collapse_shape %w     -> [M, K]
//     %p  = hip.gemm(%ctx) ins(%xf, %wf, %b) outs(...) {transB = 1}
//     %r  = tensor.expand_shape %p       -> [N, O_0, ..., O_d, M]
//     %y  = hip.transpose %r {perm = [0, d+1, 1, ..., d]} -> [N, M, O...]
//
// The Conv's bias passes straight through as the GEMM's optional C operand:
// it holds one value per output channel, which is the GEMM's column axis, so
// it broadcasts exactly as ONNX Gemm requires.
//===----------------------------------------------------------------------===//

#include "patch_embed_conv_to_gemm.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

/// The extents the rewrite needs, all compile-time known.
struct PatchEmbedPlan {
  /// Patch grid extent per spatial axis, i.e. the Conv's output spatial dims.
  llvm::SmallVector<int64_t> outSpatial;
  /// Patch extent per spatial axis, i.e. the Conv's kernel dims.
  llvm::SmallVector<int64_t> kernelSpatial;
  /// Input channels.
  int64_t inChannels = 0;
  /// Output channels, the GEMM's column count.
  int64_t outChannels = 0;
  /// Contraction extent, C * prod(kernel).
  int64_t contractExtent = 0;
  /// Patches per batch element, prod(outSpatial).
  int64_t patchCount = 0;

  /// A single patch per batch element makes both of the rewrite's permutations
  /// identities on the linear layout, so the gather and the channels-first
  /// transposes are never emitted and the whole rewrite is two reshapes.
  bool needsGather() const { return patchCount != 1; }
};

/// Decide whether a Conv partitions its input into disjoint patches, which is
/// what makes it expressible as a GEMM. On failure, `reason` names the guard
/// that declined, for the caller to hand to `notifyMatchFailure`.
std::optional<PatchEmbedPlan>
analyzePatchEmbedConv(mlir::RankedTensorType xType,
                      mlir::RankedTensorType wType,
                      mlir::RankedTensorType yType,
                      llvm::ArrayRef<int64_t> strides,
                      llvm::ArrayRef<int64_t> pads,
                      llvm::ArrayRef<int64_t> dilations, int64_t group,
                      llvm::StringRef &reason) {
  auto decline = [&](llvm::StringRef why) -> std::optional<PatchEmbedPlan> {
    reason = why;
    return std::nullopt;
  };

  int64_t rank = xType.getRank();
  if (rank < 4 || rank != wType.getRank() || rank != yType.getRank())
    return decline("conv.rank_mismatch");

  // group must be 1 (default).
  if (group != 1)
    return decline("conv.group_ne_1");

  // pads all zero: a padded patch is not a contiguous slice of the input.
  if (llvm::any_of(pads, [](int64_t p) { return p != 0; }))
    return decline("conv.has_padding");

  // A dilated patch is strided within the input, not contiguous.
  if (llvm::any_of(dilations, [](int64_t d) { return d != 1; }))
    return decline("conv.has_dilation");

  int64_t nSpatial = rank - 2;
  if (static_cast<int64_t>(strides.size()) != nSpatial)
    return decline("conv.strides_rank");

  if (xType.getDimSize(0) != yType.getDimSize(0))
    return decline("conv.batch_mismatch");

  if (xType.getElementType() != wType.getElementType() ||
      xType.getElementType() != yType.getElementType())
    return decline("conv.mixed_element_types");

  PatchEmbedPlan plan;
  plan.inChannels = xType.getDimSize(1);
  if (plan.inChannels == mlir::ShapedType::kDynamic)
    return decline("conv.cin_dynamic");
  // With group == 1 a filter spans the whole channel axis, so the contraction
  // extent is C * prod(kernel) only if the weight declares the same C.
  if (wType.getDimSize(1) != plan.inChannels)
    return decline("conv.weight_channels_mismatch");

  plan.contractExtent = plan.inChannels;
  plan.patchCount = 1;
  for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
    int64_t xs = xType.getDimSize(2 + i);
    int64_t ws = wType.getDimSize(2 + i);
    int64_t ys = yType.getDimSize(2 + i);
    if (xs == mlir::ShapedType::kDynamic || ws == mlir::ShapedType::kDynamic)
      return decline("conv.spatial_dynamic");
    if (ys == mlir::ShapedType::kDynamic)
      return decline("conv.out_spatial_dynamic");
    // The non-overlap requirement: consecutive windows must abut exactly.
    if (strides[i] != ws)
      return decline("conv.stride_ne_kernel");
    // A ragged trailing patch would need a Slice before the reshape.
    if (ws < 1 || xs < ws || xs % ws != 0)
      return decline("conv.input_not_tiled_by_kernel");
    if (ys != xs / ws)
      return decline("conv.out_spatial_mismatch");
    plan.kernelSpatial.push_back(ws);
    plan.outSpatial.push_back(ys);
    plan.patchCount *= ys;
    plan.contractExtent *= ws;
  }

  plan.outChannels = wType.getDimSize(0);
  if (plan.outChannels == mlir::ShapedType::kDynamic)
    return decline("conv.m_dynamic");

  // A 1x1 kernel satisfies `stride == kernel` trivially, but it is not a patch
  // embed and this rewrite makes it slower, so decline it.
  //
  // With every kernel extent 1 the patch is a single pixel: the split reshape
  // is an identity, and the two transposes the gather path emits degenerate
  // into a plain NCHW <-> NHWC round trip around a contraction that was already
  // a GEMM over C. Nothing is gained and two full passes over the input and the
  // output are added. The cost shows up as soon as a model has 1x1 convs in
  // bulk: detr's ResNet-50 backbone has 33, and converting them took transposes
  // from 94 calls to 161 and cost 36% of the model's runtime.
  //
  // The rewrite only pays when the patch is big enough that the GEMM dominates
  // the permutations it needs -- gemma3-4b contracts over 14x14x3 = 588 per
  // output element, Qwen's rank-5 embed over 2x16x16x3 = 1536. Requiring more
  // than one element in the patch is the weakest condition that separates those
  // from a 1x1, and it leaves the 1x1 case to the ordinary conv path, which
  // handles it as a K = C contraction with no data movement at all.
  //
  // The single-patch shapes are exempt: there both permutations are identities
  // on the linear layout and are never emitted, so the rewrite is a pure win
  // regardless of patch size.
  if (plan.needsGather() &&
      llvm::all_of(plan.kernelSpatial, [](int64_t k) { return k == 1; }))
    return decline("conv.unit_kernel");

  // The gather path splits every spatial axis in two, so the intermediate is
  // rank 2 + 2*nSpatial. hip.transpose caps out at rank 8, which a rank-5 Conv
  // hits exactly; declining beyond that keeps the Conv on a path that works
  // rather than failing a later pass.
  if (plan.needsGather() && 2 + 2 * nSpatial > 8)
    return decline("conv.gather_rank_too_high");

  return plan;
}

llvm::SmallVector<int64_t> readI64Array(mlir::ArrayAttr attr) {
  llvm::SmallVector<int64_t> values;
  if (!attr)
    return values;
  for (mlir::Attribute a : attr)
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
      values.push_back(ia.getValue().getSExtValue());
  return values;
}

struct PatchEmbedConvToGemm : public mlir::OpRewritePattern<mlir::hip::ConvOp> {
  PatchEmbedConvToGemm(mlir::MLIRContext *ctx, mlir::PatternBenefit benefit)
      : mlir::OpRewritePattern<mlir::hip::ConvOp>(ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::ConvOp conv,
                  mlir::PatternRewriter &rewriter) const override {
    // Runs pre-bufferization, where a HIP DPS op carries its result as a
    // tensor. In memref mode there is no result to replace.
    if (conv->getNumResults() != 1)
      return rewriter.notifyMatchFailure(conv, "conv.no_tensor_result");

    mlir::Value x = conv.getInput();
    mlir::Value w = conv.getWeights();
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto wType = mlir::dyn_cast<mlir::RankedTensorType>(w.getType());
    auto yType =
        mlir::dyn_cast<mlir::RankedTensorType>(conv->getResult(0).getType());
    if (!xType || !wType || !yType)
      return rewriter.notifyMatchFailure(conv, "conv.not_ranked");

    // hip.conv fills in ONNX's per-axis defaults during conversion, so every
    // geometry attribute is present and explicit. `pads` is taken at face
    // value: hip.conv has no auto_pad, so nothing else can be overriding it.
    llvm::SmallVector<int64_t> strides = readI64Array(conv.getStrides());
    llvm::SmallVector<int64_t> pads = readI64Array(conv.getPads());
    llvm::SmallVector<int64_t> dilations = readI64Array(conv.getDilations());

    llvm::StringRef reason;
    std::optional<PatchEmbedPlan> plan = analyzePatchEmbedConv(
        xType, wType, yType, strides, pads, dilations,
        static_cast<int64_t>(conv.getGroup()), reason);
    if (!plan)
      return rewriter.notifyMatchFailure(conv, reason);

    mlir::Location loc = conv.getLoc();
    mlir::Value ctx = conv.getCtx();
    mlir::Type elemType = xType.getElementType();
    const int64_t rank = xType.getRank();
    const int64_t nSpatial = rank - 2;
    const int64_t N = xType.getDimSize(0);
    const int64_t C = plan->inChannels;
    const int64_t K = plan->contractExtent;
    const int64_t M = plan->outChannels;

    // Batch is the only extent the analysis leaves dynamic, so it is the only
    // one the reshapes and the `tensor.empty` inits below have to restate.
    // Materialized once here so they all share a single `tensor.dim`.
    mlir::Value dynBatch;
    if (xType.isDynamicDim(0))
      dynBatch = mlir::tensor::DimOp::create(rewriter, loc, x, 0).getResult();
    const mlir::OpFoldResult batch =
        dynBatch ? mlir::OpFoldResult(dynBatch)
                 : mlir::OpFoldResult(rewriter.getIndexAttr(N));
    const mlir::OpFoldResult one = rewriter.getIndexAttr(1);

    // Every init built here is batch-major with the rest of its extents
    // static, which is what lets a single dynamic size cover all of them.
    auto emptyLike = [&](mlir::RankedTensorType type) -> mlir::Value {
      llvm::SmallVector<mlir::Value> dynSizes;
      if (type.isDynamicDim(0))
        dynSizes.push_back(dynBatch);
      return mlir::tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                           type.getElementType(), dynSizes)
          .getResult();
    };

    auto emitTranspose = [&](mlir::Value src, llvm::ArrayRef<int64_t> perm,
                             mlir::RankedTensorType resultType) -> mlir::Value {
      llvm::SmallVector<mlir::NamedAttribute> attrs{
          rewriter.getNamedAttr("perm", rewriter.getI64ArrayAttr(perm))};
      return mlir::hip::TransposeOp::create(
                 rewriter, loc,
                 mlir::ValueRange{ctx, src, emptyLike(resultType)}, attrs)
          ->getResult(0);
    };

    // Flatten [leading, tail...] into [leading, prod(tail)]: the weight
    // always, and the input too when there is a single patch.
    llvm::SmallVector<mlir::ReassociationIndices> headTailReassoc(2);
    headTailReassoc[0].push_back(0);
    for (int64_t i : llvm::seq<int64_t>(1, rank))
      headTailReassoc[1].push_back(i);

    // Rows of the GEMM: one per (batch, patch).
    const int64_t P = (N == mlir::ShapedType::kDynamic)
                          ? mlir::ShapedType::kDynamic
                          : N * plan->patchCount;

    mlir::Value xFlat;
    if (!plan->needsGather()) {
      // Every kernel extent equals its input extent, so [N, C, k...] already
      // holds one contiguous patch per row.
      xFlat = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, mlir::RankedTensorType::get({P, K}, elemType), x,
          headTailReassoc);
    } else {
      // Split each spatial axis into (patch index, offset within patch):
      //   [N, C, S_0, ..., S_{d-1}] -> [N, C, O_0, k_0, ..., O_{d-1}, k_{d-1}]
      llvm::SmallVector<int64_t> splitShape{N, C};
      llvm::SmallVector<mlir::OpFoldResult> splitSizes{
          batch, rewriter.getIndexAttr(C)};
      llvm::SmallVector<mlir::ReassociationIndices> splitReassoc(2 + nSpatial);
      splitReassoc[0].push_back(0);
      splitReassoc[1].push_back(1);
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
        splitShape.push_back(plan->outSpatial[i]);
        splitShape.push_back(plan->kernelSpatial[i]);
        splitSizes.push_back(rewriter.getIndexAttr(plan->outSpatial[i]));
        splitSizes.push_back(rewriter.getIndexAttr(plan->kernelSpatial[i]));
        splitReassoc[2 + i].push_back(2 + 2 * i);
        splitReassoc[2 + i].push_back(3 + 2 * i);
      }
      mlir::Value split = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, mlir::RankedTensorType::get(splitShape, elemType), x,
          splitReassoc, splitSizes);

      // Move the patch indices ahead of the channel and the intra-patch
      // offsets, so one GEMM row is one contiguous patch:
      //   [N, C, O_0, k_0, ...] -> [N, O_0, ..., O_{d-1}, C, k_0, ...]
      llvm::SmallVector<int64_t> perm{0};
      llvm::SmallVector<int64_t> permShape{N};
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
        perm.push_back(2 + 2 * i);
        permShape.push_back(plan->outSpatial[i]);
      }
      perm.push_back(1);
      permShape.push_back(C);
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
        perm.push_back(3 + 2 * i);
        permShape.push_back(plan->kernelSpatial[i]);
      }
      mlir::Value gathered = emitTranspose(
          split, perm, mlir::RankedTensorType::get(permShape, elemType));

      // [N, O...] collapses into the row axis, [C, k...] into the column one.
      llvm::SmallVector<mlir::ReassociationIndices> rowColReassoc(2);
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial + 1))
        rowColReassoc[0].push_back(i);
      for (int64_t i : llvm::seq<int64_t>(nSpatial + 1, 2 * nSpatial + 2))
        rowColReassoc[1].push_back(i);
      xFlat = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, mlir::RankedTensorType::get({P, K}, elemType),
          gathered, rowColReassoc);
    }

    // The weight is already laid out channel-major then kernel-major, which is
    // exactly the row order the input above ends up in.
    mlir::Value wFlat = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, mlir::RankedTensorType::get({M, K}, elemType), w,
        headTailReassoc);

    // The GEMM's row count is the batch times the patch count. Read it off the
    // operand that already carries it rather than re-deriving the product.
    auto gemmType = mlir::RankedTensorType::get({P, M}, elemType);
    llvm::SmallVector<mlir::Value> gemmDynSizes;
    if (gemmType.isDynamicDim(0))
      gemmDynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, xFlat, 0).getResult());
    mlir::Value gemmInit =
        mlir::tensor::EmptyOp::create(rewriter, loc, gemmType.getShape(),
                                      elemType, gemmDynSizes)
            .getResult();

    llvm::SmallVector<mlir::Value> gemmOperands{ctx, xFlat, wFlat};
    if (mlir::Value bias = conv.getBias())
      gemmOperands.push_back(bias);
    gemmOperands.push_back(gemmInit);
    llvm::SmallVector<mlir::NamedAttribute> gemmAttrs{
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(0)),
        // The weight contracts along its trailing axes, so B arrives as [M, K]
        // and the GEMM transposes it rather than the rewrite doing so.
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(1))};
    mlir::Value gemmOut =
        mlir::hip::GemmOp::create(rewriter, loc, gemmOperands, gemmAttrs)
            ->getResult(0);

    mlir::Value result;
    if (!plan->needsGather()) {
      // [N, M] and [N, M, 1, ..., 1] share a linear layout.
      llvm::SmallVector<mlir::OpFoldResult> outSizes{batch,
                                                     rewriter.getIndexAttr(M)};
      outSizes.append(nSpatial, one);
      result = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, yType, gemmOut, headTailReassoc, outSizes);
    } else {
      // The GEMM produced [N*prod(O), M], i.e. channels-last per patch. The
      // Conv declares channels-first, and [N, O..., M] -> [N, M, O...] is a
      // permutation, not a reshape -- so split the batch and patch axes back
      // out and move the channel axis forward.
      llvm::SmallVector<int64_t> nlcShape{N};
      llvm::SmallVector<mlir::OpFoldResult> nlcSizes{batch};
      llvm::SmallVector<mlir::ReassociationIndices> nlcReassoc(2);
      nlcReassoc[0].push_back(0);
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
        nlcShape.push_back(plan->outSpatial[i]);
        nlcSizes.push_back(rewriter.getIndexAttr(plan->outSpatial[i]));
        nlcReassoc[0].push_back(1 + i);
      }
      nlcShape.push_back(M);
      nlcSizes.push_back(rewriter.getIndexAttr(M));
      nlcReassoc[1].push_back(nSpatial + 1);
      mlir::Value nlc = mlir::tensor::ExpandShapeOp::create(
          rewriter, loc, mlir::RankedTensorType::get(nlcShape, elemType),
          gemmOut, nlcReassoc, nlcSizes);

      llvm::SmallVector<int64_t> outPerm{0, nSpatial + 1};
      for (int64_t i : llvm::seq<int64_t>(0, nSpatial))
        outPerm.push_back(1 + i);
      result = emitTranspose(nlc, outPerm, yType);
    }

    rewriter.replaceOp(conv, result);
    return mlir::success();
  }
};

} // namespace

void populatePatchEmbedConvToGemmPattern(mlir::RewritePatternSet &patterns,
                                         mlir::PatternBenefit benefit) {
  patterns.add<PatchEmbedConvToGemm>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
