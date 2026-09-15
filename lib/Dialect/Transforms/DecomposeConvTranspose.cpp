/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <numeric>

namespace mlir::hip {
#define GEN_PASS_DEF_DECOMPOSECONVTRANSPOSEPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Per spatial-dim quantities shared by every residue, following MIOpen's
/// implicit-gemm backward-data v4r1 decomposition (the same one MIGraphX
/// applies in `rewrite_convolution`).
///
/// Only the dilation == 1 case is handled here, which collapses the general
/// gcd(stride, dilation) arithmetic to ytilda == stride and dd == 1.
struct DimInfo {
  int64_t stride;
  int64_t kernel; // filter length Y
  int64_t ydot;   // ceil(Y / stride) -> max taps per residue
  int64_t htilda; // per-residue conv output length
  int64_t full;   // reassembled length, htilda * stride
  int64_t padLo;  // leading pad of the transposed conv
  int64_t out;    // final output length
};

/// A transposed convolution scatters each input element across `stride`
/// output positions, so it splits into `stride` dense stride-1 convolutions
/// ("residues"), one per output phase, whose results interleave.
///
/// Residue `itilda` uses filter taps {itilda, itilda+stride, ...} reversed,
/// with the weight's input/output channels swapped. The taps and the flip
/// only touch the (constant) filter, so rather than emitting slice / step /
/// reverse ops -- TOSA has no strided slice to lower a `step` to -- the
/// sub-filters are computed here and materialized as new `hip.constant`s.
///
/// Reassembly is a strided `tensor.insert_slice` per residue, which
/// bufferizes to a `memref.subview` + `memref.copy`. When every residue is
/// non-empty (stride <= kernel) the residues tile the output exactly, so the
/// destination needs no zero-fill.
class DecomposeConvTranspose : public OpRewritePattern<ConvTransposeOp> {
public:
  using OpRewritePattern<ConvTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvTransposeOp op,
                                PatternRewriter &rewriter) const override;
};

SmallVector<int64_t> getI64Values(ArrayAttr attr) {
  SmallVector<int64_t> values;
  if (!attr)
    return values;
  for (Attribute a : attr)
    values.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
  return values;
}

/// Flat index into a row-major 4-D buffer.
int64_t flatIndex(ArrayRef<int64_t> shape, int64_t a, int64_t b, int64_t c,
                  int64_t d) {
  return ((a * shape[1] + b) * shape[2] + c) * shape[3] + d;
}

LogicalResult
DecomposeConvTranspose::matchAndRewrite(ConvTransposeOp op,
                                        PatternRewriter &rewriter) const {
  if (op.getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");

  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
  if (inputType.getRank() != 4 || weightType.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "expected 2D transposed conv");

  // hip.conv lowers to tosa.conv2d, which has no grouped form.
  if (op.getGroup() != 1)
    return rewriter.notifyMatchFailure(op, "grouped transposed conv");

  // The residues become hip.conv, which lowers to tosa.conv2d only for these
  // three types. Anything else (f64 in particular) would decompose here and
  // then fail to legalize, so leave it on the MIOpen path instead.
  Type elementType = resultType.getElementType();
  if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32())
    return rewriter.notifyMatchFailure(op, "expected f16, bf16, or f32");
  if (inputType.getElementType() != elementType ||
      weightType.getElementType() != elementType)
    return rewriter.notifyMatchFailure(op, "expected matching element types");

  // The tap selection and flip are folded into new constants, so the filter
  // has to be available at compile time.
  auto weightConst = op.getWeights().getDefiningOp<ConstantOp>();
  if (!weightConst)
    return rewriter.notifyMatchFailure(op, "weights are not a hip.constant");
  auto weightAttr =
      dyn_cast_if_present<DenseElementsAttr>(weightConst.getValueAttr());
  if (!weightAttr)
    return rewriter.notifyMatchFailure(op, "weights have no inline value");

  SmallVector<int64_t> strides = getI64Values(op.getStridesAttr());
  SmallVector<int64_t> pads = getI64Values(op.getPadsAttr());
  SmallVector<int64_t> dilations = getI64Values(op.getDilationsAttr());
  if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2D spatial attributes");
  if (dilations[0] != 1 || dilations[1] != 1)
    return rewriter.notifyMatchFailure(op, "dilated transposed conv");

  // ConvTranspose weights are [C, M/group, kH, kW] -- input channels first.
  ArrayRef<int64_t> weightShape = weightType.getShape();
  const int64_t numBatch = inputType.getDimSize(0);
  const int64_t inChannels = weightShape[0];
  const int64_t outChannels = weightShape[1];
  if (inputType.getDimSize(1) != inChannels ||
      resultType.getDimSize(1) != outChannels)
    return rewriter.notifyMatchFailure(op, "incompatible channel counts");

  // Checked up front rather than at the point of use: a pattern must leave the
  // IR untouched when it fails, and by then the residues have been emitted.
  Value bias = op.getBias();
  if (bias) {
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != outChannels ||
        biasType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "expected a static 1-D bias of matching type and length M");
  }

  SmallVector<DimInfo, 2> dims;
  for (int64_t d : llvm::seq<int64_t>(2)) {
    DimInfo di;
    di.stride = strides[d];
    di.kernel = weightShape[2 + d];
    di.padLo = pads[d];
    di.out = resultType.getDimSize(2 + d);
    if (di.stride < 1 || di.padLo < 0)
      return rewriter.notifyMatchFailure(op, "expected positive stride");
    // A residue whose first tap falls outside the filter is empty, which
    // would leave holes in the destination that nothing writes.
    if (di.stride > di.kernel)
      return rewriter.notifyMatchFailure(op, "stride exceeds kernel");
    di.ydot = (di.kernel + di.stride - 1) / di.stride;
    di.htilda = inputType.getDimSize(2 + d) + di.ydot - 1;
    di.full = di.htilda * di.stride;
    // Cropping off the leading pad must leave enough for the result, which
    // also covers output_padding: the slack positions are never written by
    // any residue and the transposed conv would produce zeros there too.
    if (di.padLo + di.out > di.full)
      return rewriter.notifyMatchFailure(op, "output exceeds the residue grid");
    dims.push_back(di);
  }

  // One convolution -- and so one serial rocMLIR compilation, with its own
  // tuning-space enumeration -- is emitted per residue, and the count is the
  // product of the strides. Large strides are legal but would expand into
  // hundreds of kernels and dominate compile time, so they stay on the MIOpen
  // path. Covers the strides transposed convolutions actually use (2 and 4 for
  // upsampling, up to 8x8 here) without opening that hole.
  constexpr int64_t kMaxResidues = 64;
  if (dims[0].stride * dims[1].stride > kMaxResidues)
    return rewriter.notifyMatchFailure(op, "too many residues to be worth it");

  Location loc = op.getLoc();
  Value ctx = op.getCtx();
  SmallVector<APFloat> weightValues(weightAttr.getValues<APFloat>().begin(),
                                    weightAttr.getValues<APFloat>().end());

  auto fullType =
      resultType.clone({numBatch, outChannels, dims[0].full, dims[1].full});
  Value scattered =
      tensor::EmptyOp::create(rewriter, loc, fullType.getShape(), elementType)
          .getResult();

  const APFloat zero =
      APFloat::getZero(cast<FloatType>(elementType).getFloatSemantics());

  for (int64_t it0 : llvm::seq<int64_t>(dims[0].stride)) {
    for (int64_t it1 : llvm::seq<int64_t>(dims[1].stride)) {
      // Taps kept by this residue along each dim.
      const int64_t taps0 =
          (dims[0].kernel - it0 + dims[0].stride - 1) / dims[0].stride;
      const int64_t taps1 =
          (dims[1].kernel - it1 + dims[1].stride - 1) / dims[1].stride;

      SmallVector<int64_t, 4> subShape{outChannels, inChannels, taps0, taps1};
      SmallVector<APFloat> subValues(outChannels * inChannels * taps0 * taps1,
                                     zero);
      for (int64_t m : llvm::seq<int64_t>(outChannels))
        for (int64_t c : llvm::seq<int64_t>(inChannels))
          for (int64_t a : llvm::seq<int64_t>(taps0))
            for (int64_t b : llvm::seq<int64_t>(taps1)) {
              // Reversed tap order: position a holds tap (taps0 - 1 - a).
              const int64_t y0 = it0 + (taps0 - 1 - a) * dims[0].stride;
              const int64_t y1 = it1 + (taps1 - 1 - b) * dims[1].stride;
              subValues[flatIndex(subShape, m, c, a, b)] =
                  weightValues[flatIndex(weightShape, c, m, y0, y1)];
            }

      auto subWeightType = RankedTensorType::get(subShape, elementType);
      Value subWeight =
          ConstantOp::create(rewriter, loc, subWeightType,
                             DenseElementsAttr::get(subWeightType, subValues))
              .getResult();

      auto partialType = resultType.clone(
          {numBatch, outChannels, dims[0].htilda, dims[1].htilda});
      Value partialInit =
          tensor::EmptyOp::create(rewriter, loc, partialType.getShape(),
                                  elementType)
              .getResult();
      Value partial =
          ConvOp::create(
              rewriter, loc, TypeRange{partialType},
              ValueRange{ctx, op.getInput(), subWeight, partialInit},
              ArrayRef<NamedAttribute>{
                  rewriter.getNamedAttr(
                      "kernel_shape", rewriter.getI64ArrayAttr({taps0, taps1})),
                  rewriter.getNamedAttr("strides",
                                        rewriter.getI64ArrayAttr({1, 1})),
                  rewriter.getNamedAttr(
                      "pads", rewriter.getI64ArrayAttr({taps0 - 1, taps1 - 1,
                                                        dims[0].ydot - 1,
                                                        dims[1].ydot - 1})),
                  rewriter.getNamedAttr("dilations",
                                        rewriter.getI64ArrayAttr({1, 1})),
                  rewriter.getNamedAttr("group",
                                        rewriter.getI64IntegerAttr(1))})
              .getResult(0);

      // Residue itilda lands on output positions {itilda + k * stride}.
      SmallVector<OpFoldResult> offsets{
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(it0), rewriter.getIndexAttr(it1)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(numBatch),
                                      rewriter.getIndexAttr(outChannels),
                                      rewriter.getIndexAttr(dims[0].htilda),
                                      rewriter.getIndexAttr(dims[1].htilda)};
      SmallVector<OpFoldResult> insertStrides{
          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(dims[0].stride),
          rewriter.getIndexAttr(dims[1].stride)};
      scattered =
          tensor::InsertSliceOp::create(rewriter, loc, partial, scattered,
                                        offsets, sizes, insertStrides)
              .getResult();
    }
  }

  // Crop the padding region (and any residue slack) down to the result.
  SmallVector<OpFoldResult> cropOffsets{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(dims[0].padLo),
                                        rewriter.getIndexAttr(dims[1].padLo)};
  SmallVector<OpFoldResult> cropSizes{
      rewriter.getIndexAttr(numBatch), rewriter.getIndexAttr(outChannels),
      rewriter.getIndexAttr(dims[0].out), rewriter.getIndexAttr(dims[1].out)};
  SmallVector<OpFoldResult> cropStrides(4, rewriter.getIndexAttr(1));
  Value result =
      tensor::ExtractSliceOp::create(rewriter, loc, resultType, scattered,
                                     cropOffsets, cropSizes, cropStrides)
          .getResult();

  // One broadcast add on the final shape. The residues tile the grid, so each
  // output element belongs to exactly one of them and passing the bias to every
  // residue convolution would be equally correct -- and would fuse into the
  // convolution epilogue instead of costing a separate dispatch. Kept separate
  // here so the bias does not depend on the residue tiling.
  if (bias) {
    auto biasType = cast<RankedTensorType>(bias.getType());
    // Broadcast against [N, M, H, W] rather than the trailing axis.
    auto shapedBiasType =
        RankedTensorType::get({1, biasType.getDimSize(0), 1, 1}, elementType);
    Value shapedBias = tensor::ExpandShapeOp::create(
                           rewriter, loc, shapedBiasType, bias,
                           SmallVector<ReassociationIndices>{{0, 1, 2, 3}})
                           .getResult();
    Value biasInit = tensor::EmptyOp::create(rewriter, loc,
                                             resultType.getShape(), elementType)
                         .getResult();
    result = AddOp::create(rewriter, loc, resultType, ctx, result, shapedBias,
                           biasInit)
                 .getResult(0);
  }

  rewriter.replaceOp(op, result);
  return success();
}

struct DecomposeConvTransposePass
    : public impl::DecomposeConvTransposePassBase<DecomposeConvTransposePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeConvTranspose>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
