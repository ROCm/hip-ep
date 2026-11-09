/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipConstantData.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
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
  int64_t inLen;  // input length
  int64_t ydot;   // ceil(Y / stride) -> max taps per residue
  int64_t htilda; // per-residue conv output length
  int64_t full;   // reassembled length, htilda * stride
  int64_t padLo;  // leading pad of the transposed conv
  int64_t out;    // final output length

  // Each residue pads its input by `taps - 1` in front, so its output length
  // is `inLen + trailingPad` whatever its tap count. That makes the trailing
  // pad the single knob for sizing the grid, uniformly across residues.
  int64_t trailingPad() const { return htilda - inLen; }
};

/// A transposed convolution scatters each input element across `stride`
/// output positions, so it splits into one dense stride-1 convolution per
/// output phase ("residue"), whose results interleave. In 2-D the phases are
/// independent per axis, so the count is `strideH * strideW`.
///
/// Residue `itilda` uses filter taps {itilda, itilda+stride, ...} reversed,
/// with the weight's input/output channels swapped. The taps and the flip
/// only touch the (constant) filter, so rather than emitting slice / step /
/// reverse ops -- TOSA has no strided slice to lower a `step` to -- the
/// sub-filters are computed here and materialized as new `hip.constant`s.
///
/// When every residue has the same kernel and the same padding (the filter
/// length is a multiple of the stride, as in a 4x4 filter at stride 2), the
/// residue filters are stacked along the output channels into one wide
/// convolution, and its result is reassembled as a DepthToSpace: reshape the
/// channels to `[strideH, strideW, M]`, `hip.transpose` so each length sits
/// next to its stride, then reshape to `[N, M, Ht*strideH, Wt*strideW]`.
/// The rocMLIR anchor fusion folds that transpose into the convolution's
/// store. Emitting per-residue convolutions joined by a concat instead leaves
/// the concat to bufferize into one strided copy per residue. Residues with
/// fewer taps are padded with zero taps to the same kernel when that at most
/// doubles the multiply-adds (a 3x3 filter at stride 2), so they take the
/// same path. A single residue (stride 1) has nothing to interleave and is
/// copied onto the grid with one `tensor.insert_slice`.
///
/// Otherwise the convolutions cannot share one weight, so reassembly stays a
/// strided `tensor.insert_slice` per residue, which bufferizes to a
/// `memref.subview` + `memref.copy`. When every residue is non-empty
/// (stride <= kernel) the residues tile the grid exactly, so the destination
/// needs no zero-fill. For the same reason a bias becomes the bias of the
/// residue convolutions; only a run-time bias on the wide convolution, which
/// cannot be repeated per residue here, stays an add after the crop.
///
/// Taking a 1-D slice of the unequal-tap case, with a 3-tap filter [w0 w1 w2]
/// over a 4-long input, before:
///
///   hip.conv_transpose ins(%x : tensor<1x1x4xf32>, %w : tensor<1x1x3xf32>)
///                      {strides = [2], pads = [1, 1], output_padding = [1]}
///                      -> tensor<1x1x8xf32>
///
/// after -- residue 0 keeps taps {0, 2} reversed to [w2 w0], residue 1 keeps
/// tap {1}, each convolved at stride 1 and scattered onto alternating
/// positions of a length-10 grid, which is then cropped by the leading pad:
///
///   %r0 = hip.conv ins(%x, [w2 w0]) {strides = [1]} -> tensor<1x1x5xf32>
///   %r1 = hip.conv ins(%x, [w1])    {strides = [1]} -> tensor<1x1x5xf32>
///   %g0 = tensor.insert_slice %r0 into %empty[0][5][2]  // grid[0,2,4,6,8]
///   %g1 = tensor.insert_slice %r1 into %g0   [1][5][2]  // grid[1,3,5,7,9]
///   %y  = tensor.extract_slice %g1[1][8][1]             // drop the pad
///
/// The grid can be longer than the transposed convolution's natural extent
/// (here 10 against 9) when the filter is not a multiple of the stride. Those
/// trailing positions are still written by a residue, but the value is zero:
/// they draw only on input elements past the end, which the residue
/// convolution's own zero padding supplies. ONNX defines the `output_padding`
/// region as zero too, so it lands in the same slack -- and when there is not
/// enough of it (the stride divides the filter, leaving none) each residue's
/// trailing pad is extended to grow the grid, which adds more of exactly these
/// zero-valued positions.
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

/// Decompose a 1-D transposed convolution through the same 2-D hip.conv path
/// used by the general decomposition. The temporary height dimension is one:
///
///   [N, C, W] -> [N, C, 1, W]
///
/// Keeping W dynamic is important: the residue padding is derived solely from
/// static ConvTranspose attributes, while tensor.dim supplies the dynamic
/// extent for the temporary tensors and the final crop.
static LogicalResult decompose1DConvTranspose(ConvTransposeOp op,
                                              PatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inputType || !weightType || !resultType || !weightType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "expected ranked tensors and static weights");
  if (inputType.getRank() != 3 || weightType.getRank() != 3 ||
      resultType.getRank() != 3)
    return rewriter.notifyMatchFailure(op, "expected 1D transposed conv");
  if (inputType.isDynamicDim(1) || resultType.isDynamicDim(1))
    return rewriter.notifyMatchFailure(op,
                                       "expected static channel dimensions");
  if (op.getGroup() != 1)
    return rewriter.notifyMatchFailure(op, "grouped transposed conv");

  Type elementType = resultType.getElementType();
  if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32())
    return rewriter.notifyMatchFailure(op, "expected f16, bf16, or f32");
  if (inputType.getElementType() != elementType ||
      weightType.getElementType() != elementType)
    return rewriter.notifyMatchFailure(op, "expected matching element types");

  auto weightConst = op.getWeights().getDefiningOp<ConstantOp>();
  if (!weightConst)
    return rewriter.notifyMatchFailure(op, "weights are not a hip.constant");
  FailureOr<DenseElementsAttr> weightData =
      readConstantData(weightConst, weightType);
  if (failed(weightData))
    return rewriter.notifyMatchFailure(op, "weights are not readable here");
  SmallVector<APFloat> weightValues(weightData->getValues<APFloat>().begin(),
                                    weightData->getValues<APFloat>().end());

  SmallVector<int64_t> strides = getI64Values(op.getStridesAttr());
  SmallVector<int64_t> pads = getI64Values(op.getPadsAttr());
  SmallVector<int64_t> dilations = getI64Values(op.getDilationsAttr());
  SmallVector<int64_t> outputPadding = getI64Values(op.getOutputPaddingAttr());
  if (strides.size() != 1 || pads.size() != 2 || dilations.size() != 1 ||
      outputPadding.size() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1D spatial attributes");
  if (dilations[0] != 1)
    return rewriter.notifyMatchFailure(op, "dilated transposed conv");

  const int64_t stride = strides[0];
  const int64_t kernel = weightType.getDimSize(2);
  const int64_t padLo = pads[0];
  const int64_t padHi = pads[1];
  if (stride < 1 || padLo < 0 || padHi < 0 || stride > kernel)
    return rewriter.notifyMatchFailure(op, "unsupported stride or padding");
  if (outputPadding[0] < 0 || outputPadding[0] >= stride)
    return rewriter.notifyMatchFailure(
        op, "output_padding must be less than stride");

  const int64_t ydot = llvm::divideCeil(kernel, stride);
  // With dilation one, the grid length is inputWidth plus this static
  // extension. The second term is the static counterpart of the 2-D pass's
  // `max(natural, needed)` calculation.
  const int64_t neededYdot =
      llvm::divideCeil(kernel - padHi + outputPadding[0], stride);
  const int64_t trailingPad =
      ydot - 1 + std::max<int64_t>(0, neededYdot - ydot);

  const int64_t inChannels = weightType.getDimSize(0);
  const int64_t outChannels = weightType.getDimSize(1);
  if (inputType.getDimSize(1) != inChannels ||
      resultType.getDimSize(1) != outChannels)
    return rewriter.notifyMatchFailure(op, "incompatible batch or channels");
  if (!inputType.isDynamicDim(0) && !resultType.isDynamicDim(0) &&
      inputType.getDimSize(0) != resultType.getDimSize(0))
    return rewriter.notifyMatchFailure(op, "incompatible batch dimensions");

  Value bias = op.getBias();
  if (bias) {
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != outChannels ||
        biasType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "expected a static 1-D bias of matching type and length M");
  }

  Location loc = op.getLoc();
  auto input4Type = RankedTensorType::get(
      {inputType.getDimSize(0), inChannels, 1, inputType.getDimSize(2)},
      elementType);
  Value input4 = tensor::ExpandShapeOp::create(
                     rewriter, loc, input4Type, op.getInput(),
                     SmallVector<ReassociationIndices>{{0}, {1}, {2, 3}})
                     .getResult();

  Value inputBatch =
      tensor::DimOp::create(rewriter, loc, op.getInput(), 0).getResult();
  Value outputBatch =
      tensor::DimOp::create(rewriter, loc, op.getOutput(), 0).getResult();
  Value inputWidth =
      tensor::DimOp::create(rewriter, loc, op.getInput(), 2).getResult();
  Value outputWidth =
      tensor::DimOp::create(rewriter, loc, op.getOutput(), 2).getResult();
  Value partialWidth = arith::AddIOp::create(
      rewriter, loc, inputWidth,
      arith::ConstantIndexOp::create(rewriter, loc, ydot - 1 + trailingPad));
  auto dynamic4Type = RankedTensorType::get(
      {inputType.getDimSize(0), outChannels, 1, ShapedType::kDynamic},
      elementType);
  Value scattered =
      tensor::EmptyOp::create(rewriter, loc, dynamic4Type,
                              ValueRange{inputBatch, partialWidth})
          .getResult();
  const APFloat zero =
      APFloat::getZero(cast<FloatType>(elementType).getFloatSemantics());

  for (int64_t residue : llvm::seq<int64_t>(stride)) {
    const int64_t taps = (kernel - residue + stride - 1) / stride;
    SmallVector<int64_t, 4> subShape{outChannels, inChannels, 1, taps};
    SmallVector<APFloat> subValues(outChannels * inChannels * taps, zero);
    for (int64_t m : llvm::seq<int64_t>(outChannels))
      for (int64_t c : llvm::seq<int64_t>(inChannels))
        for (int64_t tap : llvm::seq<int64_t>(taps)) {
          const int64_t sourceTap = residue + (taps - 1 - tap) * stride;
          subValues[flatIndex(subShape, m, c, 0, tap)] =
              weightValues[(c * outChannels + m) * kernel + sourceTap];
        }

    auto subWeightType = RankedTensorType::get(subShape, elementType);
    Value subWeight =
        ConstantOp::create(rewriter, loc, subWeightType,
                           DenseElementsAttr::get(subWeightType, subValues))
            .getResult();
    Value partialInit =
        tensor::EmptyOp::create(rewriter, loc, dynamic4Type,
                                ValueRange{inputBatch, partialWidth})
            .getResult();
    Value partial =
        ConvOp::create(
            rewriter, loc, TypeRange{dynamic4Type},
            ValueRange{op.getCtx(), input4, subWeight, partialInit},
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("kernel_shape",
                                      rewriter.getI64ArrayAttr({1, taps})),
                rewriter.getNamedAttr("strides",
                                      rewriter.getI64ArrayAttr({1, 1})),
                rewriter.getNamedAttr(
                    "pads",
                    rewriter.getI64ArrayAttr({0, taps - 1, 0, trailingPad})),
                rewriter.getNamedAttr("dilations",
                                      rewriter.getI64ArrayAttr({1, 1})),
                rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(1))})
            .getResult(0);

    scattered =
        tensor::InsertSliceOp::create(
            rewriter, loc, partial, scattered,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(residue)},
            SmallVector<OpFoldResult>{inputBatch,
                                      rewriter.getIndexAttr(outChannels),
                                      rewriter.getIndexAttr(1), partialWidth},
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(stride)})
            .getResult();
  }

  auto result4Type = RankedTensorType::get(
      {resultType.getDimSize(0), outChannels, 1, ShapedType::kDynamic},
      elementType);
  Value result4 = tensor::ExtractSliceOp::create(
      rewriter, loc, result4Type, scattered,
      SmallVector<OpFoldResult>{
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(padLo)},
      SmallVector<OpFoldResult>{outputBatch, rewriter.getIndexAttr(outChannels),
                                rewriter.getIndexAttr(1), outputWidth},
      SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1)));

  if (bias) {
    auto shapedBiasType =
        RankedTensorType::get({1, outChannels, 1, 1}, elementType);
    Value shapedBias = tensor::ExpandShapeOp::create(
                           rewriter, loc, shapedBiasType, bias,
                           SmallVector<ReassociationIndices>{{0, 1, 2, 3}})
                           .getResult();
    Value biasInit =
        tensor::EmptyOp::create(rewriter, loc, result4Type,
                                ValueRange{outputBatch, outputWidth})
            .getResult();
    result4 = AddOp::create(rewriter, loc, result4Type, op.getCtx(), result4,
                            shapedBias, biasInit)
                  .getResult(0);
  }

  Value result = tensor::CollapseShapeOp::create(
                     rewriter, loc, resultType, result4,
                     SmallVector<ReassociationIndices>{{0}, {1}, {2, 3}})
                     .getResult();
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult
DecomposeConvTranspose::matchAndRewrite(ConvTransposeOp op,
                                        PatternRewriter &rewriter) const {
  if (op.getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");

  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (inputType && inputType.getRank() == 3)
    return decompose1DConvTranspose(op, rewriter);
  if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
  // The result rank is checked alongside the operands because nothing verifies
  // that it agrees with them: the type comes from the `outs` operand, which is
  // only constrained to be a tensor or memref. A rank<4 result would otherwise
  // reach getDimSize(2 + d) below and assert instead of bailing out here.
  if (inputType.getRank() != 4 || weightType.getRank() != 4 ||
      resultType.getRank() != 4)
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
  // has to be readable at compile time. Requiring an inline value would
  // decline nearly every real filter, which the importer externalizes to a
  // file-backed range. Memory-address carriers are not resolved: this pattern
  // is equally reachable from textual pass invocation, where the address
  // would be dereferenced blind.
  auto weightConst = op.getWeights().getDefiningOp<ConstantOp>();
  if (!weightConst)
    return rewriter.notifyMatchFailure(op, "weights are not a hip.constant");
  FailureOr<DenseElementsAttr> weightData =
      readConstantData(weightConst, weightType);
  if (failed(weightData))
    return rewriter.notifyMatchFailure(op, "weights are not readable here");
  DenseElementsAttr weightAttr = *weightData;

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
  // The batch is taken from the input and used for the residue convolutions,
  // the reassembly grid and the crop sizes, while the crop's result type comes
  // from the op. Nothing verifies those agree, so a mismatch would build a
  // `tensor.extract_slice` whose sizes contradict its result type -- invalid
  // IR, and a hard pass failure rather than a fallback.
  if (inputType.getDimSize(0) != resultType.getDimSize(0) ||
      inputType.getDimSize(1) != inChannels ||
      resultType.getDimSize(1) != outChannels)
    return rewriter.notifyMatchFailure(op, "incompatible batch or channels");

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
      return rewriter.notifyMatchFailure(op, "expected stride >= 1, pad >= 0");
    // A residue whose first tap falls outside the filter is empty, which
    // would leave holes in the destination that nothing writes.
    if (di.stride > di.kernel)
      return rewriter.notifyMatchFailure(op, "stride exceeds kernel");
    di.ydot = (di.kernel + di.stride - 1) / di.stride;
    di.inLen = inputType.getDimSize(2 + d);
    // The natural extent covers the scatter itself. `output_padding` can ask
    // for more, and when the stride divides the kernel there is no slack to
    // absorb it, so grow the grid to fit and let the extra positions be
    // written as zero (see the note above the pattern).
    int64_t natural = di.inLen + di.ydot - 1;
    int64_t needed = (di.padLo + di.out + di.stride - 1) / di.stride;
    di.htilda = std::max(natural, needed);
    di.full = di.htilda * di.stride;
    // ONNX requires output_padding < stride, which bounds the growth above at
    // one position per axis. Anything beyond that means the result shape does
    // not agree with the attributes, so decline rather than size off it.
    if (di.htilda > natural + 1)
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

  const APFloat zero =
      APFloat::getZero(cast<FloatType>(elementType).getFloatSemantics());

  struct Residue {
    Value partial;
    int64_t it0;
    int64_t it1;
    int64_t taps0;
    int64_t taps1;
    int64_t padBegin0;
    int64_t padBegin1;
    int64_t padEnd0;
    int64_t padEnd1;
  };
  SmallVector<Residue> residues;
  residues.reserve(dims[0].stride * dims[1].stride);

  // Taps kept by residue `it` along dim `d`.
  auto residueTaps = [&](int64_t d, int64_t it) {
    return (dims[d].kernel - it + dims[d].stride - 1) / dims[d].stride;
  };
  // A residue with fewer than `ydot` taps can be widened to `ydot` with zero
  // taps in front: its leading pad grows by the same amount, so every output
  // reads the same inputs, and all residues then share one wide convolution.
  // The zero taps cost multiply-adds, so pad only while they at most double
  // the real ones (a 3x3 filter at stride 2 needs 16 taps for 9).
  constexpr int64_t kMaxPaddedTapRatio = 2;
  int64_t realTaps = 0;
  for (int64_t it0 : llvm::seq<int64_t>(dims[0].stride))
    for (int64_t it1 : llvm::seq<int64_t>(dims[1].stride))
      realTaps += residueTaps(0, it0) * residueTaps(1, it1);
  const bool padResidues =
      dims[0].stride * dims[1].stride * dims[0].ydot * dims[1].ydot <=
      kMaxPaddedTapRatio * realTaps;

  for (int64_t it0 : llvm::seq<int64_t>(dims[0].stride)) {
    for (int64_t it1 : llvm::seq<int64_t>(dims[1].stride)) {
      const int64_t taps0 = padResidues ? dims[0].ydot : residueTaps(0, it0);
      const int64_t taps1 = padResidues ? dims[1].ydot : residueTaps(1, it1);
      const int64_t padBegin0 = taps0 - 1;
      const int64_t padBegin1 = taps1 - 1;
      const int64_t padEnd0 = dims[0].trailingPad();
      const int64_t padEnd1 = dims[1].trailingPad();
      residues.push_back({Value(), it0, it1, taps0, taps1, padBegin0, padBegin1,
                          padEnd0, padEnd1});
    }
  }

  // Writes residue `r`'s filter, [M, C, taps0, taps1] row-major, into
  // `values`: taps {it, it + stride, ...} reversed, input/output channels
  // swapped relative to the ConvTranspose weight. Positions of a padded
  // residue that fall past the filter keep the zero `values` starts with.
  auto fillSubWeight = [&](const Residue &r, MutableArrayRef<APFloat> values) {
    SmallVector<int64_t, 4> subShape{outChannels, inChannels, r.taps0, r.taps1};
    for (int64_t m : llvm::seq<int64_t>(outChannels))
      for (int64_t c : llvm::seq<int64_t>(inChannels))
        for (int64_t a : llvm::seq<int64_t>(r.taps0))
          for (int64_t b : llvm::seq<int64_t>(r.taps1)) {
            // Reversed tap order: position a holds tap (taps0 - 1 - a).
            const int64_t y0 = r.it0 + (r.taps0 - 1 - a) * dims[0].stride;
            const int64_t y1 = r.it1 + (r.taps1 - 1 - b) * dims[1].stride;
            if (y0 >= dims[0].kernel || y1 >= dims[1].kernel)
              continue;
            values[flatIndex(subShape, m, c, a, b)] =
                weightValues[flatIndex(weightShape, c, m, y0, y1)];
          }
  };

  // A stride-1 convolution with `r`'s taps and pads producing `channels`
  // output channels on the [Ht0, Ht1] residue grid.
  auto emitResidueConv = [&](const Residue &r, int64_t channels,
                             ArrayRef<APFloat> values,
                             Value convBias) -> Value {
    auto weightTy = RankedTensorType::get(
        {channels, inChannels, r.taps0, r.taps1}, elementType);
    Value weight = ConstantOp::create(rewriter, loc, weightTy,
                                      DenseElementsAttr::get(weightTy, values))
                       .getResult();
    auto outTy =
        resultType.clone({numBatch, channels, dims[0].htilda, dims[1].htilda});
    Value init =
        tensor::EmptyOp::create(rewriter, loc, outTy.getShape(), elementType)
            .getResult();
    SmallVector<Value> operands{ctx, op.getInput(), weight};
    if (convBias)
      operands.push_back(convBias);
    operands.push_back(init);
    return ConvOp::create(
               rewriter, loc, TypeRange{outTy}, operands,
               ArrayRef<NamedAttribute>{
                   rewriter.getNamedAttr(
                       "kernel_shape",
                       rewriter.getI64ArrayAttr({r.taps0, r.taps1})),
                   rewriter.getNamedAttr("strides",
                                         rewriter.getI64ArrayAttr({1, 1})),
                   rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(
                                                     {r.padBegin0, r.padBegin1,
                                                      r.padEnd0, r.padEnd1})),
                   rewriter.getNamedAttr("dilations",
                                         rewriter.getI64ArrayAttr({1, 1})),
                   rewriter.getNamedAttr("group",
                                         rewriter.getI64IntegerAttr(1))})
        .getResult(0);
  };

  const Residue &first = residues.front();
  const bool equalResidues =
      residues.size() > 1 &&
      llvm::all_of(residues, [&](const Residue &residue) {
        return residue.taps0 == first.taps0 && residue.taps1 == first.taps1 &&
               residue.padBegin0 == first.padBegin0 &&
               residue.padBegin1 == first.padBegin1 &&
               residue.padEnd0 == first.padEnd0 &&
               residue.padEnd1 == first.padEnd1;
      });

  // The residues tile the grid, so every output element, the zero slack
  // included, comes from exactly one residue convolution, and the bias can
  // be that convolution's bias. There it stays in the kernel epilogue
  // instead of costing an add after the crop.
  bool biasFolded = false;

  Value scattered;
  if (equalResidues) {
    // One convolution for all residues. Its output channel is
    // (it0 * strideW + it1) * M + m, so viewing it as [N, S0, S1, M, Ht0, Ht1]
    // and moving each stride next to its length is a DepthToSpace (DCR) --
    // one transpose, no concat.
    const int64_t ht0 = dims[0].htilda;
    const int64_t ht1 = dims[1].htilda;
    const int64_t stride0 = dims[0].stride;
    const int64_t stride1 = dims[1].stride;
    const int64_t numResidues = static_cast<int64_t>(residues.size());
    const int64_t perResidue =
        outChannels * inChannels * first.taps0 * first.taps1;
    SmallVector<APFloat> wideValues(numResidues * perResidue, zero);
    for (int64_t i : llvm::seq<int64_t>(numResidues))
      fillSubWeight(residues[i], MutableArrayRef<APFloat>(wideValues)
                                     .slice(i * perResidue, perResidue));
    // The wide bias repeats the bias once per residue, so it must be readable
    // here; a bias computed at run time stays an add after the crop.
    Value wideBias;
    if (bias)
      if (auto biasConst = bias.getDefiningOp<ConstantOp>()) {
        auto biasType = cast<RankedTensorType>(bias.getType());
        FailureOr<DenseElementsAttr> biasData =
            readConstantData(biasConst, biasType);
        if (succeeded(biasData)) {
          SmallVector<APFloat> tiled;
          tiled.reserve(numResidues * outChannels);
          for (int64_t i = 0; i < numResidues; ++i)
            tiled.append(biasData->getValues<APFloat>().begin(),
                         biasData->getValues<APFloat>().end());
          auto wideBiasType =
              RankedTensorType::get({numResidues * outChannels}, elementType);
          wideBias =
              ConstantOp::create(rewriter, loc, wideBiasType,
                                 DenseElementsAttr::get(wideBiasType, tiled))
                  .getResult();
          biasFolded = true;
        }
      }
    Value wide =
        emitResidueConv(first, numResidues * outChannels, wideValues, wideBias);

    auto splitType = RankedTensorType::get(
        {numBatch, stride0, stride1, outChannels, ht0, ht1}, elementType);
    SmallVector<ReassociationIndices> splitReassoc{{0}, {1, 2, 3}, {4}, {5}};
    Value split = tensor::ExpandShapeOp::create(rewriter, loc, splitType, wide,
                                                splitReassoc)
                      .getResult();
    auto transposedType = RankedTensorType::get(
        {numBatch, outChannels, ht0, stride0, ht1, stride1}, elementType);
    Value transposeInit =
        tensor::EmptyOp::create(rewriter, loc, transposedType.getShape(),
                                elementType)
            .getResult();
    // [N, S0, S1, M, Ht0, Ht1] -> [N, M, Ht0, S0, Ht1, S1].
    Value transposed =
        TransposeOp::create(rewriter, loc, ctx, split, transposeInit,
                            rewriter.getI64ArrayAttr({0, 3, 4, 1, 5, 2}))
            .getResult(0);
    auto gridType =
        resultType.clone({numBatch, outChannels, dims[0].full, dims[1].full});
    SmallVector<ReassociationIndices> collapseReassoc{{0}, {1}, {2, 3}, {4, 5}};
    scattered = tensor::CollapseShapeOp::create(rewriter, loc, gridType,
                                                transposed, collapseReassoc)
                    .getResult();
  } else {
    auto fullType =
        resultType.clone({numBatch, outChannels, dims[0].full, dims[1].full});
    scattered =
        tensor::EmptyOp::create(rewriter, loc, fullType.getShape(), elementType)
            .getResult();
    for (Residue &residue : residues) {
      SmallVector<APFloat> subValues(
          outChannels * inChannels * residue.taps0 * residue.taps1, zero);
      fillSubWeight(residue, subValues);
      residue.partial = emitResidueConv(residue, outChannels, subValues, bias);
      // Residue itilda lands on output positions {itilda + k * stride}.
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(residue.it0),
                                        rewriter.getIndexAttr(residue.it1)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(numBatch),
                                      rewriter.getIndexAttr(outChannels),
                                      rewriter.getIndexAttr(dims[0].htilda),
                                      rewriter.getIndexAttr(dims[1].htilda)};
      SmallVector<OpFoldResult> insertStrides{
          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(dims[0].stride),
          rewriter.getIndexAttr(dims[1].stride)};
      scattered = tensor::InsertSliceOp::create(rewriter, loc, residue.partial,
                                                scattered, offsets, sizes,
                                                insertStrides)
                      .getResult();
    }
    biasFolded = true;
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

  if (bias && !biasFolded) {
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
