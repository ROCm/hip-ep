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

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <memory>
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
/// Reassembly is a strided `tensor.insert_slice` per residue, which
/// bufferizes to a `memref.subview` + `memref.copy`. When every residue is
/// non-empty (stride <= kernel) the residues tile the grid exactly, so the
/// destination needs no zero-fill.
///
/// Taking a 1-D slice of the 2x2-stride case, with a 3-tap filter [w0 w1 w2]
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

/// Read a filter carrier's payload, whatever source it holds.
///
/// "Compile-time constant" is not the same as "inline `value`": the importer
/// externalizes initializers past a small byte threshold, so an ordinary
/// ConvTranspose filter reaches this pass as a file-backed byte range rather
/// than a `DenseElementsAttr`. Requiring an inline value would therefore
/// decline nearly every real filter, and a ConvTranspose-only module would
/// then reach the compiler with no rocMLIR kernel in it at all.
///
/// Memory-address carriers are deliberately not resolved. That address is
/// process-local and valid only while the producer that recorded it is live,
/// which this pattern cannot establish -- it is equally reachable from textual
/// pass invocation, where the address would be dereferenced blind.
FailureOr<DenseElementsAttr> readConstant(ConstantOp constant,
                                          RankedTensorType type) {
  switch (constant.getSourceKind()) {
  case ConstantOp::SourceKind::Inline: {
    // The verifier ties an inline value to the carrier's result type, but a
    // failed match has to be cheaper than an assertion if it ever does not.
    auto value = dyn_cast<DenseElementsAttr>(constant.getValueAttr());
    if (!value || value.getType() != type)
      return failure();
    return value;
  }
  case ConstantOp::SourceKind::Memory:
    return failure();
  case ConstantOp::SourceKind::File:
    break;
  }

  StringRef path = constant.getLocationAttr().getValue();
  const int64_t offset = constant.getOffsetAttr().getInt();
  const int64_t size = constant.getSizeAttr().getInt();
  const int64_t elementBytes = (type.getElementTypeBitWidth() + 7) / 8;
  if (size != type.getNumElements() * elementBytes)
    return failure();
  // The range is validated against the file before mapping: a slice running
  // past the end can otherwise be mapped and fault on access rather than
  // failing here.
  uint64_t fileSize = 0;
  if (llvm::sys::fs::file_size(path, fileSize) ||
      static_cast<uint64_t>(offset) + static_cast<uint64_t>(size) > fileSize)
    return failure();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileSlice(path, static_cast<uint64_t>(size),
                                       offset);
  if (!buffer || (*buffer)->getBufferSize() != static_cast<uint64_t>(size))
    return failure();
  DenseElementsAttr value = DenseElementsAttr::getFromRawBuffer(
      type,
      ArrayRef<char>((*buffer)->getBufferStart(), static_cast<size_t>(size)));
  if (!value)
    return failure();
  return value;
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
  // has to be readable at compile time.
  auto weightConst = op.getWeights().getDefiningOp<ConstantOp>();
  if (!weightConst)
    return rewriter.notifyMatchFailure(op, "weights are not a hip.constant");
  FailureOr<DenseElementsAttr> weightData =
      readConstant(weightConst, weightType);
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
                      "pads", rewriter.getI64ArrayAttr(
                                  {taps0 - 1, taps1 - 1, dims[0].trailingPad(),
                                   dims[1].trailingPad()})),
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
