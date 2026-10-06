/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Conv -> hip.conv. Rank-4 (2D, NCHW) and rank-5 (3D, NCDHW) inputs
/// lower directly; the custom kernel takes a spatial rank of 1, 2, or 3.
/// Rank-3 (1D) input is reshaped to rank-4 with a unit H dimension (NCL ->
/// NC1L) via tensor.expand_shape, run through the same hip.conv, then
/// collapsed back to NCL via tensor.collapse_shape. Both expand/collapse lower
/// to zero-cost metadata ops (no data movement), so 1D conv reuses the 2D
/// kernel instead of a dedicated op. The `group` attribute is preserved
/// through the 1D reshape (grouped/depthwise 1D convs -> grouped/depthwise 2D
/// convs), and dynamic result dims (batch, channels, and spatial extents) are
/// sized at runtime from the conv input + attributes.
struct ConvToHip : public mlir::RewritePattern {
  ConvToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ConvToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value weights = op->getOperand(1);

  // ONNX Conv always has 3 operands, but bias can be onnx.NoValue (NoneType)
  bool hasBias = op->getNumOperands() > 2 &&
                 !mlir::isa<mlir::NoneType>(op->getOperand(2).getType());
  mlir::Value bias = hasBias ? op->getOperand(2) : nullptr;

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

  // Rank 3 (1D), 4 (2D) and 5 (3D). A higher-benefit pattern (patch-embed
  // GEMM) still wins when it matches. Anything else, including rank 6+, stays
  // unconverted.
  const int64_t inputRank = inputType.getRank();
  if (inputRank < 3 || inputRank > 5)
    return rewriter.notifyMatchFailure(
        op, "ConvToHip only supports rank-3 (1D), rank-4 (2D) and rank-5 (3D) "
            "Conv");
  const bool is1D = (inputRank == 3);
  const int64_t spatialDims =
      inputRank - 2; // 1 for NCL, 2 for NCHW, 3 for NCDHW

  // hip.conv requires input, weights and result to share a rank. The 1D path
  // reshapes all three against that shared rank, and the dynamic-extent sizing
  // below indexes the per-spatial-axis attributes by (result dim - 2), so a
  // result outranking the input would read past them.
  //
  // This has to be settled before any IR exists: the sizing loop starts
  // emitting tensor.dim for dims 0 and 1, so a refusal after it leaves the
  // greedy driver holding a half-applied pattern, which fails the whole
  // pipeline with no diagnostic instead of leaving the op alone.
  auto weightsType = mlir::dyn_cast<mlir::RankedTensorType>(weights.getType());
  if (!weightsType || weightsType.getRank() != inputRank ||
      resultType.getRank() != inputRank)
    return rewriter.notifyMatchFailure(
        op, "conv input, weights, and result ranks must match");

  // Extract attributes from onnx.Conv
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // kernel_shape is optional in ONNX; when absent, the weight layout
    // [Cout, Cin/group, k1..kN] defines it, and the rank check above already
    // guarantees those trailing spatialDims dims exist. Exporters that leave
    // the spatial dims dynamic commonly omit the attribute. Only a statically
    // shaped W is usable, because those extents get folded into compile-time
    // constants in the sizing below; a dynamic kernel dim leaves the vector
    // empty for the arity check to refuse.
    for (int64_t i : llvm::seq<int64_t>(spatialDims)) {
      if (weightsType.isDynamicDim(2 + i)) {
        kernelShape.clear();
        break;
      }
      kernelShape.push_back(weightsType.getDimSize(2 + i));
    }
  }

  // hip.conv needs one kernel extent per spatial dim, and ConvLowering reports
  // a disagreement with emitError rather than a match failure, so letting a
  // short kernelShape through here kills the compile later instead of leaving
  // the op for someone else. That covers an omitted kernel_shape the weights
  // could not supply as well as an explicit one of the wrong arity.
  //
  // It has to be checked before any IR exists: the dynamic-extent sizing below
  // starts emitting tensor.dim for dims 0 and 1, and a bail-out after that
  // leaves the greedy driver holding a half-applied pattern, which fails the
  // whole pipeline with no diagnostic at all. The sizing also indexes
  // kernelShape per spatial axis, and the 1D path prepends a unit extent to
  // it; both already assume this holds.
  if (static_cast<int64_t>(kernelShape.size()) != spatialDims)
    return rewriter.notifyMatchFailure(
        op, "kernel_shape is neither given nor inferable from the weights, or "
            "its arity does not match the spatial rank");

  llvm::SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    for (auto a : attr)
      strides.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default strides = 1 for each spatial dimension
    strides.assign(spatialDims, 1);
  }

  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    for (auto a : attr)
      pads.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default pads = 0 (2 entries per spatial dim: begin + end)
    pads.assign(spatialDims * 2, 0);
  }

  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    for (auto a : attr)
      dilations.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default dilations = 1
    dilations.assign(spatialDims, 1);
  }

  // Same contract as kernel_shape above: ConvLowering wants one stride and one
  // dilation per spatial dim and reports a disagreement with emitError, so a
  // wrong-arity attribute has to be refused here rather than built into a
  // malformed hip.conv that kills the compile at lowering. Defaulting the
  // missing entries instead would convolve with attributes the model never
  // asked for. Checked before any IR exists, for the same reason as the
  // kernel_shape check.
  if (static_cast<int64_t>(strides.size()) != spatialDims ||
      static_cast<int64_t>(dilations.size()) != spatialDims)
    return rewriter.notifyMatchFailure(
        op, "conv strides/dilations arity does not match the spatial rank");

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();

  // Resolve auto_pad into explicit pads. Only NOTSET keeps the `pads` read
  // above; every other mode overrides them, and hip.conv carries the explicit
  // form only, so leaving a mode unresolved would quietly convolve with the
  // wrong padding and return wrong results instead of failing. Inferring
  // kernel_shape widens what reaches here, which is why it belongs in this
  // change. The SAME_UPPER / SAME_LOWER budget split -- halve pad_total and
  // give the odd pad to the end for SAME_UPPER, to the begin for SAME_LOWER --
  // follows PoolConversion, which follows onnx-mlir's customComputeShape.
  //
  // Like the checks above, this settles before any op is created: the sizing
  // loop below starts emitting tensor.dim, so refusing after it would leave
  // the greedy driver holding a half-applied pattern and fail the whole
  // pipeline with no diagnostic.
  llvm::StringRef autoPad = "NOTSET";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("auto_pad"))
    autoPad = attr.getValue();

  if (autoPad == "VALID") {
    pads.assign(spatialDims * 2, 0);
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    // Splitting the pad budget needs both the input and output extents, and
    // only a static pair folds into the compile-time constants below.
    for (int64_t i : llvm::seq<int64_t>(spatialDims)) {
      if (inputType.isDynamicDim(2 + i) || resultType.isDynamicDim(2 + i))
        return rewriter.notifyMatchFailure(
            op, "conv auto_pad=SAME_* requires static spatial dims");
    }
    pads.assign(spatialDims * 2, 0);
    for (int64_t i : llvm::seq<int64_t>(spatialDims)) {
      const int64_t st = strides[i];
      const int64_t dil = dilations[i];
      const int64_t effK = (kernelShape[i] - 1) * dil + 1;
      int64_t padTotal = (resultType.getDimSize(2 + i) - 1) * st + effK -
                         inputType.getDimSize(2 + i);
      if (padTotal < 0)
        padTotal = 0;
      const int64_t half = padTotal / 2;
      pads[i] = (autoPad == "SAME_UPPER") ? half : padTotal - half;
      pads[spatialDims + i] = padTotal - pads[i];
    }
  } else if (autoPad != "NOTSET") {
    return rewriter.notifyMatchFailure(op, "conv unknown auto_pad value");
  }

  // Checked after auto_pad rather than with the strides/dilations pair above:
  // VALID and SAME_* replace `pads` wholesale with a correctly sized vector,
  // so only NOTSET can carry an explicit attribute of the wrong arity this far.
  if (static_cast<int64_t>(pads.size()) != spatialDims * 2)
    return rewriter.notifyMatchFailure(
        op, "conv pads arity does not match the spatial rank");

  // The rank-3 (1D) case is handled by reshaping to a rank-4 (2D) conv with a
  // unit H dimension and collapsing the result back. `conv2dResultType` is the
  // type fed to hip.conv; for 1D it is the NC1L' rank-4 type, and for 2D/3D
  // it is the original result type. For 1D, `is1D` drives the destination
  // reshape below.
  mlir::RankedTensorType conv2dResultType = resultType;

  // NCL <-> NC1L reassociation: identity on N and C, split/merge the trailing
  // spatial dim against a unit H. Shared by the input/weights expand and the
  // init/result reshape below.
  llvm::SmallVector<mlir::ReassociationIndices> reassoc1d = {{0}, {1}, {2, 3}};

  // Resolve the runtime size of every dynamic result dim BEFORE the 1D reshape
  // below rewrites `input`/`weights`/attrs into their H=1 2D forms — this block
  // must see the ORIGINAL rank-N operands and attributes.
  //   - dim 0 (batch)      -> input's batch extent
  //   - dim 1 (out chans)  -> weights' dim 0 (Cout)
  //   - dim >= 2 (spatial) -> the conv output formula for that axis:
  //       out = (in + pad_begin + pad_end - dilation*(kernel-1) - 1)/stride + 1
  // resultDynSize[d] stays null for static dims (extent lives in resultType).
  //
  // Before (only a dynamic batch could be sized -> non-batch dynamic bailed):
  //   %n = tensor.dim %input, 0 ; tensor.empty(%n) : tensor<?x128x64xf16>
  // After (dynamic spatial dims too, e.g. a strided down-sampling conv):
  //   %h  = tensor.dim %input, 2
  //   %ho = arith ((%h + addend) floordiv stride + 1)
  //   tensor.empty(%n, %ho) : tensor<?x128x?x64xf16>
  //
  // The result rank matches the input's, and kernel_shape, strides, dilations
  // and pads all carry one entry per spatial axis (two for pads) by the time
  // the loop runs, so (dimIdx - 2) indexes every one of them in range. All of
  // those checks deliberately precede op creation.
  llvm::SmallVector<mlir::Value> resultDynSize(resultType.getRank(),
                                               mlir::Value());
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx == 0) {
      resultDynSize[dimIdx] =
          mlir::tensor::DimOp::create(rewriter, loc, input, /*index=*/0);
      continue;
    }
    if (dimIdx == 1) {
      // Output channels equal the weight tensor's first dim (Cout).
      resultDynSize[dimIdx] =
          mlir::tensor::DimOp::create(rewriter, loc, weights, /*index=*/0);
      continue;
    }
    const int64_t s = dimIdx - 2;     // spatial axis index (0-based)
    const int64_t k = kernelShape[s]; // guaranteed present by the check above
    const int64_t st = strides[s];
    const int64_t dil = dilations[s];
    const int64_t pb = pads[s];
    const int64_t pe = pads[spatialDims + s];
    // Everything except the (dynamic) input extent is a compile-time constant.
    const int64_t addend = pb + pe - dil * (k - 1) - 1;
    mlir::Value inExtent =
        mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx);
    mlir::Value addendC =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, addend);
    mlir::Value num =
        mlir::arith::AddIOp::create(rewriter, loc, inExtent, addendC);
    mlir::Value strideC =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, st);
    // Conv output extents are >= 0 for a valid convolution, so signed
    // division (divsi) matches ONNX's floor semantics on the non-negative
    // numerator here.
    mlir::Value divd =
        mlir::arith::DivSIOp::create(rewriter, loc, num, strideC);
    mlir::Value oneC = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    resultDynSize[dimIdx] =
        mlir::arith::AddIOp::create(rewriter, loc, divd, oneC);
  }

  if (is1D) {
    // The shared 2D path treats NCL as NC[H=1]L. Every 1D attribute maps onto
    // the W axis of that view, including the dilation — the unit H axis takes
    // the identity (k=1, stride 1, dilation 1, pad 0), so nothing about the 1D
    // problem is lost. `group` is preserved verbatim too (a depthwise [C,1,K]
    // filter reshapes to [C,1,1,K] with group=C), so grouped and depthwise 1D
    // convolutions ride the same path.
    //
    // Expand a rank-3 NCL operand to rank-4 NC1L (unit H before the spatial
    // dim). Dynamic source dims are carried into output_shape via tensor.dim so
    // a dynamic batch or spatial extent survives the reshape.
    auto expandTo = [&](mlir::Value v,
                        mlir::RankedTensorType srcTy) -> mlir::Value {
      llvm::SmallVector<int64_t> shape4(srcTy.getShape().begin(),
                                        srcTy.getShape().end());
      shape4.insert(shape4.end() - 1, 1); // insert H=1 before the spatial dim
      auto ty4 = mlir::RankedTensorType::get(shape4, srcTy.getElementType());
      // output_shape maps rank-4 positions back to the rank-3 source: 0,1 are
      // N,C; position 2 is the inserted unit H; position 3 is the spatial dim
      // (source index 2). Static dims use an index attr; dynamic dims a
      // tensor.dim of the source.
      llvm::SmallVector<mlir::OpFoldResult> outShape;
      for (int64_t i4 : llvm::seq<int64_t>(4)) {
        if (i4 == 2) {
          outShape.push_back(rewriter.getIndexAttr(1));
          continue;
        }
        const int64_t origIdx = (i4 < 2) ? i4 : 2;
        if (srcTy.isDynamicDim(origIdx))
          outShape.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, v, origIdx)
                  .getResult());
        else
          outShape.push_back(rewriter.getIndexAttr(srcTy.getDimSize(origIdx)));
      }
      return mlir::tensor::ExpandShapeOp::create(rewriter, loc, ty4, v,
                                                 reassoc1d, outShape);
    };

    input = expandTo(input, inputType);       // [N,Cin,Lin]  -> [N,Cin,1,Lin]
    weights = expandTo(weights, weightsType); // [Cout,Cin,K] -> [Cout,Cin,1,K]

    // Rank-4 result type [N, Cout, 1, Lout].
    llvm::SmallVector<int64_t> res4(resultType.getShape().begin(),
                                    resultType.getShape().end());
    res4.insert(res4.end() - 1, 1);
    conv2dResultType =
        mlir::RankedTensorType::get(res4, resultType.getElementType());

    // Promote the 1D attribute vectors to their 2D (H=1) equivalents.
    //   kernel_shape [K]      -> [1, K]
    //   strides      [s]      -> [1, s]
    //   pads         [b, e]   -> [0, b, 0, e]  (H top/bottom = 0)
    //   dilations    [d]      -> [1, d]
    // The arity checks above pin spatialDims == 1 here: one kernel extent, one
    // stride, one dilation, two pads.
    kernelShape.insert(kernelShape.begin(), 1);
    strides.insert(strides.begin(), 1);
    const int64_t padBegin = pads[0];
    const int64_t padEnd = pads[1];
    pads = {0, padBegin, 0, padEnd};
    dilations = {1, dilations[0]};
  }

  // Create the output (destination) tensor at the ORIGINAL result rank, then —
  // for 1D — expand it to the rank-4 NC1L' view used as the conv `outs`. The
  // conv result is later collapsed back to rank-3. Because
  // collapse_shape(expand_shape(init)) folds to `init`, the value feeding the
  // return aliases the destination buffer directly — bufferization write-
  // throughs it to the output parameter exactly like the rank-4 path, leaving
  // NO transient alloc (a lone transient would not be pooled and would lower
  // to the undefined hip_device_malloc).
  //
  // Dynamic dims are sized from `resultDynSize` (resolved above from the conv
  // INPUT + attributes, never from op->getResult(0)): sourcing an extent from
  // the op's own result is self-referential — replaceOp would remap the DimOp
  // onto the freshly-created hip.conv result while the DimOp stays positioned
  // before it, a use-before-def dominance error.
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank()))
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(resultDynSize[dimIdx]);

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  if (is1D) {
    // Expand the rank-3 init to the rank-4 NC1L' conv `outs`. Positions map
    // like expandTo above: 0,1 -> N,C; 2 -> unit H; 3 -> spatial (source idx
    // 2). Dynamic dims reuse the already-resolved resultDynSize values.
    llvm::SmallVector<mlir::OpFoldResult> outShape;
    for (int64_t i4 : llvm::seq<int64_t>(4)) {
      if (i4 == 2) {
        outShape.push_back(rewriter.getIndexAttr(1));
        continue;
      }
      const int64_t origIdx = (i4 < 2) ? i4 : 2;
      if (resultType.isDynamicDim(origIdx))
        outShape.push_back(resultDynSize[origIdx]);
      else
        outShape.push_back(
            rewriter.getIndexAttr(resultType.getDimSize(origIdx)));
    }
    init = mlir::tensor::ExpandShapeOp::create(rewriter, loc, conv2dResultType,
                                               init, reassoc1d, outShape);
  }

  // Build operands vector: context, input, weights, [bias], init
  llvm::SmallVector<mlir::Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(init);

  // Build attributes. The 1D rewrite has promoted them to the unit-H 2D form;
  // rank-4 and rank-5 keep the spatial rank of the original op.
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape",
                                        rewriter.getI64ArrayAttr(kernelShape)));
  attrs.push_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides)));
  attrs.push_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads)));
  attrs.push_back(
      rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr(dilations)));
  attrs.push_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::ConvOp::create(rewriter, loc, operands, attrs);

  if (is1D) {
    // Collapse the NC1L' conv result back to NCL'. Zero-cost metadata op; folds
    // against the init's expand_shape so the destination buffer is reused.
    auto collapsed = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, resultType, hipOp.getResult(0), reassoc1d);
    rewriter.replaceOp(op, collapsed.getResult());
    return mlir::success();
  }

  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
