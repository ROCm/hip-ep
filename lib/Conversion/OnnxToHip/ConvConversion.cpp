/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Conv -> hip.conv. Rank-4 input lowers directly to a 2D conv. Rank-3
/// (1D) input is reshaped to rank-4 with a unit H dimension (NCL -> NC1L) via
/// tensor.expand_shape, run through the same hip.conv, then collapsed back to
/// NCL via tensor.collapse_shape. Both expand/collapse lower to zero-cost
/// metadata ops (no data movement), so 1D conv reuses the 2D MIOpen path
/// instead of a dedicated op/kernel.
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

  // Only rank-3 (1D conv) and rank-4 (2D conv) are supported. Rank-5 (3D conv)
  // has no runtime path today — leave it to whatever other pattern (if any)
  // claims it.
  const int64_t inputRank = inputType.getRank();
  if (inputRank != 3 && inputRank != 4)
    return rewriter.notifyMatchFailure(
        op, "ConvToHip only supports rank-3 (1D) and rank-4 (2D) Conv");
  const bool is1D = (inputRank == 3);
  const int64_t spatialDims = inputRank - 2; // 1 for NCL, 2 for NCHW

  // Extract attributes from onnx.Conv
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  }

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

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();

  // The rank-3 (1D) case is handled by reshaping to a rank-4 (2D) conv with a
  // unit H dimension and collapsing the result back. `conv2dResultType` is the
  // type fed to hip.conv; for 1D it is the NC1L' rank-4 type, for 2D it is the
  // original result type. For 1D, `is1D` drives the destination reshape below.
  mlir::RankedTensorType conv2dResultType = resultType;

  // NCL <-> NC1L reassociation: identity on N and C, split/merge the trailing
  // spatial dim against a unit H. Shared by the input/weights expand and the
  // init/result reshape below.
  llvm::SmallVector<mlir::ReassociationIndices> reassoc1d = {{0}, {1}, {2, 3}};

  if (is1D) {
    // The shared 2D MIOpen path treats NCL as NC[H=1]L. It does not honor
    // dilation != 1 or group != 1 in this H=1 reinterpretation; bail rather
    // than silently miscompile.
    if (!dilations.empty() && dilations[0] != 1)
      return rewriter.notifyMatchFailure(
          op, "1D Conv with dilation != 1 is not supported");
    if (group != 1)
      return rewriter.notifyMatchFailure(
          op, "1D Conv with group != 1 is not supported");
    if (inputType.getRank() != 3 || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "1D Conv requires a static rank-3 input shape");

    auto weightsType = mlir::cast<mlir::RankedTensorType>(weights.getType());

    auto expandTo = [&](mlir::Value v,
                        mlir::RankedTensorType srcTy) -> mlir::Value {
      llvm::SmallVector<int64_t> shape4(srcTy.getShape().begin(),
                                        srcTy.getShape().end());
      shape4.insert(shape4.end() - 1, 1); // insert H=1 before the spatial dim
      auto ty4 = mlir::RankedTensorType::get(shape4, srcTy.getElementType());
      // All dims are static (guarded above), so the output_shape is a list of
      // index attrs.
      llvm::SmallVector<mlir::OpFoldResult> outShape;
      for (int64_t d : shape4)
        outShape.push_back(rewriter.getIndexAttr(d));
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
    //   dilations    [d] / {} -> [1, 1]
    kernelShape.insert(kernelShape.begin(), 1);
    strides.insert(strides.begin(), 1);
    int64_t padBegin = pads.empty() ? 0 : pads[0];
    int64_t padEnd = pads.size() > 1 ? pads[1] : padBegin;
    pads = {0, padBegin, 0, padEnd};
    dilations = {1, 1};
  }

  // Create the output (destination) tensor at the ORIGINAL result rank, then —
  // for 1D — expand it to the rank-4 NC1L' view used as the conv `outs`. The
  // conv result is later collapsed back to rank-3. Because
  // collapse_shape(expand_shape(init)) folds to `init`, the value feeding the
  // return aliases the destination buffer directly — bufferization write-
  // throughs it to the output parameter exactly like the rank-4 path, leaving
  // NO transient alloc (a lone transient would not be pooled and would lower
  // to the undefined hip_device_malloc).
  // Size the destination init's dynamic dims from the conv INPUT, not the conv
  // result. Sourcing from op->getResult(0) is self-referential: replaceOp later
  // remaps the DimOp's operand to the freshly-created hip.conv result, but the
  // DimOp stays positioned *before* the conv it now reads from — a
  // use-before-def / dominance error that aborts the pipeline. This only
  // manifests when an output dim is dynamic (e.g. a dynamic batch), so static-
  // shape convs never hit it. For a standard convolution the only dynamic
  // output dim is the batch N (dim 0), which equals the input's batch dim;
  // output channels are static (Cout from weights) and the spatial extents are
  // statically inferred here. A dynamic non-batch output dim cannot be sized
  // from the input alone, so bail (CPU fallback) rather than emit a wrong size.
  //
  // Before (buggy — %dst defined below its use):
  //   %n   = tensor.dim %dst, 0
  //   %ini = tensor.empty(%n)
  //   %dst = hip.conv ..., %ini
  // After:
  //   %n   = tensor.dim %input, 0
  //   %ini = tensor.empty(%n)
  //   %dst = hip.conv ..., %ini
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx != 0)
      return rewriter.notifyMatchFailure(
          op, "ConvToHip can only size a dynamic batch dim from the input; "
              "dynamic non-batch output dims are unsupported");
    dynSizes.push_back(
        mlir::tensor::DimOp::create(rewriter, loc, input, /*index=*/0));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  if (is1D) {
    llvm::SmallVector<mlir::OpFoldResult> outShape;
    for (int64_t d : conv2dResultType.getShape())
      outShape.push_back(rewriter.getIndexAttr(d));
    init = mlir::tensor::ExpandShapeOp::create(rewriter, loc, conv2dResultType,
                                               init, reassoc1d, outShape);
  }

  // Build operands vector: context, input, weights, [bias], init
  llvm::SmallVector<mlir::Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(init);

  // Build attributes (always 2D form by this point).
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
