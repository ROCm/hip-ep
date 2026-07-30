/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>

namespace mlir {
namespace hip {
namespace {

/// onnx.Conv -> hip.conv. Rank-4 input lowers directly to a 2D conv. Rank-3
/// (1D) input is reshaped to rank-4 with a unit H dimension (NCL -> NC1L) via
/// tensor.expand_shape, run through the same hip.conv, then collapsed back to
/// NCL via tensor.collapse_shape. Both expand/collapse lower to zero-cost
/// metadata ops (no data movement), so 1D conv reuses the 2D MIOpen path
/// instead of a dedicated op/kernel. The `group` attribute is preserved
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
    const int64_t s = dimIdx - 2; // spatial axis index (0-based)
    if (s >= static_cast<int64_t>(kernelShape.size()))
      return rewriter.notifyMatchFailure(
          op, "dynamic spatial output dim requires an explicit kernel_shape");
    const int64_t k = kernelShape[s];
    const int64_t st =
        (s < static_cast<int64_t>(strides.size())) ? strides[s] : 1;
    const int64_t dil =
        (s < static_cast<int64_t>(dilations.size())) ? dilations[s] : 1;
    const int64_t pb = (s < static_cast<int64_t>(pads.size())) ? pads[s] : 0;
    const int64_t pe = (spatialDims + s < static_cast<int64_t>(pads.size()))
                           ? pads[spatialDims + s]
                           : 0;
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
    // The shared 2D MIOpen path treats NCL as NC[H=1]L. The H=1 reshape hard-
    // codes dilations to [1,1] below, so it cannot preserve a non-unit spatial
    // dilation — bail on dilation != 1. `group` IS preserved verbatim on the
    // 2D hip.conv (a depthwise [C,1,K] filter reshapes to [C,1,1,K] with
    // group=C), so grouped/depthwise 1D convs are supported.
    if (!dilations.empty() && dilations[0] != 1)
      return rewriter.notifyMatchFailure(
          op, "1D Conv with dilation != 1 is not supported");

    auto weightsType = mlir::cast<mlir::RankedTensorType>(weights.getType());

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

class ConvToGemm : public RewritePattern {
public:
  ConvToGemm(MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*patternBenefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult ConvToGemm::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  const auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
  const int64_t inputRank = inputType.getRank();

  // Only supports 2D
  if (inputRank != 4)
    return rewriter.notifyMatchFailure(op, "Only 2d supported");

  // Not handling dynamic shapes now
  if (llvm::any_of(llvm::seq(inputRank),
                   [&inputType](int i) { return inputType.isDynamicDim(i); }))
    return rewriter.notifyMatchFailure(op, "dynamic dim");

  const auto inputShape = inputType.getShape();
  int64_t N = inputShape[0];
  int64_t C = inputShape[1];
  int64_t H = inputShape[2];
  int64_t W = inputShape[3];

  // Only batch size 1 supported
  if (N != 1)
    return rewriter.notifyMatchFailure(op, "N != 1");

  const int64_t spatialDims = inputRank - 2;
  const auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
  }
  int64_t kh = kernelShape[0];
  int64_t kw = kernelShape[1];

  SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<ArrayAttr>("strides")) {
    for (auto a : attr)
      strides.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default strides = 1 for each spatial dimension
    strides.assign(spatialDims, 1);
  }
  int64_t stride_h = strides[0];
  int64_t stride_w = strides[1];

  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    for (auto a : attr)
      pads.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default pads = 0 (2 entries per spatial dim: begin + end)
    pads.assign(spatialDims * 2, 0);
  }
  int64_t pad_h0 = pads[0];
  int64_t pad_h1 = pads[1];
  int64_t pad_w0 = pads[2];
  int64_t pad_w1 = pads[3];

  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    for (auto a : attr)
      dilations.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default dilations = 1
    dilations.assign(spatialDims, 1);
  }
  int64_t dil_h = dilations[0];
  int64_t dil_w = dilations[1];
  if (dil_h != 1 || dil_w != 1)
    return rewriter.notifyMatchFailure(op, "dilation != 1");

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();
  if (group != 1)
    return rewriter.notifyMatchFailure(op, "group != 1");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  int64_t H_out = (H + pad_h0 + pad_h1 - kh) / stride_h + 1;
  int64_t W_out = (W + pad_w0 + pad_w1 - kw) / stride_w + 1;
  auto im2colResultType = inputType.clone({C * kh * kw, H_out * W_out});

  auto wgtType = cast<RankedTensorType>(op->getOperand(1).getType());

  auto loc = op->getLoc();
  Value im2colInit =
      tensor::EmptyOp::create(rewriter, loc, im2colResultType, {});
  auto im2colOp = hip::Im2d2ColOp::create(
      rewriter, loc, im2colResultType, context, op->getOperand(0), im2colInit,
      rewriter.getI64ArrayAttr(kernelShape), rewriter.getI64ArrayAttr(strides),
      rewriter.getI64ArrayAttr(pads), rewriter.getI64ArrayAttr(dilations),
      rewriter.getI64ArrayAttr({H_out, W_out}));

  auto OC = wgtType.getShape()[0];
  auto newWgtType = wgtType.clone({OC, C * kh * kw});
  Value newWgtShape = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIndexTensorAttr(newWgtType.getShape()));
  Value newWgt = tensor::ReshapeOp::create(rewriter, loc, newWgtType,
                                           op->getOperand(1), newWgtShape);

  auto gemmOutType = inputType.clone({OC, H_out * W_out});
  auto gemmInit = tensor::EmptyOp::create(rewriter, loc, gemmOutType, {});
  SmallVector<Value> operands = {context, newWgt, im2colOp.getResult(0)};
  if (op->getNumOperands() > 2) {
    auto bias = op->getOperand(2);
    auto biasType = cast<RankedTensorType>(bias.getType());
    if (biasType.getRank() == 1) {
      auto biasDim = biasType.getDimSize(0);
      Value biasShape = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexTensorAttr({biasDim, 1}));
      bias = tensor::ReshapeOp::create(
          rewriter, loc, biasType.clone({biasDim, 1}), bias, biasShape);
    }
    operands.push_back(bias);
  }
  operands.push_back(gemmInit);

  auto hipGemm = hip::GemmOp::create(rewriter, loc, gemmOutType, operands);

  Value finalShape = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIndexTensorAttr(resultType.getShape()));
  auto finalReshape = tensor::ReshapeOp::create(
      rewriter, loc, resultType, hipGemm.getResult(0), finalShape);
  rewriter.replaceOp(op, finalReshape);

  return success();
}

} // namespace

void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ConvToGemm>(ctx);
}

} // namespace hip
} // namespace mlir
