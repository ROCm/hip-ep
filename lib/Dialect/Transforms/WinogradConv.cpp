/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- WinogradConv.cpp - fp16 3x3 convs as Winograd F(2x2, 3x3) ----------===//
//
// Rewrites an eligible fp16 3x3 conv, and the activation that follows it, into
// one hip.winograd_conv. The compiler later builds a kernel for each one.
//
// Before:
//   %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x64x72x84xf16>,
//                         tensor<64x64x3x3xf16>, tensor<64xf16>)
//                     outs(%e0 : tensor<1x64x72x84xf16>) {kernel_shape = [3,
//                     3], strides = [1, 1], pads = [1, 1, 1, 1], dilations =
//                     [1, 1], group = 1}
//   %y = hip.leaky_relu(%ctx) ins(%c : ...) outs(%e1 : ...) {alpha = 0.1}
// After:
//   %y = hip.winograd_conv(%ctx) ins(%x, %w, %b : ...) outs(%e1 : ...)
//                              {activation = "leaky_relu", alpha = 0.1}
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipConstantData.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>

#include <algorithm>

namespace mlir::hip {
#define GEN_PASS_DEF_WINOGRADCONVPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

bool isI64Array(ArrayAttr attr, ArrayRef<int64_t> want) {
  if (!attr || attr.size() != want.size())
    return false;
  for (auto [a, w] : llvm::zip_equal(attr, want)) {
    auto i = dyn_cast<IntegerAttr>(a);
    if (!i || i.getInt() != w)
      return false;
  }
  return true;
}

// Whether the Winograd kernel beats the rocMLIR conv for this NCHW fp16 shape.
// The rule is MIGraphX's NCHW fp16 heuristic (winograd_f23_profitable,
// src/targets/gpu/prefuse_ops.cpp), which the kernel was tuned with, minus the
// band where hip-ep's tuned rocMLIR convs are faster: large channel counts at
// small spatial sizes (both channel counts >= 511, side <= 32).
bool isProfitable(int64_t inC, int64_t outC, int64_t height, int64_t width) {
  const int64_t spatial = std::min(height, width);
  const int64_t minC = std::min(inC, outC), maxC = std::max(inC, outC);
  if (minC >= 511 && spatial <= 32)
    return false;
  if (inC < 16 && (maxC > 16 || inC < 8))
    return false;
  if (inC * outC >= 700000)
    return false;
  if (minC >= 224) {
    if (spatial <= 32)
      return true;
    return spatial >= 48 && outC >= 288 && outC <= 384;
  }
  if (minC >= 128 && spatial >= 128)
    return false;
  return true;
}

// A producer that the rocMLIR conv would read as a view (an upsample, a
// transpose, a reshape or a slice) costs a copy when the conv runs on its own
// kernel, which can outweigh the Winograd gain.
bool isAbsorbedView(Value input) {
  Operation *def = input.getDefiningOp();
  return def && isa<ResizeOp, TransposeOp, tensor::ExtractSliceOp,
                    tensor::ExpandShapeOp, tensor::CollapseShapeOp>(def);
}

bool isEligible(ConvOp conv) {
  auto input = dyn_cast<RankedTensorType>(conv.getInput().getType());
  auto weights = dyn_cast<RankedTensorType>(conv.getWeights().getType());
  auto output = dyn_cast<RankedTensorType>(conv.getOutput().getType());
  if (!input || !weights || !output || !input.hasStaticShape() ||
      !weights.hasStaticShape() || !output.hasStaticShape() ||
      input.getRank() != 4 || weights.getRank() != 4)
    return false;
  if (!input.getElementType().isF16() || !weights.getElementType().isF16() ||
      !output.getElementType().isF16())
    return false;
  if (!isI64Array(conv.getKernelShape(), {3, 3}) ||
      !isI64Array(conv.getStrides(), {1, 1}) ||
      !isI64Array(conv.getDilations(), {1, 1}) ||
      !isI64Array(conv.getPads(), {1, 1, 1, 1}) || conv.getGroup() != 1)
    return false;
  // The runtime builds the filter store once per session, so the filter must
  // be a model constant.
  if (!conv.getWeights().getDefiningOp<ConstantOp>())
    return false;
  if (Value bias = conv.getBias()) {
    auto b = dyn_cast<RankedTensorType>(bias.getType());
    if (!b || !b.getElementType().isF16())
      return false;
  }
  if (isAbsorbedView(conv.getInput()))
    return false;
  ArrayRef<int64_t> x = input.getShape(), w = weights.getShape();
  return isProfitable(x[1], w[0], x[2], x[3]);
}

// True for a constant whose every element is 0.
bool isZeroSplat(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!constant || !type || !type.hasStaticShape() ||
      type.getNumElements() > 64 || !isa<FloatType>(type.getElementType()))
    return false;
  FailureOr<DenseElementsAttr> data = readConstantData(constant, type);
  if (failed(data))
    return false;
  return llvm::all_of(data->getValues<APFloat>(),
                      [](const APFloat &v) { return v.isZero(); });
}

struct Epilogue {
  Operation *op = nullptr;
  StringRef activation = "none";
  double alpha = 0.0;
  Value output;
};

// The activation that consumes the conv result alone and keeps its type.
Epilogue findEpilogue(ConvOp conv) {
  Epilogue e;
  Value result = conv->getResult(0);
  if (!result.hasOneUse())
    return e;
  Operation *user = *result.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getResult(0).getType() != result.getType())
    return e;
  if (auto leaky = dyn_cast<LeakyReluOp>(user)) {
    e = {user, "leaky_relu", leaky.getAlpha().convertToDouble(),
         leaky.getOutput()};
  } else if (auto max = dyn_cast<MaxOp>(user)) {
    Value other = max.getLhs() == result ? max.getRhs() : max.getLhs();
    if (isZeroSplat(other))
      e = {user, "relu", 0.0, max.getOutput()};
  }
  return e;
}

struct WinogradConvPass : public impl::WinogradConvPassBase<WinogradConvPass> {
  void runOnOperation() override {
    SmallVector<ConvOp> convs;
    getOperation().walk([&](ConvOp op) {
      if (isEligible(op))
        convs.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (ConvOp conv : convs) {
      Epilogue e = findEpilogue(conv);
      Operation *replaced = e.op ? e.op : conv.getOperation();
      Value output = e.op ? e.output : conv.getOutput();
      rewriter.setInsertionPoint(replaced);
      auto wg = WinogradConvOp::create(
          rewriter, conv.getLoc(), replaced->getResultTypes(), conv.getCtx(),
          conv.getInput(), conv.getWeights(), conv.getBias(), output,
          rewriter.getStringAttr(e.activation),
          rewriter.getF64FloatAttr(e.alpha), rewriter.getStringAttr(""),
          rewriter.getStringAttr(""), rewriter.getI64IntegerAttr(0),
          rewriter.getI64IntegerAttr(0), rewriter.getBoolAttr(false));
      rewriter.replaceOp(replaced, wg->getResults());
      if (e.op)
        rewriter.eraseOp(conv);
    }
  }
};

} // namespace
} // namespace mlir::hip
