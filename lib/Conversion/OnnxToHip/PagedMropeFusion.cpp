/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "OnnxToHipUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include <cmath>
#include <optional>

namespace mlir::hip {
namespace {
int64_t integer(Operation *op, StringRef name, int64_t fallback = 0) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  return attr ? attr.getInt() : fallback;
}
bool custom(Operation *op, StringRef name) {
  if (!op || op->getName().getStringRef() != "onnx.Custom")
    return false;
  auto function = op->getAttrOfType<StringAttr>("function_name");
  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  return function && function.getValue() == name && domain &&
         domain.getValue() == "com.microsoft";
}
bool view(Operation *op) {
  if (!op || op->getNumResults() != 1 || op->getNumOperands() == 0)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "onnx.Squeeze" || name == "onnx.Unsqueeze" ||
         name == "onnx.Reshape";
}
Value sourceOfViews(Value value) {
  while (view(value.getDefiningOp()))
    value = value.getDefiningOp()->getOperand(0);
  return value;
}
struct RotaryChain {
  Operation *cast, *rotary;
  Value floatInput;
  SmallVector<int64_t> sections;
  int64_t dimension, layout;
  float scale;
};
std::optional<RotaryChain> matchRotary(Value value, int64_t heads,
                                       int64_t dimension) {
  Operation *cast = onnxProducer(value, "onnx.Cast");
  auto result = dyn_cast<RankedTensorType>(value.getType());
  if (!cast || !value.hasOneUse() || integer(cast, "to") != 10 || !result ||
      result.getRank() != 2 || !result.getElementType().isF16())
    return std::nullopt;
  Value input = cast->getOperand(0), cursor = input;
  auto flat = dyn_cast<RankedTensorType>(input.getType());
  if (!flat || flat.getRank() != 2 || !flat.getElementType().isF32() ||
      flat.getShape() != result.getShape() ||
      flat.getDimSize(1) != heads * dimension)
    return std::nullopt;
  while (view(cursor.getDefiningOp())) {
    if (!cursor.hasOneUse())
      return std::nullopt;
    cursor = cursor.getDefiningOp()->getOperand(0);
  }
  Operation *rotary = cursor.getDefiningOp();
  if (!custom(rotary, "MRotaryEmbedding") || !cursor.hasOneUse() ||
      rotary->getNumOperands() != 4 || rotary->getNumResults() != 1 ||
      integer(rotary, "interleaved") != 0 ||
      integer(rotary, "is_packed_batching") != 1 ||
      integer(rotary, "num_heads") != heads)
    return std::nullopt;
  auto tensor = dyn_cast<RankedTensorType>(rotary->getOperand(0).getType());
  auto positions = dyn_cast<RankedTensorType>(rotary->getOperand(1).getType());
  auto cos = dyn_cast<RankedTensorType>(rotary->getOperand(2).getType());
  if (!tensor || tensor.getRank() != 3 || tensor.getDimSize(0) != 1 ||
      !tensor.getElementType().isF32() ||
      tensor.getDimSize(2) != heads * dimension || tensor != cursor.getType() ||
      !positions || positions.getRank() != 3 || positions.getDimSize(0) != 3 ||
      positions.getDimSize(1) != 1 ||
      !positions.getElementType().isInteger(64) || !cos || cos.getRank() != 2 ||
      !cos.getElementType().isF32() || cos != rotary->getOperand(3).getType())
    return std::nullopt;
  SmallVector<int64_t> sections;
  if (auto attr = rotary->getAttrOfType<ArrayAttr>("mrope_section")) {
    for (Attribute a : attr) {
      auto n = dyn_cast<IntegerAttr>(a);
      if (!n)
        return std::nullopt;
      sections.push_back(n.getInt());
    }
  } else if (auto attr =
                 rotary->getAttrOfType<DenseI64ArrayAttr>("mrope_section")) {
    sections.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  }
  const int64_t r = integer(rotary, "rotary_embedding_dim");
  const int64_t layout = integer(rotary, "mrope_layout");
  if (r <= 0 || (r & 1) || r > dimension || cos.getDimSize(1) != r / 2 ||
      sections.size() != 3 ||
      llvm::any_of(sections, [&](int64_t x) { return x < 0 || x > r / 2; }) ||
      sections[0] + sections[1] + sections[2] != r / 2 ||
      (layout != 0 && layout != 1))
    return std::nullopt;
  auto scale = rotary->getAttrOfType<FloatAttr>("scale");
  const float s = scale ? scale.getValueAsDouble() : 1.0f;
  if (!std::isfinite(s))
    return std::nullopt;
  return RotaryChain{cast, rotary, input, sections, r, layout, s};
}

struct PagedMropeFusion : RewritePattern {
  PagedMropeFusion(MLIRContext *ctx) : RewritePattern("onnx.Custom", 2, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!custom(op, "PagedAttention") || op->getNumOperands() < 8 ||
        op->getNumOperands() > 17 || integer(op, "do_rotary") != 0 ||
        integer(op, "rotary_interleaved") != 0)
      return failure();
    auto present = [&](unsigned i) {
      return i < op->getNumOperands() &&
             !isa<NoneType>(op->getOperand(i).getType());
    };
    if (present(8) || present(9))
      return failure();
    auto pool = dyn_cast<RankedTensorType>(op->getOperand(3).getType());
    if (!pool || pool.getRank() != 4 || pool.isDynamicDim(3))
      return failure();
    const int64_t h = integer(op, "num_heads"), g = integer(op, "kv_num_heads"),
                  d = pool.getDimSize(3);
    if (h <= 0 || g <= 0 || d <= 0 || h > INT32_MAX || g > INT32_MAX ||
        d > INT32_MAX)
      return failure();
    auto q = matchRotary(op->getOperand(0), h, d);
    auto k = matchRotary(op->getOperand(1), g, d);
    if (!q || !k || q->rotary == k->rotary || q->dimension != k->dimension ||
        q->sections != k->sections || q->layout != k->layout ||
        q->scale != k->scale ||
        q->rotary->getOperand(2) != k->rotary->getOperand(2) ||
        q->rotary->getOperand(3) != k->rotary->getOperand(3) ||
        sourceOfViews(q->rotary->getOperand(1)) !=
            sourceOfViews(k->rotary->getOperand(1)))
      return failure();
    Value positions = sourceOfViews(q->rotary->getOperand(1));
    auto positionType = dyn_cast<RankedTensorType>(positions.getType());
    if (!positionType || positionType.getRank() != 2 ||
        positionType.getDimSize(0) != 3)
      positions = q->rotary->getOperand(1);
    Value none;
    for (Value v : op->getOperands())
      if (isa<NoneType>(v.getType())) {
        none = v;
        break;
      }
    if (!none) {
      OperationState absent(op->getLoc(), "onnx.NoValue");
      absent.addTypes(rewriter.getNoneType());
      none = rewriter.create(absent)->getResult(0);
    }
    SmallVector<Value> operands = llvm::to_vector(op->getOperands());
    operands.resize(17, none);
    operands[0] = q->floatInput;
    operands[1] = k->floatInput;
    operands[8] = q->rotary->getOperand(2);
    operands[9] = q->rotary->getOperand(3);
    operands.push_back(positions);
    OperationState state(op->getLoc(), "onnx.Custom");
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    state.attributes.set("function_name",
                         rewriter.getStringAttr("PagedAttentionMRoPE"));
    state.attributes.set("domain_name", rewriter.getStringAttr("hip.internal"));
    state.attributes.set("rotary_dim",
                         rewriter.getI64IntegerAttr(q->dimension));
    state.attributes.set("mrope_section",
                         rewriter.getI64ArrayAttr(q->sections));
    state.attributes.set("mrope_layout", rewriter.getI64IntegerAttr(q->layout));
    state.attributes.set("mrope_scale", rewriter.getF32FloatAttr(q->scale));
    auto fused = rewriter.create(state);
    rewriter.replaceOp(op, fused->getResults());
    rewriter.eraseOp(q->cast);
    rewriter.eraseOp(k->cast);
    rewriter.replaceOp(q->rotary, ValueRange{q->rotary->getOperand(0)});
    rewriter.replaceOp(k->rotary, ValueRange{k->rotary->getOperand(0)});
    return success();
  }
};
} // namespace
void populatePagedMropeFusionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<PagedMropeFusion>(ctx);
}
} // namespace mlir::hip
