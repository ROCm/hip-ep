/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "OnnxToHipUtils.h"

namespace mlir::hip {
namespace {
struct PagedAttentionToHip : RewritePattern {
  PagedAttentionToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto name = op->getAttrOfType<StringAttr>("function_name");
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    const bool mrope = name && domain &&
                       name.getValue() == "PagedAttentionMRoPE" &&
                       domain.getValue() == "hip.internal";
    if (!mrope && (!name || name.getValue() != "PagedAttention" || !domain ||
                   domain.getValue() != "com.microsoft"))
      return failure();
    // The ONNX schema owns the optional/default-zero attribute. It is not
    // part of the HIP op or runtime contract until soft-capping is supported.
    if (Attribute value = op->getAttr("softcap")) {
      auto cap = dyn_cast<FloatAttr>(value);
      if (!cap || cap.getValueAsDouble() != 0.0)
        return rewriter.notifyMatchFailure(op,
                                           "nonzero softcap is unsupported");
    }
    if (op->getNumOperands() < 8 ||
        op->getNumOperands() > (mrope ? 18u : 17u) ||
        (mrope && op->getNumOperands() != 18) || op->getNumResults() < 1 ||
        op->getNumResults() > 3)
      return rewriter.notifyMatchFailure(
          op, "PagedAttention expects 8-17 slots and 1-3 outputs");
    auto slot = [&](unsigned i) -> Value {
      if (i >= op->getNumOperands() ||
          isa<NoneType>(op->getOperand(i).getType()))
        return {};
      return op->getOperand(i);
    };
    for (unsigned i = 0; i < 8; ++i)
      if (!slot(i) || !isa<RankedTensorType>(slot(i).getType()))
        return rewriter.notifyMatchFailure(
            op, "separate Q/K/V, pools and request metadata are required");
    for (unsigned i = 10; i < 16; ++i)
      if (slot(i))
        return rewriter.notifyMatchFailure(
            op, "slot_mapping/head_sink/qk_norm/quantized-cache inputs are not "
                "implemented");
    if (!op->getAttrOfType<IntegerAttr>("num_heads") ||
        !op->getAttrOfType<IntegerAttr>("kv_num_heads"))
      return rewriter.notifyMatchFailure(op, "head attributes are required");
    auto intAttr = [&](StringRef key, int64_t fallback) {
      auto attr = op->getAttrOfType<IntegerAttr>(key);
      return attr ? attr.getInt() : fallback;
    };
    auto strAttr = [&](StringRef key, StringRef fallback) {
      auto attr = op->getAttrOfType<StringAttr>(key);
      return attr ? attr.getValue() : fallback;
    };
    auto defaultCacheType = [&](StringRef key) {
      StringRef value = strAttr(key, "");
      // The ORT importer represents an unspecified string with one NUL byte.
      return value.empty() || (value.size() == 1 && value.front() == '\0');
    };
    if (intAttr("is_causal", 1) != 1 || intAttr("v_head_size", 0) != 0 ||
        intAttr("rotary_offset", 0) != 0 ||
        strAttr("k_quant_type", "NONE") != "NONE" ||
        strAttr("v_quant_type", "NONE") != "NONE" ||
        !defaultCacheType("k_cache_dtype") ||
        !defaultCacheType("v_cache_dtype") ||
        strAttr("kv_cache_layout", "SEPARATE") != "SEPARATE")
      return rewriter.notifyMatchFailure(
          op,
          "noncausal/quantized/latent/offset Page modes are not implemented");
    if (auto epsilon = op->getAttrOfType<FloatAttr>("qk_norm_epsilon"))
      if (epsilon.getValueAsDouble() != static_cast<double>(1e-6f))
        return rewriter.notifyMatchFailure(
            op, "nondefault Q/K norm epsilon is not implemented");
    SmallVector<RankedTensorType> outputs;
    for (Value v : op->getResults()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t)
        return rewriter.notifyMatchFailure(op, "ranked outputs are required");
      outputs.push_back(t);
    }
    const unsigned resultCount = outputs.size();
    if (outputs.size() < 2)
      outputs.push_back(cast<RankedTensorType>(slot(3).getType()));
    if (outputs.size() < 3)
      outputs.push_back(cast<RankedTensorType>(slot(4).getType()));
    // Present pools are aliases, not tensor.empty allocations.
    if (outputs[1] != slot(3).getType() || outputs[2] != slot(4).getType())
      return rewriter.notifyMatchFailure(
          op, "present cache types must match input pools");
    auto ctx = getContextArg(op, rewriter);
    if (failed(ctx))
      return failure();
    OperationState state(op->getLoc(), mrope ? "hip.paged_mrope_attention"
                                             : "hip.paged_attention");
    state.addOperands(
        {*ctx, slot(0), slot(1), slot(2), slot(5), slot(6), slot(7)});
    for (unsigned i : {8u, 9u})
      if (slot(i))
        state.addOperands(slot(i));
    if (mrope) {
      if (!slot(8) || !slot(9) || !slot(17))
        return rewriter.notifyMatchFailure(
            op, "fused MRoPE requires tables and explicit positions");
      state.addOperands(slot(17));
    }
    if (slot(16))
      state.addOperands(slot(16));
    state.addOperands(
        {createEmptyTensor(rewriter, op->getLoc(), outputs[0], slot(0)),
         slot(3), slot(4)});
    state.addTypes(outputs);
    // ONNX integer attributes may be signed; Hip I64Attr is signless.
    for (StringRef attr : {"num_heads", "kv_num_heads", "local_window_size"})
      if (auto a = op->getAttrOfType<IntegerAttr>(attr))
        state.addAttribute(attr, rewriter.getI64IntegerAttr(a.getInt()));
    if (mrope) {
      for (StringRef attr :
           {"rotary_dim", "mrope_section", "mrope_layout", "mrope_scale"}) {
        Attribute a = op->getAttr(attr);
        if (!a)
          return rewriter.notifyMatchFailure(
              op, "missing internal MRoPE attribute");
        state.addAttribute(attr, a);
      }
    } else {
      for (StringRef attr : {"do_rotary", "rotary_interleaved"})
        if (auto a = op->getAttrOfType<IntegerAttr>(attr))
          state.addAttribute(attr, rewriter.getI64IntegerAttr(a.getInt()));
    }
    for (StringRef attr : {"scale"})
      if (auto a = op->getAttrOfType<FloatAttr>(attr))
        state.addAttribute(attr,
                           rewriter.getF32FloatAttr(a.getValueAsDouble()));
    if (!mrope)
      state.addAttribute("operand_segment_sizes",
                         rewriter.getDenseI32ArrayAttr(
                             {1, 1, 1, 1, 1, 1, 1, slot(8) ? 1 : 0,
                              slot(9) ? 1 : 0, slot(16) ? 1 : 0, 1, 1, 1}));
    auto converted = rewriter.create(state);
    rewriter.replaceOp(op, converted->getResults().take_front(resultCount));
    return success();
  }
};
} // namespace
void populatePagedAttentionConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<PagedAttentionToHip>(ctx);
}
} // namespace mlir::hip
