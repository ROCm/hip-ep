/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ComposeDimSpecs.cpp - Build model-output DimSpec trees ------------===//
//
// Walks the IR backwards from each result of `@main_graph` to compose a
// fully-resolved DimSpec tree per dynamic result dim. Composition uses the
// per-op `shape_interface::getResultDimSpec` dispatcher and bottoms out at
// func-arg shape/value accesses, Static constants, or RuntimeSlot leaves
// published by Category-C ops upstream.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/debug_log.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_COMPOSEDIMSPECSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Resolve the DimSpec for `v`'s dim `dim_index` by walking the producer
// chain. Substitutes operand references in the producer's spec until all
// `InputDim`/`InputValueI64` leaves point at func-arg indices (not local
// operand indices).
//
// This wraps the dispatch logic in HipShapeInterface so that the compose
// pass can iterate to a fixed point (a producer's spec may reference one
// of its operands, which in turn produces a spec referencing another, and
// so on).
DimSpec composeSpecForValue(Value v, unsigned dim_index);

// Translate operand-relative DimSpec leaves into func-arg-relative leaves
// by walking back through the producer of each `InputDim` / `InputValueI64`
// leaf. The producer-op's spec uses operand indices in its own scope; this
// helper traverses to the func-arg scope.
DimSpec composeSpecFromOpSpec(Operation *op, const DimSpec &op_spec);

DimSpec composeSpecForValue(Value v, unsigned dim_index) {
  // Bottom case: func-arg → InputDim leaf with EP-relative index.
  DimSpec via_iface = shape_interface::resolveDimFromValue(v, dim_index);
  if (!via_iface.nodes().empty()) {
    // If the result is itself a Static / InputDim / InputValueI64 leaf,
    // no further composition is needed.
    auto kind = via_iface.root().kind;
    if (kind == DimSpecKind::Static || kind == DimSpecKind::InputDim ||
        kind == DimSpecKind::InputValueI64 ||
        kind == DimSpecKind::RuntimeSlot) {
      return via_iface;
    }
    // Otherwise, this is a producer-op-attached spec referring to its
    // own operand indices; descend.
    Operation *producer = v.getDefiningOp();
    if (!producer)
      return via_iface;
    return composeSpecFromOpSpec(producer, via_iface);
  }
  return DimSpec();
}

DimSpec composeSpecFromOpSpec(Operation *op, const DimSpec &op_spec) {
  // Walk the op_spec tree; for any InputDim / InputValueI64 leaf whose
  // input_index is an operand-relative index of `op`, substitute the
  // composed spec from that operand.
  if (op_spec.nodes().empty())
    return op_spec;

  // Collect operand-relative leaves to substitute.
  llvm::DenseMap<int32_t, DimSpec> substitutions;
  auto needSubstitute = [&](const DimSpecNode &n) -> bool {
    return n.kind == DimSpecKind::InputDim ||
           n.kind == DimSpecKind::InputValueI64;
  };

  // Build composed subtrees for each unique leaf occurrence. Use an
  // ad-hoc key: encode (kind, input_index, dim_index_or_flat_offset)
  // into a slot id namespace separate from real Category-C slots.
  // Substitution map: we reuse the same `substituteSlots` machinery by
  // first lifting operand-relative leaves to fake slot ids, then
  // remapping. To avoid that complexity, do a manual tree walk and
  // produce a fresh composed DimSpec.
  //
  // Implementation: clone op_spec, substituting in-place.
  DimSpec composed;
  std::vector<DimSpecNode> &out_nodes = composed.mutableNodes();

  // Recursive cloner.
  std::function<int32_t(int32_t)> clone = [&](int32_t src_idx) -> int32_t {
    const DimSpecNode &src = op_spec.nodes()[src_idx];
    if (needSubstitute(src)) {
      // Resolve the operand-relative leaf:
      int32_t operand_idx = src.input_index;
      if (operand_idx < 0 ||
          operand_idx >= (int32_t)op->getNumOperands()) {
        // Defensive: copy unchanged (will be diagnosed by verify).
        out_nodes.push_back(src);
        return (int32_t)out_nodes.size() - 1;
      }
      Value operand = op->getOperand(operand_idx);
      DimSpec sub;
      if (src.kind == DimSpecKind::InputDim) {
        sub = composeSpecForValue(operand, (unsigned)src.dim_index);
      } else {
        // InputValueI64
        sub = shape_interface::resolveValueFromI64Tensor(operand,
                                                         src.flat_offset);
      }
      if (sub.nodes().empty()) {
        // No resolution → leave the leaf as-is (caller should diagnose).
        out_nodes.push_back(src);
        return (int32_t)out_nodes.size() - 1;
      }
      int32_t base = (int32_t)out_nodes.size();
      for (const auto &n : sub.nodes()) {
        DimSpecNode c = n;
        if (c.lhs >= 0)
          c.lhs += base;
        if (c.rhs >= 0)
          c.rhs += base;
        out_nodes.push_back(c);
      }
      return base;
    }
    if (src.kind == DimSpecKind::Static ||
        src.kind == DimSpecKind::RuntimeSlot) {
      out_nodes.push_back(src);
      return (int32_t)out_nodes.size() - 1;
    }
    // Binary node — clone with recursive children.
    int32_t my_idx = (int32_t)out_nodes.size();
    out_nodes.push_back(src);
    out_nodes[my_idx].lhs = -1;
    out_nodes[my_idx].rhs = -1;
    int32_t lhs_idx = clone(src.lhs);
    int32_t rhs_idx = clone(src.rhs);
    out_nodes[my_idx].lhs = lhs_idx;
    out_nodes[my_idx].rhs = rhs_idx;
    return my_idx;
  };

  if (op_spec.nodes().empty())
    return op_spec;
  clone(0);
  return composed;
}

class ComposeDimSpecsPass
    : public impl::ComposeDimSpecsPassBase<ComposeDimSpecsPass> {
public:
  using impl::ComposeDimSpecsPassBase<
      ComposeDimSpecsPass>::ComposeDimSpecsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc) {
      // Some pipelines (post-bufferize-to-out-params) move @main_graph to
      // llvm.func; this pass is intentionally scheduled BEFORE that hop,
      // i.e. before convert-hip-to-llvm. If main_graph isn't a func.func
      // here, treat as a no-op so we don't accidentally fail downstream
      // pipelines.
      return;
    }

    // The bufferize-to-out-params pass converts func results into out-
    // memref-params at positions [num_orig_inputs .. end). We need to
    // distinguish *original* results from out-params.
    //
    // Strategy: the OnnxToHip pass attaches `hipdnn.output_shapes` to the
    // module before bufferization. After bufferize-to-out-params, the
    // function args layout is [hip.context, ...inputs..., ...output out-
    // params...]. The number of original outputs equals the number of
    // out-param args added (one per result, which is the
    // `outputElementSizes` count). We rely on `hipdnn.output_shapes`
    // being present and consistent with the trailing func args.
    auto outputShapesAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
    if (!outputShapesAttr)
      return;
    size_t numOutputs = outputShapesAttr.size();
    if (numOutputs == 0)
      return;

    auto funcType = mainFunc.getFunctionType();
    size_t numArgs = funcType.getNumInputs();
    if (numOutputs > numArgs)
      return;

    // The output out-param slots are the last `numOutputs` arguments.
    // The producer of each output's *content* is the value bound to that
    // out-param via the `bufferization.materialize_in_destination` /
    // `copy` ops or via the final assignment. For DimSpec purposes we
    // only need the static MLIR memref shape — bufferize-to-out-params
    // preserves the source result type as the out-param's memref type
    // (after `IdentityLayoutMap` boundary conversion).
    //
    // For each output i, look at the memref type of out-param argument
    // (numArgs - numOutputs + i). Its memref shape mirrors the original
    // result shape.

    // hipdnn.output_shapes is a per-output DenseI64ArrayAttr (rank+shape).
    // For each dynamic dim we synthesize a DimSpec by following the SSA
    // value that the function writes into the out-param.
    //
    // Locating the writer: the bufferize-to-out-params pass emits either:
    //   - `bufferization.materialize_in_destination %src in writable
    //      %out_param` (preserved if the bufferization pipeline didn't
    //      lower it), or
    //   - a `memref.copy %src, %out_param`, or
    //   - the value is computed in place into %out_param (DPS), so the
    //      producer IS the consumer.
    //
    // In our pipeline `buildBufferDeallocationPipeline` runs after
    // `bufferization.materialize_in_destination`, and the standard
    // bufferization.materialize_in_destination is converted to
    // memref.copy by `createConvertBufferizationToMemRefPass`. So by the
    // time ComposeDimSpecs runs (right before convert-hip-to-llvm), the
    // writer of each out-param is a `memref.copy` whose source operand
    // is the actual producer.

    // To stay general, walk the entry block in reverse and for each
    // out-param argument, find the last op whose results / operands touch
    // it. For memref.copy that's operand(1); for DPS we trace through
    // the DestinationStyleOpInterface init.
    Block &entry = mainFunc.getBody().front();

    llvm::SmallVector<Attribute> perOutputDimSpecAttrs(numOutputs, nullptr);
    int32_t maxSlotIdSeen = -1;

    // Walk every op in main_graph and collect slot ids referenced by op
    // attributes. Without this, slots claimed by Category-C wrappers
    // (e.g. `wrap_nonzero` slot 0) whose published dim is never reflected
    // in a function-output `RuntimeSlot` leaf — typical when the
    // intermediate output is consumed only by something like `Shape()`
    // that converts the dim into a static-typed payload — would not
    // contribute to `hipdnn.dyn_dim_slots_count`. The runtime then aborts
    // at publish_dim time with "slot_id N out of range [0, 0)". This
    // mirrors the same scan done by ComposeDimSpecs for output-bound
    // slots, but covers ALL slot publishers regardless of whether their
    // slot is referenced from a function output.
    mainFunc.getBody().walk([&](Operation *op) {
      if (auto sidAttr = op->getAttrOfType<IntegerAttr>("slot_id")) {
        int32_t s = static_cast<int32_t>(sidAttr.getInt());
        if (s > maxSlotIdSeen)
          maxSlotIdSeen = s;
      }
      if (auto sidsAttr = op->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
        for (int32_t s : sidsAttr.asArrayRef())
          if (s > maxSlotIdSeen)
            maxSlotIdSeen = s;
      }
      // Phase 2 array-form schema: `hipdnn.output_slot_ids` is an outer
      // ArrayAttr keyed on result index, each entry a DenseI32ArrayAttr
      // of length == rank with -1 for non-dyn dims. Without this scan
      // the propagator slots reserved by ReservePropagatorSlotsPass do
      // not contribute to `hipdnn.dyn_dim_slots_count`, and at runtime
      // the propagator wrapper aborts with "slot_id N out of range".
      if (auto grid =
              op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
        for (Attribute outer : grid) {
          if (auto perDim = dyn_cast<DenseI32ArrayAttr>(outer)) {
            for (int32_t s : perDim.asArrayRef()) {
              if (s > maxSlotIdSeen)
                maxSlotIdSeen = s;
            }
          }
        }
      }
    });

    // Resolve the source value (and, if applicable, the producing op
    // when the out-param is consumed via DPS) for a given out-param
    // block-argument. The two cases:
    //   (1) memref.copy / bufferization.materialize_in_destination — the
    //       writer is a copy whose operand(0) is a distinct SSA value
    //       (the "true" source). `source = operand(0)`,
    //       `dps_producer = nullptr`.
    //   (2) DPS — the writer is a HIP op (or similar) that takes the
    //       out-param as its DPS init operand and writes directly into
    //       it. Looking up the InputDim from the func-arg memref would
    //       give the trivially wrong answer ("output's dim N == own dim
    //       N") because the func-arg memref carries the dynamic-shape
    //       placeholder, not the producer's true spec. Instead, return
    //       the producer op so the caller can read its
    //       `output_dim_specs` attribute directly.
    struct SourceInfo {
      Value source;
      Operation *dps_producer = nullptr;
      unsigned dps_result_index = 0;
    };
    auto resolveSourceForOutParam = [&](BlockArgument outArg) -> SourceInfo {
      SourceInfo info;
      for (auto &op : llvm::reverse(entry)) {
        if (op.getName().getStringRef() == "memref.copy" &&
            op.getNumOperands() == 2 && op.getOperand(1) == outArg) {
          info.source = op.getOperand(0);
          return info;
        }
        if (op.getName().getStringRef() ==
                "bufferization.materialize_in_destination" &&
            op.getNumOperands() >= 2 && op.getOperand(1) == outArg) {
          info.source = op.getOperand(0);
          return info;
        }
        for (unsigned k = 0; k < op.getNumOperands(); ++k) {
          if (op.getOperand(k) == outArg) {
            info.source = outArg;
            info.dps_producer = &op;
            // Some DPS ops have results that mirror their init operands
            // (e.g. tensor world), others (after bufferize-to-out-params)
            // have ZERO results and write only into the out-param in
            // place. The two-stage lookup below tries the result-list
            // first, then falls back to the operand index. Either index
            // is only consulted by callers that read producer-attached
            // `output_dim_specs`, which is keyed on the original op
            // result index — so for 0-result ops we leave the result
            // index at 0 and rely on the attribute itself to be present
            // with a single per-result entry.
            for (unsigned r = 0; r < op.getNumResults(); ++r) {
              if (op.getResult(r) == outArg) {
                info.dps_result_index = r;
                break;
              }
            }
            return info;
          }
        }
      }
      return info;
    };

    for (size_t i = 0; i < numOutputs; ++i) {
      size_t outArgIdx = numArgs - numOutputs + i;
      auto outArg = entry.getArgument(outArgIdx);
      auto memrefType = llvm::dyn_cast<MemRefType>(outArg.getType());
      if (!memrefType)
        continue;
      SourceInfo srcInfo = resolveSourceForOutParam(outArg);

      llvm::SmallVector<Attribute> perDimAttrs;
      perDimAttrs.reserve(memrefType.getRank());
      bool anyResolved = false;
      bool allStatic = true;
      for (int64_t d = 0; d < memrefType.getRank(); ++d) {
        DimSpec ds;
        if (memrefType.isDynamicDim(d)) {
          allStatic = false;
          if (srcInfo.dps_producer) {
            // DPS: read the producer op's per-dim spec directly so the
            // RuntimeSlot leaf (or any other producer-attached
            // information) is preserved instead of being shadowed by an
            // InputDim leaf pointing at the out-param.
            DimSpec via_attr = shape_interface::getResultDimSpec(
                srcInfo.dps_producer, srcInfo.dps_result_index, (unsigned)d);
            if (!via_attr.nodes().empty()) {
              auto kind = via_attr.root().kind;
              if (kind == DimSpecKind::Static ||
                  kind == DimSpecKind::RuntimeSlot) {
                // Universal leaves (no operand reference): use as-is.
                // Static is a literal; RuntimeSlot is keyed on a global
                // slot id namespace that doesn't shift across ops.
                ds = via_attr;
              } else {
                // InputDim / InputValueI64 / arithmetic trees attached
                // to a producer op all use OPERAND-RELATIVE indices in
                // the producer's own scope. They must be re-walked
                // through the producer's operands to be lifted into
                // func-arg-relative indices the EP resolver can read.
                ds = composeSpecFromOpSpec(srcInfo.dps_producer, via_attr);
              }
            }
          } else if (srcInfo.source) {
            ds = composeSpecForValue(srcInfo.source, (unsigned)d);
          }
        } else {
          ds = DimSpec::makeStatic(memrefType.getDimSize(d));
        }
        if (ds.nodes().empty()) {
          // Could not resolve — store an empty marker attribute.
          // Empty ArrayAttr serializes to "no information"; the EP
          // resolver will treat as "unknown" and fall back to -1.
          perDimAttrs.push_back(ArrayAttr::get(module.getContext(), {}));
          continue;
        }
        anyResolved = true;
        for (int32_t slot : ds.collectSlotIds()) {
          if (slot > maxSlotIdSeen)
            maxSlotIdSeen = slot;
        }
        perDimAttrs.push_back(ds.serializeAsArrayAttr(module.getContext()));
      }
      // Always attach the per-output array (one entry per dim) so the
      // EP resolver can index without bounds-checks. Even if all dims
      // were static, having the explicit Static leaves is cheap and
      // future-proofs serialization.
      if (anyResolved || allStatic) {
        perOutputDimSpecAttrs[i] =
            ArrayAttr::get(module.getContext(), perDimAttrs);
      }
    }

    // Store as module attribute `hipdnn.output_dim_specs` — an ArrayAttr
    // with one ArrayAttr per output, each containing per-dim ArrayAttrs
    // (the serialized DimSpec node lists).
    llvm::SmallVector<Attribute> outerEntries;
    outerEntries.reserve(numOutputs);
    for (auto attr : perOutputDimSpecAttrs) {
      if (!attr) {
        outerEntries.push_back(ArrayAttr::get(module.getContext(), {}));
      } else {
        outerEntries.push_back(attr);
      }
    }
    module->setAttr("hipdnn.output_dim_specs",
                    ArrayAttr::get(module.getContext(), outerEntries));
    // Defensive floor: if ReservePropagatorSlotsPass bumped
    // `hipdnn.next_dyn_slot_id` higher than any slot id we walked
    // (e.g. an allocated slot id was canonicalised away by an
    // intervening pass), still reserve enough table entries for the
    // pre-bumped high water. This is symmetric to the body-walk above
    // and protects against future passes that touch the schema in
    // ways the walk doesn't observe.
    int32_t finalCount = maxSlotIdSeen + 1;
    if (auto a =
            module->getAttrOfType<IntegerAttr>("hipdnn.next_dyn_slot_id")) {
      int32_t reserved = static_cast<int32_t>(a.getInt());
      if (reserved > finalCount)
        finalCount = reserved;
    }
    module->setAttr("hipdnn.dyn_dim_slots_count",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 32),
                                     finalCount));
  }
};

} // namespace
} // namespace hip
} // namespace mlir
