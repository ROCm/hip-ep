/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SlotLifetimeCoalesce.cpp - Slot id coalescing by DimSpec + lifetime ===//
//
// Phase 3 of the slot-buffer-coalesce design
// (docs/design/slot-buffer-coalesce.md).
//
// When two slot-bound values have STRUCTURALLY IDENTICAL DimSpecs (after
// `DimSpec::canonicalize()`) and NON-OVERLAPPING lifetimes within a single
// `@main_graph` execution, they can safely share one dyn-pool buffer at
// runtime. This pass picks a single representative slot id per equivalence
// class and rewrites every reference (publisher attr, consumer attr,
// `RuntimeSlot(N)` leaves inside `output_dim_specs`, and the module-level
// `hipdnn.output_dim_specs`) so the rest of the pipeline sees a smaller,
// contiguous slot id range.
//
// Output-bound slots (consumed by func results with `hipdnn.output_index`)
// live PAST `inference_compute` return. We model their lifetime as
// `[def, +inf)` so they never coalesce with intermediates whose last use
// lands inside the function body but whose def is earlier. Two
// output-bound slots cannot share storage at all -- both must remain
// alive simultaneously for the EP-side resolver to read them post-sync.
//
// Pipeline position: between hip-annotate-input-dim-slots and
// convert-hip-to-llvm. Annotation runs first so the per-operand
// `hipdnn.input_dim_slots` / `hipdnn.input_slot_buffers` attributes get
// rewritten in step 5 along with everything else.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

#define DEBUG_TYPE "hip-slot-lifetime-coalesce"

STATISTIC(NumSlotsBefore,
          "Number of distinct slot ids referenced before coalescing");
STATISTIC(NumSlotsAfter,
          "Number of distinct slot ids referenced after coalescing");
STATISTIC(NumSlotsRewritten,
          "Number of slot id references rewritten by the coalescer");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SLOTLIFETIMECOALESCEPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// SlotInfo: per-slot record built in Phase A
//===----------------------------------------------------------------------===//

struct SlotInfo {
  int32_t slot_id = -1;
  // Producer op (the slot publisher). Tracked for diagnostics and so the
  // post-rewrite renumber can resolve "which slot id was this originally?"
  // if needed.
  Operation *producer = nullptr;
  // Which result + dim of `producer` is this slot bound to.
  unsigned producer_result_index = 0;
  unsigned producer_dim_index = 0;
  // Canonical DimSpec serialized as a stable string. Two slots compare
  // equal-class iff their `canon_bytes` strings match.
  std::string canon_bytes;
  // Lifetime indices over a linear walk of @main_graph (def order). +inf
  // for output-bound slots (consumed by func results with
  // hipdnn.output_index).
  int64_t def_idx = std::numeric_limits<int64_t>::max();
  int64_t last_use_idx = std::numeric_limits<int64_t>::min();
  bool output_bound = false;
};

constexpr int64_t kInfLifetime = std::numeric_limits<int64_t>::max();

// True when slot intervals overlap. Output-bound slots have lastUse = +inf,
// which forces overlap with anything defined at or before def_idx of the
// output-bound one.
bool lifetimesOverlap(const SlotInfo &a, const SlotInfo &b) {
  if (a.def_idx == kInfLifetime || b.def_idx == kInfLifetime)
    return true;
  return !(a.last_use_idx < b.def_idx || b.last_use_idx < a.def_idx);
}

// Stable string serialization of a canonicalised DimSpec. We use the
// op's `output_dim_specs` per-dim ArrayAttr text form -- two
// canonicalised trees serialise to identical text iff they're
// structurally equal.
std::string serializeCanonical(const DimSpec &ds) {
  std::string out;
  llvm::raw_string_ostream os(out);
  for (const auto &n : ds.nodes()) {
    os << "(" << (int)n.kind << "," << n.value << "," << n.input_index << ","
       << n.dim_index << "," << n.flat_offset << "," << n.slot_id << ","
       << n.lhs << "," << n.rhs << ")";
  }
  return os.str();
}

// Pull the DimSpec for (result, dim) from an op's `output_dim_specs`
// attribute -- the shape of the attribute is
// ArrayAttr<ArrayAttr<ArrayAttr>> i.e. outer = per-result, middle =
// per-dim, inner = the DimSpec node list. Returns an empty DimSpec when
// the attribute is missing or the indices are out of range.
DimSpec readOutputDimSpec(Operation *op, unsigned r, unsigned d) {
  auto outer = op->getAttrOfType<ArrayAttr>("output_dim_specs");
  if (!outer || r >= outer.size())
    return DimSpec();
  auto perResult = llvm::dyn_cast<ArrayAttr>(outer[r]);
  if (!perResult || d >= perResult.size())
    return DimSpec();
  auto specAttr = llvm::dyn_cast<ArrayAttr>(perResult[d]);
  if (!specAttr)
    return DimSpec();
  return DimSpec::parseFromArrayAttr(specAttr);
}

// Read the slot id grid attached to `op`, falling back to the legacy
// scalar `slot_id` / array-form `slot_ids` schemas. Returns one entry
// per (result, dim), -1 entries where no slot was published.
//
// Layout: outer[result] -> array<i32: per-dim slot ids>. When the
// legacy `slot_id` (scalar) attr is used, the grid is filled in for
// result 0 dim D where D is the first dynamic dim of result 0 (matching
// the publisher convention).
SmallVector<SmallVector<int32_t, 4>, 1> readSlotGrid(Operation *op) {
  SmallVector<SmallVector<int32_t, 4>, 1> grid;
  const unsigned nr = op->getNumResults();
  grid.reserve(nr);

  if (auto outer = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
    for (unsigned r = 0; r < nr; ++r) {
      SmallVector<int32_t, 4> perDim;
      if (r < outer.size()) {
        if (auto pr = llvm::dyn_cast<DenseI32ArrayAttr>(outer[r])) {
          for (int32_t s : pr.asArrayRef())
            perDim.push_back(s);
        }
      }
      grid.push_back(std::move(perDim));
    }
    return grid;
  }
  if (auto slotIds = op->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
    // Legacy: one dim grid per result-0; treat as result-0 array.
    SmallVector<int32_t, 4> perDim0;
    for (int32_t s : slotIds.asArrayRef())
      perDim0.push_back(s);
    grid.push_back(std::move(perDim0));
    for (unsigned r = 1; r < nr; ++r)
      grid.push_back({});
    return grid;
  }
  if (auto slot = op->getAttrOfType<IntegerAttr>("slot_id")) {
    // Scalar legacy: bind to the first dynamic dim of result 0.
    SmallVector<int32_t, 4> perDim0;
    if (nr > 0) {
      if (auto rt = llvm::dyn_cast<ShapedType>(op->getResult(0).getType())) {
        if (rt.hasRank()) {
          int32_t found = (int32_t)slot.getInt();
          for (int64_t d = 0; d < rt.getRank(); ++d) {
            if (rt.isDynamicDim(d)) {
              perDim0.resize((size_t)d + 1, -1);
              perDim0[d] = found;
              break;
            }
          }
        }
      }
    }
    grid.push_back(std::move(perDim0));
    for (unsigned r = 1; r < nr; ++r)
      grid.push_back({});
    return grid;
  }
  return grid;
}

//===----------------------------------------------------------------------===//
// IR rewrite: substitute remap on every slot-id-bearing attribute
//===----------------------------------------------------------------------===//

// Apply `remap` to a DenseI32ArrayAttr in place by constructing a fresh
// attribute with the substituted values. Returns the new attribute (or
// `attr` unchanged when no entry is in `remap`).
Attribute remapDenseI32Array(MLIRContext *ctx, DenseI32ArrayAttr attr,
                             const DenseMap<int32_t, int32_t> &remap,
                             uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<int32_t> out;
  out.reserve(attr.size());
  bool changed = false;
  for (int32_t s : attr.asArrayRef()) {
    if (s >= 0) {
      auto it = remap.find(s);
      if (it != remap.end() && it->second != s) {
        out.push_back(it->second);
        ++rewriteCount;
        changed = true;
        continue;
      }
    }
    out.push_back(s);
  }
  if (!changed)
    return attr;
  return b.getDenseI32ArrayAttr(out);
}

// Same shape as the dim-slot pair attribute: ArrayAttr<ArrayAttr<DenseI32ArrayAttr>>
// for `hipdnn.input_dim_slots` (per-operand list of [dim_idx, slot_id]
// pairs).
Attribute
remapInputDimSlotsAttr(MLIRContext *ctx, ArrayAttr attr,
                       const DenseMap<int32_t, int32_t> &remap,
                       uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> outer;
  outer.reserve(attr.size());
  bool changed = false;
  for (Attribute opAttr : attr) {
    auto perOperand = llvm::dyn_cast<ArrayAttr>(opAttr);
    if (!perOperand) {
      outer.push_back(opAttr);
      continue;
    }
    SmallVector<Attribute> inner;
    inner.reserve(perOperand.size());
    for (Attribute pairAttr : perOperand) {
      auto pair = llvm::dyn_cast<DenseI32ArrayAttr>(pairAttr);
      if (!pair || pair.size() != 2) {
        inner.push_back(pairAttr);
        continue;
      }
      int32_t dim = pair.asArrayRef()[0];
      int32_t slot = pair.asArrayRef()[1];
      if (slot >= 0) {
        auto it = remap.find(slot);
        if (it != remap.end() && it->second != slot) {
          slot = it->second;
          ++rewriteCount;
          changed = true;
        }
      }
      std::array<int32_t, 2> repl = {dim, slot};
      inner.push_back(b.getDenseI32ArrayAttr(repl));
    }
    outer.push_back(b.getArrayAttr(inner));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(outer);
}

// Shape: ArrayAttr<ArrayAttr<DenseI32ArrayAttr>> for
// `hipdnn.output_slot_ids` (per-result list of per-dim slot ids).
Attribute remapOutputSlotIdsAttr(MLIRContext *ctx, ArrayAttr attr,
                                 const DenseMap<int32_t, int32_t> &remap,
                                 uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> outer;
  outer.reserve(attr.size());
  bool changed = false;
  for (Attribute resAttr : attr) {
    auto perResult = llvm::dyn_cast<DenseI32ArrayAttr>(resAttr);
    if (!perResult) {
      outer.push_back(resAttr);
      continue;
    }
    Attribute newAttr =
        remapDenseI32Array(ctx, perResult, remap, rewriteCount);
    if (newAttr != Attribute(perResult))
      changed = true;
    outer.push_back(newAttr);
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(outer);
}

// Walk every RuntimeSlot leaf in a serialized DimSpec ArrayAttr and
// rewrite its slot id field (index 5 in the 8-field encoding) per
// `remap`. Returns the new attribute -- or `attr` unchanged when there's
// nothing to rewrite.
Attribute remapDimSpecArrayAttr(MLIRContext *ctx, ArrayAttr attr,
                                const DenseMap<int32_t, int32_t> &remap,
                                uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> nodeAttrs;
  nodeAttrs.reserve(attr.size());
  bool changed = false;
  for (Attribute nodeAttr : attr) {
    auto fields = llvm::dyn_cast<DenseI64ArrayAttr>(nodeAttr);
    if (!fields || fields.size() < 8) {
      nodeAttrs.push_back(nodeAttr);
      continue;
    }
    SmallVector<int64_t> vals(fields.asArrayRef().begin(),
                              fields.asArrayRef().end());
    // kind == 3 (RuntimeSlot) with slot_id at index 5.
    if (vals[0] == 3 && vals[5] >= 0) {
      auto it = remap.find((int32_t)vals[5]);
      if (it != remap.end() && it->second != vals[5]) {
        vals[5] = it->second;
        changed = true;
        ++rewriteCount;
      }
    }
    nodeAttrs.push_back(b.getDenseI64ArrayAttr(vals));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(nodeAttrs);
}

// Walk the per-op `output_dim_specs` (ArrayAttr<ArrayAttr<ArrayAttr>>:
// per-result -> per-dim -> DimSpec node list) and rewrite every
// RuntimeSlot leaf.
Attribute remapOutputDimSpecsAttr(MLIRContext *ctx, ArrayAttr attr,
                                  const DenseMap<int32_t, int32_t> &remap,
                                  uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> resEntries;
  resEntries.reserve(attr.size());
  bool changed = false;
  for (Attribute resA : attr) {
    auto perResult = llvm::dyn_cast<ArrayAttr>(resA);
    if (!perResult) {
      resEntries.push_back(resA);
      continue;
    }
    SmallVector<Attribute> dimEntries;
    dimEntries.reserve(perResult.size());
    for (Attribute dimA : perResult) {
      auto perDim = llvm::dyn_cast<ArrayAttr>(dimA);
      if (!perDim) {
        dimEntries.push_back(dimA);
        continue;
      }
      Attribute newPerDim =
          remapDimSpecArrayAttr(ctx, perDim, remap, rewriteCount);
      if (newPerDim != Attribute(perDim))
        changed = true;
      dimEntries.push_back(newPerDim);
    }
    resEntries.push_back(b.getArrayAttr(dimEntries));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(resEntries);
}

// Walk the module-level `hipdnn.output_dim_specs` (same nesting shape
// as the per-op one) and rewrite slot ids.
Attribute remapModuleOutputDimSpecsAttr(
    MLIRContext *ctx, ArrayAttr attr,
    const DenseMap<int32_t, int32_t> &remap, uint64_t &rewriteCount) {
  return remapOutputDimSpecsAttr(ctx, attr, remap, rewriteCount);
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

class SlotLifetimeCoalescePass
    : public impl::SlotLifetimeCoalescePassBase<SlotLifetimeCoalescePass> {
public:
  using impl::SlotLifetimeCoalescePassBase<
      SlotLifetimeCoalescePass>::SlotLifetimeCoalescePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto *hipDialect = ctx->getLoadedDialect<HipDialect>();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc)
      return;

    if (disableCoalesce)
      return;

    //
    // Phase A: assign a sequential index to every op in @main_graph
    // (def-use ordering proxy) and collect slot publishers + consumers.
    //
    DenseMap<Operation *, int64_t> opIndex;
    SmallVector<Operation *> orderedOps;
    int64_t idx = 0;
    mainFunc.walk([&](Operation *op) {
      if (op == mainFunc.getOperation())
        return;
      opIndex[op] = idx++;
      orderedOps.push_back(op);
    });

    // Step A.1: collect every published slot. Each slot id maps to a
    // single SlotInfo (we de-dup by slot id since the same slot can
    // appear on multiple result-dim entries on the same op only in
    // rare publisher patterns; one info per slot is enough for
    // lifetime + canonical key purposes).
    DenseMap<int32_t, SlotInfo> infoBySlot;
    DenseMap<Operation *, SmallVector<int32_t, 4>> slotsByOp;

    for (Operation *op : orderedOps) {
      if (op->getDialect() != hipDialect)
        continue;
      auto grid = readSlotGrid(op);
      if (grid.empty())
        continue;
      SmallVector<int32_t, 4> slotsThisOp;
      for (unsigned r = 0; r < grid.size(); ++r) {
        for (unsigned d = 0; d < grid[r].size(); ++d) {
          int32_t s = grid[r][d];
          if (s < 0)
            continue;
          auto &info = infoBySlot[s];
          // If we haven't seen this slot before, bind its producer
          // identity here. A slot may appear in multiple (r, d) on
          // the same op (legacy convention for multi-dim publishers
          // — though today we only have one publisher per slot).
          if (info.slot_id < 0) {
            info.slot_id = s;
            info.producer = op;
            info.producer_result_index = r;
            info.producer_dim_index = d;
            info.def_idx = opIndex[op];
            // Default lastUse = def -- we'll bump via walks below.
            info.last_use_idx = info.def_idx;
            // DimSpec from this (r, d).
            DimSpec ds = readOutputDimSpec(op, r, d);
            ds.canonicalize();
            info.canon_bytes = serializeCanonical(ds);
          }
          slotsThisOp.push_back(s);
        }
      }
      if (!slotsThisOp.empty())
        slotsByOp[op] = std::move(slotsThisOp);
    }

    if (infoBySlot.empty())
      return;

    NumSlotsBefore = infoBySlot.size();

    // Step A.2: walk consumers to compute last_use_idx per slot.
    // Two consumer attributes reference a slot:
    //   - `hipdnn.input_dim_slots`: ArrayAttr<ArrayAttr<DenseI32ArrayAttr>>
    //     per-operand list of [dim_idx, slot_id] pairs.
    //   - `hipdnn.input_slot_buffers`: DenseI32ArrayAttr -- per-operand
    //     producer slot id (-1 for "no slot").
    for (Operation *op : orderedOps) {
      int64_t myIdx = opIndex[op];
      if (auto a = op->getAttrOfType<ArrayAttr>("hipdnn.input_dim_slots")) {
        for (Attribute opAttr : a) {
          auto perOperand = llvm::dyn_cast<ArrayAttr>(opAttr);
          if (!perOperand)
            continue;
          for (Attribute pairAttr : perOperand) {
            auto pair = llvm::dyn_cast<DenseI32ArrayAttr>(pairAttr);
            if (!pair || pair.size() != 2)
              continue;
            int32_t s = pair.asArrayRef()[1];
            auto it = infoBySlot.find(s);
            if (it != infoBySlot.end() &&
                myIdx > it->second.last_use_idx)
              it->second.last_use_idx = myIdx;
          }
        }
      }
      if (auto a =
              op->getAttrOfType<DenseI32ArrayAttr>("hipdnn.input_slot_buffers")) {
        for (int32_t s : a.asArrayRef()) {
          if (s < 0)
            continue;
          auto it = infoBySlot.find(s);
          if (it != infoBySlot.end() && myIdx > it->second.last_use_idx)
            it->second.last_use_idx = myIdx;
        }
      }
    }

    // Step A.3: classify output-bound slots. A slot is output-bound
    // when any RuntimeSlot(N) leaf appears in the module-level
    // `hipdnn.output_dim_specs` (those leaves get read by the EP-side
    // resolver after stream sync). We also widen the lifetime of an
    // output-bound slot to +inf so it never coalesces with an
    // intermediate that ends mid-graph.
    if (auto outDS =
            module->getAttrOfType<ArrayAttr>("hipdnn.output_dim_specs")) {
      for (Attribute resA : outDS) {
        auto perResult = llvm::dyn_cast<ArrayAttr>(resA);
        if (!perResult)
          continue;
        for (Attribute dimA : perResult) {
          auto perDim = llvm::dyn_cast<ArrayAttr>(dimA);
          if (!perDim)
            continue;
          DimSpec ds = DimSpec::parseFromArrayAttr(perDim);
          for (int32_t s : ds.collectSlotIds()) {
            auto it = infoBySlot.find(s);
            if (it != infoBySlot.end()) {
              it->second.output_bound = true;
              it->second.last_use_idx = kInfLifetime;
            }
          }
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[hip-slot-lifetime-coalesce] Collected "
                   << infoBySlot.size() << " slots:\n";
      for (auto &[s, info] : infoBySlot) {
        llvm::dbgs() << "  slot " << s << ": def=" << info.def_idx
                     << " lastUse=" << info.last_use_idx
                     << " outBound=" << info.output_bound
                     << " canon=\"" << info.canon_bytes << "\"\n";
      }
    });

    //
    // Phase B: group by canonical DimSpec bytes, then first-fit-decreasing
    // bin-pack within each group on lifetime. Output-bound slots get
    // their own bin (lastUse = +inf) -- they never share with an
    // intermediate-only slot that ends earlier.
    //
    // std::map for deterministic iteration order keyed on string; the
    // bin-pack output (and consequently the slot remap) is then
    // independent of DenseMap probe order. We store slot ids (not
    // SlotInfo* into `infoBySlot`) because the DenseMap can rehash
    // when grown by Phase A.* (any new keyed access path beyond
    // initial population would silently invalidate the pointers).
    std::map<std::string, SmallVector<int32_t>> byCanonical;
    for (auto &[s, info] : infoBySlot) {
      byCanonical[info.canon_bytes].push_back(s);
    }

    // Build the remap: every slot in the same bin maps to the bin's
    // representative (smallest original slot id).
    DenseMap<int32_t, int32_t> remap;
    for (auto &[key, group] : byCanonical) {
      // Sort by lifetime length descending so first-fit-decreasing
      // packs the longest lifetimes first (better packing).
      std::sort(group.begin(), group.end(),
                [&](int32_t a, int32_t b) {
                  const SlotInfo &ia = infoBySlot[a];
                  const SlotInfo &ib = infoBySlot[b];
                  int64_t la = (ia.last_use_idx == kInfLifetime
                                    ? kInfLifetime
                                    : ia.last_use_idx - ia.def_idx);
                  int64_t lb = (ib.last_use_idx == kInfLifetime
                                    ? kInfLifetime
                                    : ib.last_use_idx - ib.def_idx);
                  if (la != lb)
                    return la > lb;
                  return a < b;
                });
      SmallVector<SmallVector<int32_t>> bins;
      for (int32_t s : group) {
        SlotInfo &info = infoBySlot[s];
        bool placed = false;
        for (auto &bin : bins) {
          bool conflicts = false;
          for (int32_t existingSlot : bin) {
            if (lifetimesOverlap(info, infoBySlot[existingSlot])) {
              conflicts = true;
              break;
            }
          }
          if (!conflicts) {
            bin.push_back(s);
            placed = true;
            break;
          }
        }
        if (!placed)
          bins.push_back({s});
      }
      for (auto &bin : bins) {
        int32_t rep = std::numeric_limits<int32_t>::max();
        for (int32_t s : bin)
          rep = std::min(rep, s);
        for (int32_t s : bin)
          remap[s] = rep;
      }
    }

    // No-op short-circuit when nothing actually merged.
    bool anyChange = false;
    for (auto it = remap.begin(); it != remap.end(); ++it)
      if (it->first != it->second) {
        anyChange = true;
        break;
      }
    if (!anyChange) {
      NumSlotsAfter = infoBySlot.size();
      return;
    }

    //
    // Phase C: contiguous renumber. Surviving rep slot ids are
    // compacted to 0..K-1 so the runtime's slot table stays dense.
    //
    SmallVector<int32_t> survivors;
    for (auto it = remap.begin(); it != remap.end(); ++it) {
      if (it->first == it->second)
        survivors.push_back(it->first);
    }
    std::sort(survivors.begin(), survivors.end());
    DenseMap<int32_t, int32_t> compact;
    for (size_t i = 0; i < survivors.size(); ++i)
      compact[survivors[i]] = (int32_t)i;
    // Compose: original -> rep -> compact.
    DenseMap<int32_t, int32_t> finalRemap;
    for (auto it = remap.begin(); it != remap.end(); ++it)
      finalRemap[it->first] = compact[it->second];

    LLVM_DEBUG({
      llvm::dbgs() << "[hip-slot-lifetime-coalesce] Remap (orig -> final):\n";
      for (auto &[orig, fin] : finalRemap)
        llvm::dbgs() << "  " << orig << " -> " << fin << "\n";
    });

    //
    // Phase D: IR rewrite. Walk every op once and substitute on every
    // slot-bearing attribute.
    //
    uint64_t rewrites = 0;
    mainFunc.walk([&](Operation *op) {
      // Publisher-side attrs:
      if (auto a = op->getAttrOfType<IntegerAttr>("slot_id")) {
        int32_t s = (int32_t)a.getInt();
        if (s >= 0) {
          auto it = finalRemap.find(s);
          if (it != finalRemap.end() && it->second != s) {
            op->setAttr("slot_id",
                        IntegerAttr::get(IntegerType::get(ctx, 32),
                                         it->second));
            ++rewrites;
          }
        }
      }
      if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
        Attribute newAttr = remapDenseI32Array(ctx, a, finalRemap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("slot_ids", newAttr);
      }
      if (auto a = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
        Attribute newAttr =
            remapOutputSlotIdsAttr(ctx, a, finalRemap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.output_slot_ids", newAttr);
      }
      // Per-op output_dim_specs (RuntimeSlot leaves).
      if (auto a = op->getAttrOfType<ArrayAttr>("output_dim_specs")) {
        Attribute newAttr =
            remapOutputDimSpecsAttr(ctx, a, finalRemap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("output_dim_specs", newAttr);
      }
      // Consumer-side attrs (from AnnotateInputDimSlotsPass):
      if (auto a = op->getAttrOfType<ArrayAttr>("hipdnn.input_dim_slots")) {
        Attribute newAttr =
            remapInputDimSlotsAttr(ctx, a, finalRemap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.input_dim_slots", newAttr);
      }
      if (auto a =
              op->getAttrOfType<DenseI32ArrayAttr>("hipdnn.input_slot_buffers")) {
        Attribute newAttr = remapDenseI32Array(ctx, a, finalRemap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.input_slot_buffers", newAttr);
      }
    });

    // Module-level attrs:
    if (auto a = module->getAttrOfType<ArrayAttr>("hipdnn.output_dim_specs")) {
      Attribute newAttr =
          remapModuleOutputDimSpecsAttr(ctx, a, finalRemap, rewrites);
      if (newAttr != Attribute(a))
        module->setAttr("hipdnn.output_dim_specs", newAttr);
    }

    // Update the slot count to the new contiguous range size.
    Builder b(ctx);
    module->setAttr("hipdnn.dyn_dim_slots_count",
                    b.getI32IntegerAttr((int32_t)survivors.size()));
    // Bump the next-slot-id counter so any post-coalesce reservation
    // (today there is none, but the invariant should hold) starts
    // beyond the new range.
    module->setAttr("hipdnn.next_dyn_slot_id",
                    b.getI32IntegerAttr((int32_t)survivors.size()));

    NumSlotsAfter = survivors.size();
    NumSlotsRewritten = rewrites;
  }
};

} // namespace
} // namespace hip
} // namespace mlir
