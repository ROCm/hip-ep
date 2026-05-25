/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DimSpecsDebug.cpp - hip-dump-dim-specs / hip-verify-dim-specs -----===//
//
// Analysis-only passes for the data-dependent dynamic output shape feature.
// Neither pass mutates the IR.
//
//   --hip-dump-dim-specs    Pretty-print every per-op `output_dim_specs`
//                           (and the special `element_dim_specs` on
//                           `hip.shape`), plus the module-level
//                           `hipdnn.output_dim_specs` produced by
//                           `--hip-compose-dim-specs`, to stderr.
//
//   --hip-verify-dim-specs  Structurally validate every DimSpec it can
//                           find (trees well-formed, slot ids in range,
//                           every dynamic result dim of @main_graph
//                           covered). Returns failure on the first
//                           problem so it composes with `--check-result`
//                           inside a debug pipeline.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_DUMPDIMSPECSPASS
#define GEN_PASS_DEF_VERIFYDIMSPECSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// The attribute names below match the conventions the rest of the pipeline
// already uses (`HipOps.td` for per-op, `ComposeDimSpecs.cpp` for the
// module-level composed form). Keep them in one place so any rename is a
// single edit.
constexpr llvm::StringLiteral kPerOpOutputDimSpecsAttr = "output_dim_specs";
constexpr llvm::StringLiteral kHipShapeElementDimSpecsAttr =
    "element_dim_specs";
constexpr llvm::StringLiteral kModuleOutputDimSpecsAttr =
    "hipdnn.output_dim_specs";
constexpr llvm::StringLiteral kModuleDynDimSlotsCountAttr =
    "hipdnn.dyn_dim_slots_count";

// Convenience: extract an ArrayAttr-valued attribute and pretty-print its
// DimSpec contents under a header. Used for both the per-op `output_dim_specs`
// (one outer ArrayAttr per result, each holding one inner ArrayAttr per dim)
// and `element_dim_specs` (one outer ArrayAttr of N entries, one per output
// element of a `hip.shape` op).
void dumpPerOpArrayAttr(llvm::raw_ostream &os, llvm::StringRef header,
                        ::mlir::ArrayAttr attr) {
  if (!attr || attr.empty())
    return;
  os << "    " << header << ":\n";
  for (size_t i = 0; i < attr.size(); ++i) {
    auto inner = llvm::dyn_cast<::mlir::ArrayAttr>(attr[i]);
    if (!inner) {
      os << "      [" << i << "] <not an ArrayAttr>\n";
      continue;
    }
    for (size_t j = 0; j < inner.size(); ++j) {
      auto leaf = llvm::dyn_cast<::mlir::ArrayAttr>(inner[j]);
      os << "      [" << i << "][" << j << "] = ";
      if (!leaf) {
        os << "<not an ArrayAttr>\n";
        continue;
      }
      DimSpec ds = DimSpec::parseFromArrayAttr(leaf);
      if (ds.nodes().empty())
        os << "<empty / unparseable>";
      else
        os << ds.toString();
      os << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// --hip-dump-dim-specs
//===----------------------------------------------------------------------===//

class DumpDimSpecsPass
    : public impl::DumpDimSpecsPassBase<DumpDimSpecsPass> {
public:
  using impl::DumpDimSpecsPassBase<DumpDimSpecsPass>::DumpDimSpecsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &os = llvm::errs();

    os << "=== hip-dump-dim-specs ===\n";

    // 1) Per-op output_dim_specs / element_dim_specs on every HIP op
    //    in the module. Walk all ops (not just `@main_graph`) because
    //    debug / outlined helpers may also carry these attrs.
    os << "[per-op DimSpecs]\n";
    bool any_op_with_dimspec = false;
    module.walk([&](Operation *op) {
      auto opNs = op->getDialect()
                      ? op->getDialect()->getNamespace()
                      : llvm::StringRef("");
      if (opNs != "hip")
        return;
      auto output_dim_specs = op->getAttrOfType<::mlir::ArrayAttr>(
          kPerOpOutputDimSpecsAttr);
      auto element_dim_specs = op->getAttrOfType<::mlir::ArrayAttr>(
          kHipShapeElementDimSpecsAttr);
      if ((!output_dim_specs || output_dim_specs.empty()) &&
          (!element_dim_specs || element_dim_specs.empty())) {
        return;
      }
      any_op_with_dimspec = true;
      os << "  " << op->getName() << " @ " << op->getLoc() << "\n";
      dumpPerOpArrayAttr(os, "output_dim_specs[result][dim]",
                         output_dim_specs);
      dumpPerOpArrayAttr(os, "element_dim_specs[result][elem]",
                         element_dim_specs);
    });
    if (!any_op_with_dimspec) {
      os << "  (no HIP op carries output_dim_specs / element_dim_specs)\n";
    }

    // 2) Module-level composed DimSpecs produced by --hip-compose-dim-specs.
    os << "[module-level hipdnn.output_dim_specs]\n";
    auto modAttr = module->getAttrOfType<::mlir::ArrayAttr>(
        kModuleOutputDimSpecsAttr);
    if (!modAttr) {
      os << "  <attribute not present — has --hip-compose-dim-specs run?>\n";
    } else {
      DimSpec::printOutputDimSpecsAttr(modAttr, os);
    }

    // 3) Slot count summary.
    if (auto slotsAttr = module->getAttrOfType<::mlir::IntegerAttr>(
            kModuleDynDimSlotsCountAttr)) {
      os << "[hipdnn.dyn_dim_slots_count] = " << slotsAttr.getInt() << "\n";
    } else {
      os << "[hipdnn.dyn_dim_slots_count] <not set>\n";
    }
  }
};

//===----------------------------------------------------------------------===//
// --hip-verify-dim-specs
//===----------------------------------------------------------------------===//

class VerifyDimSpecsPass
    : public impl::VerifyDimSpecsPassBase<VerifyDimSpecsPass> {
public:
  using impl::VerifyDimSpecsPassBase<VerifyDimSpecsPass>::VerifyDimSpecsPassBase;

  // Collect every slot_id referenced inside `ds`. Recurses through binary
  // nodes; visits each node exactly once via a flat node-array sweep.
  static void collectSlotIds(const DimSpec &ds,
                             llvm::SmallVectorImpl<int32_t> &out) {
    for (const auto &n : ds.nodes()) {
      if (n.kind == DimSpecKind::RuntimeSlot)
        out.push_back(n.slot_id);
    }
  }

  // Verify a single tree, emitting one diagnostic per failure to
  // `errStream` and returning the cumulative pass/fail.
  bool verifyOneTree(const DimSpec &ds, llvm::StringRef context,
                     int32_t dyn_slot_bound,
                     llvm::raw_ostream &errStream) {
    if (ds.nodes().empty())
      return true; // "no spec" is signalled separately by callers
    std::string err;
    if (!ds.verify(err)) {
      errStream << context
                << ": malformed DimSpec tree (DimSpec::verify failed): "
                << err << "\n";
      return false;
    }
    llvm::SmallVector<int32_t, 4> slots;
    collectSlotIds(ds, slots);
    bool ok = true;
    for (int32_t s : slots) {
      if (s < 0) {
        errStream << context << ": RuntimeSlot leaf has slot_id=" << s
                  << " (must be >= 0)\n";
        ok = false;
      } else if (dyn_slot_bound >= 0 && s >= dyn_slot_bound) {
        errStream << context << ": RuntimeSlot leaf has slot_id=" << s
                  << " but hipdnn.dyn_dim_slots_count=" << dyn_slot_bound
                  << " (must be < bound)\n";
        ok = false;
      }
    }
    return ok;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &os = llvm::errs();
    bool ok = true;

    int32_t dyn_slot_bound = -1;
    if (auto slotsAttr = module->getAttrOfType<::mlir::IntegerAttr>(
            kModuleDynDimSlotsCountAttr)) {
      dyn_slot_bound = static_cast<int32_t>(slotsAttr.getInt());
    }

    // 1) Per-op DimSpec attrs must each be structurally well-formed.
    //    We iterate at the op level so the diagnostic carries the op
    //    location.
    module.walk([&](Operation *op) {
      auto ns = op->getDialect()
                    ? op->getDialect()->getNamespace()
                    : llvm::StringRef("");
      if (ns != "hip")
        return;
      auto checkArrayAttr = [&](::mlir::ArrayAttr a, llvm::StringRef label) {
        if (!a)
          return;
        for (size_t i = 0; i < a.size(); ++i) {
          auto inner = llvm::dyn_cast<::mlir::ArrayAttr>(a[i]);
          if (!inner) {
            os << op->getName() << " @ " << op->getLoc() << ": " << label
               << "[" << i << "] is not an ArrayAttr\n";
            ok = false;
            continue;
          }
          for (size_t j = 0; j < inner.size(); ++j) {
            auto leaf = llvm::dyn_cast<::mlir::ArrayAttr>(inner[j]);
            if (!leaf) {
              os << op->getName() << " @ " << op->getLoc() << ": " << label
                 << "[" << i << "][" << j << "] is not an ArrayAttr\n";
              ok = false;
              continue;
            }
            DimSpec ds = DimSpec::parseFromArrayAttr(leaf);
            std::string ctx;
            llvm::raw_string_ostream rs(ctx);
            rs << op->getName() << " @ " << op->getLoc() << " " << label
               << "[" << i << "][" << j << "]";
            rs.flush();
            ok &= verifyOneTree(ds, ctx, dyn_slot_bound, os);
          }
        }
      };
      checkArrayAttr(op->getAttrOfType<::mlir::ArrayAttr>(
                         kPerOpOutputDimSpecsAttr),
                     kPerOpOutputDimSpecsAttr);
      checkArrayAttr(op->getAttrOfType<::mlir::ArrayAttr>(
                         kHipShapeElementDimSpecsAttr),
                     kHipShapeElementDimSpecsAttr);
    });

    // 2) Module-level composed attr (the EP / GenerateInterface consumer).
    //    Cross-check three invariants against `hipdnn.output_shapes` (the
    //    pre-bufferization snapshot of each model output's rank+shape,
    //    attached by OnnxToHip and consumed by ComposeDimSpecs after
    //    bufferize-to-out-params has erased the original FuncType results):
    //      (a) outer rank == #outputs
    //      (b) per-output inner rank == that output's rank
    //      (c) every dynamic ('?'/-1) output dim has a non-empty DimSpec
    //          entry (otherwise the EP falls back to "-1 unknown" silently)
    auto modAttr = module->getAttrOfType<::mlir::ArrayAttr>(
        kModuleOutputDimSpecsAttr);
    auto outputShapesAttr =
        module->getAttrOfType<::mlir::ArrayAttr>("hipdnn.output_shapes");
    if (modAttr && outputShapesAttr) {
      if (modAttr.size() != outputShapesAttr.size()) {
        os << "hipdnn.output_dim_specs outer size " << modAttr.size()
           << " does not match hipdnn.output_shapes size "
           << outputShapesAttr.size() << "\n";
        ok = false;
      } else {
        auto parsed = DimSpec::parseOutputDimSpecsAttr(modAttr);
        for (size_t i = 0; i < parsed.size(); ++i) {
          auto shapeArr = llvm::dyn_cast<::mlir::DenseI64ArrayAttr>(
              outputShapesAttr[i]);
          if (!shapeArr) {
            os << "hipdnn.output_shapes[" << i
               << "] is not a DenseI64ArrayAttr\n";
            ok = false;
            continue;
          }
          int64_t rank = shapeArr.size();
          const auto &dimSpecs = parsed[i];
          // Empty inner vector is the "ComposeDimSpecs did not populate
          // this output" convention. Allowed when the output has no
          // dynamic dims (all shapeArr entries >= 0).
          if (dimSpecs.empty()) {
            for (int64_t d = 0; d < rank; ++d) {
              if (shapeArr[d] < 0) {
                os << "Output[" << i << "] dim[" << d
                   << "] is dynamic but hipdnn.output_dim_specs[" << i
                   << "] is empty\n";
                ok = false;
              }
            }
            continue;
          }
          if ((int64_t)dimSpecs.size() != rank) {
            os << "hipdnn.output_dim_specs[" << i << "] size "
               << dimSpecs.size()
               << " does not match Output[" << i << "] rank " << rank
               << "\n";
            ok = false;
            continue;
          }
          for (int64_t d = 0; d < rank; ++d) {
            std::string ctx;
            llvm::raw_string_ostream rs(ctx);
            rs << "hipdnn.output_dim_specs[" << i << "] dim[" << d << "]";
            rs.flush();
            ok &= verifyOneTree(dimSpecs[d], ctx, dyn_slot_bound, os);
            if (shapeArr[d] < 0 && dimSpecs[d].nodes().empty()) {
              os << "Output[" << i << "] dim[" << d
                 << "] is dynamic but its DimSpec is empty (composition "
                    "failed silently)\n";
              ok = false;
            }
          }
        }
      }
    }

    // 3) Phase 3 invariants:
    //    (i)  Every `hipdnn.output_slot_ids` grid `-1` entry must
    //         correspond to a non-dynamic dim of the op's result;
    //         every non-`-1` entry must correspond to a dynamic dim.
    //         A `-1` on a dynamic dim means the publisher forgot to
    //         reserve it; a non-`-1` on a static dim is a slot leak.
    //    (ii) Output-bound slots (slot ids referenced by
    //         `hipdnn.output_dim_specs` -- those live PAST stream-sync)
    //         may not share their slot id with another op's publisher
    //         (would silently overwrite the value the EP reads after
    //         sync).
    DenseSet<int32_t> outputBoundSlots;
    if (modAttr) {
      auto parsed = DimSpec::parseOutputDimSpecsAttr(modAttr);
      for (const auto &perOut : parsed) {
        for (const auto &ds : perOut) {
          for (int32_t s : ds.collectSlotIds())
            outputBoundSlots.insert(s);
        }
      }
    }
    DenseMap<int32_t, Operation *> firstPublisherOfSlot;
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (mainFunc) {
      mainFunc.walk([&](Operation *op) {
        auto ns = op->getDialect()
                      ? op->getDialect()->getNamespace()
                      : llvm::StringRef("");
        if (ns != "hip")
          return;
        // (i) `hipdnn.output_slot_ids` grid vs. dynamic-dim mask.
        if (auto arr = op->getAttrOfType<::mlir::ArrayAttr>(
                "hipdnn.output_slot_ids")) {
          for (unsigned r = 0; r < arr.size() && r < op->getNumResults();
               ++r) {
            auto perResult = llvm::dyn_cast<::mlir::DenseI32ArrayAttr>(arr[r]);
            if (!perResult)
              continue;
            auto rt = llvm::dyn_cast<ShapedType>(op->getResult(r).getType());
            if (!rt || !rt.hasRank())
              continue;
            for (int64_t d = 0; d < (int64_t)perResult.size() && d < rt.getRank();
                 ++d) {
              int32_t s = perResult.asArrayRef()[d];
              bool isDyn = rt.isDynamicDim(d);
              if (s >= 0 && !isDyn) {
                os << op->getName() << " @ " << op->getLoc()
                   << ": hipdnn.output_slot_ids[" << r << "][" << d
                   << "]=" << s << " on a STATIC dim (slot leak)\n";
                ok = false;
              }
              // A -1 on a dynamic dim is allowed when the dim resolves
              // entirely Cat-A / Cat-B (no slot needed). The reservation
              // pass only assigns slots when the DimSpec contains a
              // RuntimeSlot leaf, so a -1 on a dynamic dim is a legitimate
              // "no slot needed" marker.
            }
          }
        }
        // (ii) Slot-publisher uniqueness via the legacy attrs as well.
        auto recordPublisher = [&](int32_t s, Operation *op) {
          if (s < 0)
            return;
          auto it = firstPublisherOfSlot.find(s);
          if (it == firstPublisherOfSlot.end()) {
            firstPublisherOfSlot[s] = op;
            return;
          }
          if (it->second == op)
            return;
          // Two distinct publisher ops sharing a slot id -- legal only
          // when the slot is NOT output-bound (intermediate coalescing).
          if (outputBoundSlots.count(s)) {
            os << op->getName() << " @ " << op->getLoc()
               << ": output-bound slot " << s
               << " is also published by " << it->second->getName() << " @ "
               << it->second->getLoc()
               << " -- output-bound slots cannot be coalesced\n";
            ok = false;
          }
        };
        if (auto a = op->getAttrOfType<::mlir::IntegerAttr>("slot_id"))
          recordPublisher((int32_t)a.getInt(), op);
        if (auto a = op->getAttrOfType<::mlir::DenseI32ArrayAttr>("slot_ids"))
          for (int32_t s : a.asArrayRef())
            recordPublisher(s, op);
        if (auto arr = op->getAttrOfType<::mlir::ArrayAttr>(
                "hipdnn.output_slot_ids")) {
          for (Attribute resA : arr) {
            if (auto pr = llvm::dyn_cast<::mlir::DenseI32ArrayAttr>(resA))
              for (int32_t s : pr.asArrayRef())
                recordPublisher(s, op);
          }
        }
      });
    }

    if (!ok) {
      signalPassFailure();
      return;
    }
    os << "hip-verify-dim-specs: OK\n";
  }
};

} // namespace
} // namespace hip
} // namespace mlir
