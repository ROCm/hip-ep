/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DedupDPSInits.cpp - liveness-safe tensor.empty DPS dedup -----------===//
//
// Merges `tensor.empty` ops that feed DPS `outs` operands when the buffers
// they will become have non-overlapping live ranges, AND have identical
// dynamic-size operands (so the shared buffer is provably the same size).
//
// Each HIP DPS op (`hip.gemm`, `hip.matmul`, `hip.slice`, …) takes a
// `tensor.empty` as its destination-passing-style init. `tensor.empty` is
// side-effect-free, so a stock CSE merges ALL same-typed empties — after
// bufferize the DPS ops share one buffer and clobber each other when
// simultaneously live (Whisper encoder: cosine ~0.6). Removing CSE entirely
// is safe for correctness but prevents buffer reuse across transformer
// layers, so the GPU activation pool grows several-fold on deep models and
// by more than an order of magnitude on a VLM vision tower, where each
// layer's attention buffers land in their own pool domain with no
// cross-layer reuse (gemma3 vision, 1 image: ~30 GB without this dedup vs
// ~1.2 GB with — matching the CSE-on baseline).
//
// This pass replaces that unsafe CSE with two independent safety conditions,
// BOTH of which must hold before two empties are merged:
//
//   (1) Live ranges must not overlap. The live range of an empty's buffer
//       runs from the empty to the LAST op that reads or writes the buffer
//       *or any value that aliases it* after bufferize. Aliases are produced
//       by (a) a DPS op writing the buffer in place (its result aliases the
//       `outs` init) and (b) view ops (`tensor.expand_shape`/`collapse_shape`/
//       `cast`/`extract_slice`, anything `ViewLikeOpInterface`). A naive
//       "last direct user of the DPS result" underestimates this — a result
//       consumed by an `insert_slice`/view whose own result is read much
//       later keeps the buffer live until that later read — and merging on
//       the underestimate clobbers a still-live buffer (gemma3 vision: a
//       slice's `axes` staging buffer was reused while live → garbage axis
//       value → runtime abort). We therefore walk the buffer's alias set
//       transitively and take the max index over all of its users.
//
//   (2) Dynamic-size operands must be PROVABLY EQUAL. Grouping by result type
//       alone is unsafe for `tensor<?x…>`: two empties of the same static type
//       can have different dynamic-extent operands and thus different runtime
//       sizes. Merging them aliases a consumer onto a buffer that may be too
//       small → out-of-bounds (gemma3 vision: a same-typed dynamic empty was
//       merged onto a differently-sized one → garbage/abort at runtime). SSA
//       identity is too strict: each transformer layer recomputes its extent
//       with a fresh SSA value that is *structurally identical* (same function
//       of the symbolic batch/seq dim). We therefore prove extent-equality with
//       global value numbering (GVN): a single bottom-up pass over the entry
//       block assigns every value an equivalence-class id, equal iff produced
//       by structurally-identical effect-free op chains over the same leaves.
//       Two empties have the same size iff their dynamic-size operands share
//       ids. This is exactly the equivalence a stock CSE folds to (CSE merges
//       ops with identical operands, having folded the shape arithmetic to a
//       fixpoint first) — so we recover CSE's full merging power, but without
//       CSE's unsafe empty-merge and WITH the liveness guard CSE lacks. (A
//       prior depth-limited recursive structural compare was too weak: the
//       per-dim extent of a 4-D `reshape(-1)` is a deep product/divsi tree of
//       `hip.readback_scalar`s — pre-bufferize the readback is effect-free, so
//       GVN/CSE fold it, but a fixed recursion depth exhausts on the tree and
//       wrongly reports "not equal", leaving the buffers unmerged → pool blow-up
//       on a VLM vision tower.)
//
// Before (two non-overlapping, same-size DPS consumers share one empty):
//   %sz = ...                            // dynamic extent, shared SSA value
//   %e0 = tensor.empty(%sz) : tensor<?x8xf16>
//   %a  = hip.matmul ... outs(%e0)       // last read at index 10
//   %e1 = tensor.empty(%sz) : tensor<?x8xf16>
//   %b  = hip.matmul ... outs(%e1)       // first def at index 30
// After: %e1 erased, %b uses %e0 (ranges [.,10] and [30,.] disjoint; same %sz).
//
// Kept separate (overlapping live ranges via a transitive alias):
//   %e0 = tensor.empty() : tensor<2xi64>
//   %s  = hip.slice ... outs(%e0)        // %s aliases %e0
//   %v  = tensor.insert_slice %s into …  // %s read here, but %v read LATER
//   ... %v consumed at index 900 ...     // %e0 live until 900 → no merge
//
// Kept separate (same static type, different dynamic size operand):
//   %e0 = tensor.empty(%p) : tensor<?x8xf16>
//   %e1 = tensor.empty(%q) : tensor<?x8xf16>   // %p != %q → never merged
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

namespace mlir {
namespace hip {
#define GEN_PASS_DEF_DEDUPDPSINITSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"
} // namespace hip
} // namespace mlir

using namespace mlir;

namespace {

struct EmptyInfo {
  tensor::EmptyOp emptyOp;
  unsigned defIndex;
  unsigned lastConsumerIndex;
};

// Global value numbering over a single block. Assigns each SSA value an
// equivalence-class id such that two values share an id iff they are GUARANTEED
// to hold the same runtime value — i.e. they are produced by structurally
// identical effect-free op chains over leaves that are themselves equal
// (same block arg / same constant / same upstream class). Built bottom-up in
// program order (defs precede uses within a block), so it captures arbitrarily
// deep computations without recursion depth limits — matching the equivalence a
// stock CSE folds to. Effectful ops (and their results) get unique ids: their
// value can depend on mutable state, so two textually-distinct ones are not
// assumed equal. NOTE on `hip.readback_scalar`: pre-bufferize its operand is a
// `tensor` and its `getEffects` attaches no effect (the Read only appears once
// the operand is a `memref`), so it is correctly treated as a pure function of
// its tensor operand here — exactly why two readbacks of the same shape slice
// unify, letting per-layer extent arithmetic prove equal.
class BlockValueNumbering {
public:
  explicit BlockValueNumbering(Block &block) {
    for (Operation &op : block) {
      bool pure = isMemoryEffectFree(&op) && op.getNumRegions() == 0;
      if (!pure) {
        for (Value r : op.getResults())
          idOf[r] = nextId++; // opaque: unique, never unifies with anything
        continue;
      }
      // Structural key shared by all results of the op (result index appended
      // per-result below): op name, attribute dictionary, operand class ids,
      // and result types. Two ops with identical keys compute identical values.
      std::string base;
      llvm::raw_string_ostream os(base);
      os << op.getName().getStringRef() << '|';
      op.getAttrDictionary().print(os);
      os << '|';
      for (Value o : op.getOperands())
        os << leafOrClassId(o) << ',';
      os << '|';
      for (Type t : op.getResultTypes()) {
        t.print(os);
        os << ',';
      }
      os.flush();
      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        std::string key = base + "|r" + std::to_string(i);
        auto [it, inserted] = classOf.try_emplace(key, nextId);
        if (inserted)
          ++nextId;
        idOf[res] = it->second;
      }
    }
  }

  // Two values provably equal at runtime?
  bool equal(Value a, Value b) {
    return a == b || leafOrClassId(a) == leafOrClassId(b);
  }

private:
  // Class id for a value, assigning a fresh unique id to leaves not produced by
  // a pure block op (function/block args, results computed before this block).
  unsigned leafOrClassId(Value v) {
    auto it = idOf.find(v);
    if (it != idOf.end())
      return it->second;
    unsigned id = nextId++;
    idOf[v] = id;
    return id;
  }

  llvm::DenseMap<Value, unsigned> idOf;
  std::map<std::string, unsigned> classOf;
  unsigned nextId = 0;
};

struct DedupDPSInitsPass
    : public hip::impl::DedupDPSInitsPassBase<DedupDPSInitsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    Block *block = &funcOp.getBody().front();
    if (block->empty())
      return;

    // Global value numbering: lets `sameDynSizes` prove two empties' dynamic
    // extents equal across layers without a recursion-depth limit.
    BlockValueNumbering vn(*block);

    // Build op → sequential-index map over the entry block.
    DenseMap<Operation *, unsigned> opIndex;
    unsigned idx = 0;
    for (Operation &op : *block)
      opIndex[&op] = idx++;
    const unsigned lastIndex = idx - 1;

    // Map any op (possibly nested in a region of an entry-block op) to its
    // enclosing entry-block index. Ops that don't resolve into this block
    // (none expected in a single-block func graph) conservatively extend the
    // live range to the end of the block.
    auto indexOf = [&](Operation *user) -> unsigned {
      Operation *resolved = user;
      if (resolved->getBlock() != block)
        resolved = block->findAncestorOpInBlock(*resolved);
      if (!resolved)
        return lastIndex;
      auto it = opIndex.find(resolved);
      return it != opIndex.end() ? it->second : lastIndex;
    };

    // Last index at which `empty`'s buffer (or any value aliasing it after
    // bufferize) is read or written. Walks the alias set transitively:
    //  - a DPS op using an alias as its `outs` init writes in place; its
    //    tied result aliases the buffer (covers hip.* DPS ops AND
    //    tensor.insert_slice's dest),
    //  - a ViewLikeOpInterface op (expand/collapse/cast/extract_slice) whose
    //    source is an alias produces a result that aliases the buffer.
    // Every other use is a plain read that ends a chain at its own index.
    auto computeLastConsumer = [&](tensor::EmptyOp empty) -> unsigned {
      unsigned last = opIndex[empty.getOperation()];
      SmallVector<Value, 8> worklist{empty.getResult()};
      SmallPtrSet<Value, 8> seen{empty.getResult()};
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        for (OpOperand &use : v.getUses()) {
          Operation *user = use.getOwner();
          last = std::max(last, indexOf(user));

          // DPS init (outs) use → result aliases the buffer.
          if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
            if (dps.isDpsInit(&use)) {
              OpResult tied = dps.getTiedOpResult(&use);
              if (tied && seen.insert(tied).second)
                worklist.push_back(tied);
              continue;
            }
          }
          // View op with `v` as its source → result aliases the buffer.
          if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
            if (view.getViewSource() == v) {
              for (Value res : user->getResults())
                if (seen.insert(res).second)
                  worklist.push_back(res);
            }
          }
        }
      }
      return last;
    };

    // Collect tensor.empty ops that feed at least one DPS init.
    SmallVector<EmptyInfo> empties;
    for (Operation &op : *block) {
      auto emptyOp = dyn_cast<tensor::EmptyOp>(&op);
      if (!emptyOp)
        continue;

      bool feedsDPS = false;
      for (OpOperand &use : emptyOp.getResult().getUses()) {
        auto dstOp = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
        if (dstOp && dstOp.isDpsInit(&use)) {
          feedsDPS = true;
          break;
        }
      }
      if (!feedsDPS)
        continue;

      empties.push_back(
          {emptyOp, opIndex[&op], computeLastConsumer(emptyOp)});
    }

    if (empties.empty())
      return;

    // Two empties may share a buffer only if their (statically known) result
    // types match AND their dynamic-size operands are PROVABLY EQUAL via the
    // GVN above — NOT merely SSA-identical, so a per-layer recomputed extent
    // (a fresh SSA value that is structurally identical to the rep's) still
    // unifies. Grouping by type is necessary but not sufficient for
    // `tensor<?x…>`.
    auto sameDynSizes = [&vn](tensor::EmptyOp a, tensor::EmptyOp b) {
      ValueRange da = a.getDynamicSizes();
      ValueRange db = b.getDynamicSizes();
      if (da.size() != db.size())
        return false;
      for (auto [x, y] : llvm::zip(da, db))
        if (!vn.equal(x, y))
          return false;
      return true;
    };

    // Group by result type (dynamic-operand equality is checked at merge time
    // since same-typed empties can still differ in their `?` extent sources).
    DenseMap<Type, SmallVector<unsigned>> typeGroups;
    for (auto [i, info] : llvm::enumerate(empties))
      typeGroups[info.emptyOp.getType()].push_back(i);

    // Greedy merge within each type group: for each empty (in textual order),
    // reuse the earliest representative whose live range does not overlap and
    // whose dynamic-size operands match. Otherwise it becomes a new rep.
    unsigned mergeCount = 0;
    for (auto &[type, indices] : typeGroups) {
      SmallVector<unsigned> reps; // indices into `empties`

      for (unsigned i : indices) {
        EmptyInfo &info = empties[i];
        bool merged = false;

        for (unsigned repIdx : reps) {
          EmptyInfo &rep = empties[repIdx];
          bool overlap = !(info.lastConsumerIndex < rep.defIndex ||
                           rep.lastConsumerIndex < info.defIndex);
          if (overlap || !sameDynSizes(info.emptyOp, rep.emptyOp))
            continue;

          // The rep is textually earlier, so it (and its dynamic-size
          // operands) dominates this empty's later uses.
          info.emptyOp.getResult().replaceAllUsesWith(rep.emptyOp.getResult());
          info.emptyOp->erase();
          rep.lastConsumerIndex =
              std::max(rep.lastConsumerIndex, info.lastConsumerIndex);
          rep.defIndex = std::min(rep.defIndex, info.defIndex);
          merged = true;
          ++mergeCount;
          break;
        }

        if (!merged)
          reps.push_back(i);
      }
    }

    if (mergeCount > 0)
      llvm::errs() << "[DedupDPSInits] merged " << mergeCount
                   << " tensor.empty ops (liveness-safe)\n";
  }
};

} // namespace
