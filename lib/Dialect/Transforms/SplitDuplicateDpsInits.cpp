/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SplitDuplicateDpsInits.cpp - de-alias same-op DPS init operands ---===//
//
// Gives every init (DPS-out) operand of an op its own buffer when CSE has
// collapsed several of their seed `tensor.empty` ops onto one value. Runs
// AFTER CSE and BEFORE one-shot-bufferize.
//
// Why this pass exists
// --------------------
// `tensor.empty` is `Pure`, so CSE freely deduplicates same-typed empties.
// That is harmless -- and desirable -- for single-use scratch: it keeps the
// buffer count, and the downstream `--hip-pool-allocs` domain count, low.
//
// It is WRONG, however, when CSE collapses the distinct seeds of two *init*
// operands of the SAME op onto one value. After bufferization both tied
// results then share one buffer, and an op whose kernel reads back its own
// outputs miscompiles. The canonical victim is `hip.gqa`: its kernel writes
// then reads K from `present_key` and V from `present_value`, so aliasing the
// two makes it read V as K and silently produce wrong attention output.
//
// One-Shot Bufferize does not save us here. Its analysis separates two writing
// inits that alias only when it can find a read-after-write conflict on a tied
// result -- i.e. when the results are LIVE. When the results are dead (e.g. a
// present-KV output that no downstream op consumes), there is no such read, the
// inits stay merged, and the kernel-internal read silently consumes the wrong
// bytes.
//
// Relation to upstream
// --------------------
// Upstream handles the identical hazard structurally, by running
// `bufferization::EmptyTensorToAllocTensor` over EVERY `tensor.empty` before
// bufferize (see IREE `addIREEComprehensiveBufferizePasses` and the LLVM
// sparsifier pipeline). `bufferization.alloc_tensor` is documented to not alias
// any other buffer and is not `Pure`, so it survives CSE and forces a distinct
// allocation. That blanket conversion is correct but also blocks every benign
// empty merge, multiplying the buffer (and pool-domain) count on
// activation-heavy graphs. This pass applies the same `alloc_tensor` mechanism
// surgically -- only to the duplicated same-op inits that actually need it --
// preserving the benign merges and their pool packing.
//
// Safety
// ------
// Only duplicates whose seed is a `tensor.empty` are rewritten. A
// `tensor.empty` has undefined contents, so substituting a fresh uninitialized
// `bufferization.alloc_tensor` is value-preserving. A non-empty duplicate
// init may carry contents the op reads (a read-before-write / accumulate init),
// so it is intentionally left untouched rather than risk dropping live data.
//
// Before (post-CSE; one %e feeds both present_* inits of one gqa):
//
//   %e = tensor.empty() : tensor<AxBxCxDxf16>
//   hip.gqa(...) outs(%out, %e, %e : ...)   // present_key aliases
//   present_value
//
// After:
//
//   %e  = tensor.empty()               : tensor<AxBxCxDxf16>
//   %e2 = bufferization.alloc_tensor() : tensor<AxBxCxDxf16>
//   hip.gqa(...) outs(%out, %e, %e2 : ...)  // distinct buffers
//
// Pipeline placement
// ------------------
// In `buildOnnxToHipPipelineTail`, immediately after the post-shape-inference
// CSE (which produces the duplication this pass repairs) and before
// `one-shot-bufferize` (which would otherwise lock in the shared buffer).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-split-duplicate-dps-inits"

STATISTIC(NumInitsSplit,
          "Number of duplicate same-op DPS init operands re-pointed at a fresh "
          "bufferization.alloc_tensor");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SPLITDUPLICATEDPSINITSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct SplitDuplicateDpsInitsPass
    : public impl::SplitDuplicateDpsInitsPassBase<SplitDuplicateDpsInitsPass> {

  void runOnOperation() override {
    getOperation().walk([](DestinationStyleOpInterface dpsOp) {
      // Init values already seen on THIS op. The first occurrence keeps its
      // original `tensor.empty`; later occurrences of the same value are each
      // re-pointed at their own fresh allocation.
      llvm::SmallPtrSet<Value, 4> seen;
      for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
        // Only `tensor.empty` seeds are the CSE-merge hazard, and only they are
        // safe to replace (undefined contents). Anything else is left intact.
        auto emptyOp = init.get().getDefiningOp<tensor::EmptyOp>();
        if (!emptyOp || seen.insert(init.get()).second)
          continue;

        OpBuilder b(dpsOp);
        auto fresh = bufferization::AllocTensorOp::create(
            b, emptyOp.getLoc(), emptyOp.getType(), emptyOp.getDynamicSizes());
        init.set(fresh);
        ++NumInitsSplit;
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] " << dpsOp->getName() << " init #"
                   << init.getOperandNumber()
                   << ": split shared tensor.empty -> fresh alloc_tensor\n");
      }
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
