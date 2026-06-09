/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SetOutputAllocatorAttr.cpp - mark module as allocator mode ---------===//
//
// Sets the `hipdnn.output_allocator` unit attribute on the module. That
// attribute is the single source of truth for allocator mode -- downstream
// `convert-hip-to-llvm` and `generate-interface` read it to emit the 2-arg
// (state, inputs) ABI instead of the classic 3-arg out-param ABI.
//
// Intentionally tiny and decoupled from the IR rewrite (`hip-use-output-
// allocator`): the allocator pipeline runs BOTH at "slot 4.5", but keeping the
// mode marker in its own pass means it can be deleted in a single step once the
// allocator path is the only path (classic out-params removed) -- the rewrite
// pass needs no change. See docs/design/output-allocator-design.md.
//
// Before:
//   module { func.func @main_graph(...) { ... } }
//
// After:
//   module attributes {hipdnn.output_allocator} {
//     func.func @main_graph(...) { ... }    // bodies untouched
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "hip-set-output-allocator-attr"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SETOUTPUTALLOCATORATTRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct SetOutputAllocatorAttrPass
    : impl::SetOutputAllocatorAttrPassBase<SetOutputAllocatorAttrPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->setAttr("hipdnn.output_allocator", UnitAttr::get(&getContext()));
  }
};

} // namespace

} // namespace hip
} // namespace mlir
