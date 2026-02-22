/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// HIP Buffer Deallocation Pass
//===----------------------------------------------------------------------===//
// Automatically inserts hip.free operations for locally allocated buffers
// following RAII-style lifetime management.
//
// Key behaviors:
// - Tracks all hip.alloc operations in each function
// - Inserts hip.free before each return in LIFO (stack) order
// - Does NOT free function arguments (caller-owned)
// - Handles multiple return paths correctly
//
// Example transformation:
//   func.func @test(%ctx: !hip.context) -> i32 {
//     %buf1 = hip.alloc(%ctx) : memref<...>
//     %buf2 = hip.alloc(%ctx) : memref<...>
//     // ... use buffers ...
//     return %result : i32
//   }
//
// After pass:
//   func.func @test(%ctx: !hip.context) -> i32 {
//     %buf1 = hip.alloc(%ctx) : memref<...>
//     %buf2 = hip.alloc(%ctx) : memref<...>
//     // ... use buffers ...
//     hip.free(%ctx, %buf2)  // LIFO order
//     hip.free(%ctx, %buf1)
//     return %result : i32
//   }
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "HipPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

class HipBufferDeallocationPass
    : public PassWrapper<HipBufferDeallocationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HipBufferDeallocationPass)

  StringRef getArgument() const final { return "hip-buffer-deallocation"; }

  StringRef getDescription() const final {
    return "Insert hip.free operations for locally allocated buffers in LIFO "
           "order";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Track all allocations in this function (in order)
    SmallVector<AllocOp, 4> allocations;

    // Find all hip.alloc operations
    funcOp.walk([&](AllocOp allocOp) { allocations.push_back(allocOp); });

    // If no allocations, nothing to do
    if (allocations.empty()) {
      return;
    }

    // Insert hip.free operations before each return
    funcOp.walk([&](func::ReturnOp returnOp) {
      OpBuilder builder(returnOp);

      // Free in reverse (LIFO) order
      for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        AllocOp allocOp = *it;
        Value buffer = allocOp.getMemref();
        Value context = allocOp.getHandle();

        // Create hip.free operation
        builder.create<FreeOp>(returnOp.getLoc(), context, buffer);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace hip {

std::unique_ptr<Pass> createHipBufferDeallocationPass() {
  return std::make_unique<HipBufferDeallocationPass>();
}

} // namespace hip
} // namespace mlir
