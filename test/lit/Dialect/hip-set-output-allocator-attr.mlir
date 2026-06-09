// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck test for --hip-set-output-allocator-attr.
//
// The pass stamps the `hipdnn.output_allocator` unit attribute on the module --
// the single source of truth that convert-hip-to-llvm + generate-interface read
// to select the allocator ABI. It ONLY adds the attribute; it does not rewrite
// any function body (the returned-alloc -> hip.alloc_output rewrite is
// hip-use-output-allocator's job). Here the returned `memref.alloc` must survive
// untouched, proving the two passes are cleanly separated.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-set-output-allocator-attr %s 2>&1 | FileCheck %s

// CHECK: module attributes {hipdnn.output_allocator}
// CHECK-LABEL: func.func @main_graph
// CHECK: memref.alloc
// CHECK-NOT: hip.alloc_output
// CHECK: return
module {
  func.func @main_graph(%ctx: !hip.context, %m: index) -> memref<?xf16> {
    %out = memref.alloc(%m) : memref<?xf16>
    return %out : memref<?xf16>
  }
}
