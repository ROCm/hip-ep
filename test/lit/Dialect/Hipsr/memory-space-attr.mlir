// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests that #hipsr.mem<...> parses and prints, and that a bad kind errors.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----
// All four kinds print back unchanged as a memref memory space.
// CHECK-LABEL: func.func @roundtrip_spaces
// CHECK-SAME:    memref<4xf32, #hipsr.mem<device>>
// CHECK-SAME:    memref<4xf32, #hipsr.mem<host>>
// CHECK-SAME:    memref<4xf32, #hipsr.mem<pinned>>
// CHECK-SAME:    memref<4xf32, #hipsr.mem<managed>>
func.func @roundtrip_spaces(%d: memref<4xf32, #hipsr.mem<device>>,
                            %h: memref<4xf32, #hipsr.mem<host>>,
                            %p: memref<4xf32, #hipsr.mem<pinned>>,
                            %m: memref<4xf32, #hipsr.mem<managed>>) {
  return
}

// -----
// Also works as a plain attribute on an op, not just as a memref space.
// CHECK-LABEL: func.func @attr_on_op
// CHECK-SAME:    hipsr.space = #hipsr.mem<pinned>
func.func @attr_on_op() attributes {hipsr.space = #hipsr.mem<pinned>} {
  return
}

// -----
// An unknown kind fails to parse.
// expected-error @+2 {{to be one of: host, device, pinned, managed}}
// expected-error @+1 {{failed to parse Hipsr_MemorySpaceAttr parameter 'kind'}}
func.func @bad_kind(%x: memref<4xf32, #hipsr.mem<bogus>>) {
  return
}
