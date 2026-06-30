// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pins the #hip.mem<...> memory-space attribute: parse/print round-trip of all
// four kinds (as a memref space and as a standalone attribute) and rejection of
// an unknown kind keyword.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Aspect A: every memory-space kind round-trips as a memref space, in custom
// syntax, unchanged.
// CHECK-LABEL: func.func @roundtrip_spaces
// CHECK-SAME:    memref<4xf32, #hip.mem<device>>
// CHECK-SAME:    memref<4xf32, #hip.mem<host>>
// CHECK-SAME:    memref<4xf32, #hip.mem<pinned>>
// CHECK-SAME:    memref<4xf32, #hip.mem<managed>>
func.func @roundtrip_spaces(%d: memref<4xf32, #hip.mem<device>>,
                            %h: memref<4xf32, #hip.mem<host>>,
                            %p: memref<4xf32, #hip.mem<pinned>>,
                            %m: memref<4xf32, #hip.mem<managed>>) {
  return
}

// -----

// Aspect B: the attribute also round-trips as a standalone discardable
// attribute (not only as a memref space).
// CHECK-LABEL: func.func @attr_as_discardable
// CHECK-SAME:    hipdnn.space = #hip.mem<pinned>
func.func @attr_as_discardable() attributes {hipdnn.space = #hip.mem<pinned>} {
  return
}

// -----

// Aspect C: an unknown kind keyword is a parse error (the enum parser and the
// attribute parser each emit a diagnostic).
// expected-error @+2 {{to be one of: device, host, pinned, managed}}
// expected-error @+1 {{failed to parse Hip_MemorySpaceAttr parameter 'kind'}}
func.func @bad_kind(%x: memref<4xf32, #hip.mem<bogus>>) {
  return
}
