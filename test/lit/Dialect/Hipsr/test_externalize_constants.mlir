// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Tests -hipsr-externalize-constants (Phase 2). The pass assigns each opted-in
// hipsr.constant a 64-byte aligned cumulative constants-file offset and stamps
// offset/size on the op, append-only (value/source are kept). No FileSystem is
// injected in these runs, so the constants-file write (Phase 2) is skipped and the
// pass is a pure IR transform -- which is what lets these cases be checked
// without any on-disk file.
//
// One case per distinct behavior (no source-kind duplication: file_source and
// mem_source produce the same observable IR, so only file_source is checked):
//   1. inline value        -> offset/size added, value kept
//   2. file_source         -> offset/size added, source kept (size from attr)
//   3. two constants       -> offsets are cumulative and 64-byte aligned
//   4. externalize = false -> opted out, untouched
//   5. already externalized -> skipped, not re-stamped (idempotent)

// RUN: hip-mlir-opt --hipsr-externalize-constants %s -split-input-file | FileCheck %s

// -----
// Inline value: byte size (4 x f32 = 16) recorded at offset 0, value retained.

// CHECK-LABEL: func.func @inline_value
// CHECK: hipsr.constant {offset = 0 : i64, size = 16 : i64, value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : tensor<4xf32>
func.func @inline_value() -> tensor<4xf32> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// File-backed source: size is taken from the attr (the file is never read),
// and the source attr is preserved alongside the new offset/size.

// CHECK-LABEL: func.func @file_source
// CHECK: hipsr.constant {offset = 0 : i64, size = 1000 : i64, source = #hipsr.file_source<"w.bin", 100, 1000>} : tensor<100xf32>
func.func @file_source() -> tensor<100xf32> {
  %0 = hipsr.constant {source = #hipsr.file_source<"w.bin", 100, 1000>} : tensor<100xf32>
  return %0 : tensor<100xf32>
}

// -----
// Cumulative placement: the first constant occupies [0, 1000); the second is
// padded up to the next 64-byte boundary (1024), proving alignment + running
// offset across constants.

// CHECK-LABEL: func.func @cumulative_alignment
// CHECK: hipsr.constant {offset = 0 : i64, size = 1000 : i64, source = #hipsr.file_source<"w.bin", 0, 1000>} : tensor<250xf32>
// CHECK: hipsr.constant {offset = 1024 : i64, size = 8 : i64, value = dense<[5.000000e+00, 6.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
func.func @cumulative_alignment() -> (tensor<250xf32>, tensor<2xf32>) {
  %0 = hipsr.constant {source = #hipsr.file_source<"w.bin", 0, 1000>} : tensor<250xf32>
  %1 = hipsr.constant {value = dense<[5.0, 6.0]> : tensor<2xf32>} : tensor<2xf32>
  return %0, %1 : tensor<250xf32>, tensor<2xf32>
}

// -----
// Module-scoped: the pass runs on the whole module, so offsets accumulate
// across functions (the second func's constant lands after the first, aligned)
// rather than restarting at 0 per function.

// CHECK-LABEL: func.func @first
// CHECK: hipsr.constant {offset = 0 : i64, size = 1000 : i64, source = #hipsr.file_source<"w.bin", 0, 1000>} : tensor<250xf32>
// CHECK-LABEL: func.func @second
// CHECK: hipsr.constant {offset = 1024 : i64, size = 8 : i64, value = dense<[7.000000e+00, 8.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
func.func @first() -> tensor<250xf32> {
  %0 = hipsr.constant {source = #hipsr.file_source<"w.bin", 0, 1000>} : tensor<250xf32>
  return %0 : tensor<250xf32>
}
func.func @second() -> tensor<2xf32> {
  %0 = hipsr.constant {value = dense<[7.0, 8.0]> : tensor<2xf32>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----
// Opt-out: externalize = false is left exactly as-is (no offset/size added).

// CHECK-LABEL: func.func @opt_out
// CHECK: hipsr.constant {externalize = false, value = dense<{{.*}}> : tensor<4xf32>} : tensor<4xf32>
// CHECK-NOT: offset
func.func @opt_out() -> tensor<4xf32> {
  %0 = hipsr.constant {externalize = false, value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Idempotent: an already-externalized constant (offset present) is skipped, so
// its pre-existing offset survives instead of being recomputed to 0.

// CHECK-LABEL: func.func @already_externalized
// CHECK: hipsr.constant {offset = 999 : i64, size = 16 : i64, value = dense<{{.*}}> : tensor<4xf32>} : tensor<4xf32>
func.func @already_externalized() -> tensor<4xf32> {
  %0 = hipsr.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>, offset = 999 : i64, size = 16 : i64} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
