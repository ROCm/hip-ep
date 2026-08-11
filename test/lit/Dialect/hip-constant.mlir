// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --canonicalize %t/valid.mlir | FileCheck %s
// RUN: hip-mlir-opt --verify-diagnostics %t/errors.mlir

// The carrier remains after canonicalization: it is pure but deliberately has
// no fold that can bypass externalization policy.
// CHECK-LABEL: func.func @valid
// CHECK: hip.constant {value = dense<7> : tensor<si8>} : tensor<si8>
// CHECK: hip.constant {value = dense<9> : tensor<ui8>} : tensor<ui8>
//--- valid.mlir
func.func @valid() -> (tensor<si8>, tensor<ui8>) {
  %signed = hip.constant {value = dense<7> : tensor<si8>} : tensor<si8>
  %unsigned = hip.constant {value = dense<9> : tensor<ui8>} : tensor<ui8>
  return %signed, %unsigned : tensor<si8>, tensor<ui8>
}

//--- errors.mlir
func.func @both_sources() -> tensor<2xi8> {
  // expected-error @+1 {{inline source must contain only `value`}}
  %0 = hip.constant {value = dense<[1, 2]> : tensor<2xi8>, location = "/tmp/w", offset = 0 : i64, size = 2 : i64} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @incomplete_source() -> tensor<2xi8> {
  // expected-error @+1 {{external source requires `location`, `offset`, and `size` together}}
  %0 = hip.constant {location = "/tmp/w"} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @missing_source() -> tensor<2xi8> {
  // expected-error @+1 {{requires exactly one source: `value` or complete `location`/`offset`/`size`}}
  %0 = hip.constant : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @value_type_mismatch() -> tensor<2xi8> {
  // expected-error @+1 {{inline `value` type 'tensor<2xi16>' does not match result type 'tensor<2xi8>'}}
  %0 = hip.constant {value = dense<[1, 2]> : tensor<2xi16>} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @size_mismatch() -> tensor<2xi32> {
  // expected-error @+1 {{external source byte size 7 does not match result byte size 8}}
  %0 = hip.constant {location = "/tmp/w", offset = 0 : i64, size = 7 : i64} : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func.func @zero_external_size() -> tensor<0xi8> {
  // expected-error @+1 {{external source `size` must be positive}}
  %0 = hip.constant {location = "/tmp/w", offset = 0 : i64, size = 0 : i64} : tensor<0xi8>
  return %0 : tensor<0xi8>
}

func.func @negative_file_offset() -> tensor<2xi8> {
  // expected-error @+1 {{file source `offset` must be non-negative}}
  %0 = hip.constant {location = "/tmp/w", offset = -1 : i64, size = 2 : i64} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @file_range_overflow() -> tensor<2xi8> {
  // expected-error @+1 {{file source range overflows int64}}
  %0 = hip.constant {location = "/tmp/w", offset = 9223372036854775806 : i64, size = 2 : i64} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @null_memory_address() -> tensor<2xi8> {
  // expected-error @+1 {{memory-address source has null address}}
  %0 = hip.constant {location = "*/_ORT_MEM_ADDR_/*", offset = 0 : i64, size = 2 : i64} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @origin_without_order() -> tensor<2xi8> {
  // expected-error @+1 {{compiler-owned `origin` and `order` must be present together}}
  %0 = hip.constant {origin = "onnx-imported", value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @order_without_origin() -> tensor<2xi8> {
  // expected-error @+1 {{compiler-owned `origin` and `order` must be present together}}
  %0 = hip.constant {order = 0 : i64, value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @unknown_origin() -> tensor<2xi8> {
  // expected-error @+1 {{has unknown compiler-owned origin `plugin`}}
  %0 = hip.constant {order = 0 : i64, origin = "plugin", value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @negative_order() -> tensor<2xi8> {
  // expected-error @+1 {{compiler-owned order must be non-negative}}
  %0 = hip.constant {order = -1 : i64, origin = "onnx-imported", value = dense<[1, 2]> : tensor<2xi8>} : tensor<2xi8>
  return %0 : tensor<2xi8>
}

func.func @dynamic_shape() -> tensor<?xi8> {
  // expected-error @+1 {{requires a statically shaped ranked tensor result}}
  %0 = hip.constant {location = "/tmp/w", offset = 0 : i64, size = 2 : i64} : tensor<?xi8>
  return %0 : tensor<?xi8>
}

func.func @unsupported_type() -> tensor<2xindex> {
  // expected-error @+1 {{has unsupported element type 'index'}}
  %0 = hip.constant {location = "/tmp/w", offset = 0 : i64, size = 16 : i64} : tensor<2xindex>
  return %0 : tensor<2xindex>
}

func.func @shape_overflow() -> tensor<3037000500x3037000500xi8> {
  // expected-error @+1 {{result element count overflows int64}}
  %0 = hip.constant {location = "/tmp/w", offset = 0 : i64, size = 0 : i64} : tensor<3037000500x3037000500xi8>
  return %0 : tensor<3037000500x3037000500xi8>
}
