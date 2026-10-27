// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-shapes --split-input-file --verify-diagnostics %s

// The tool-only fixture returns successful but contract-invalid shape lists.
// Explicit checks must reject them in release builds without relying on assert.
// The fixture is destination-style, with one destination per result.

func.func @wrong_result_count(%e: tensor<?xf32>) {
  // expected-error @+1 {{'hip.test_malformed_reify_dps' op --hip-infer-shapes: successful reification returned 0 shape vector(s) for 1 result(s); expected exactly one vector per result}}
  %0 = "hip.test_malformed_reify_dps"(%e) {kind = "result_count"} : (tensor<?xf32>) -> tensor<?xf32>
  return
}

// -----

func.func @wrong_rank(%e: tensor<?x?xf32>) {
  // expected-error @+1 {{'hip.test_malformed_reify_dps' op --hip-infer-shapes: successful reification returned 1 dimension(s) for result #0 of rank 2}}
  %0 = "hip.test_malformed_reify_dps"(%e) {kind = "rank"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return
}

// -----

func.func @static_contradiction(%e: tensor<2x?xf32>) {
  // expected-error @+1 {{'hip.test_malformed_reify_dps' op --hip-infer-shapes: successful reification returned extent 3 for result #0, dimension #0, contradicting existing static extent 2}}
  %0 = "hip.test_malformed_reify_dps"(%e) {kind = "static_contradiction"} : (tensor<2x?xf32>) -> tensor<2x?xf32>
  return
}

// -----

func.func @negative_extent(%e: tensor<?xf32>) {
  // expected-error @+1 {{'hip.test_malformed_reify_dps' op --hip-infer-shapes: successful reification returned negative extent -2 for result #0, dimension #0}}
  %0 = "hip.test_malformed_reify_dps"(%e) {kind = "negative_extent"} : (tensor<?xf32>) -> tensor<?xf32>
  return
}
