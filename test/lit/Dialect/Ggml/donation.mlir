// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: ggml-analyses --donation --split-input-file --verify-diagnostics %s | FileCheck %s

// Independent updates and a direct state result permit donation.
// CHECK-LABEL: module attributes {test.case = "eligible"}
// CHECK: input_donation_eligible = true
// CHECK-SAME: storage_id = 42 : i64
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "eligible"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> tensor<1x1x4x8xf16, #cache>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64, storage_id = 42 : i64}]} {
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %written : tensor<1x1x4x8xf16, #cache>
}
}

// -----

// A later read of the old state prevents donation.
// CHECK-LABEL: module attributes {test.case = "late_old_read"}
// CHECK: input_donation_eligible = false
// CHECK: state 'state.0' is not donatable: a later operation reads the state's old bytes
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "late_old_read"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> tensor<1x1x4x8xf16, #cache>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64}]} {
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %old = "ggml.add"(%state, %state) : (tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %written : tensor<1x1x4x8xf16, #cache>
}
}

// -----

// An alias created before the write still names the old bytes.
// CHECK-LABEL: module attributes {test.case = "late_old_alias"}
// CHECK: input_donation_eligible = false
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "late_old_alias"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> tensor<1x1x4x8xf16, #cache>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64}]} {
  %alias = "ggml.reshape"(%state) : (tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %old = "ggml.add"(%alias, %alias) : (tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %written : tensor<1x1x4x8xf16, #cache>
}
}

// -----

// A later view of the replacement is data-dependent on the write.
// CHECK-LABEL: module attributes {test.case = "replacement_alias"}
// CHECK: input_donation_eligible = true
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "replacement_alias"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> tensor<1x1x4x8xf16, #cache>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64}]} {
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %alias = "ggml.reshape"(%written) : (tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %new = "ggml.add"(%alias, %alias) : (tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %written : tensor<1x1x4x8xf16, #cache>
}
}

// -----

// A same-shaped independent result cannot donate this state's buffer.
// CHECK-LABEL: module attributes {test.case = "unrelated_result"}
// CHECK: input_donation_eligible = false
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "unrelated_result"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> tensor<1x1x4x8xf16, #cache>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64}]} {
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  %copy = "ggml.add"(%written, %written) : (tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %copy : tensor<1x1x4x8xf16, #cache>
}
}

// -----

// Two public results must not claim ownership of the same storage.
// CHECK-LABEL: module attributes {test.case = "duplicate_results"}
// CHECK: input_donation_eligible = false
#cache = #ggml.layout<ne = [8, 4, 1, 1], nb = [2, 16, 64, 64], offset = 0>
#updates = #ggml.layout<ne = [8, 1, 1, 1], nb = [4, 32, 32, 32], offset = 0>
#indices = #ggml.layout<ne = [1, 1, 1, 1], nb = [8, 8, 8, 8], offset = 0>
module attributes {test.case = "duplicate_results"} {
func.func @main_graph(%state: tensor<1x1x4x8xf16, #cache>, %updates: tensor<1x1x1x8xf32, #updates>,
                      %indices: tensor<1x1x1x1xi64, #indices>) -> (tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>)
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
      output_index = 0 : i64}]} {
  %written = "ggml.set_rows"(%updates, %indices, %state)
      : (tensor<1x1x1x8xf32, #updates>, tensor<1x1x1x1xi64, #indices>, tensor<1x1x4x8xf16, #cache>) -> tensor<1x1x4x8xf16, #cache>
  return %written, %written : tensor<1x1x4x8xf16, #cache>, tensor<1x1x4x8xf16, #cache>
}
}

// -----

// No state metadata is legal and is not replaced with an invented contract.
// CHECK-LABEL: module attributes {test.case = "no_pairs"}
// CHECK-NOT: ggml.state_pairs
module attributes {test.case = "no_pairs"} {
func.func @main_graph() { return }
}

// -----

// An explicitly empty list also requires no donation decisions.
// CHECK-LABEL: module attributes {test.case = "empty_pairs"}
// CHECK-NOT: ggml.state_pairs
module attributes {test.case = "empty_pairs"} {
func.func @main_graph() attributes {ggml.state_pairs = []} { return }
}

// -----

// A declaration with state pairs cannot be used as an executable donation proof.
// expected-error@+1 {{state donation requires a single-block body}}
func.func private @main_graph(tensor<1xf16>) -> tensor<1xf16>
    attributes {ggml.state_pairs = [{name = "state.0", input_index = 0 : i64,
                                    output_index = 0 : i64}]}
