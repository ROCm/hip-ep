// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --hipsr-generate-interface | FileCheck %s

// Two expanded memrefs become a packed wrapper and runtime trampolines.
// CHECK-LABEL: module attributes {hip.constants_file = "weights.bin", hipdnn.constant_offsets = array<i64: 0>, hipdnn.constant_sizes = array<i64: 16>, hipdnn.input_ranks = array<i64: 1, 2>} {
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_json(
// CHECK-NEXT: llvm.func @hipdnn_ep_state_cleanup(!llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__hipsr_input_ranks(dense<[1, 2]> : tensor<2xi64>) {addr_space = 0 : i32} : !llvm.array<2 x i64>
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_compute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_blob(
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.func private @main_graph(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_2]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_2]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_11:.*]] = llvm.extractvalue %[[VAL_10]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_12:.*]] = llvm.extractvalue %[[VAL_10]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_10]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_10]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_10]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_17:.*]] = llvm.extractvalue %[[VAL_10]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_18:.*]] = llvm.call @main_graph_internal(%arg0, %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[VAL_19:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL_19]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func private @main_graph_internal(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: llvm.return %[[VAL_0]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(168 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_init(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %arg2, %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_compute(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__hipsr_input_ranks : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.addressof @main_graph : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_compute(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_cleanup(%arg0: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.call @hipdnn_ep_state_cleanup(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_get_metadata_json() -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_json : !llvm.ptr
// CHECK-NEXT: llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }

module attributes {
  hip.constants_file = "weights.bin",
  hipdnn.constant_sizes = array<i64: 16>,
  hipdnn.constant_offsets = array<i64: 0>,
  hipdnn.input_ranks = array<i64: 1, 2>
} {
  llvm.func @main_graph(
      %state: !llvm.ptr,
      %a_alloc: !llvm.ptr<1>, %a_aligned: !llvm.ptr<1>, %a_offset: i64,
      %a_size: i64, %a_stride: i64,
      %b_alloc: !llvm.ptr<1>, %b_aligned: !llvm.ptr<1>, %b_offset: i64,
      %b_size0: i64, %b_size1: i64, %b_stride0: i64, %b_stride1: i64)
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {
    %result = llvm.mlir.poison
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.return %result
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  }
}

// -----

// Op-state slots pass the init function instead of a null pointer.
// CHECK-LABEL: module attributes {hipdnn.input_ranks = array<i64: 1>, hipdnn.num_op_state_slots = 1 : i32} {
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_json(
// CHECK-NEXT: llvm.func @hipdnn_ep_state_cleanup(!llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__hipsr_input_ranks(dense<1> : tensor<1xi64>) {addr_space = 0 : i32} : !llvm.array<1 x i64>
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_compute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @__metadata_blob(
// CHECK-NEXT: llvm.func @hipdnn_ep_inference_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.func @hipdnn_ep_op_states_init_fn(%arg0: !llvm.ptr) -> i32 {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL_0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func private @main_graph(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_2]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_2]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_2]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_2]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_8:.*]] = llvm.call @main_graph_internal(%arg0, %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %[[VAL_9]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func private @main_graph_internal(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64, %arg5: i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.return %[[VAL_0]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_init(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(104 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.addressof @hipdnn_ep_op_states_init_fn : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_init(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %arg2, %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_compute(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__hipsr_input_ranks : !llvm.ptr
// CHECK-NEXT: %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[VAL_2:.*]] = llvm.mlir.addressof @main_graph : !llvm.ptr
// CHECK-NEXT: %[[VAL_3:.*]] = llvm.call @hipdnn_ep_inference_compute(%arg0, %arg1, %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_3]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_cleanup(%arg0: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.call @hipdnn_ep_state_cleanup(%arg0) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.return %[[VAL_0]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inference_get_metadata_json() -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT: %[[VAL_0:.*]] = llvm.mlir.addressof @__metadata_json : !llvm.ptr
// CHECK-NEXT: llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }

module attributes {
  hipdnn.num_op_state_slots = 1 : i32,
  hipdnn.input_ranks = array<i64: 1>
} {
  llvm.func @hipdnn_ep_op_states_init_fn(%state: !llvm.ptr) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
  llvm.func @main_graph(
      %state: !llvm.ptr,
      %a_alloc: !llvm.ptr<1>, %a_aligned: !llvm.ptr<1>, %a_offset: i64,
      %a_size: i64, %a_stride: i64)
      -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> {
    %result = llvm.mlir.poison
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %result
        : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  }
}
