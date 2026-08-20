// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  llvm.func @wrap_already_recorded(!llvm.ptr) -> i32
  llvm.func @hipdnn_ep_state_record_status(!llvm.ptr, i32) -> i32
  llvm.func @read_scalar_i32(!llvm.ptr) -> i32
  llvm.func @hipdnn_ep_loop_frame_destroy(!llvm.ptr) -> i32
  llvm.func @hipdnn_ep_copy_output(!llvm.ptr) -> i32

  // An existing recorder use remains singular.
  // CHECK-LABEL: llvm.func @already_recorded
  // CHECK: %[[STATUS:.*]] = llvm.call @wrap_already_recorded
  // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]])
  // CHECK-NEXT: llvm.return
  llvm.func @already_recorded(%state: !llvm.ptr) {
    %status = llvm.call @wrap_already_recorded(%state) : (!llvm.ptr) -> i32
    %recorded = llvm.call @hipdnn_ep_state_record_status(%state, %status)
        : (!llvm.ptr, i32) -> i32
    llvm.return
  }

  // A consumed i32 data value is not a status and remains untouched.
  // CHECK-LABEL: llvm.func @consumed_data
  // CHECK: %[[VALUE:.*]] = llvm.call @read_scalar_i32
  // CHECK-NEXT: %{{.*}} = llvm.zext %[[VALUE]] : i32 to i64
  llvm.func @consumed_data(%state: !llvm.ptr) -> i64 {
    %value = llvm.call @read_scalar_i32(%state) : (!llvm.ptr) -> i32
    %wide = llvm.zext %value : i32 to i64
    llvm.return %wide : i64
  }

  // Later stack layers add these non-wrap status calls. They are part of the
  // policy before those layers rebase onto this prerequisite.
  // CHECK-LABEL: llvm.func @future_status_calls
  // CHECK: %[[DESTROY:.*]] = llvm.call @hipdnn_ep_loop_frame_destroy
  // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[DESTROY]])
  // CHECK: %[[COPY:.*]] = llvm.call @hipdnn_ep_copy_output
  // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[COPY]])
  llvm.func @future_status_calls(%state: !llvm.ptr) {
    %destroy =
        llvm.call @hipdnn_ep_loop_frame_destroy(%state) : (!llvm.ptr) -> i32
    %copy = llvm.call @hipdnn_ep_copy_output(%state) : (!llvm.ptr) -> i32
    llvm.return
  }
}
