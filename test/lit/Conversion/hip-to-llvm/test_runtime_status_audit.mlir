// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt --convert-hip-to-llvm %s 2>&1 | FileCheck %s

// A future direct runtime call cannot bypass the shared status policy by
// silently discarding an external i32 result.
// CHECK: unused i32 result from 'unconsumed_status'

module {
  llvm.func @unconsumed_status(!llvm.ptr) -> i32

  llvm.func @main_graph(%state: !llvm.ptr, %inputs: !llvm.ptr) -> i32 {
    %unused = llvm.call @unconsumed_status(%state) : (!llvm.ptr) -> i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
}
