// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// A graph input is always device memory, so every host buffer here is built
// inside the function.

// A tensor destination ties the result to `init`, so the result type prints.
// CHECK-LABEL: func.func @copy_d2h_tensor(
// CHECK-SAME:    %[[CTX:[^:]+]]: !hipsr.context,
// CHECK-SAME:    %[[SRC:[^:]+]]: tensor<1xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[INIT:.+]] = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %{{.+}} = hipsr.copy_d2h(%[[CTX]]) ins(%[[SRC]] : tensor<1xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @copy_d2h_tensor(
    %ctx: !hipsr.context, %src: tensor<1xi64, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
  %result = hipsr.copy_d2h(%ctx)
      ins(%src : tensor<1xi64, #hipsr.mem<device>>)
      outs(%init : tensor<1xi64, #hipsr.mem<host>>)
      : tensor<1xi64, #hipsr.mem<host>>
  return
}

// -----

// A memref destination is the buffer, so there is no result and no trailing
// result type. `{{$}}` is what pins that absence.
// CHECK-LABEL: func.func @copy_d2h_memref(
// CHECK-SAME:    %[[CTX:[^:]+]]: !hipsr.context,
// CHECK-SAME:    %[[SRC:[^:]+]]: memref<2x3xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[INIT:.+]] = memref.alloc() : memref<2x3xf16, #hipsr.mem<host>>
// CHECK-NEXT:    hipsr.copy_d2h(%[[CTX]]) ins(%[[SRC]] : memref<2x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : memref<2x3xf16, #hipsr.mem<host>>){{$}}
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @copy_d2h_memref(
    %ctx: !hipsr.context, %src: memref<2x3xf16, #hipsr.mem<device>>) {
  %init = memref.alloc() : memref<2x3xf16, #hipsr.mem<host>>
  hipsr.copy_d2h(%ctx)
      ins(%src : memref<2x3xf16, #hipsr.mem<device>>)
      outs(%init : memref<2x3xf16, #hipsr.mem<host>>)
  return
}

// -----

// A copy moves bytes; it does not convert them.
func.func @copy_d2h_element_mismatch(
    %ctx: !hipsr.context, %src: tensor<4xf32, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<4xf16, #hipsr.mem<host>>
  // expected-error@+1 {{failed to verify that all of {src, init} have same element type}}
  %result = hipsr.copy_d2h(%ctx)
      ins(%src : tensor<4xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4xf16, #hipsr.mem<host>>)
      : tensor<4xf16, #hipsr.mem<host>>
  return
}

// -----

// The destination holds exactly what the source holds.
func.func @copy_d2h_shape_mismatch(
    %ctx: !hipsr.context, %src: tensor<4xi64, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<8xi64, #hipsr.mem<host>>
  // expected-error@+1 {{failed to verify that all of {src, init} have same shape}}
  %result = hipsr.copy_d2h(%ctx)
      ins(%src : tensor<4xi64, #hipsr.mem<device>>)
      outs(%init : tensor<8xi64, #hipsr.mem<host>>)
      : tensor<8xi64, #hipsr.mem<host>>
  return
}

// -----

// There is nothing to transfer if the source is already on the host.
func.func @copy_d2h_host_source(%ctx: !hipsr.context) {
  %src = memref.alloc() : memref<4xi64, #hipsr.mem<host>>
  %init = memref.alloc() : memref<4xi64, #hipsr.mem<host>>
  // expected-error@+1 {{operand #1 must be ranked device tensor or device memref}}
  hipsr.copy_d2h(%ctx)
      ins(%src : memref<4xi64, #hipsr.mem<host>>)
      outs(%init : memref<4xi64, #hipsr.mem<host>>)
  return
}

// -----

// A device destination would leave the result unreadable on the host, which is
// the only reason to run the op.
func.func @copy_d2h_device_destination(
    %ctx: !hipsr.context, %src: memref<4xi64, #hipsr.mem<device>>) {
  %init = memref.alloc() : memref<4xi64, #hipsr.mem<device>>
  // expected-error@+1 {{operand #2 must be ranked host tensor or host memref}}
  hipsr.copy_d2h(%ctx)
      ins(%src : memref<4xi64, #hipsr.mem<device>>)
      outs(%init : memref<4xi64, #hipsr.mem<device>>)
  return
}
