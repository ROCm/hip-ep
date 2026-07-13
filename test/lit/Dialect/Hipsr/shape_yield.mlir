// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Round-trips hipsr.shape_yield. It is a shape-region terminator, so it is
// placed inside a generic (unregistered) region-carrying op for the test.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --allow-unregistered-dialect %s \
// RUN:   | hip-mlir-opt --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @shape_yield_dims
func.func @shape_yield_dims() {
  "test.shape_region_holder"() ({
    %0 = arith.constant 2 : index
    %1 = arith.constant 512 : index
    // CHECK: hipsr.shape_yield %{{.+}}, %{{.+}}
    hipsr.shape_yield %0, %1
  }) : () -> ()
  return
}

// CHECK-LABEL: func.func @shape_yield_scalar
func.func @shape_yield_scalar() {
  "test.shape_region_holder"() ({
    // CHECK: hipsr.shape_yield
    hipsr.shape_yield
  }) : () -> ()
  return
}
