// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-partition-pool-domains %s

// expected-error @+1 {{hipsr-partition-pool-domains only supports single-block functions in phase 1}}
func.func @multi_block() {
  return
^bb1:
  return
}

// -----

func.func @existing_domain(%arg: i32) -> i32 {
  // expected-error @+1 {{hipsr-partition-pool-domains does not support existing pool domains}}
  %0 = hipsr.pool_domain(%arg : i32) {
  ^bb0(%domain_arg: i32):
    hipsr.pool_domain_yield %domain_arg : i32
  } -> i32
  return %0 : i32
}
