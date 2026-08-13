// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics \
// RUN:   -hipsr-partition-pool-domains %s

// Multi-block functions are not supported.
// expected-error @+1 {{hipsr-partition-pool-domains only supports single-block functions}}
func.func @multi_block() {
  return
^bb1:
  return
}

// -----

// Function argument zero must be the HIPSR context.
// expected-error @+1 {{hipsr-partition-pool-domains requires function argument zero to be !hipsr.context}}
func.func @missing_context() {
  return
}

// -----

// A non-context first argument is not accepted.
// expected-error @+1 {{hipsr-partition-pool-domains requires function argument zero to be !hipsr.context}}
func.func @wrong_context(%arg: i32) {
  return
}
