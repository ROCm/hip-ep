/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- silu.cpp - SiLU activation (stub) ----------------------------------===//
//
// SiLU(x) = x * sigmoid(x).  No MIOpen/hipBLASLt equivalent.
// Stub implementation: zeroes the output buffer.
// TODO: implement as a simple HIP element-wise kernel.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>

extern "C" void hip_silu(void * /*handle*/, void * /*input*/, void *output) {
  (void)output;
  fprintf(stderr,
          "[hip_silu] stub called -- output not zeroed (no size info)\n");
}
