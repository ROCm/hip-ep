//===- silu.cpp - SiLU activation (stub) ----------------------------------===//
//
// SiLU(x) = x * sigmoid(x).  No MIOpen/hipBLASLt equivalent.
// Stub implementation: zeroes the output buffer.
// TODO: implement as a simple HIP element-wise kernel.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdio>

extern "C" void hip_silu(void * /*handle*/, void * /*input*/, void *output) {
  (void)output;
  fprintf(stderr, "[hip_silu] stub called -- output not zeroed (no size info)\n");
}
