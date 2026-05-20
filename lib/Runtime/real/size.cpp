/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>

// ONNX Size runtime wrapper (dynamic-shape path only).
//
// Statically-shaped Size ops are folded to arith.constant in
// SizeConversion.cpp and never reach this symbol. This wrapper is only
// called when the input has at least one dynamic dim; the HipToLLVM
// lowering computes `num_elements = prod(input.shape)` as a runtime i64
// using compile-time constants for static dims and `MemRefDescriptor.
// sizes[]` for dynamic dims, then calls us to deposit that 8-byte value
// into the rank-0 i64 output buffer that lives on the GPU.
//
// Implementation: a single 8-byte hipMemcpyHostToDevice async copy on
// the runtime's stream. The src is a local int64_t on the C stack --
// pageable host memory + H2D in HIP semantics behaves synchronously
// (driver stages the bytes into a pinned scratch buffer before
// returning), so reading the stack slot after the call returns is
// safe.
int wrap_size(RuntimeState *state, void *output, int64_t num_elements) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_size\n");
    return -1;
  }
  if (!output) {
    fprintf(stderr, "Invalid output pointer in wrap_size\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_size(num_elements=%lld)\n",
                    (long long)num_elements);

  void *stream = hipdnn_ep_state_get_stream(state);
  // Re-use the existing H2D wrapper for consistent error handling.
  // 8 bytes total -- the cost of taking the address of the local is
  // dwarfed by the kernel-launch overhead of any neighbouring op.
  int64_t value = num_elements;
  return wrap_hipMemcpyH2D(output, &value, static_cast<int64_t>(sizeof(value)),
                           stream);
}
