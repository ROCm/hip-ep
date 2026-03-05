/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace hip_compilation {

// C-compatible tensor descriptor passed across the DLL boundary.
// Memory is caller-owned CPU memory; the DLL copies H2D/D2H internally.
struct tensor_t {
  void *data;
  int64_t *shape;
  size_t rank;
  size_t element_size;
};

// Contiguous array of tensor_t descriptors (inputs or outputs).
struct span_t {
  tensor_t *data;
  size_t count;
};

// Lifecycle:
//   init_fn(&state);            // once
//   compute_fn(state, in, out); // N times
//   cleanup_fn(state);          // once
typedef int (*init_fn)(void **out_state);
typedef int (*compute_fn)(void *state, span_t *inputs, span_t *outputs);
typedef int (*cleanup_fn)(void *state);

} // namespace hip_compilation
